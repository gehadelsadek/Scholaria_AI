"""
qdrant_upsert.py — التخزين في الـ collection الموحّدة.

collection واحدة لكل المصادر والـ tenants. العزل بفلترة الـ payload
وقت البحث، مش بفصل الـ collections.

⚠️ نسخة واحدة بالحرف عند المسارين.
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ingestion.shared.embedding import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    VECTOR_SIZE,
)
from ingestion.shared.schema import ChunkPayload, make_point_id

COLLECTION = "rag_platform"

# محلي افتراضيًا. للسيرفر المشترك: QDRANT_URL=http://host:6333
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_data")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_client() -> QdrantClient:
    """
    QDRANT_URL موجود → سحابي/سيرفر مشترك (مطلوب للدمج بين المسارين)
    مش موجود → محلي على الديسك
    """
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient, collection: str = COLLECTION) -> None:
    """بتنشئ الـ collection لو مش موجودة."""
    existing = [c.name for c in client.get_collections().collections]

    if collection in existing:
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )
    print(f"[qdrant] اتعملت collection: {collection}")


def upsert_payloads(
    client: QdrantClient,
    payloads: list[ChunkPayload],
    dense_vectors: list,
    sparse_vectors: list,
    collection: str = COLLECTION,
) -> None:
    """
    بتخزن الـ chunks. الـ point id حتمي، فإعادة التشغيل بتحدّث مش بتكرّر.
    """
    points = []

    for payload, dense, sparse in zip(payloads, dense_vectors, sparse_vectors):
        points.append(
            PointStruct(
                id=make_point_id(
                    payload.tenant_id,
                    payload.source_type,
                    payload.content_id,
                    payload.content_version,
                    payload.chunk_index,
                ),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=payload.to_dict(),
            )
        )

    client.upsert(collection_name=collection, points=points)
    print(f"[qdrant] اتخزن {len(points)} chunk")


# =========================================================
#  الحذف
# =========================================================


def delete_content(
    client: QdrantClient,
    tenant_id: str,
    content_id: int,
    source_type: str,
    version: int | None = None,
    collection: str = COLLECTION,
) -> None:
    """بتحذف vectors محتوى معيّن — كل إصداراته أو إصدار محدد."""
    must = [
        FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id))),
        FieldCondition(key="content_id", match=MatchValue(value=content_id)),
        FieldCondition(key="source_type", match=MatchValue(value=source_type)),
    ]

    if version is not None:
        must.append(
            FieldCondition(key="content_version", match=MatchValue(value=version))
        )

    client.delete(collection_name=collection, points_selector=Filter(must=must))
    print(f"[qdrant] اتحذف محتوى {content_id} ({source_type})")


def delete_old_versions(
    client: QdrantClient,
    tenant_id: str,
    content_id: int,
    source_type: str,
    keep_version: int,
    collection: str = COLLECTION,
) -> None:
    """
    بتحذف كل إصدارات المحتوى ما عدا الحالي.
    البند 6.2: حذف الإصدار السابق بعد نجاح الجديد.
    """
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id))),
                FieldCondition(key="content_id", match=MatchValue(value=content_id)),
                FieldCondition(key="source_type", match=MatchValue(value=source_type)),
            ],
            must_not=[
                FieldCondition(
                    key="content_version", match=MatchValue(value=keep_version)
                )
            ],
        ),
    )


def count_content(
    client: QdrantClient,
    tenant_id: str,
    content_id: int,
    collection: str = COLLECTION,
) -> int:
    """بتعد الـ chunks الموجودة لمحتوى معيّن — للتحقق."""
    return client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id))),
                FieldCondition(key="content_id", match=MatchValue(value=content_id)),
            ]
        ),
    ).count

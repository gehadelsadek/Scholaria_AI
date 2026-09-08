"""
تخزين واسترجاع من Qdrant — حسب البند 4.1 و 6.2 في الوثيقة.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVectorParams,
    SparseVector,
)

COLLECTION = "scholaria_content"
VECTOR_SIZE = 1024


def get_client(path="qdrant_data"):
    """
    وضع محلي — تخزين على الديسك مباشرة بدون سيرفر.
    للإنتاج: QdrantClient(url="http://localhost:6333") حسب البند 4.1.
    """
    return QdrantClient(path=path)


def ensure_collection(client, vector_size=VECTOR_SIZE):
    """بتنشئ الـ collection بـ dense + sparse vectors."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        print(f"[qdrant] اتعملت collection: {COLLECTION} (hybrid)")

    return client


def upsert_chunks(client, records, dense_vectors, sparse_vectors):
    """بتخزن الـ chunks بـ dense + sparse vectors."""
    points = []

    for rec, dense, sparse in zip(records, dense_vectors, sparse_vectors):
        points.append(
            PointStruct(
                id=rec["vectorId"],
                vector={
                    "dense": dense.tolist(),
                    "sparse": SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    **rec["metadata"],
                    "text": rec["rawText"],
                    "chunkIndex": rec["chunkIndex"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[qdrant] اتخزن {len(points)} chunk (dense + sparse)")


def delete_content(client, content_id, version=None):
    """
    بتحذف vectors محتوى معين.
    البند 6.2: حذف الإصدار السابق بعد نجاح الجديد.
    """
    conditions = [FieldCondition(key="contentId", match=MatchValue(value=content_id))]

    if version is not None:
        conditions.append(
            FieldCondition(key="contentVersion", match=MatchValue(value=version))
        )

    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=conditions),
    )
    print(f"[qdrant] اتحذف محتوى {content_id}" + (f" v{version}" if version else ""))


def delete_old_versions(client, content_id, keep_version):
    """
    بتحذف كل إصدارات المحتوى ما عدا الإصدار الحالي.
    البند 6.2 بند 7: حذف الإصدار السابق بعد نجاح الجديد.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchExcept

    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(key="contentId", match=MatchValue(value=content_id)),
            ],
            must_not=[
                FieldCondition(
                    key="contentVersion", match=MatchValue(value=keep_version)
                ),
            ],
        ),
    )
    print(f"[qdrant] اتحذفت إصدارات قديمة لمحتوى {content_id} (فضل v{keep_version})")


def count_content(client, content_id):
    """بتعد الـ chunks الموجودة لمحتوى معين — للتحقق."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    result = client.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="contentId", match=MatchValue(value=content_id))]
        ),
    )
    return result.count

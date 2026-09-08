"""
qdrant_upsert.py
================

Qdrant client + collection موحّدة (unified collection) لكل أنواع
المحتوى، وupsert للـ chunks بعد ما يتحوّلوا لـ ChunkPayload (schema.py).

العزل بين tenants/courses بيتم بفلترة الـ payload وقت البحث
(tenant_id, course_id) - مش بعمل collection منفصلة لكل عميل، عشان
نتجنب انفجار عدد الـ collections مع نمو المنصة.

كان جزء من ده في Embed_upsert_qdrant.py القديم (اللي كان فيه كمان
تحميل الموديلات) - دلوقتي اتقسم: embedding.py للموديلات، والملف ده
للـ Qdrant بس.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    Modifier,
)

from .embedding import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .schema import ChunkPayload


QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "rag_platform"   # collection واحدة موحّدة لكل المصادر

UPSERT_BATCH_SIZE = 64


def get_qdrant_client(path: str = QDRANT_PATH) -> QdrantClient:
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if collection_name in [c.name for c in client.get_collections().collections]:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            # modifier=IDF عشان الـ scoring يقرب لـ BM25 الحقيقي بدل TF خام
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )


def make_point_id(tenant_id: str, source: str, start, end) -> str:
    """
    id ثابت (deterministic): إعادة تشغيل نفس المصدر بتعمل update لنفس
    النقطة بدل ما تكررها. مربوط بالـ tenant_id عشان مفيش تصادم لو
    tenant تاني عنده مصدر بنفس الاسم بالظبط.
    """
    key = f"{tenant_id}:{source}:{start}:{end}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def build_point(payload: ChunkPayload, dense_vector: list, sparse_vector) -> PointStruct:
    return PointStruct(
        id=make_point_id(payload.tenant_id, payload.source, payload.start_s, payload.end_s),
        vector={
            DENSE_VECTOR_NAME: dense_vector,
            SPARSE_VECTOR_NAME: SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            ),
        },
        payload=payload.to_dict(),
    )


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    collection_name: str = COLLECTION_NAME,
) -> None:
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=points[i : i + UPSERT_BATCH_SIZE])


def upsert_chunks(
    client: QdrantClient,
    payloads: list[ChunkPayload],
    dense_vectors: list,
    sparse_vectors: list,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    الدالة الرئيسية اللي أي pipeline (فيديو أو ملفات) بيستدعيها بعد ما
    يجهز الـ payloads (schema.py) والـ vectors (embedding.py) بتاعته.
    """
    ensure_collection(client, collection_name=collection_name)
    points = [
        build_point(p, dv, sv)
        for p, dv, sv in zip(payloads, dense_vectors, sparse_vectors)
    ]
    upsert_points(client, points, collection_name=collection_name)
    return len(points)

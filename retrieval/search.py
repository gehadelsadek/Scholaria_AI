"""
search.py — البحث الهجين في الـ collection الموحّدة.

dense (المعنى) + BM25 (الكلمات المفتاحية)، مدموجين بـ RRF.
الفلتر بيتطبق على المسارين — العزل مضمون في الاتنين.
"""

from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from ingestion.shared.embedding import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    encode_dense_query,
    encode_sparse_query,
)
from ingestion.shared.qdrant_upsert import COLLECTION
from retrieval.filtering import build_access_filter


def hybrid_search(
    client,
    query: str,
    tenant_id: str,
    course_id: str | None = None,
    source_type: str | None = None,
    top_k: int = 10,
    prefetch_limit: int = 50,
    collection: str = COLLECTION,
):
    """
    بحث هجين.

    source_type=None    → الملفات والفيديو مع بعض (الافتراضي)
    source_type="file"  → الملفات بس
    source_type="video" → الفيديو بس
    """
    access = build_access_filter(tenant_id, course_id, source_type)

    dense = encode_dense_query(query)
    sparse = encode_sparse_query(query)

    return client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                filter=access,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using=SPARSE_VECTOR_NAME,
                filter=access,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points


def dense_search(
    client,
    query: str,
    tenant_id: str,
    course_id: str | None = None,
    source_type: str | None = None,
    top_k: int = 10,
    collection: str = COLLECTION,
):
    """بحث dense فقط — للمقارنة مع الهجين."""
    return client.query_points(
        collection_name=collection,
        query=encode_dense_query(query),
        using=DENSE_VECTOR_NAME,
        query_filter=build_access_filter(tenant_id, course_id, source_type),
        limit=top_k,
    ).points

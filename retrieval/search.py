"""
البحث في الـ unified collection — البند 6.5.
Hybrid: dense (معنى) + sparse (كلمات) مدموجين بـ RRF.
"""

from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

from ingestion.shared.qdrant_upsert import COLLECTION
from retrieval.filtering import build_access_filter


def search(client, query_vector, tenant_id, course_id, top_k=20):
    """بحث dense فقط — للمقارنة."""
    return client.query_points(
        collection_name=COLLECTION,
        query=query_vector.tolist(),
        using="dense",
        query_filter=build_access_filter(tenant_id, course_id),
        limit=top_k,
    ).points


def hybrid_search(
    client, dense_vec, sparse_vec, tenant_id, course_id, top_k=20, prefetch_limit=50
):
    """
    بحث hybrid بدمج RRF.
    الفلتر بيتطبق على المسارين — العزل مضمون في الاتنين.
    """
    access = build_access_filter(tenant_id, course_id)

    return client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec.tolist(),
                using="dense",
                filter=access,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                filter=access,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points

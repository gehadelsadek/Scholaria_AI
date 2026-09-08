"""
بناء فلاتر العزل — البند 6.5 و 15.
الفلترة إلزامية قبل أي بحث: TenantId + CourseId + Published.
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue


def build_access_filter(tenant_id, course_id, published=True):
    """
    فلتر العزل الأساسي.
    البند 18.2: صفر تسرب بين المقررات والمؤسسات.
    """
    conditions = [
        FieldCondition(key="tenantId", match=MatchValue(value=tenant_id)),
        FieldCondition(key="courseId", match=MatchValue(value=course_id)),
    ]

    if published is not None:
        conditions.append(
            FieldCondition(key="published", match=MatchValue(value=published))
        )

    return Filter(must=conditions)

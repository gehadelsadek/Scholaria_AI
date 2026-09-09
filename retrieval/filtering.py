"""
filtering.py — فلاتر العزل واختيار المصدر.

العزل (tenant + course) إلزامي في كل بحث — البند 6.5 و 15.
اختيار المصدر (ملفات / فيديو / الاتنين) اختياري.
"""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from ingestion.shared.schema import SourceType


def build_access_filter(
    tenant_id: str,
    course_id: str | None = None,
    source_type: str | None = None,
    published: bool | None = True,
) -> Filter:
    """
    بتبني فلتر البحث.

    tenant_id   إلزامي — البند 18.2: صفر تسرب بين المؤسسات
    course_id   إلزامي عمليًا — سيبه None بس لو بتدوري عبر كل مقررات المؤسسة
    source_type None = الملفات والفيديو مع بعض
                "file" = الملفات بس
                "video" = الفيديو بس
    """
    if not tenant_id:
        raise ValueError("tenant_id إلزامي — البحث من غيره بيكسر العزل")

    must = [FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id)))]

    if course_id is not None:
        must.append(
            FieldCondition(key="course_id", match=MatchValue(value=str(course_id)))
        )

    if source_type is not None:
        if source_type not in SourceType.ALL:
            raise ValueError(f"source_type غير معروف: {source_type!r}")
        must.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )

    if published is not None:
        must.append(FieldCondition(key="published", match=MatchValue(value=published)))

    return Filter(must=must)

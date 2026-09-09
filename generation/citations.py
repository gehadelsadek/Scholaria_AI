"""
citations.py — تنسيق المراجع حسب نوع المصدر.

الملف → "صفحة 54"
الفيديو → "12:40 – 14:05"
"""

from ingestion.shared.schema import SourceType


def format_timestamp(seconds) -> str:
    """ثواني → mm:ss أو hh:mm:ss."""
    if seconds is None:
        return "?"

    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_reference(payload: dict) -> str:
    """نص المرجع المعروض للطالب."""
    if payload.get("source_type") == SourceType.VIDEO:
        start = format_timestamp(payload.get("start_s"))
        end = format_timestamp(payload.get("end_s"))
        return f"{start} – {end}"

    page = payload.get("page_number")
    return f"صفحة {page}" if page is not None else "?"


def build_citation(payload: dict, index: int) -> dict:
    """
    مرجع كامل — البند 6.6.
    الـ frontend بيستخدم source_type عشان يعرف يفتح إيه.
    """
    is_video = payload.get("source_type") == SourceType.VIDEO

    return {
        "index": index,
        "source": payload.get("source"),
        "source_type": payload.get("source_type"),
        "reference": format_reference(payload),
        "page_number": payload.get("page_number"),
        "start_s": payload.get("start_s"),
        "end_s": payload.get("end_s"),
        "deep_link": payload.get("deep_link") if is_video else None,
        "content_id": payload.get("content_id"),
    }

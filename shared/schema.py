"""
schema.py
=========

تعريف موحّد لشكل الـ payload المخزّن في Qdrant لكل أنواع المحتوى
(فيديو أو ملفات)، عشان:
    - الـ retrieval يقدر يفلتر على tenant_id / course_id / source_type
      من غير ما يهتم المصدر جاي منين (video pipeline ولا files pipeline).
    - كل pipeline (video_chunking.py دلوقتي، وfile_chunking.py لاحقًا)
      يعدّي من هنا قبل ما يوصل لـ qdrant_upsert.py، فمفيش payload
      "يهرب" من غير tenant_id/course_id.

القاعدة: collection واحدة موحّدة لكل المصادر (مش collection لكل tenant
أو لكل نوع محتوى) - العزل بيتم عن طريق فلترة الـ payload وقت البحث
(شوف retrieval/filtering.py).
"""

from dataclasses import dataclass, asdict
from typing import Optional


# =========================================================
# Source types المدعومة
# =========================================================

class SourceType:
    VIDEO = "video"
    FILE = "file"


# =========================================================
# Payload schema
# =========================================================

@dataclass
class ChunkPayload:
    # --- multi-tenancy (إلزامي لكل chunk) ---
    tenant_id: str
    course_id: str

    # --- محتوى ---
    text: str          # النص المعروض للـ LLM (مع الهيدر: اسم الفيديو/الملف + التوقيت)
    raw_text: str       # النص الخام من غير هيدر - ده اللي بيتبعت للـ reranker
    source: str         # اسم الفيديو أو الملف
    source_type: str    # SourceType.VIDEO أو SourceType.FILE

    # --- خاص بالفيديو (فاضية لو المصدر ملف) ---
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    deep_link: Optional[str] = None

    # --- خاص بالملفات (فاضية لو المصدر فيديو) ---
    page_number: Optional[int] = None

    def validate(self) -> "ChunkPayload":
        if not self.tenant_id or not self.course_id:
            raise ValueError(
                "tenant_id و course_id إلزاميين لكل chunk قبل الرفع لـ Qdrant"
            )
        if not self.text or not self.text.strip():
            raise ValueError("text مينفعش يكون فاضي")
        if self.source_type not in (SourceType.VIDEO, SourceType.FILE):
            raise ValueError(f"source_type غير معروف: {self.source_type!r}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def build_video_payload(chunk: dict, tenant_id: str, course_id: str) -> ChunkPayload:
    """
    بيحوّل الـ dict الناتج من video_chunking.chunk_video_transcript()
    لـ ChunkPayload موحّد، وبيتحقق منه على طول (fail fast لو ناقص حاجة).
    """
    return ChunkPayload(
        tenant_id=tenant_id,
        course_id=course_id,
        text=chunk.get("text", ""),
        raw_text=chunk.get("raw_text") or chunk.get("text", ""),
        source=chunk.get("source", "?"),
        source_type=SourceType.VIDEO,
        start_s=chunk.get("start_s"),
        end_s=chunk.get("end_s"),
        deep_link=chunk.get("deep_link"),
    ).validate()

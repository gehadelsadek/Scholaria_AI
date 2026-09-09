"""
schema.py — الـ payload الموحّد لكل المصادر (ملفات + فيديو).

collection واحدة لكل المصادر، والعزل بفلترة الـ payload وقت البحث
(retrieval/filtering.py) — مش بـ collection لكل tenant.

⚠️ نسخة واحدة بالحرف عند المسارين. أي اختلاف في أسماء الحقول يكسر الفلترة.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional


class SourceType:
    FILE = "file"
    VIDEO = "video"

    ALL = (FILE, VIDEO)


# =========================================================
#  الـ payload
# =========================================================


@dataclass
class ChunkPayload:
    # --- multi-tenancy — إلزامي لكل chunk ---
    tenant_id: str
    course_id: str

    # --- المحتوى ---
    text: str  # المعروض للـ LLM (مع الهيدر)
    raw_text: str  # الخام — للـ reranker والعرض للمستخدم
    source: str  # اسم الملف أو الفيديو
    source_type: str  # SourceType.FILE أو VIDEO

    # --- الهوية والإصدار ---
    content_id: int
    chunk_index: int
    content_version: int = 1
    published: bool = True
    language: str = "ar"
    checksum: str = ""

    # --- خاص بالفيديو ---
    video_id: Optional[str] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    deep_link: Optional[str] = None

    # --- خاص بالملفات ---
    file_id: Optional[int] = None
    page_number: Optional[int] = None
    topic_id: Optional[int] = None
    topic_pages: Optional[str] = None  # JSON string — Qdrant مبيقبلش lists

    def validate(self) -> "ChunkPayload":
        if not self.tenant_id or self.course_id in (None, ""):
            raise ValueError("tenant_id و course_id إلزاميين قبل الرفع لـ Qdrant")

        if not self.text or not self.text.strip():
            raise ValueError("text مينفعش يكون فاضي")

        if self.source_type not in SourceType.ALL:
            raise ValueError(f"source_type غير معروف: {self.source_type!r}")

        # لازم مرجع قابل للفتح (البند 7.4)
        if self.page_number is None and self.start_s is None:
            raise ValueError("مفيش مرجع (صفحة أو توقيت)")

        return self

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================
#  معرّف النقطة
# =========================================================


def make_point_id(
    tenant_id, source_type, content_id, content_version, chunk_index
) -> str:
    """
    معرّف حتمي — نفس المدخلات تدي نفس الـ id دايمًا،
    فإعادة التشغيل بتحدّث بدل ما تكرّر (البند 6.2).

    source_type جزء من المفتاح، فمفيش تصادم بين الملفات والفيديو
    حتى لو الـ content_id اتكرر.
    """
    key = f"{tenant_id}:{source_type}:{content_id}:{content_version}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# =========================================================
#  أدوات مساعدة
# =========================================================


def file_checksum(path) -> str:
    """بصمة ملف — البند 16.2: مفيش إعادة embedding لو مااتغيرش."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def text_checksum(text: str) -> str:
    """بصمة نص — للمصادر اللي مالهاش ملف (زي الترانسكربت)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_language(text: str, threshold: float = 0.3) -> str:
    """بتحدد اللغة الغالبة من نسبة الحروف العربية."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "ar"
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic / len(letters) >= threshold else "en"


# =========================================================
#  البناء من مخرجات كل pipeline
# =========================================================


def build_file_payload(chunk: dict, doc_info: dict, chunk_index: int) -> ChunkPayload:
    """
    بيحوّل chunk من file_chunking لـ payload موحّد.
    chunk المتوقع: text, raw_text, source, primary_page, pages, topic_id, topic_pages
    """
    raw = chunk.get("raw_text") or chunk.get("rawText") or chunk["text"]
    pages = chunk.get("pages") or []
    page = chunk.get("primary_page") or (pages[0] if pages else None)

    return ChunkPayload(
        tenant_id=str(doc_info["tenant_id"]),
        course_id=str(doc_info["course_id"]),
        text=chunk["text"],
        raw_text=raw,
        source=chunk.get("source") or doc_info.get("title", "?"),
        source_type=SourceType.FILE,
        content_id=doc_info["content_id"],
        chunk_index=chunk_index,
        content_version=doc_info.get("content_version", 1),
        published=doc_info.get("published", True),
        language=doc_info.get("language") or detect_language(raw),
        checksum=doc_info.get("checksum", ""),
        file_id=doc_info.get("file_id"),
        page_number=page,
        topic_id=chunk.get("topic_id"),
        topic_pages=json.dumps(chunk.get("topic_pages", [])),
    ).validate()


def build_video_payload(chunk: dict, doc_info: dict, chunk_index: int) -> ChunkPayload:
    """
    بيحوّل chunk من video_chunking لـ payload موحّد.
    chunk المتوقع: text, raw_text, source, start_s, end_s, deep_link
    """
    raw = chunk.get("raw_text") or chunk.get("rawText") or chunk["text"]

    return ChunkPayload(
        tenant_id=str(doc_info["tenant_id"]),
        course_id=str(doc_info["course_id"]),
        text=chunk["text"],
        raw_text=raw,
        source=chunk.get("source") or doc_info.get("title", "?"),
        source_type=SourceType.VIDEO,
        content_id=doc_info["content_id"],
        chunk_index=chunk_index,
        content_version=doc_info.get("content_version", 1),
        published=doc_info.get("published", True),
        language=doc_info.get("language") or detect_language(raw),
        checksum=doc_info.get("checksum", ""),
        video_id=doc_info.get("video_id"),
        start_s=chunk.get("start_s"),
        end_s=chunk.get("end_s"),
        deep_link=chunk.get("deep_link"),
    ).validate()

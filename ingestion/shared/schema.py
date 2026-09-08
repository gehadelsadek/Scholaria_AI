"""
الشكل الموحد للـ chunk — حسب البند 6.4 في وثيقة المواصفات الفنية.
"""

import hashlib
import json
import os
import uuid

# القيم الافتراضية لحد ما الربط بالـ Backend يحصل
DEFAULTS = {
    "tenantId": "dev-tenant",
    "courseId": 0,
    "sectionId": None,
    "contentType": "Lecture",
    "language": "ar",
    "published": True,
}

CONTENT_TYPES = ["Lecture", "Article", "Quiz", "Flashcard", "Assignment"]
SOURCE_TYPES = ["pdf", "video"]


def make_vector_id(content_id, version, chunk_index):
    """معرف ثابت بصيغة UUID — Qdrant بيقبل UUID أو int بس."""
    raw = f"{content_id}:{version}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def file_checksum(path):
    """
    بصمة الملف — لو مااتغيرتش، مفيش داعي لإعادة الفهرسة.
    البند 16.2: عدم إعادة Embedding عند ثبات Checksum.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def detect_language(text, threshold=0.3):
    """بتحدد اللغة الغالبة من نسبة الحروف العربية."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "ar"
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic / len(letters) >= threshold else "en"


def _to_seconds(value):
    """ثواني صحيحة — الـ deepLink بيستخدم ?t=760."""
    return int(round(value)) if value is not None else None


def build_metadata(chunk, doc_info, chunk_index):
    """
    بتبني الـ metadata الإلزامية لكل chunk حسب البند 6.4.
    """
    content_id = doc_info["contentId"]
    version = doc_info.get("contentVersion", 1)
    text = chunk.get("raw_text") or chunk.get("rawText") or chunk["text"]

    meta = {
        # ===== العزل والصلاحيات (البند 6.5 + 15) =====
        "tenantId": doc_info.get("tenantId", DEFAULTS["tenantId"]),
        "courseId": doc_info.get("courseId", DEFAULTS["courseId"]),
        "sectionId": doc_info.get("sectionId", DEFAULTS["sectionId"]),
        "contentId": content_id,
        "contentType": doc_info.get("contentType", DEFAULTS["contentType"]),
        "sourceType": doc_info.get("sourceType", "pdf"),
        # ===== المرجع (citation) =====
        "title": doc_info.get("title", ""),
        "fileId": doc_info.get("fileId"),
        "pageNumber": chunk.get("primary_page")
        or (chunk["pages"][0] if chunk.get("pages") else None),
        "topicId": chunk.get("topic_id"),
        "topicPages": json.dumps(chunk.get("topic_pages", [])),
        # ===== حقول الفيديو — فاضية في الـ PDF =====
        "videoId": doc_info.get("videoId"),
        "videoStartTime": _to_seconds(chunk.get("start_s")),
        "videoEndTime": _to_seconds(chunk.get("end_s")),
        # ===== الإصدار والحالة =====
        "language": doc_info.get("language") or detect_language(text),
        "contentVersion": version,
        "published": doc_info.get("published", DEFAULTS["published"]),
        "checksum": doc_info.get("checksum", ""),
    }

    return {
        "vectorId": make_vector_id(content_id, version, chunk_index),
        "chunkIndex": chunk_index,
        "text": chunk["text"],  # بالسياق — للـ embedding
        "rawText": text,  # الأصلي — للعرض
        "metadata": meta,
    }


def validate(record):
    """بتتأكد إن الحقول الإلزامية موجودة قبل الفهرسة."""
    required = [
        "tenantId",
        "courseId",
        "contentId",
        "contentType",
        "sourceType",
        "language",
        "contentVersion",
        "published",
    ]

    meta = record["metadata"]
    missing = [f for f in required if meta.get(f) is None]

    if missing:
        raise ValueError(f"حقول ناقصة: {missing}")

    if meta["contentType"] not in CONTENT_TYPES:
        raise ValueError(f"contentType غير صالح: {meta['contentType']}")

    if meta["sourceType"] not in SOURCE_TYPES:
        raise ValueError(f"sourceType غير صالح: {meta['sourceType']}")

    # لازم مرجع للـ citation: صفحة أو توقيت
    if meta.get("pageNumber") is None and meta.get("videoStartTime") is None:
        raise ValueError("مفيش مرجع (صفحة أو توقيت)")

    return True

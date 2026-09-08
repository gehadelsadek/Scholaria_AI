"""
ingestion/files/file_pipeline.py
تنسيق مسار الملفات كامل: استخراج → تنظيف → تقطيع → metadata → embedding → تخزين.
"""

import hashlib
import os
import re

from ingestion.files.pdf_extraction import extract_pdf
from ingestion.files.file_chunking import clean_pages, chunk_semantic, enrich_chunks
from ingestion.files.registry import (
    load_registry,
    save_registry,
    needs_indexing,
    register,
)
from ingestion.shared.schema import build_metadata, validate, file_checksum
from ingestion.shared.storage import save_chunks
from ingestion.shared.embedding import embed_documents, embed_sparse_documents
from ingestion.shared.qdrant_upsert import (
    get_client,
    ensure_collection,
    upsert_chunks,
    COLLECTION,
)

DATA_DIR = "data"

# الإعدادات دي بتيجي من الـ Backend في النظام الحقيقي (البند 6.2)
TENANT_ID = "dev-tenant"
COURSE_ID = 1
START_CONTENT_ID = 1000


# ============================================================
#  إزالة التكرار — بتشتغل بعد تجميع كل الملفات
# ============================================================


def _normalize_for_hash(text):
    """توحيد النص للمقارنة (مسافات وحالة الأحرف)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def remove_duplicates(records):
    """
    بتشيل الـ chunks المتطابقة عبر كل الملفات.
    بتحفظ مواضع النسخ المكررة بدل ما ترميها — مفيد للـ citation.
    """
    seen = {}
    unique = []

    for r in records:
        h = hashlib.md5(_normalize_for_hash(r["text"]).encode("utf-8")).hexdigest()

        if h in seen:
            meta = r.get("metadata", {})
            seen[h].setdefault("duplicateOf", []).append(
                {
                    "contentId": meta.get("contentId"),
                    "pageNumber": meta.get("pageNumber"),
                }
            )
            continue

        seen[h] = r
        unique.append(r)

    removed = len(records) - len(unique)
    print(f"[dedup] اتشال {removed} chunk مكرر ({len(unique)} فاضلين)")
    return unique


# ============================================================
#  معالجة ملف واحد
# ============================================================


def process_pdf(path, registry, content_id):
    """
    بتعالج ملف واحد: استخراج → تنظيف → تقطيع → metadata.
    بترجع records جاهزة للـ embedding.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    checksum = file_checksum(path)

    should_index, version = needs_indexing(path, checksum, registry)

    if not should_index:
        print(f"⏭️  {name} — مااتغيرش، تم التخطي")
        return [], registry

    print(f"\n{'='*60}")
    print(f"📄 {name}  (contentId={content_id}, v{version})")
    print(f"{'='*60}")

    pages = extract_pdf(path)
    cleaned = clean_pages(pages)
    chunks = chunk_semantic(cleaned, source=name)
    chunks = enrich_chunks(chunks)

    doc_info = {
        "contentId": content_id,
        "contentVersion": version,
        "sourceType": "pdf",
        "title": name,
        "checksum": checksum,
        "contentType": "Lecture",
        "tenantId": TENANT_ID,
        "courseId": COURSE_ID,
    }

    records = []
    for i, c in enumerate(chunks):
        rec = build_metadata(c, doc_info, i)
        validate(rec)
        records.append(rec)

    print(f"chunks: {len(records)}")
    registry = register(path, checksum, version, content_id, len(records), registry)

    return records, registry


# ============================================================
#  تنسيق المسار
# ============================================================


def extract_all(data_dir=DATA_DIR):
    """
    بتمشي على كل ملفات PDF وترجع الـ records المحدّثة.
    الملفات اللي مااتغيرتش بتتخطى (البند 16.2).
    """
    registry = load_registry()

    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf") and not f.startswith("._")
    ]

    print(f"لقيت {len(files)} ملف PDF")

    # نحجز الـ ids مقدمًا عشان فشل ملف ميكسرش التسلسل
    used_ids = {e["contentId"] for e in registry.values()}
    next_id = max(used_ids) + 1 if used_ids else START_CONTENT_ID

    all_records = []
    for path in files:
        entry = registry.get(os.path.basename(path))
        if entry:
            cid = entry["contentId"]
        else:
            cid = next_id
            next_id += 1

        try:
            recs, registry = process_pdf(path, registry, content_id=cid)
            all_records.extend(recs)
        except Exception as e:
            print(f"❌ فشل {os.path.basename(path)}: {e}")

    save_registry(registry)
    return remove_duplicates(all_records) if all_records else []


def index_records(records):
    """
    بتعمل embedding للـ records وتخزنها في Qdrant.
    dense للمعنى + sparse للكلمات المفتاحية (البند 6.5).
    """
    texts = [r["text"] for r in records]
    raw_texts = [r.get("rawText", r["text"]) for r in records]

    print("\n[embed] بيولّد dense vectors...")
    dense_vectors = embed_documents(texts)

    print("[embed] بيولّد sparse vectors...")
    sparse_vectors = embed_sparse_documents(raw_texts)

    client = get_client()
    try:
        ensure_collection(client, vector_size=dense_vectors.shape[1])
        upsert_chunks(client, records, dense_vectors, sparse_vectors)

        info = client.get_collection(COLLECTION)
        print(f"✅ الإجمالي في الـ DB: {info.points_count} نقطة")
    finally:
        client.close()


def main():
    records = extract_all()

    if not records:
        print("\n✅ كل الملفات محدّثة — مفيش شغل جديد")
        return

    print(f"\n✅ {len(records)} chunk جديد/محدّث")

    # نحفظ نسخة للمراجعة البشرية
    save_chunks(records, name="all_chunks")

    index_records(records)


if __name__ == "__main__":
    main()

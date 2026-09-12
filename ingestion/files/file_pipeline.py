"""
ingestion/files/file_pipeline.py
تنسيق مسار الملفات كامل: استخراج → تنظيف → تقطيع → payload → embedding → تخزين.
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
from ingestion.shared.schema import build_file_payload, file_checksum
from ingestion.shared.embedding import encode_dense, encode_sparse
from ingestion.shared.qdrant_upsert import (
    get_client,
    ensure_collection,
    upsert_payloads,
    COLLECTION,
)

DATA_DIR = "data"

# الإعدادات دي بتيجي من الـ Backend في النظام الحقيقي (البند 6.2)
TENANT_ID = "acme"
COURSE_ID = "agile-101"  # string — موحّد مع مسار الفيديو
START_CONTENT_ID = 1000


# ============================================================
#  إزالة التكرار — بتشتغل بعد تجميع كل الملفات
# ============================================================


def _normalize_for_hash(text):
    """توحيد النص للمقارنة (مسافات وحالة الأحرف)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def remove_duplicates(payloads):
    """
    بتشيل الـ chunks المتطابقة عبر كل الملفات.
    بتشتغل على ChunkPayload objects مش dicts.
    """
    seen = set()
    unique = []

    for p in payloads:
        h = hashlib.md5(_normalize_for_hash(p.text).encode("utf-8")).hexdigest()

        if h in seen:
            continue

        seen.add(h)
        unique.append(p)

    removed = len(payloads) - len(unique)
    print(f"[dedup] اتشال {removed} chunk مكرر ({len(unique)} فاضلين)")
    return unique


# ============================================================
#  معالجة ملف واحد
# ============================================================


def process_pdf(path, registry, content_id):
    """
    بتعالج ملف واحد: استخراج → تنظيف → تقطيع → payload.
    بترجع ChunkPayload objects جاهزة للـ embedding.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    checksum = file_checksum(path)

    should_index, version = needs_indexing(path, checksum, registry)

    if not should_index:
        print(f"⏭️  {name} — مااتغيرش، تم التخطي")
        return [], registry

    print(f"\n{'='*60}")
    print(f"📄 {name}  (content_id={content_id}, v{version})")
    print(f"{'='*60}")

    pages = extract_pdf(path)
    cleaned = clean_pages(pages)
    chunks = chunk_semantic(cleaned, source=name)
    chunks = enrich_chunks(chunks)

    doc_info = {
        "tenant_id": TENANT_ID,
        "course_id": COURSE_ID,
        "content_id": content_id,
        "content_version": version,
        "title": name,
        "checksum": checksum,
    }

    # build_file_payload بتعمل البناء والتحقق مع بعض
    payloads = [build_file_payload(c, doc_info, i) for i, c in enumerate(chunks)]

    print(f"chunks: {len(payloads)}")
    registry = register(path, checksum, version, content_id, len(payloads), registry)

    return payloads, registry


# ============================================================
#  تنسيق المسار
# ============================================================


def extract_all(data_dir=DATA_DIR):
    """
    بتمشي على كل ملفات PDF وترجع الـ payloads المحدّثة.
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

    all_payloads = []
    for path in files:
        entry = registry.get(os.path.basename(path))
        if entry:
            cid = entry["contentId"]
        else:
            cid = next_id
            next_id += 1

        try:
            payloads, registry = process_pdf(path, registry, content_id=cid)
            all_payloads.extend(payloads)
        except Exception as e:
            print(f"❌ فشل {os.path.basename(path)}: {e}")

    save_registry(registry)
    return remove_duplicates(all_payloads) if all_payloads else []


def index_payloads(payloads):
    """
    بتعمل embedding وتخزن في الـ collection الموحّدة.
    dense للمعنى + sparse للكلمات المفتاحية (البند 6.5).
    """
    print("\n[embed] بيولّد dense vectors...")
    dense = encode_dense([p.text for p in payloads])

    print("[embed] بيولّد sparse vectors...")
    sparse = encode_sparse([p.raw_text for p in payloads])

    client = get_client()
    try:
        ensure_collection(client)
        upsert_payloads(client, payloads, dense, sparse)

        info = client.get_collection(COLLECTION)
        print(f"✅ الإجمالي في الـ DB: {info.points_count} نقطة")
    finally:
        client.close()


def main():
    payloads = extract_all()

    if not payloads:
        print("\n✅ كل الملفات محدّثة — مفيش شغل جديد")
        return

    print(f"\n✅ {len(payloads)} chunk جديد/محدّث")
    index_payloads(payloads)


if __name__ == "__main__":
    main()

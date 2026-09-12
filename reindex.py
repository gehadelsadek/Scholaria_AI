"""
reindex.py — رفع الـ chunks المحفوظة لـ Qdrant من غير إعادة استخراج.

بيقرا output/all_chunks.json (الـ schema القديم المتداخل) ويحوّله
للـ payload الموحّد، وبعدها embedding ورفع على دفعات.

الـ vectors بتتحفظ محليًا، فلو الرفع فشل مش هتعيدي الـ embedding.

Usage:
    python reindex.py              # عادي
    python reindex.py --fresh      # يتجاهل الـ vectors المحفوظة ويعيد حسابها
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from ingestion.shared.embedding import encode_dense, encode_sparse
from ingestion.shared.qdrant_upsert import (
    COLLECTION,
    ensure_collection,
    get_client,
    upsert_payloads,
)
from ingestion.shared.schema import ChunkPayload, SourceType

# لازم يطابقوا اللي في video_pipeline.py — وإلا الفلترة بتفصل المصدرين
TENANT_ID = "acme"
COURSE_ID = "agile-101"

CHUNKS_FILE = Path("output/all_chunks.json")
DENSE_CACHE = Path("output/dense.npy")
SPARSE_CACHE = Path("output/sparse.pkl")

# الرفع على دفعات — الطلب الكبير بيقطع الاتصال بالسحابة
UPLOAD_BATCH = 64


# =========================================================
#  التحويل من الـ schema القديم
# =========================================================


def convert(record: dict) -> ChunkPayload:
    """
    بتحوّل سجل من الشكل القديم (metadata متداخلة + camelCase)
    للـ payload الموحّد المسطّح.
    """
    meta = record.get("metadata", {})

    return ChunkPayload(
        tenant_id=TENANT_ID,
        course_id=COURSE_ID,
        text=record["text"],
        raw_text=record.get("rawText") or record["text"],
        source=meta.get("title") or "?",
        source_type=SourceType.FILE,
        content_id=meta.get("contentId", 0),
        chunk_index=record.get("chunkIndex", 0),
        content_version=meta.get("contentVersion", 1),
        published=meta.get("published", True),
        language=meta.get("language", "ar"),
        checksum=meta.get("checksum", ""),
        file_id=meta.get("fileId"),
        page_number=meta.get("pageNumber"),
        topic_id=meta.get("topicId"),
        topic_pages=meta.get("topicPages"),
    ).validate()


def load_payloads() -> list[ChunkPayload]:
    """بتقرا الملف وتحوّل السجلات، وتتخطى اللي مالوش مرجع صالح."""
    if not CHUNKS_FILE.exists():
        sys.exit(f"❌ {CHUNKS_FILE} مش موجود — شغّلي file_pipeline الأول")

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    payloads, skipped = [], 0

    for record in data:
        try:
            payloads.append(convert(record))
        except (ValueError, KeyError):
            skipped += 1

    if skipped:
        print(f"⚠️  اتخطى {skipped} سجل (مرجع ناقص أو حقول مفقودة)")

    if not payloads:
        sys.exit("❌ مفيش سجلات صالحة")

    print(f"📄 {len(payloads)} chunk جاهزة")
    return payloads


# =========================================================
#  الـ vectors — مع كاش عشان الفشل ميعيدش الحساب
# =========================================================


def build_vectors(payloads, use_cache=True):
    """
    بتولّد الـ vectors أو تقرا المحفوظة.
    الكاش بيتلغى لو عدد الـ chunks اتغير.
    """
    cached = use_cache and DENSE_CACHE.exists() and SPARSE_CACHE.exists()

    if cached:
        dense = np.load(DENSE_CACHE)
        if len(dense) == len(payloads):
            print("[embed] بيقرا الـ vectors المحفوظة...")
            with SPARSE_CACHE.open("rb") as f:
                return dense.tolist(), pickle.load(f)
        print("⚠️  الكاش قديم (عدد مختلف) — هيعيد الحساب")

    print("[embed] بيولّد dense vectors...")
    dense = encode_dense([p.text for p in payloads])

    print("[embed] بيولّد sparse vectors...")
    sparse = encode_sparse([p.raw_text for p in payloads])

    DENSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(DENSE_CACHE, np.array(dense))
    with SPARSE_CACHE.open("wb") as f:
        pickle.dump(sparse, f)
    print("[embed] اتحفظوا للاستخدام الجاي")

    return dense, sparse


# =========================================================
#  الرفع
# =========================================================


def upload(payloads, dense, sparse):
    """بترفع على دفعات — الطلب الكبير بيقطع الاتصال بالسحابة."""
    client = get_client()

    try:
        ensure_collection(client)

        total = len(payloads)
        for start in range(0, total, UPLOAD_BATCH):
            end = min(start + UPLOAD_BATCH, total)
            upsert_payloads(
                client,
                payloads[start:end],
                dense[start:end],
                sparse[start:end],
            )
            print(f"   ↑ {end}/{total}")

        print(
            f"\n✅ الإجمالي في الـ DB: {client.get_collection(COLLECTION).points_count}"
        )
    finally:
        client.close()


def main():
    use_cache = "--fresh" not in sys.argv

    payloads = load_payloads()
    dense, sparse = build_vectors(payloads, use_cache=use_cache)
    upload(payloads, dense, sparse)


if __name__ == "__main__":
    main()

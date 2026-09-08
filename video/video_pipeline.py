"""
video_pipeline.py
==================

بتاع الفيديو بس (orchestration) - بيربط:
    transcription.py   (سحب الترانسكريبت من Vimeo)
        -> video_chunking.py   (تنظيف + تقطيع + إثراء)
        -> shared/embedding.py (dense + sparse)
        -> shared/qdrant_upsert.py (upsert في الـ collection الموحّدة)

ده بديل تشغيل السكريبتات الأربعة القديمة (transcribe_fetch.py ->
clean_video_transcripts.py -> chunk_video_transcripts.py ->
Embed_upsert_qdrant.py) واحد ورا التاني يدويًا مع ملفات وسيطة على
الدِسك؛ دلوقتي كله بيحصل في الميموري لكل فيديو، وأي pipeline تاني
(files_pipeline.py مثلاً) بيستخدم نفس shared/.

Usage:
    python -m ingestion.video.video_pipeline <folder_id>
    (عدّلي TENANT_ID / COURSE_ID تحت الأول قبل التشغيل)
"""
from ingestion.video import transcription
from ingestion.video.video_chunking import (
    chunk_video_transcript,
    load_semantic_model,
    USE_SEMANTIC_CHUNKING,
)
from ingestion.shared.embedding import (
    load_embedding_model,
    load_sparse_model,
    encode_dense,
    encode_sparse,
)
from ingestion.shared.qdrant_upsert import get_qdrant_client, upsert_chunks
from ingestion.shared.schema import build_video_payload


# =========================================================
# ✏️ عدّلي هنا قبل كل تشغيل لعميل/كورس مختلف
# =========================================================
# المنصة multi-tenant، لكن كل تشغيلة للسكريبت ده بتخص عميل وكورس
# واحد بس - فـ tenant_id/course_id ثابتين (static) هنا بدل ما يتحلّوا
# من جدول/registry. لو محتاجة تجهّزي أكتر من عميل، غيّري القيمتين دول
# وشغّلي السكريبت تاني لكل عميل على حدة.
TENANT_ID = "acme"
COURSE_ID = "agile-101"
# =========================================================
# End of configuration
# =========================================================


def ingest_video(
    video_id: str,
    video_name: str | None = None,
    video_url: str | None = None,
    tenant_id: str = TENANT_ID,
    course_id: str = COURSE_ID,
    client=None,
    dense_model=None,
    sparse_model=None,
    semantic_model=None,
) -> int:
    """
    بتاخد فيديو واحد من Vimeo لحد ما يبقى مفهرس في Qdrant. بترجع عدد
    الـ chunks اللي اتعملهم upsert (0 لو معندوش ترانسكريبت).

    tenant_id/course_id بياخدوا قيمتهم الافتراضية من TENANT_ID/COURSE_ID
    فوق، لكن ممكن تتجاوزيهم صراحةً في نداء واحد لو احتجتي (مثلاً سكريبت
    بيلف على أكتر من عميل).

    الموديلات والـ client اختياريين كـ parameters عشان ingest_folder()
    تحمّلهم مرة واحدة وتشاركهم بين كل الفيديوهات بدل ما كل فيديو يحمّل
    نسخته الخاصة.
    """
    if not tenant_id or not course_id:
        raise ValueError("TENANT_ID و COURSE_ID لازم يكونوا متعرّفين فوق في الملف")

    print(f"[{video_id}] Fetching transcript...")
    raw_data = transcription.fetch_transcript(video_id, video_name)
    if raw_data is None:
        return 0

    print(f"[{video_id}] Cleaning + chunking...")
    result = chunk_video_transcript(
        raw_data,
        tenant_id=tenant_id,
        course_id=course_id,
        video_url=video_url,
        semantic_model=semantic_model,
    )
    chunks = result["chunks"]
    if not chunks:
        print(f"[{video_id}] No chunks produced, skipping.")
        return 0

    print(f"[{video_id}] Embedding ({len(chunks)} chunks)...")
    texts = [c.get("raw_text") or c.get("text", "") for c in chunks]
    dense_vectors = encode_dense(texts, model=dense_model)
    sparse_vectors = encode_sparse(texts, model=sparse_model)

    payloads = [build_video_payload(c, tenant_id=tenant_id, course_id=course_id) for c in chunks]

    print(f"[{video_id}] Upserting to Qdrant...")
    client = client or get_qdrant_client()
    n = upsert_chunks(client, payloads, dense_vectors, sparse_vectors)

    print(f"[{video_id}] Done: {n} points upserted.")
    return n


def ingest_folder(folder_id: str, tenant_id: str = TENANT_ID, course_id: str = COURSE_ID) -> None:
    """بتاخد كل الفيديوهات في مجلد Vimeo معين وتمشيهم في نفس الـ pipeline،
    كلهم بنفس tenant_id/course_id (الثابتين فوق ما لم تتجاوزيهم هنا)،
    مع تحميل الموديلات مرة واحدة بس وتشاركها بين كل الفيديوهات."""
    videos = transcription.get_videos_from_folder(folder_id)
    print(f"Found {len(videos)} videos in folder {folder_id} -> tenant={tenant_id}, course={course_id}")

    dense_model = load_embedding_model()
    sparse_model = load_sparse_model()
    semantic_model = load_semantic_model() if USE_SEMANTIC_CHUNKING else None
    client = get_qdrant_client()

    total_videos, total_chunks = 0, 0
    for v in videos:
        try:
            n = ingest_video(
                v["id"],
                video_name=v["name"],
                tenant_id=tenant_id,
                course_id=course_id,
                client=client,
                dense_model=dense_model,
                sparse_model=sparse_model,
                semantic_model=semantic_model,
            )
            if n:
                total_videos += 1
                total_chunks += n
        except Exception as e:
            print(f"! Error ingesting {v['name']}: {e}")

    print("\nDone.")
    print(f"Videos ingested: {total_videos}/{len(videos)}")
    print(f"Total chunks upserted: {total_chunks}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.video.video_pipeline <folder_id>")
        print(f"(هيشتغل بـ TENANT_ID={TENANT_ID!r}, COURSE_ID={COURSE_ID!r} المكتوبين فوق في الملف)")
    else:
        ingest_folder(sys.argv[1])

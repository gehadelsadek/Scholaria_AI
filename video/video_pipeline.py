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
    from ingestion.video.video_pipeline import ingest_video, ingest_folder

    ingest_video("123456789", tenant_id="acme", course_id="agile-101")
    ingest_folder("30332812", tenant_id="acme", course_id="agile-101")
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


def ingest_video(
    video_id: str,
    tenant_id: str,
    course_id: str,
    video_name: str | None = None,
    video_url: str | None = None,
    client=None,
    dense_model=None,
    sparse_model=None,
    semantic_model=None,
) -> int:
    """
    بتاخد فيديو واحد من Vimeo لحد ما يبقى مفهرس في Qdrant. بترجع عدد
    الـ chunks اللي اتعملهم upsert (0 لو معندوش ترانسكريبت).

    الموديلات والـ client اختياريين كـ parameters عشان ingest_folder()
    تحمّلهم مرة واحدة وتشاركهم بين كل الفيديوهات بدل ما كل فيديو يحمّل
    نسخته الخاصة.
    """
    if not tenant_id or not course_id:
        raise ValueError("tenant_id و course_id إلزاميين")

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


def ingest_folder(folder_id: str, tenant_id: str, course_id: str) -> None:
    """بتاخد كل الفيديوهات في مجلد Vimeo معين وتمشيهم في نفس الـ pipeline،
    مع تحميل الموديلات مرة واحدة بس وتشاركها بين كل الفيديوهات."""
    videos = transcription.get_videos_from_folder(folder_id)
    print(f"Found {len(videos)} videos in folder {folder_id}")

    dense_model = load_embedding_model()
    sparse_model = load_sparse_model()
    semantic_model = load_semantic_model() if USE_SEMANTIC_CHUNKING else None
    client = get_qdrant_client()

    total_videos, total_chunks = 0, 0
    for v in videos:
        try:
            n = ingest_video(
                v["id"],
                tenant_id=tenant_id,
                course_id=course_id,
                video_name=v["name"],
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

    if len(sys.argv) < 4:
        print("Usage: python -m ingestion.video.video_pipeline <folder_id> <tenant_id> <course_id>")
    else:
        ingest_folder(sys.argv[1], tenant_id=sys.argv[2], course_id=sys.argv[3])

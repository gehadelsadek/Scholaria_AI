"""
video_chunking.py
==================

منطق تقطيع الفيديو الخاص: تنظيف segments الترانسكريبت (كان في
clean_video_transcripts.py) + التقطيع لـ chunks (كان في
chunk_video_transcripts.py) - اتدمجوا هنا لأنهم خطوة واحدة منطقيًا
(cleaned segments بيتاكلوا على طول من chunking).

⚠️ ملاحظة على النسخة القديمة: كان فيه نسختين من enrich_video_chunks()
متعرّفين في نفس الملف (التانية بتدّي غير مكتملة ومبترجعش حاجة)،
و chunk_video_json() كانت بتوقف نص الطريق من غير ما تكتب الملف
الناتج ولا ترجّع الـ chunks. الإصلاحين اتعملوا هنا: نسخة واحدة من
enrich_video_chunks (بـ tenant_id/course_id إلزاميين)، و
chunk_video_transcript() بترجّع النتيجة كاملة عشان video_pipeline.py
يقدر يكمّل عليها (embedding + upsert) من غير ما يعدّي على ملفات
وسيطة على الدِسك.

نظام الفجوات الثلاثي (Three-tier gap system):
    - فجوة صغيرة  (< gap_small_sec)      -> مسافة عادية (نفس الجملة مستمرة)
    - فجوة متوسطة (gap_small..gap_large) -> "\n"        (فاصل فكرة فرعية)
    - فجوة كبيرة  (>= gap_large_sec)     -> flush كامل  (chunk جديد)
      بشرط إن الـ buffer أصلًا >= min_chars.

إشارة رابعة: Semantic boundary via bge-m3 - لو التشابه بين جملة واللي
بعدها نزل تحت threshold معيّن، ده معناه تغيّر موضوع حقيقي، حتى لو
الفجوة الزمنية صغيرة. لو الموديل مش متاح، الكود بيرجع تلقائيًا لسلوك
الفجوات الزمنية بس من غير ما يفشل.

Overlap: بيتفعل بس عند flush حقيقي، وبياخد segments كاملة (مش قص حروف
عشوائي) عشان يفضل مربوط بتوقيت حقيقي.
"""
import json
from pathlib import Path

import numpy as np


# =========================================================
# Cleaning (كان clean_video_transcripts.py)
# =========================================================

def clean_text(text: str) -> str:
    """تنظيف نص segment واحد مع الحفاظ على علامات الترقيم."""
    import re

    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?؟،؛:])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"([!?؟])\1+", r"\1", text)
    return text.strip()


def clean_segments(pages: list[dict]) -> list[dict]:
    """
    بتاخد segments خام (زي اللي راجعة من transcription.py: timestamp_s,
    end_s, text) وترجّع نفس الشكل بعد التنظيف، مع إسقاط أي segment فاضي.
    """
    cleaned = []
    for segment in pages:
        text = clean_text(segment.get("text", ""))
        if not text:
            continue

        timestamp_s = segment.get("timestamp_s", 0)
        end_s = segment.get("end_s", timestamp_s)

        cleaned.append({
            "timestamp_s": timestamp_s,
            "end_s": end_s,
            "text": text,
        })
    return cleaned


# =========================================================
# ⚙️ Configuration — عدّل هنا بس
# =========================================================

# المفتاح الرئيسي: شغّل/وقف الـ semantic chunking من هنا مباشرة.
# لو False -> الكود مش هيحاول يعمل import ولا load للموديل خالص (0 تكلفة).
USE_SEMANTIC_CHUNKING = True

CHARS_PER_TOKEN = 2.5

SEMANTIC_MODEL_NAME = "BAAI/bge-m3"
SEMANTIC_THRESHOLD = 0.55   # لازم يتضبط على عينة حقيقية من الداتا بتاعتك
SEMANTIC_BATCH_SIZE = 32
SENTENCE_END_CHARS = (".", "!", "?", "؟")

# فولدر تحميل موديل الـ semantic chunking (منفصل عن embedding.py المشترك
# لأن ده استخدام داخلي وقت التقطيع بس، مش الـ embedding النهائي اللي
# بيتخزن في Qdrant - لكن نفس الموديل BAAI/bge-m3 فعليًا).
MODEL_CACHE_DIR = Path("models_cache")


def tokens_to_chars(tokens):
    return int(tokens * CHARS_PER_TOKEN)


PROFILES = {
    "video": {
        "min_chars": tokens_to_chars(200),
        "max_chars": tokens_to_chars(700),
        "overlap_chars": tokens_to_chars(80),
        "gap_small_sec": 0.8,   # أقل منها -> نفس الجملة، مسافة عادية
        "gap_large_sec": 3.0,   # أكبر منها أو تساويها -> flush كامل
    },
}

# =========================================================
# End of configuration
# =========================================================


_model_cache = {}


def load_semantic_model(model_name=SEMANTIC_MODEL_NAME, cache_dir=MODEL_CACHE_DIR):
    """
    بيحمّل الموديل مرة واحدة بس ويكاشه في _model_cache طول عمر الـ process.
    الاستيراد lazy جوه الدالة عشان لو USE_SEMANTIC_CHUNKING = False الكود
    متعملوش import ولا يحاول يحمّل حاجة خالص.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("! sentence_transformers مش متثبت. تشغيل من غير semantic chunking.")
        return None

    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        _model_cache[model_name] = model
        return model
    except Exception as e:
        print(f"! فشل تحميل الموديل {model_name}: {e}. تشغيل من غير semantic chunking.")
        return None


# =========================================================
# Semantic boundary detection
# =========================================================

def compute_semantic_boundaries(segments, model, threshold=SEMANTIC_THRESHOLD):
    """
    بيرجع dict: {seg_index -> True/False} بيقول هل الحد بعد الـ segment ده
    (آخر segment في جملة) فيه "قفزة موضوع" حسب bge-m3.
    """
    if model is None:
        return {}

    non_empty = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if text:
            non_empty.append((i, text))

    if len(non_empty) < 2:
        return {}

    unit_texts = []
    unit_last_idx = []
    buf_texts, buf_last_idx = [], None
    for idx, text in non_empty:
        buf_texts.append(text)
        buf_last_idx = idx
        if text.endswith(SENTENCE_END_CHARS):
            unit_texts.append(" ".join(buf_texts))
            unit_last_idx.append(buf_last_idx)
            buf_texts = []
    if buf_texts:
        unit_texts.append(" ".join(buf_texts))
        unit_last_idx.append(buf_last_idx)

    if len(unit_texts) < 2:
        return {}

    embeddings = model.encode(
        unit_texts,
        normalize_embeddings=True,
        batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=False,
    )

    boundary_flags = {}
    for i in range(len(unit_texts) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        boundary_flags[unit_last_idx[i]] = sim < threshold

    return boundary_flags


# =========================================================
# Core chunking (كان chunk_video_transcripts.py)
# =========================================================

def _joiner_for_gap(gap, gap_small_sec):
    """يحدد نوع الوصلة بين segment واللي قبله حسب حجم الفجوة الزمنية."""
    if gap < gap_small_sec:
        return " "      # استكمال نفس الجملة
    return "\n"          # فاصل فكرة فرعية (بدون flush)


def chunk_video_segments(
    segments,
    source="",
    profile="video",
    semantic_model=None,
    semantic_threshold=SEMANTIC_THRESHOLD,
):
    cfg = PROFILES[profile]
    min_chars = cfg["min_chars"]
    max_chars = cfg["max_chars"]
    overlap_chars = cfg["overlap_chars"]
    gap_small = cfg["gap_small_sec"]

    # سقف الدمج الخلفي (merge-back fix): بدل ما نسمح بالدمج لحد max_chars،
    # نحصره في هامش قريب من min_chars عشان منبلعش نقط flush شرعية.
    merge_cap = min_chars + overlap_chars

    boundary_flags = compute_semantic_boundaries(segments, semantic_model, semantic_threshold)

    chunks = []

    # الـ buffer list of segments (مش نص واحد) عشان نقدر نحسب overlap
    # بتوقيت حقيقي عند الـ flush.
    buffer = []
    prev_end = None

    def buffer_text_len():
        return sum(len(s["joiner"]) + len(s["text"]) for s in buffer)

    def buffer_to_text(segs):
        return "".join(s["joiner"] + s["text"] for s in segs)

    def take_overlap_tail(segs):
        """بياخد آخر segments من الـ buffer لحد ما يوصل overlap_chars،
        بدون ما يقطع أي segment من نصه."""
        carry = []
        acc = 0
        for s in reversed(segs):
            carry.insert(0, s)
            acc += len(s["joiner"]) + len(s["text"])
            if acc >= overlap_chars:
                break
        return carry

    def flush():
        nonlocal buffer
        if not buffer:
            return

        carry_count = 0
        for s in buffer:
            if s.get("is_carry"):
                carry_count += 1
            else:
                break

        new_segs = buffer[carry_count:]
        if not new_segs:
            buffer = []
            return

        text_full = buffer_to_text(buffer).strip()
        text_new_only = buffer_to_text(new_segs)

        if not text_full:
            buffer = []
            return

        start_s = buffer[0]["start_s"]
        end_s = buffer[-1]["end_s"]

        # دمج الـ chunk الصغير جدًا في اللي قبله - بسقف أضيق (merge_cap)
        # بدل max_chars، عشان منبلعش نقط flush شرعية وسط الملف.
        if len(text_full) < min_chars and chunks:
            last = chunks[-1]
            if len(last["text"]) + len(text_new_only) <= merge_cap:
                last["text"] += text_new_only
                last["end_s"] = end_s
                buffer = []
                return

        chunks.append({
            "text": text_full,
            "source": source,
            "start_s": start_s,
            "end_s": end_s,
        })

        if overlap_chars > 0 and len(text_full) > overlap_chars * 1.5:
            carry = take_overlap_tail(buffer)
        else:
            carry = []
        buffer = []
        for i, s in enumerate(carry):
            buffer.append({
                "text": s["text"],
                "start_s": s["start_s"],
                "end_s": s["end_s"],
                "joiner": "" if i == 0 else s["joiner"],
                "is_carry": True,
            })

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start_s = seg.get("timestamp_s", 0)
        end_s = seg.get("end_s", start_s)

        gap = (start_s - prev_end) if prev_end is not None else None

        # أي flush على فجوة لازم يحصل *قبل* إضافة الـ segment الجديد.
        # فجوة كبيرة أو متوسطة: مفيش split إلا بعد min_chars - الفرق
        # الوحيد هو الوصلة (مسافة vs \n).
        previous_semantic_shift = boundary_flags.get(i - 1, False) if i > 0 else False

        if buffer:
            if gap is not None and gap >= gap_small and buffer_text_len() >= min_chars:
                flush()
            elif buffer_text_len() >= min_chars and previous_semantic_shift:
                # مفيش فجوة كفاية، لكن bge-m3 شايف إن الموضوع اتغيّر عند
                # نهاية آخر جملة اتضافت (i-1)
                flush()

        if gap is None or not buffer:
            joiner = ""
        else:
            joiner = _joiner_for_gap(gap, gap_small)

        if len(text) > max_chars:
            flush()
            for part_text, part_start, part_end in split_long_segment(
                text, start_s, end_s, max_chars
            ):
                chunks.append(
                    {"text": part_text, "source": source, "start_s": part_start, "end_s": part_end}
                )
            prev_end = end_s
            continue

        buffer.append(
            {"text": text, "start_s": start_s, "end_s": end_s, "joiner": joiner, "is_carry": False}
        )
        prev_end = end_s

        # آخر فرصة: لو وصلنا max_chars فعليًا بعد الإضافة، لازم flush فوري.
        if buffer_text_len() >= max_chars:
            flush()

    flush()

    chunks = [c for c in chunks if len(c["text"].strip()) >= 30]
    chunks = _merge_undersized_forward(chunks, min_chars, max_chars)
    return chunks


def _text_overlap_len(a, b):
    a, b = a.strip(), b.strip()
    max_n = min(len(a), len(b))
    for n in range(max_n, 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def _merge_undersized_forward(chunks, min_chars, max_chars):
    """يدمج أي chunk أقصر من min_chars في اللي بعده لو المجموع يدخل في
    max_chars. ده شبكة أمان لحالة المقدمة القصيرة."""
    if len(chunks) < 2:
        return chunks

    merged = [chunks[0]]
    for c in chunks[1:]:
        last = merged[-1]
        if len(last["text"]) < min_chars and len(last["text"]) + len(c["text"]) <= max_chars:
            ol = _text_overlap_len(last["text"], c["text"])
            extra = c["text"][ol:]
            if extra:
                joiner = "" if extra.startswith("\n") else "\n"
                last["text"] += joiner + extra
            last["end_s"] = c["end_s"]
        else:
            merged.append(c)
    return merged


def split_long_segment(text, start_s, end_s, max_chars):
    total_duration = end_s - start_s
    total_len = len(text) or 1

    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_chars and current:
            parts.append(current.strip())
            current = ""
        current += ("\n" if current else "") + line
    if current.strip():
        parts.append(current.strip())

    result = []
    cursor = start_s
    for part in parts:
        ratio = len(part) / total_len
        part_end = min(cursor + total_duration * ratio, end_s)
        result.append((part, cursor, part_end))
        cursor = part_end
    return result


# =========================================================
# Enrichment
# =========================================================

def _format_timestamp(seconds):
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def enrich_video_chunks(
    chunks: list[dict],
    tenant_id: str,
    course_id: str,
    video_title: str | None = None,
    video_url: str | None = None,
) -> list[dict]:
    """
    نسخة واحدة موحّدة (النسخة القديمة كان فيها اتنين: واحدة أصلية وواحدة
    بتضيف tenant_id/course_id بس بتوقف نص الطريق من غير ما ترجع حاجة).
    """
    if not tenant_id or not course_id:
        raise ValueError("tenant_id و course_id إلزاميين لكل chunk")

    for c in chunks:
        c["type"] = "video"
        c["tenant_id"] = tenant_id
        c["course_id"] = course_id

        title = video_title or c["source"].replace("-", " ").replace("_", " ")
        start_fmt = _format_timestamp(c.get("start_s"))
        end_fmt = _format_timestamp(c.get("end_s"))
        header = f"[الفيديو: {title} | من {start_fmt} إلى {end_fmt}]"

        if video_url and c.get("start_s") is not None:
            c["deep_link"] = f"{video_url}?t={int(c['start_s'])}s"

        c["raw_text"] = c["text"]
        c["text"] = f"{header}\n{c['text']}"

    return chunks


# =========================================================
# High-level entrypoint: raw transcript -> clean -> chunk -> enrich
# =========================================================

def chunk_video_transcript(
    raw_data: dict,
    tenant_id: str,
    course_id: str,
    video_url: str | None = None,
    semantic_model=None,
) -> dict:
    """
    بتاخد الـ dict الخام الراجع من transcription.py (source_name, pages)
    وترجّع dict فيه source_name والـ chunks الجاهزة (منظّفة، مقطّعة،
    ومُثراة بالـ headers + tenant/course) - جاهزة تدخل على طول لـ
    embedding.py ثم qdrant_upsert.py في video_pipeline.py، من غير أي
    ملفات وسيطة على الدِسك.

    لو محتاجة تشغيل ملف-لملف زي الأسلوب القديم (raw_extracted/ ->
    cleaned_videos/ -> chunked_videos/) استخدمي chunk_video_file() تحت.
    """
    if not tenant_id or not course_id:
        raise ValueError("tenant_id و course_id إلزاميين - راجعي Q2 في decision log")

    source_name = raw_data.get("source_name", "unknown")
    raw_pages = raw_data.get("pages", [])

    cleaned_pages = clean_segments(raw_pages)

    chunks = chunk_video_segments(
        cleaned_pages, source=source_name, profile="video", semantic_model=semantic_model
    )
    chunks = enrich_video_chunks(
        chunks,
        tenant_id=tenant_id,
        course_id=course_id,
        video_title=source_name,
        video_url=video_url,
    )

    return {"source_name": source_name, "chunks": chunks}


# =========================================================
# File-based helpers (اختياري - لتشغيل/اختبار محلي بنفس أسلوب السكريبتات
# القديمة، بدل استدعاء video_pipeline.py الكامل)
# =========================================================

INPUT_DIR = Path("raw_extracted")
OUTPUT_DIR = Path("chunked_videos")


def chunk_video_file(
    input_path: Path,
    output_path: Path,
    tenant_id: str,
    course_id: str,
    semantic_model=None,
) -> int:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    result = chunk_video_transcript(
        raw_data, tenant_id=tenant_id, course_id=course_id, semantic_model=semantic_model
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return len(result["chunks"])


def chunk_all_videos(tenant_id: str, course_id: str) -> None:
    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {INPUT_DIR}")
        return

    # الموديل بيتحمّل مرة واحدة بس هنا وبيتشارك بين كل الفيديوهات.
    semantic_model = load_semantic_model() if USE_SEMANTIC_CHUNKING else None

    total_files, total_chunks = 0, 0
    for input_path in json_files:
        try:
            output_path = OUTPUT_DIR / input_path.name
            n_chunks = chunk_video_file(
                input_path, output_path, tenant_id, course_id, semantic_model=semantic_model
            )
            total_files += 1
            total_chunks += n_chunks
            print(f"Chunked: {input_path.name} -> {n_chunks} chunks")
        except Exception as e:
            print(f"! Error chunking {input_path.name}: {e}")

    print("\nDone.")
    print(f"Files chunked: {total_files}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python video_chunking.py <tenant_id> <course_id>")
    else:
        chunk_all_videos(tenant_id=sys.argv[1], course_id=sys.argv[2])

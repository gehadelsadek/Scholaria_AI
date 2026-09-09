"""
تنظيف وتقطيع ملفات PDF.

المراحل: تطبيع → تنظيف الصفحات → تقسيم دلالي → إثراء بالسياق.

⚠️ normalize_arabic لازم يكون متطابق مع نسخة مسار الفيديو،
وإلا نفس النص هيدي vectors مختلفة.
"""

import re
import unicodedata
from collections import Counter

import numpy as np

from ingestion.shared.embedding import encode_dense

CHARS_PER_TOKEN = 2.5


def tokens_to_chars(tokens):
    return int(tokens * CHARS_PER_TOKEN)


# البروفايلات — البند 6.3
# الأرقام معايرة بالقياس مش منسوخة من الوثيقة (شوف README)
PROFILES = {
    "pdf": {
        "min_chars": tokens_to_chars(100),
        "max_chars": tokens_to_chars(700),
        "overlap_chars": tokens_to_chars(90),
    },
}


# ============================================================
#  0) التطبيع العربي
#  ⚠️ نسخة مطابقة لازم تكون في مسار الفيديو كمان
# ============================================================

BIDI_CHARS = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
INVISIBLE_CHARS = "\u200b\u200c\u200d\ufeff"
TATWEEL = "\u0640"
DIACRITICS = re.compile(r"[\u064b-\u0652\u0670]")


def normalize_arabic(text):
    """
    تطبيع النص العربي:
    - NFKC بترجع الحروف من presentation forms لشكلها الطبيعي
    - إزالة حروف التحكم والتشكيل والتطويل
    - توحيد أشكال الألف والياء والتاء المربوطة
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    for ch in BIDI_CHARS + INVISIBLE_CHARS + TATWEEL:
        text = text.replace(ch, "")

    text = DIACRITICS.sub("", text)

    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
#  1) اكتشاف الترميز المكسور
# ============================================================

BROKEN_ENCODING = re.compile(
    r"[\u0600-\u06FF\uFB50-\uFEFF][a-zA-Z‚„†ˆ‰ŠŒ''"
    "•–—˜™›Ÿ¡-¿]"
    r"|[a-zA-Z‚„†ˆ‰ŠŒ''"
    "•–—˜™›Ÿ¡-¿][\u0600-\u06ff\ufb50-\ufeff]"
)


def is_broken_encoding(text, min_arabic=20, threshold=2):
    """
    بتكتشف النص العربي المكسور بسبب ترميز الخط في الـ PDF.
    مؤشرها: حروف لاتينية أو رموز خاصة ملتصقة بحروف عربية.
    """
    arabic = len(re.findall(r"[\u0600-\u06FF\uFB50-\uFEFF]", text))
    if arabic < min_arabic:
        return False
    return len(BROKEN_ENCODING.findall(text)) >= threshold


# ============================================================
#  2) إزالة الـ boilerplate
# ============================================================

# أنماط بتظهر جوه النص مش كسطر مستقل
INLINE_BOILERPLATE = [
    re.compile(
        r"B\s*Y\s*:?\s*E\s*L?\s*S\s*a\s*y\s*e\s*d\s*M\s*o\s*h\s*s\s*e\s*n", re.I
    ),
    re.compile(
        r"(P[fg]MP|PMP|PBA|RMP|ACP|PMOCP|CSPP|CPMAI|P3O|CP)"
        r"(\s*,\s*(P[fg]MP|PMP|PBA|RMP|ACP|PMOCP|CSPP|CPMAI|P3O|CP|SP)){2,}",
        re.I,
    ),
    re.compile(r"©?\s*w{0,3}\.?pm-?tricks\.com[^\n]{0,60}", re.I),
    re.compile(r"All\s*rights?\s*reserved[^\n]{0,30}", re.I),
    re.compile(r"Do\s*not\s*share", re.I),
]


def strip_inline_boilerplate(text):
    """بتشيل أنماط الـ boilerplate اللي جوه النص."""
    for pattern in INLINE_BOILERPLATE:
        text = pattern.sub(" ", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_boilerplate(pages, threshold=0.5, max_len=120):
    """بتكتشف السطور المتكررة عبر الصفحات تلقائيًا."""
    counter = Counter()

    for page in pages:
        lines = {ln.strip() for ln in page["text"].split("\n") if ln.strip()}
        counter.update(lines)

    min_count = max(2, int(len(pages) * threshold))

    return {
        line
        for line, count in counter.items()
        if count >= min_count and len(line) <= max_len
    }


def remove_boilerplate(text, boilerplate):
    """بتشيل السطور المتكررة من صفحة."""
    kept = [
        ln for ln in text.split("\n") if ln.strip() and ln.strip() not in boilerplate
    ]
    return "\n".join(kept)


# ============================================================
#  3) فلترة جودة الـ OCR
# ============================================================

VALID_CHARS = re.compile(r"[\u0600-\u06FFa-zA-Z0-9\s.,:;!?()\-/%]")


def ocr_quality_score(text):
    """نسبة الحروف السليمة في النص (من 0 لـ 1)."""
    if not text or len(text) < 10:
        return 0.0
    return len(VALID_CHARS.findall(text)) / len(text)


def orphan_ratio(text):
    """نسبة الكلمات اليتيمة (حرف أو حرفين) — مؤشر فشل الـ OCR."""
    words = text.split()
    if not words:
        return 1.0
    return len([w for w in words if len(w) <= 2]) / len(words)


def filter_ocr_lines(text, max_orphans=0.30, min_quality=0.75):
    """تنظيف على مستوى السطر: تشيل السطور الوحشة وتسيب الكويسة."""
    kept = []

    for line in text.split("\n"):
        line = line.strip()

        if len(line) < 8 or len(line.split()) < 3:
            continue
        if orphan_ratio(line) > max_orphans:
            continue
        if ocr_quality_score(line) < min_quality:
            continue

        kept.append(line)

    return "\n".join(kept)


# ============================================================
#  4) تنظيف الصفحات — المدخل الرئيسي
# ============================================================


def clean_text(text):
    """تطبيع عربي مشترك + إزالة boilerplate خاصة بالملفات."""
    if not text:
        return ""
    return strip_inline_boilerplate(normalize_arabic(text))


def clean_pages(pages, min_ocr_length=80, max_ocr_orphans=0.30):
    """
    بتنضف كل صفحات ملف واحد.
    بتطبق فلتر جودة إضافي على الصفحات اللي جاية من OCR.
    """
    cleaned = []

    for p in pages:
        text = clean_text(p["text"])
        is_ocr = p.get("ocr", False)

        if is_ocr and text:
            text = filter_ocr_lines(text)

            if len(text) < min_ocr_length or orphan_ratio(text) > max_ocr_orphans:
                print(f"[clean] صفحة {p['page']} اترفضت (OCR ضعيف)")
                continue

        cleaned.append({"page": p["page"], "text": text, "ocr": is_ocr})

    boilerplate = find_boilerplate(cleaned)
    print(f"[clean] اتكتشف {len(boilerplate)} سطر متكرر هيتشال")

    result = []
    for p in cleaned:
        text = remove_boilerplate(p["text"], boilerplate)
        if text.strip():
            result.append({"page": p["page"], "text": text.strip(), "ocr": p["ocr"]})

    return result


# ============================================================
#  5) التقسيم الدلالي
# ============================================================


def _clean_tail(text, overlap):
    """بترجع آخر جزء من النص، مقطوع عند حدود كلمة مش وسطها."""
    if not overlap or len(text) <= overlap:
        return text

    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def find_boundaries(vectors, percentile=25, min_gap=2):
    """
    بتلاقي نقاط التغير في المعنى.
    العتبة نسبية — بتتحسب من الملف نفسه مش رقم ثابت.
    """
    if len(vectors) < 3:
        return set()

    sims = [float(np.dot(vectors[i], vectors[i + 1])) for i in range(len(vectors) - 1)]
    threshold = np.percentile(sims, percentile)

    boundaries = []
    last = -min_gap
    for i, s in enumerate(sims):
        if s < threshold and (i - last) >= min_gap:
            boundaries.append(i + 1)
            last = i

    return set(boundaries)


def split_by_lines(text, max_chars, overlap=0):
    """تقسيم صفحة ضخمة عند حدود السطور."""
    parts, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_chars and current:
            parts.append(current.strip())
            current = _clean_tail(current, overlap)
        current += ("\n" if current else "") + line
    if current.strip():
        parts.append(current.strip())
    return parts


def _make(text, source, pages, primary, tid, topic_pages):
    """بتبني chunk مع بيانات الترابط الدلالي."""
    return {
        "text": text,
        "source": source,
        "pages": pages,
        "primary_page": primary,
        "topic_id": tid,
        "topic_pages": topic_pages.get(tid, []),
    }


def chunk_semantic(
    pages,
    source="",
    profile="pdf",
    percentile=25,
    min_gap=2,
):
    """
    الـ semantic بيحدد المواضيع، والصفحة هي وحدة الـ chunk.
    الترابط الدلالي بيتحفظ في metadata بدل ما يتدمج في نص واحد.
    """
    cfg = PROFILES[profile]
    min_chars = cfg["min_chars"]
    max_chars = cfg["max_chars"]
    overlap = cfg["overlap_chars"]

    valid = [p for p in pages if p["text"].strip()]
    if not valid:
        return []

    # 1) نلاقي الحدود الدلالية
    vectors = encode_dense([p["text"] for p in valid])
    boundaries = find_boundaries(vectors, percentile, min_gap)

    # 2) نسمّي موضوع لكل صفحة
    topic_id = 0
    topic_of = {}
    for i, page in enumerate(valid):
        if i in boundaries:
            topic_id += 1
        topic_of[page["page"]] = topic_id

    topic_pages = {}
    for pg, tid in topic_of.items():
        topic_pages.setdefault(tid, []).append(pg)
    for tid in topic_pages:
        topic_pages[tid].sort()

    # 3) كل صفحة = chunk (مع دمج الصغيرة جدًا في نفس الموضوع)
    chunks = []
    pending = None

    for page in valid:
        text = page["text"].strip()
        pg = page["page"]
        tid = topic_of[pg]

        if pending:
            if topic_of[pending["page"]] == tid:
                merged = pending["text"] + "\n\n" + text
                primary = pg if len(text) >= len(pending["text"]) else pending["page"]
                pages_list = sorted({pending["page"], pg})
                pending = None
                text, primary_page, pgs = merged, primary, pages_list
            else:
                chunks.append(
                    _make(
                        pending["text"],
                        source,
                        [pending["page"]],
                        pending["page"],
                        topic_of[pending["page"]],
                        topic_pages,
                    )
                )
                pending = None
                primary_page, pgs = pg, [pg]
        else:
            primary_page, pgs = pg, [pg]

        if len(text) < min_chars:
            pending = {"text": text, "page": primary_page}
            continue

        if len(text) <= max_chars:
            chunks.append(_make(text, source, pgs, primary_page, tid, topic_pages))
        else:
            for part in split_by_lines(text, max_chars, overlap):
                chunks.append(_make(part, source, pgs, primary_page, tid, topic_pages))

    if pending:
        chunks.append(
            _make(
                pending["text"],
                source,
                [pending["page"]],
                pending["page"],
                topic_of[pending["page"]],
                topic_pages,
            )
        )

    chunks = [c for c in chunks if len(c["text"].strip()) >= 30]

    print(
        f"[chunk] {len(valid)} صفحة → {len(chunks)} chunk "
        f"({topic_id + 1} موضوع دلالي)"
    )
    return chunks


# ============================================================
#  6) الإثراء بالسياق
# ============================================================


def enrich_chunks(chunks, doc_title=None):
    """
    بتضيف سياق المستند في أول كل chunk عشان تحسن الـ embedding.
    النص الأصلي بيتحفظ في raw_text للعرض للمستخدم.

    ملاحظة: نوع المصدر بيتحدد من doc_info["sourceType"] في الـ schema،
    مش من هنا — مصدر واحد للحقيقة.
    """
    for c in chunks:
        title = doc_title or c["source"].replace("-", " ").replace("_", " ")
        page = c.get("primary_page") or (c["pages"][0] if c.get("pages") else "?")

        c["raw_text"] = c["text"]
        c["text"] = f"[المستند: {title} | صفحة {page}]\n{c['text']}"

    return chunks

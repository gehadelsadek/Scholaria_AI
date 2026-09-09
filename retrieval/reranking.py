"""
إعادة ترتيب المرشحين — البند 6.5:
استرجاع 20–30 مرشح ثم Rerank وإبقاء أفضل 3–5.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("RERANK_MODEL", "qwen/qwen3-8b")
MAX_SNIPPET = 250  # نقص كل مرشح عشان نقلل الـ tokens

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY مش موجود في .env")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
    return _client


SYSTEM_PROMPT = """Score each document 0-10 on how well it answers the question.
10=direct answer, 7=partial, 4=related only, 0=unrelated.
Text may be Arabic, English, or garbled from OCR.
Score EVERY document. Return ONLY JSON:
[{"id":1,"score":8},{"id":2,"score":3}]"""


def _build_docs_block(candidates):
    """بتجهز نص المرشحين للـ prompt."""
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c.payload.get("raw_text", "").replace("\n", " ")[:MAX_SNIPPET]
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def _parse_scores(raw, n):
    """
    بتستخرج الدرجات من رد الموديل.
    البند 5.5: يجب التحقق من الـJSON ورفضه عند عدم التطابق.
    """
    # نشيل أي code fences أو نص زيادة
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("مفيش JSON في الرد")

    data = json.loads(match.group(0))

    scores = {}
    for item in data:
        idx = int(item["id"])
        if 1 <= idx <= n:
            scores[idx] = float(item["score"])

    if not scores:
        raise ValueError("مفيش درجات صالحة")

    return scores


def rerank(query, candidates, top_n=5, timeout=60):
    """
    بترتب المرشحين حسب صلتهم بالسؤال.
    لو فشلت، بترجع الترتيب الأصلي (fail-safe).
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocuments:\n{_build_docs_block(candidates)}",
                },
            ],
            temperature=0,
            max_tokens=300,
            timeout=timeout,
        )
        # u = response.usage
        # cached = (
        #     getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        # )
        # print(
        #     f"[rerank] in={u.prompt_tokens} cached={cached} out={u.completion_tokens}"
        # )

        # raw = response.choices[0].message.content
        raw = response.choices[0].message.content
        scores = _parse_scores(raw, len(candidates))

        missing = len(candidates) - len(scores)  # ← هنا
        if missing > len(candidates) * 0.3:
            print(f"[rerank] ⚠️ الموديل قيّم {len(scores)} من {len(candidates)} بس")

        # نرتب حسب الدرجة (اللي مالوش درجة ياخد -1)
        ranked = sorted(
            enumerate(candidates, 1),
            key=lambda pair: scores.get(pair[0], -1),
            reverse=True,
        )

        return [(cand, scores.get(idx, -1)) for idx, cand in ranked[:top_n]]

        # result = []
        # for idx, cand in ranked[:top_n]:
        #     cand.rerank_score = scores.get(idx, -1)
        #     result.append(cand)

        # return result

    except Exception as e:
        print(f"[rerank] ⚠️ فشل ({e}) — الترتيب الأصلي")
        return [(c, None) for c in candidates[:top_n]]

"""
api/main.py — واجهة FastAPI لطبقة الأسئلة.

endpoint واحد: POST /ask

الـ tenant_id و course_id بييجوا من headers مش من الـ body.
⚠️ البند 15 (SEC-002): في النظام النهائي لازم يتشتقوا من Authentication
Claims — الـ headers هنا خطوة وسط لحد ما الباك إند يضيف الـ auth.
Usage:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from generation.answerer import answer
from ingestion.shared.qdrant_upsert import COLLECTION, get_client
from ingestion.shared.schema import SourceType
from retrieval.reranking import rerank
from retrieval.search import hybrid_search

CANDIDATES = 10
TOP_N = 5

# العتبة معايرة بالقياس — شوف README
MIN_SCORE = 4
RELATIVE_CUTOFF = 0.5

NO_EVIDENCE = "المعلومة دي مش متوفرة."

# الـ client بيتشارك بين الطلبات — فتحه لكل طلب بطيء
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """بيفتح الاتصال مرة واحدة عند التشغيل ويقفله عند الإغلاق."""
    global _client
    _client = get_client()
    yield
    _client.close()


app = FastAPI(
    title="Scholaria RAG API",
    description="أسئلة على محتوى المقرر — ملفات وفيديو",
    version="1.0",
    lifespan=lifespan,
)


# =========================================================
#  العقود
# =========================================================


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    source_type: str | None = Field(
        None,
        description='"file" أو "video" — اتركه فارغًا للبحث في الاتنين',
    )
    history: list[dict] = Field(
        default_factory=list,
        description="جولات المحادثة السابقة: [{role, content}]",
    )


class Citation(BaseModel):
    index: int
    source: str | None = None
    source_type: str | None = None
    reference: str | None = None
    page_number: int | None = None
    start_s: float | None = None
    end_s: float | None = None
    deep_link: str | None = None
    content_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]


# =========================================================
#  منطق الفلترة — نفس اللي في ask.py
# =========================================================


def _filter_useful(reranked):
    """
    عتبة نسبية مش رقم ثابت: السؤال اللي إجابته واضحة بياخد عتبة أعلى،
    والموزّع بياخد أوسع.
    """
    scored = [(r, s) for r, s in reranked if s is not None]

    if not scored:
        return [r for r, _ in reranked]  # الـ rerank فشل

    top = max(s for _, s in scored)

    if top < MIN_SCORE:
        return []

    cutoff = max(MIN_SCORE, top * RELATIVE_CUTOFF)
    return [r for r, s in scored if s >= cutoff]


def _empty_response() -> AskResponse:
    return AskResponse(answer=NO_EVIDENCE, grounded=False, citations=[])


# =========================================================
#  Endpoints
# =========================================================


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    x_tenant_id: str = Header(..., alias="X-Tenant-Id"),
    x_course_id: str = Header(..., alias="X-Course-Id"),
):
    """
    بتجاوب على سؤال من محتوى المقرر.

    العزل إلزامي: كل بحث بيتفلتر بـ tenant_id + course_id + published
    قبل حساب التشابه (البند 6.5 و 18.2).
    """
    if body.source_type and body.source_type not in SourceType.ALL:
        raise HTTPException(
            status_code=400,
            detail=f"source_type لازم يكون {SourceType.FILE} أو {SourceType.VIDEO}",
        )

    try:
        candidates = hybrid_search(
            _client,
            body.question,
            tenant_id=x_tenant_id,
            course_id=x_course_id,
            source_type=body.source_type,
            top_k=CANDIDATES,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"تعذّر البحث: {exc}") from exc

    if not candidates:
        return _empty_response()

    useful = _filter_useful(rerank(body.question, candidates, top_n=TOP_N))

    if not useful:
        return _empty_response()

    result = answer(body.question, useful, history=body.history)

    return AskResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        citations=[Citation(**c) for c in result["citations"]],
    )


@app.get("/health")
def health():
    """بيتأكد إن الاتصال بـ Qdrant شغال وبيرجّع عدد النقاط."""
    try:
        info = _client.get_collection(COLLECTION)
        return {"status": "ok", "points": info.points_count}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

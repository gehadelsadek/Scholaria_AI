"""
اختبار تفاعلي: اسأل أي سؤال وشوف الإجابة مع مراجعها.
"""

from ingestion.shared.embedding import embed_query, embed_sparse_query

from ingestion.shared.qdrant_upsert import get_client
from retrieval.search import hybrid_search
from generation.answerer import answer, summarize_history
from retrieval.reranking import rerank

TENANT, COURSE = "dev-tenant", 1
CANDIDATES = 10
TOP_N = 5
KEEP_TURNS = 3

# الحد الأدنى المطلق — تحته مفيش حاجة مرتبطة
MIN_SCORE = 3
# نسبة من أعلى درجة — بتتكيف مع كل سؤال
RELATIVE_CUTOFF = 0.5

NO_EVIDENCE = "المعلومة دي مش متوفرة."

client = get_client()

history = []
summary = None


# متابعة للإجابة الأخيرة
FOLLOWUP = [
    "وضح",
    "اشرح",
    "بسط",
    "كمان",
    "اكتر",
    "أكثر",
    "مثال",
    "بالتفصيل",
    "explain",
    "more",
    "example",
    "elaborate",
    "clarify",
    "expand",
    "simplify",
]


def _build_search_query(query):
    """
    سؤال المتابعة بيتبحث بالسؤال السابق — كلمات المتابعة نفسها
    (وضح، اكتر) بتلوّث الاسترجاع لأنها مالهاش معنى في المحتوى.
    """
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if not user_msgs:
        return query

    # low = query.lower()

    if any(h in query.lower() for h in FOLLOWUP):
        return f"{user_msgs[-1]} {query}"

    return query


def _filter_useful(reranked):
    """
    بتفلتر المرشحين بعتبة نسبية مش رقم ثابت.
    السؤال اللي إجابته واضحة بياخد عتبة أعلى، والموزّع بياخد أوسع.
    """
    scored = [(r, s) for r, s in reranked if s is not None]

    # الـ rerank فشل — نسيب ترتيب الاسترجاع
    if not scored:
        return [r for r, _ in reranked]

    top = max(s for _, s in scored)

    if top < MIN_SCORE:
        return []

    cutoff = max(MIN_SCORE, top * RELATIVE_CUTOFF)
    return [r for r, s in scored if s >= cutoff]


def ask(query):
    global history, summary

    search_query = _build_search_query(query)

    dense = embed_query(search_query)
    sparse = embed_sparse_query(search_query)
    candidates = hybrid_search(client, dense, sparse, TENANT, COURSE, top_k=CANDIDATES)

    print(f"\n{'='*70}")
    print(f"❓ {query}")
    print(f"{'='*70}")

    if not candidates:
        print(f"\n💬 {NO_EVIDENCE}")
        return

    useful = _filter_useful(rerank(search_query, candidates, top_n=TOP_N))

    if not useful:
        print(f"\n💬 {NO_EVIDENCE}")
        return

    result = answer(query, useful, history=history, summary=summary)

    print(f"\n💬 {result['answer']}")

    if result["citations"]:
        print(f"\n📚 المراجع:")
        for c in result["citations"]:
            print(f"   [{c['index']}] {c['source']} — صفحة {c['page']}")

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["answer"]})

    # الجولات اللي هتتشال بتتلخّص قبل ما تروح
    if len(history) > KEEP_TURNS * 2:
        old = history[: -(KEEP_TURNS * 2)]
        new_summary = summarize_history(old)
        if new_summary:
            summary = f"{summary} {new_summary}" if summary else new_summary
        history[:] = history[-(KEEP_TURNS * 2) :]


def reset():
    """بتبدأ محادثة جديدة — بتمسح التاريخ والملخص."""
    global history, summary
    history.clear()
    summary = None


print("⏳ بيجهّز الموديلات...")
embed_query("تجربة")
embed_sparse_query("تجربة")
print("✅ جاهز\n")
print("اكتب سؤالك | 'جديد' لمحادثة جديدة | 'خروج' للإنهاء")

try:
    while True:
        q = input("\n💬 سؤالك: ").strip()
        if not q:
            continue
        if q in ("خروج", "exit", "quit", "q"):
            break
        if q in ("جديد", "new", "clear"):
            reset()
            print("🔄 محادثة جديدة")
            continue
        ask(q)
except (EOFError, KeyboardInterrupt):
    pass
except Exception:
    import traceback

    traceback.print_exc()
finally:
    client.close()
    print("\n👋 انتهى")

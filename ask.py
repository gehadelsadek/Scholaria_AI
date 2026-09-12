"""
ask.py — واجهة تفاعلية للاختبار.

أوامر:
    ملفات / files   → البحث في الملفات بس
    فيديو / video   → البحث في الفيديو بس
    الكل / all      → الاتنين (الافتراضي)
    جديد / new      → محادثة جديدة
    خروج / exit
"""

from generation.answerer import answer, summarize_history
from generation.citations import format_reference
from ingestion.shared.qdrant_upsert import get_client
from ingestion.shared.schema import SourceType
from retrieval.reranking import rerank
from retrieval.search import hybrid_search

TENANT_ID = "acme"
COURSE_ID = "agile-101"

CANDIDATES = 10
TOP_N = 5
KEEP_TURNS = 3

MIN_SCORE = 4

RELATIVE_CUTOFF = 0.5

NO_EVIDENCE = "المعلومة دي مش متوفرة."

client = get_client()

history = []
summary = None
source_filter = None  # None = الملفات والفيديو مع بعض

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
    "page",
    "source",
    "elaborate",
    "where",
    "فين",
    "منين",
    "مصدر",
    "صفحه",
    "صفحة",
]


def _build_search_query(query):
    """
    سؤال المتابعة بيتبحث بالسؤال السابق — كلمات المتابعة نفسها
    (وضح، اكتر) بتلوّث الاسترجاع لأنها مالهاش معنى في المحتوى.
    """
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if not user_msgs:
        return query

    if any(h in query.lower() for h in FOLLOWUP):
        return user_msgs[-1]

    return query


def _filter_useful(reranked):
    """
    عتبة نسبية مش رقم ثابت — السؤال اللي إجابته واضحة بياخد عتبة أعلى،
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


def ask(query):
    global history, summary

    search_query = _build_search_query(query)

    candidates = hybrid_search(
        client,
        search_query,
        tenant_id=TENANT_ID,
        course_id=COURSE_ID,
        source_type=source_filter,
        top_k=CANDIDATES,
    )

    scope = {
        SourceType.FILE: "الملفات",
        SourceType.VIDEO: "الفيديو",
    }.get(source_filter, "الملفات + الفيديو")

    print(f"\n{'='*70}")
    print(f"❓ {query}")
    print(f"   [{scope}]")
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
            icon = "🎬" if c["source_type"] == SourceType.VIDEO else "📄"
            print(f"   [{c['index']}] {icon} {c['source']} — {c['reference']}")

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["answer"]})

    if len(history) > KEEP_TURNS * 2:
        old = history[: -(KEEP_TURNS * 2)]
        new_summary = summarize_history(old)
        if new_summary:
            summary = f"{summary} {new_summary}" if summary else new_summary
        history[:] = history[-(KEEP_TURNS * 2) :]


def reset():
    global history, summary
    history.clear()
    summary = None


def set_scope(value):
    global source_filter
    source_filter = value
    label = {
        SourceType.FILE: "الملفات بس",
        SourceType.VIDEO: "الفيديو بس",
    }.get(value, "الملفات + الفيديو")
    print(f"🔎 البحث في: {label}")


def main():
    global history, summary

    print("⏳ بيجهّز الموديلات...")
    try:
        hybrid_search(
            client, "تجربة", tenant_id=TENANT_ID, course_id=COURSE_ID, top_k=1
        )
    except Exception:
        print("⚠️ الـ collection لسه فاضية — شغّلي الفهرسة الأول")
    print("✅ جاهز\n")

    print("اكتب سؤالك | 'ملفات' | 'فيديو' | 'الكل' | 'جديد' | 'خروج'")

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
            if q in ("ملفات", "files"):
                set_scope(SourceType.FILE)
                continue
            if q in ("فيديو", "video"):
                set_scope(SourceType.VIDEO)
                continue
            if q in ("الكل", "all"):
                set_scope(None)
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


if __name__ == "__main__":
    main()

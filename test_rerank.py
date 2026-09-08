from ingestion.shared.embedding import embed_query, embed_sparse_query
from ingestion.shared.qdrant_upsert import get_client, hybrid_search
from retrieval.reranking import rerank

TENANT, COURSE = "dev-tenant", 1
client = get_client()


def show(label, results, reranked=False):
    print(f"\n--- {label} ---")
    if not results:
        print("   ⚠️ مفيش نتايج")
        return

    for i, item in enumerate(results, 1):
        if reranked:
            r, rscore = item
            extra = f" (rerank: {rscore})" if rscore is not None else ""
        else:
            r, extra = item, ""

        p = r.payload
        print(f"{i}. [{r.score:.3f}]{extra} {p.get('title')} | ص{p.get('pageNumber')}")
        print(f"   {p.get('text','').replace(chr(10),' ')[:130]}...")


def compare(query):
    print(f"\n{'='*70}")
    print(f"❓ {query}")
    print(f"{'='*70}")

    dense = embed_query(query)
    sparse = embed_sparse_query(query)

    candidates = hybrid_search(client, dense, sparse, TENANT, COURSE, top_k=20)

    show("قبل الـ rerank (أول 3)", candidates[:3])
    show("بعد الـ rerank", rerank(query, candidates, top_n=3), reranked=True)


try:
    for q in [
        "ما هي مخاطر المشروع وكيفية الاستجابة لها؟",
        "ازاي بنحسب الـ Critical Path؟",
        "ما هي أنواع العقود في المشتريات؟",
    ]:
        compare(q)
except Exception:
    import traceback

    traceback.print_exc()
finally:
    client.close()

from ingestion.shared.embedding import embed_query, embed_sparse_query
from ingestion.shared.qdrant_upsert import get_client
from retrieval.search import hybrid_search

TENANT, COURSE = "dev-tenant", 1
client = get_client()


def run(query, tenant=TENANT, course=COURSE, top_k=3):
    dense = embed_query(query)
    sparse = embed_sparse_query(query)
    results = hybrid_search(client, dense, sparse, tenant, course, top_k=top_k)

    print(f"\n{'='*65}")
    print(f"❓ {query}")
    print(f"   [tenant={tenant} | course={course}]")
    print(f"{'='*65}")

    if not results:
        print("   ⚠️ مفيش نتايج")
        return results

    for rank, r in enumerate(results, 1):
        p = r.payload
        text = p.get("text", "").replace("\n", " ")
        print(
            f"\n{rank}. [{r.score:.3f}] {p.get('title')} | صفحة {p.get('pageNumber')}"
        )
        print(f"   {text[:200]}...")

    return results


if __name__ == "__main__":
    try:
        # 1) البحث العادي
        for q in [
            "ما هي مخاطر المشروع وكيفية الاستجابة لها؟",
            "كيف يتم إدارة نطاق المشروع؟",
            "ما هي أنواع العقود في المشتريات؟",
            "إيه هو دور الـ Scrum Master؟",
            "ازاي بنحسب الـ Critical Path؟",
        ]:
            run(q)

        # 2) اختبار العزل
        print(f"\n\n{'#'*65}")
        print("# اختبار العزل")
        print(f"{'#'*65}")

        wrong_course = run("ما هي قيم ومبادئ اجايل؟", course=999)
        wrong_tenant = run("ما هي قيم ومبادئ اجايل؟", tenant="other-tenant")

        print(f"\n{'='*65}")
        if not wrong_course and not wrong_tenant:
            print("✅ العزل شغال — صفر تسرب")
        else:
            print("❌ تسرب! النتايج ظهرت رغم اختلاف الفلتر")
        print(f"{'='*65}")

    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        client.close()

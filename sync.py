"""
مزامنة Qdrant مع الملفات الفعلية:
- حذف vectors الملفات المشالة
- حذف الإصدارات القديمة بعد التحديث
البند 6.2 و 7.4 و 14.
"""

import json
import numpy as np
from ingestion.files.registry import (
    load_registry,
    save_registry,
    find_orphans,
    unregister,
)
from ingestion.shared.qdrant_upsert import (
    get_client,
    ensure_collection,
    upsert_chunks,
    delete_content,
    delete_old_versions,
    count_content,
    COLLECTION,
)


def sync():
    registry = load_registry()
    client = get_client()
    ensure_collection(client)

    # ===== 1) حذف الملفات اللي اتشالت =====
    orphans = find_orphans(registry)

    if orphans:
        print(f"\n🗑️  لقيت {len(orphans)} ملف اتشال:")
        for name, entry in orphans.items():
            before = count_content(client, entry["contentId"])
            delete_content(client, entry["contentId"])
            print(f"   - {name} (اتحذف {before} chunk)")
            registry = unregister(name, registry)
        save_registry(registry)
    else:
        print("✅ مفيش ملفات محذوفة")

    # ===== 2) حذف الإصدارات القديمة =====
    print(f"\n🔄 بيتأكد من الإصدارات...")
    for name, entry in registry.items():
        delete_old_versions(client, entry["contentId"], entry["version"])

    # ===== 3) تقرير نهائي =====
    info = client.get_collection(COLLECTION)
    print(f"\n{'='*55}")
    print(f"📊 الإجمالي في الـ DB: {info.points_count} chunk")
    print(f"   الملفات المفهرسة: {len(registry)}")
    for name, entry in registry.items():
        n = count_content(client, entry["contentId"])
        print(f"   - {name[:40]} → v{entry['version']} | {n} chunk")
    print(f"{'='*55}")

    client.close()


if __name__ == "__main__":
    sync()

"""
سجل الملفات المفهرسة — بيمنع إعادة فهرسة ملف مااتغيرش.
اي ملف جديد يتفهرس، ياخد contentId جديد و version=1
ملف اتعدل الـ checksum اتغير → version=2 → vectorIds جديدة تمامًا
ملف زي ما هو يتخطى — صفر تكلفة API
"""

import json
import os

REGISTRY_PATH = "output/registry.json"


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def needs_indexing(file_path, checksum, registry):
    """
    بترجع (محتاج فهرسة؟, رقم الإصدار).
    - ملف جديد           → (True, 1)
    - ملف اتعدّل         → (True, version + 1)
    - ملف زي ما هو       → (False, version)
    """
    key = os.path.basename(file_path)
    entry = registry.get(key)

    if entry is None:
        return True, 1

    if entry["checksum"] != checksum:
        return True, entry["version"] + 1

    return False, entry["version"]


def register(file_path, checksum, version, content_id, chunk_count, registry):
    """بيسجل الملف بعد نجاح الفهرسة."""
    registry[os.path.basename(file_path)] = {
        "checksum": checksum,
        "version": version,
        "contentId": content_id,
        "chunkCount": chunk_count,
    }
    return registry


def next_content_id(registry, start=1000):
    """بيولّد contentId جديد — مؤقت لحد ما ييجي من الـ Backend."""
    used = [e["contentId"] for e in registry.values()]
    return max(used) + 1 if used else start


def find_orphans(registry, data_dir="data"):
    """
    بتلاقي الملفات اللي اتشالت من data/ بس لسه مفهرسة.
    البند 7.4: المحتوى المحذوف ما يعودش في نتائج البحث.
    """
    existing = {
        f
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf") and not f.startswith("._")
    }

    return {name: entry for name, entry in registry.items() if name not in existing}


def unregister(file_name, registry):
    """بتشيل ملف من السجل بعد حذف vectors بتاعته."""
    registry.pop(file_name, None)
    return registry

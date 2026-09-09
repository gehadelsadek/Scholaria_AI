"""
storage.py — تصدير الـ chunks لملفات JSON/TXT للمراجعة اليدوية
(مش خطوة إلزامية في أي pipeline — أداة مساعدة وقت التصحيح).
"""
import json
import os


def save_chunks(chunks, output_dir="output", name="chunks"):
    """بتحفظ الـ chunks في JSON (للاستخدام البرمجي) وTXT (للمراجعة البشرية)."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON — دا اللي هيتقري بعدين وقت الـ embeddings
    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # TXT — عشان تراجعي بعينك
    txt_path = os.path.join(output_dir, f"{name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
            page = c.get("primary_page") or (c["pages"][0] if c.get("pages") else "?")
            f.write(f"{'='*60}\n")
            f.write(f"#{i} | صفحات: {page}\n")
            f.write(f"{'='*60}\n")
            f.write(c.get("text", "") + "\n\n")

    print(f"[storage] اتحفظ {len(chunks)} chunk في:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")

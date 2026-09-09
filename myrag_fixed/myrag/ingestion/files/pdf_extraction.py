import io
import os
import tempfile

import pymupdf
import pytesseract
from PIL import Image

from ingestion.files.file_chunking import is_broken_encoding, strip_inline_boilerplate

# مسار مؤقت اختياري لملفات OCR — يتفعّل بس لو اتحدد صراحةً عن طريق
# متغير بيئة، عشان الكود يفضل شغال بشكل طبيعي على أي نظام تشغيل
# (Windows/Linux/Mac) من غير ما يفرض مسار جهاز حد بعينه.
_TMP = os.getenv("PDF_OCR_TMP_DIR")
if _TMP:
    os.makedirs(_TMP, exist_ok=True)
    os.environ["TMP"] = _TMP
    os.environ["TEMP"] = _TMP
    tempfile.tempdir = _TMP

# لو Tesseract مش متعرف تلقائيًا على جهازك، حددي مساره عن طريق متغير
# البيئة TESSERACT_CMD بدل ما تتعدل هنا (عشان الكود يفضل قابل للتشغيل
# على أي جهاز/نظام تشغيل من غير تعديل).
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def _extract_text_layer(page):
    """بتستخرج النص المكتوب مرتبًا حسب موضعه البصري."""
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    text_blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
    return "\n".join(b[4].strip() for b in text_blocks).strip()


def _ocr_page(page, dpi=250, lang="ara+eng"):
    """بتحول الصفحة لصورة وتعمل عليها OCR."""
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang).strip()


def _useful_length(text):
    """طول النص بعد إزالة الـ boilerplate — ده اللي يحدد الحاجة للـ OCR."""
    return len(strip_inline_boilerplate(text).strip())


def extract_pdf(pdf_path, use_ocr=True, min_text_length=50, dpi=250):
    """
    الدالة الرئيسية: بتستخرج النص من PDF.
    بترجع لـ OCR لو النص قليل أو لو الترميز مكسور.
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    ocr_count = 0
    broken_count = 0

    for i in range(len(doc)):
        page = doc[i]
        text = _extract_text_layer(page)
        used_ocr = False

        broken = is_broken_encoding(text)
        needs_ocr = _useful_length(text) < min_text_length or broken

        if use_ocr and needs_ocr:
            ocr_text = _ocr_page(page, dpi=dpi)

            if broken and len(ocr_text) > 30:
                text = ocr_text  # نستبدل حتى لو أقصر
                used_ocr = True
                broken_count += 1
            elif _useful_length(ocr_text) > _useful_length(text):
                text = ocr_text
                used_ocr = True

            if used_ocr:
                ocr_count += 1

        pages.append(
            {
                "page": i + 1,
                "text": text,
                "ocr": used_ocr,
            }
        )

    doc.close()
    print(
        f"[pdf] {len(pages)} صفحة | OCR لـ {ocr_count} "
        f"(منهم {broken_count} ترميز مكسور)"
    )
    return pages

"""
توليد الـ vectors — dense (المعنى) و sparse (الكلمات المفتاحية).

⚠️ الملف ده لازم يكون نسخة واحدة بالحرف عند مسار الملفات ومسار الفيديو.
أي اختلاف في الموديل أو الأبعاد = vectors مش قابلة للمقارنة.
"""

import os

# مسار تخزين الموديلات — C: مساحته ضيقة
os.environ["HF_HOME"] = r"F:\hf_cache"
FASTEMBED_CACHE = r"F:\fastembed_cache"

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

# ⚠️ متفق عليها بين المسارين: bge-m3 / 1024 بُعد / COSINE
DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"

_dense = None
_sparse = None


# ============================================================
#  Dense — المعنى
# ============================================================


def get_dense_model():
    """بنحمّل الموديل مرة واحدة بس (تحميله تقيل)."""
    global _dense
    if _dense is None:
        print(f"[embed] بيحمّل {DENSE_MODEL}...")
        _dense = SentenceTransformer(DENSE_MODEL)
        print(f"[embed] جاهز | أبعاد الـ vector: {_dense.get_embedding_dimension()}")
    return _dense


def embed_documents(texts, batch_size=8):
    """
    بتحول نصوص المستندات لـ vectors.
    bge-m3 مش محتاجة بادئات — النص بيتبعت زي ما هو.
    """
    return get_dense_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def embed_query(query):
    """
    بتحول سؤال المستخدم لـ vector.
    bge-m3 بتستخدم نفس التمثيل للسؤال والمستند.
    """
    return get_dense_model().encode(query, normalize_embeddings=True)


# ============================================================
#  Sparse — الكلمات المفتاحية (BM25)
# ============================================================


def get_sparse_model():
    global _sparse
    if _sparse is None:
        print(f"[sparse] بيحمّل {SPARSE_MODEL}...")
        _sparse = SparseTextEmbedding(
            model_name=SPARSE_MODEL,
            cache_dir=FASTEMBED_CACHE,
        )
        print("[sparse] جاهز")
    return _sparse


def embed_sparse_documents(texts):
    """بترجع sparse vectors للمستندات."""
    return list(get_sparse_model().embed(texts))


def embed_sparse_query(query):
    """بترجع sparse vector للسؤال."""
    return list(get_sparse_model().query_embed(query))[0]

"""
embedding.py — توليد الـ vectors، موحّد بين كل المصادر.

Dense:  BAAI/bge-m3   (1024-dim, cosine)
Sparse: Qdrant/bm25   (fastembed, IDF عند الفهرسة)

⚠️ نسخة واحدة بالحرف عند المسارين.
أي اختلاف في الموديل أو الأبعاد = vectors مش قابلة للمقارنة.
"""

import os
from pathlib import Path

# مسار تخزين الموديلات — عدّله حسب جهازك لو C: ضيق
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", "models_cache"))
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

VECTOR_SIZE = 1024
EMBED_BATCH_SIZE = 32

# الموديلات متكاشة على مستوى الـ module عشان أي pipeline في نفس الـ process
# يشارك نفس النسخة المحمّلة بدل ما كل واحد يحمّل نسخته.
_dense_model = None
_sparse_model = None


# =========================================================
#  تحميل الموديلات
# =========================================================


def load_dense_model() -> SentenceTransformer:
    """بيحمّل bge-m3 مرة واحدة بس ويكاشه."""
    global _dense_model
    if _dense_model is None:
        print(f"[embed] بيحمّل {DENSE_MODEL}...")
        _dense_model = SentenceTransformer(DENSE_MODEL, cache_folder=str(MODEL_CACHE))
        print("[embed] جاهز")
    return _dense_model


def load_sparse_model() -> SparseTextEmbedding:
    """بيحمّل BM25 مرة واحدة بس."""
    global _sparse_model
    if _sparse_model is None:
        print(f"[sparse] بيحمّل {SPARSE_MODEL}...")
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL, cache_dir=str(MODEL_CACHE)
        )
        print("[sparse] جاهز")
    return _sparse_model


# =========================================================
#  الفهرسة
# =========================================================


def encode_dense(texts, model=None):
    """dense vectors وقت الفهرسة."""
    model = model or load_dense_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]


def encode_sparse(texts, model=None):
    """
    sparse embeddings وقت الفهرسة.
    .embed() مش .query_embed() — BM25 بيحسب IDF وقت الفهرسة
    وبيوزن الكلمات وقت السؤال بشكل مختلف.
    """
    model = model or load_sparse_model()
    return list(model.embed(texts))


# =========================================================
#  البحث
# =========================================================


def encode_dense_query(query: str, model=None):
    """dense vector لسؤال واحد."""
    model = model or load_dense_model()
    return model.encode([query], normalize_embeddings=True)[0].tolist()


def encode_sparse_query(query: str, model=None):
    """sparse embedding لسؤال واحد — query_embed مش embed."""
    model = model or load_sparse_model()
    return list(model.query_embed(query))[0]

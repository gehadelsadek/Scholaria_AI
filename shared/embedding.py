"""
embedding.py
============

تحميل موديلات الـ embedding (dense + sparse) واستخدامها - موحّد بين
كل أنواع المحتوى (فيديو، ملفات، ...) عشان أي pipeline يستخدم نفس
الموديل بدل ما يحمّله لوحده (كان اسمها Embed_upsert_qdrant.py وكانت
فيها تحميل الموديلات + الـ upsert مع بعض؛ دلوقتي الـ upsert اتفصل في
qdrant_upsert.py).

Dense:  BAAI/bge-m3   (1024-dim, cosine)
Sparse: Qdrant/bm25    (fastembed, IDF-weighted عند الفهرسة)

Usage:
    from ingestion.shared.embedding import (
        load_embedding_model, load_sparse_model,
        encode_dense, encode_sparse, encode_dense_query, encode_sparse_query,
    )
"""
from pathlib import Path

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "Qdrant/bm25"
MODEL_CACHE = Path("models_cache")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

EMBED_BATCH_SIZE = 32

# بنكاش الموديلات على مستوى الـ module عشان أي pipeline (video/files)
# في نفس الـ process يشارك نفس النسخة المحمّلة، بدل ما كل pipeline
# يحمّل نسخته الخاصة.
_dense_model: SentenceTransformer | None = None
_sparse_model: SparseTextEmbedding | None = None


def load_embedding_model() -> SentenceTransformer:
    """بيحمّل BGE-M3 مرة واحدة بس ويكاشه طول عمر الـ process الحالي."""
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE))
    return _dense_model


def load_sparse_model() -> SparseTextEmbedding:
    """بيحمّل موديل BM25 (fastembed) مرة واحدة بس."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME, cache_dir=str(MODEL_CACHE)
        )
    return _sparse_model


def encode_dense(texts: list[str], model: SentenceTransformer | None = None) -> list[list[float]]:
    """dense vectors وقت الفهرسة (indexing)."""
    model = model or load_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def encode_sparse(texts: list[str], model: SparseTextEmbedding | None = None) -> list:
    """sparse embeddings وقت الفهرسة - .embed() مش .query_embed()، عشان
    BM25 بيحسب IDF وقت الفهرسة و weight الكلمات وقت السؤال بشكل مختلف."""
    model = model or load_sparse_model()
    return list(model.embed(texts))


def encode_dense_query(query: str, model: SentenceTransformer | None = None) -> list[float]:
    """dense vector لسؤال واحد وقت البحث."""
    model = model or load_embedding_model()
    return model.encode([query], normalize_embeddings=True)[0].tolist()


def encode_sparse_query(query: str, model: SparseTextEmbedding | None = None):
    """sparse embedding لسؤال واحد وقت البحث (query_embed، مش embed)."""
    model = model or load_sparse_model()
    return list(model.query_embed(query))[0]

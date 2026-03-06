"""
Shared embedding helper for indexing and search.
Uses sentence-transformers (384-dim) when available; else placeholder for compatibility.
"""

from typing import List
import os

# Try to quiet down transformer / HF logs and progress bars.
# This does NOT change the embeddings, it only affects console noise.
os.environ.setdefault("DISABLE_TQDM", "1")
try:
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    # If transformers is not available for some reason, just skip log tuning.
    pass


# 384 matches VectorDB index dimension (all-MiniLM-L6-v2)
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_tried = False


def _get_model():
    global _model, _tried
    if _tried:
        return _model
    _tried = True
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    except ImportError:
        _model = None
    return _model


def encode(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Encode text to a vector. Uses real embeddings if sentence-transformers is installed,
    otherwise a hash-based placeholder (KNN results will be poor).
    """
    model = _get_model()
    if model is not None:
        vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()
    # Fallback: same placeholder as original index_sbicard (hash-based)
    h = hash(text) % 10000
    return [0.1 + (h % 10) / 1000.0] * dim


def using_real_embeddings() -> bool:
    """True if sentence-transformers is available and will be used."""
    return _get_model() is not None

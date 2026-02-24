"""
Search strategies for VectorDB (Strategy Pattern).

Provides KNN (semantic), BM25 (keyword), and Hybrid (weighted combination)
search with score normalization and configurable weights.
"""

from vector_db.search.base import SearchStrategy
from vector_db.search.bm25 import BM25SearchStrategy
from vector_db.search.hybrid import HybridSearchStrategy
from vector_db.search.knn import KNNSearchStrategy

__all__ = [
    "SearchStrategy",
    "KNNSearchStrategy",
    "BM25SearchStrategy",
    "HybridSearchStrategy",
]

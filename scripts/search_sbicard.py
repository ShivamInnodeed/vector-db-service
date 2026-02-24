"""
Search indexed sbicard data via VectorDB: semantic (KNN) and hybrid (KNN + BM25).

Returns links and headers for each result. Uses hardcoded search queries;
config from .env (ELASTICSEARCH_URL etc.).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient

INDEX_NAME = "sbicard_chunks"
VECTOR_DIM = 384

# Hardcoded search queries (keywords) – returns relevant links + headers from trained data
SEARCH_QUERIES = [
    "credit card apply",
    "pay bill online",
    "login",
    "rewards",
    "contact help",
    "mobile app download",
    "NEFT payment",
    "lost card report",
    "shivam sharma",
]


def placeholder_embedding(query: str, dim: int = VECTOR_DIM):
    """Placeholder query embedding (same as index: hash-based). For real semantic use an embedding model."""
    h = hash(query) % 10000
    return [0.1 + (h % 10) / 1000.0] * dim


def format_result(r) -> str:
    link = (r.metadata or {}).get("link", "")
    header = (r.metadata or {}).get("header", "")
    return f"  [{header}] {link}" if header else f"  {link}"


def main():
    client = VectorDBClient()
    top_k = 5
    print("Searching indexed sbicard data (semantic + hybrid). Top results: link + header.\n")
    for query in SEARCH_QUERIES:
        print(f"Query: «{query}»")
        query_embedding = placeholder_embedding(query)
        # 1) Semantic (KNN)
        try:
            results_knn = client.search(
                query_embedding=query_embedding,
                index=INDEX_NAME,
                top_k=top_k,
            )
            print("  Semantic (KNN):")
            for r in results_knn[:3]:
                print(format_result(r))
        except Exception as e:
            print(f"  Semantic error: {e}")
        # 2) Hybrid (KNN + BM25 weighted)
        try:
            results_hybrid = client.search_hybrid(
                query_embedding=query_embedding,
                query_text=query,
                index=INDEX_NAME,
                top_k=top_k,
                knn_weight=0.4,
                bm25_weight=0.6,
            )
            print("  Hybrid (KNN+BM25):")
            for r in results_hybrid[:3]:
                print(format_result(r))
        except Exception as e:
            print(f"  Hybrid error: {e}")
        print()


if __name__ == "__main__":
    main()

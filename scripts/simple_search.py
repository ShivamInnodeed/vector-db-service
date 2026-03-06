"""
Fast search over indexed sbicard data — no sentence-transformers model load.

Uses placeholder (hash-based) query vectors for KNN, so startup is instant.
BM25 and hybrid (KNN+BM25) run the same; only semantic quality is lower than
search_sbicard.py (which loads the real embedding model).

Use this for quick checks. For best semantic results, use search_sbicard.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient

INDEX_NAME = "sbicard_chunks"
VECTOR_DIM = 384

SEARCH_QUERIES = [
    # "credit card apply",
    # "pay bill online",
    # "login",
    # "rewards",
    # "contact help",
    # "mobile app download",
    "NEFT payment",
    "lost card report",
]


def placeholder_embedding(query: str, dim: int = VECTOR_DIM):
    """Hash-based placeholder vector — no model load, instant."""
    h = hash(query) % 10000
    return [0.1 + (h % 10) / 1000.0] * dim


def format_result(r) -> str:
    link = (r.metadata or {}).get("link", "")
    header = (r.metadata or {}).get("header", "")
    return f"  [{header}] {link}" if header else f"  {link}"


def main():
    client = VectorDBClient()
    top_k = 5
    print("Simple search (no model load). KNN uses placeholder; BM25 unchanged.\n")
    for query in SEARCH_QUERIES:
        print(f"Query: «{query}»")
        q_embedding = placeholder_embedding(query)
        try:
            results_knn = client.search(
                query_embedding=q_embedding,
                index=INDEX_NAME,
                top_k=top_k,
            )
            print("  Semantic (KNN):")
            for r in results_knn[:3]:
                print(format_result(r))
        except Exception as e:
            print(f"  Semantic error: {e}")
        try:
            results_bm25 = client.search_bm25(
                query_text=query,
                index=INDEX_NAME,
                top_k=top_k,
            )
            print("  BM25 (keyword):")
            for r in results_bm25[:3]:
                print(format_result(r))
        except Exception as e:
            print(f"  BM25 error: {e}")
        try:
            results_hybrid = client.search_hybrid(
                query_embedding=q_embedding,
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

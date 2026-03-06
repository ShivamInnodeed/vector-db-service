"""
Search indexed sbicard data via VectorDB: semantic (KNN), BM25 (keyword) and hybrid (KNN + BM25).

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
from scripts.embedding_utils import encode, using_real_embeddings, EMBEDDING_DIM

INDEX_NAME = "sbicard_chunks"

# Hardcoded search queries (keywords) – returns relevant links + headers from trained data
# SEARCH_QUERIES = [
#     "How can I pay my SBI credit card bill through NELT?",
#     "Use NEFT and pay your SBI Card bills from any bank",
#     "NEFT payment",
#     "lost card report",
#     # "pay bill online",
#     # "login",
#     # "rewards",
#     # "contact help",
#     # "mobile app download",
#     # "NEFT payment",
#     # "lost card report",
#     # "shivam sharma",
#     # "signature card",
#     # "signup",
#     # "signin",
#     # "wallet",
#     # "wallet balance",
#     # "wallet transfer",
#     # "wallet recharge",
#     # "wallet withdrawal",
#     # "wallet statement",
#     # "wallet history",
#     # "wallet statement",
#     # "recommendations",
#     # "recommendation",
# ]

SEARCH_QUERIES = [
    "credit card apply",
    "pay bill online",
    "login",
    "rewards",
    "contact help",
    "mobile app download",
    "NEFT payment",
    "lost card report",
    "How can I pay my SBI credit card bill through NEFT?",
    "I lost my SBI credit card, how do I report and block it?",
]

def query_embedding(query: str):
    """Query vector (real embeddings if sentence-transformers installed, else placeholder)."""
    return encode(query, EMBEDDING_DIM)


def format_result(r) -> str:
    link = (r.metadata or {}).get("link", "")
    header = (r.metadata or {}).get("header", "")
    return f"  [{header}] {link}" if header else f"  {link}"


def main():
    client = VectorDBClient()
    top_k = 5
    print("Searching indexed sbicard data (semantic + BM25 + hybrid). Top results: link + header.\n")
    if using_real_embeddings():
        print("Using real embeddings for semantic/hybrid.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")
    for query in SEARCH_QUERIES:
        print(f"Query: «{query}»")
        q_embedding = query_embedding(query)
        # 1) Semantic (KNN)
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
        # 2) BM25 (keyword-only)
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
        # 3) Hybrid (KNN + BM25 weighted)
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

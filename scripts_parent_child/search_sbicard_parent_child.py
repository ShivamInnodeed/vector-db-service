"""
Search sbicard data indexed with parent–child style chunks.

Children: short text used for search (embedding + BM25).
Parent snippet: longer context snippet stored in metadata and
shown to the user as a \"long header\" / short answer along with
the URL.

This script lives in scripts_parent_child/ so we can compare
results side–by–side with the original scripts/search_sbicard.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient
from scripts.embedding_utils import encode, using_real_embeddings, EMBEDDING_DIM

INDEX_NAME = "sbicard_parent_child"

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
    return encode(query, EMBEDDING_DIM)


def format_result_long(r) -> str:
    meta = r.metadata or {}
    link = meta.get("link", "")
    header = meta.get("header", "")
    parent_snippet = (meta.get("parent_snippet") or "").strip()
    snippet_short = (parent_snippet[:260] + "…") if len(parent_snippet) > 260 else parent_snippet
    lines = []
    if header:
        lines.append(f"  [{header}] {link}")
    else:
        lines.append(f"  {link}")
    if snippet_short:
        lines.append(f"    {snippet_short}")
    return "\n".join(lines)


def main():
    client = VectorDBClient()
    top_k = 5
    print("Searching sbicard_parent_child (children for search, parent snippet for display).\n")
    if using_real_embeddings():
        print("Using real embeddings for semantic/hybrid.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")

    for query in SEARCH_QUERIES:
        print(f"Query: «{query}»")
        q_embedding = query_embedding(query)

        # Hybrid (KNN + BM25) is the most relevant for comparison here.
        try:
            results_hybrid = client.search_hybrid(
                query_embedding=q_embedding,
                query_text=query,
                index=INDEX_NAME,
                top_k=top_k,
                knn_weight=0.4,
                bm25_weight=0.6,
            )
            print("  Hybrid (KNN+BM25) with long snippet:")
            for r in results_hybrid[:3]:
                print(format_result_long(r))
        except Exception as e:
            print(f"  Hybrid error: {e}")

        print()
 

if __name__ == "__main__":
    main()


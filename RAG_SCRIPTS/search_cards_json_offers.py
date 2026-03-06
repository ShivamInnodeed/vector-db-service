"""
Search over credit_card_offers_rag_json index built from data/cards_json_data.

Supports:
- Hybrid search (KNN + BM25) via VectorDBClient.search_hybrid
- Optionally, pure KNN and pure BM25 for debugging

Displays results grouped by card_name with rich snippets.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient  # type: ignore
from scripts.embedding_utils import EMBEDDING_DIM, encode, using_real_embeddings  # type: ignore
from RAG_SCRIPTS.intent_detection import detect_intents  # type: ignore
from RAG_SCRIPTS.ranking_utils import rerank_results  # type: ignore


INDEX_NAME = "credit_card_offers_rag_json"


# Predefined test queries array (used when no CLI args are passed)
DEFAULT_QUERIES: List[str] = [
    # "best cashback sbi credit card",
    # "which sbi card gives movie discount",
    # "flipkart cashback credit card sbi",
    # "sbi credit card with low annual fee",
    # "best sbi card for travel rewards",
    # "which sbi credit card gives maximum cashback?",
    # "20% discount are on which sbi card",
    # "best sbicard for shopping cashbak",
    # "good rewards card from sbi",
    # "sbi credit card offers for movies or dining",
    # "Which SBI card is best for shopping and cashback?"
    # "best cashback sbi card",
    # "sbi card movie offers",
    # "Which SBI credit card gives the highest cashback on online shopping like Flipkart, Myntra or Amazon?",
    # "I mostly spend on groceries and dining, which SBI credit card will give me the best rewards or cashback?",
    # "Which SBI credit card has the lowest annual fee but still gives good shopping rewards? ",
    # "Are there any SBI credit cards that give special discounts or cashback for fuel purchases and utility bill payments?",
    # "Which SBI card is best for frequent travelers who want travel rewards, flight benefits and partner discounts?",
    # "I want an SBI credit card mainly for online shopping and occasional movie bookings, which card should I choose?",
    # "sbicard cashback for myntra flipkart shopping which card best",
    # "Which SBI credit card offers both cashback on shopping and rewards on dining or entertainment purchases?"
    
    # Relevant queries (Credit Card Offers)
    # "Which SBI credit card gives the best cashback on online shopping like Flipkart or Myntra?",
    # "Is there any SBI credit card that offers rewards for dining and grocery purchases?",
    # "Which SBI card has the lowest annual fee but still gives good cashback on shopping?",
    # "Are there SBI credit cards that give discounts on movie tickets or entertainment bookings?",
    # "Which SBI credit card is best for travel rewards and airline partner benefits?",

    # # Irrelevant queries (Noise testing)
    # "What is the weather in Delhi tomorrow?",
    # "How to cook pasta at home step by step?",
    # "Who won the last FIFA World Cup?",
    # "Best programming language to learn in 2025?",
    # "How to improve battery life on an Android phone?"


  "Which SBI card gives highest cashback on Flipkart?",
  "Best low fee SBI card for grocery and dining",
  "SBI travel card with IndiGo benefits?",
  "Compare Flipkart SBI vs SimplyCLICK cashback rates",
  "What is weather in Mumbai tomorrow?",
  "SBI card with lowest annual fee but good rewards",
  "Best SBI card for Amazon Pay and movie tickets",
  "How to make biryani recipe?",
  "SBI cards offering lounge access under ₹1000 fee",
  "Which SBI card maximizes rewards for IT professionals (online shopping + fuel)?",
  "Winner of IPL 2026?",
  "Myntra + Cleartrip rewards comparison across SBI cards",
  "SBI SimplySAVE vs SimplyCLICK for students",
  "Best Python framework for web dev?",
  "SBI cards with fuel surcharge waiver + groceries",
  "Calculate effective ROI: Flipkart SBI vs Cashback SBI for ₹50k annual Flipkart spend",
  "How to fix iPhone battery drain?",
  "No fee SBI cards for international travel",
  "SBI card eligibility for freshers with 4LPA salary",
  "Latest Bollywood movie reviews"
]


def query_embedding(query: str):
    return encode(query, EMBEDDING_DIM)


def format_chunk(r) -> str:
    meta = r.metadata or {}
    chunk_type = meta.get("chunk_type", "")
    source_url = meta.get("source_url", "")
    parent_snippet = (meta.get("parent_snippet") or "").strip()
    snippet_short = (parent_snippet[:260] + "…") if len(parent_snippet) > 260 else parent_snippet
    line = f"    [{chunk_type}] {source_url}" if source_url else f"    [{chunk_type}]"
    if snippet_short:
        line += f"\n      {snippet_short}"
    return line


def print_query_embeddings(queries: List[str]) -> None:
    """Print embeddings for a list of queries for debugging."""
    if not queries:
        return

    if using_real_embeddings():
        print("Embeddings below are computed with MiniLM-L6-v2 (dim={}).")
        # print(f"Embeddings below are computed with MiniLM-L6-v2 (dim={EMBEDDING_DIM}).")
    else:
        print(
            "WARNING: sentence-transformers model not loaded; using placeholder embeddings instead of MiniLM-L6-v2."
        )

    for query in queries:
        if not query:
            continue
        emb = query_embedding(query)
        print(f"\nQuery: «{query}»")
        print(f"Embedding length: {len(emb)}")
        # print(f"Embedding vector: {emb}")


def run_query_hybrid(client: VectorDBClient, query: str, top_k: int = 10) -> None:
    # User-facing contract: how search results should be interpreted
    print(f"\nQuery: «{query}»")

    intents = detect_intents(query)
    if intents:
        print(f"Detected intents: {intents}")
    else:
        print("Detected intents: []")
    emb = query_embedding(query)

    try:
        results = client.search_hybrid(
            query_embedding=emb,
            query_text=query,
            index=INDEX_NAME,
            top_k=top_k,
            knn_weight=0.4,
            bm25_weight=0.6,
        )
    except Exception as e:
        print(f"  Hybrid search error: {e}")
        return

    if not results:
        print("  No results.")
        return

    # Re-rank results using metadata and numeric features only (no external data).
    ranked_results = rerank_results(results, intents)

    grouped: DefaultDict[str, List] = defaultdict(list)
    for r in ranked_results:
        meta = r.metadata or {}
        card_name = meta.get("card_name") or "Unknown card"
        grouped[card_name].append(r)

    print("  Top matches by card (hybrid):")
    for card_name, chunks in list(grouped.items())[:5]:
        meta0 = chunks[0].metadata or {}
        category = meta0.get("category", "")
        issuer = meta0.get("issuer", "")
        header = card_name
        if category and category != "unknown":
            header += f" [{category}]"
        if issuer and issuer != "unknown":
            header += f" – {issuer}"
        print(f"  {header}")
        for r in chunks[:3]:
            print(format_chunk(r))


def main() -> None:
    client = VectorDBClient()
    print(f"Searching index '{INDEX_NAME}' for JSON-based credit card offers.")
    if using_real_embeddings():
        print("Using real embeddings for semantic search.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        # Use predefined queries array when running without CLI arguments
        queries = list(DEFAULT_QUERIES)

    # Print embeddings for all provided queries (MiniLM-L6-v2 when real embeddings are enabled)
    print_query_embeddings(queries)

    for q in queries:
        run_query_hybrid(client, q, top_k=12)


if __name__ == "__main__":
    main()


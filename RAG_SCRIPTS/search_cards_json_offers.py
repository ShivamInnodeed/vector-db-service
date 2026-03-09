"""
Search over credit_card_offers_rag_json index built from data/cards_json_data.

Supports:
- Hybrid search (KNN + BM25) via VectorDBClient.search_hybrid
- Optionally, pure KNN and pure BM25 for debugging

Displays results grouped by card_name with rich snippets.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient  # type: ignore
from scripts.embedding_utils import EMBEDDING_DIM, encode, using_real_embeddings  # type: ignore
from RAG_SCRIPTS.intent_detection import detect_intents  # type: ignore
from RAG_SCRIPTS.entity_extraction import extract_entities, ExtractedEntities  # type: ignore
from RAG_SCRIPTS.synonym_expansion import expand_synonyms  # type: ignore
from RAG_SCRIPTS.ranking_utils import (  # type: ignore
    get_relevance_threshold,
    rerank_results_with_scores,
)


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


#   "Which SBI card gives highest cashback on Flipkart?",
#   "Best low fee SBI card for grocery and dining",
#   "SBI travel card with IndiGo benefits?",
#   "Compare Flipkart SBI vs SimplyCLICK cashback rates",
#   "What is weather in Mumbai tomorrow?",
#   "SBI card with lowest annual fee but good rewards",
#   "Best SBI card for Amazon Pay and movie tickets",
#   "How to make biryani recipe?",
#   "SBI cards offering lounge access under ₹1000 fee",
#   "Which SBI card maximizes rewards for IT professionals (online shopping + fuel)?",
#   "Winner of IPL 2026?",
#   "Myntra + Cleartrip rewards comparison across SBI cards",
#   "SBI SimplySAVE vs SimplyCLICK for students",
#   "Best Python framework for web dev?",
#   "SBI cards with fuel surcharge waiver + groceries",
#   "Calculate effective ROI: Flipkart SBI vs Cashback SBI for ₹50k annual Flipkart spend",
#   "How to fix iPhone battery drain?",
#   "No fee SBI cards for international travel",
#   "SBI card eligibility for freshers with 4LPA salary",
#   "Latest Bollywood movie reviews"

# 1️⃣ simple cashback query
"Best SBI card for online shopping cashback",

# 2️⃣ partner merchant query
"Which SBI credit card gives rewards on Myntra purchases?",

# 3️⃣ category specific query
"SBI credit card with maximum benefits for dining and movies",

# 4️⃣ comparison query
"Compare Flipkart SBI Card and Cashback SBI Card for online shopping rewards",

# 5️⃣ constraint query
"SBI credit cards with annual fee below 1000 and good reward points",

# 6️⃣ multi-intent query
"Best SBI credit card for travel, airport lounge access, and reward points",

# 7️⃣ realistic user query
"I spend a lot on Amazon, groceries, and fuel — which SBI credit card is best?",

# 8️⃣ edge semantic query
"SBI card that gives the most benefits for e-commerce purchases",

# 9️⃣ irrelevant query (should return nothing)
"How to train a neural network using PyTorch?",

# 🔟 tricky mixed query
"Which SBI credit card gives good cashback on Flipkart and also movie ticket offers?"

]


def query_embedding(query: str):
    return encode(query, EMBEDDING_DIM)


def _parse_annual_fee_from_meta(meta: Dict[str, Any]) -> float:
    """Parse numeric annual fee from metadata (e.g. '499', '₹999'). Unknown/missing -> 0."""
    val = meta.get("annual_fee")
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s or s.lower() == "unknown":
        return 0.0
    m = re.findall(r"\d+", s)
    return float(m[0]) if m else 0.0


def build_filter_from_entities(entities: ExtractedEntities) -> Optional[Dict[str, Any]]:
    """
    Build an Elasticsearch filter from extracted entities (categories only).
    Index stores category-like values in category_tags and partner_category_tags, not partner names;
    partners are used only for rerank-time boost.
    max_annual_fee is applied as a hard constraint after search (see apply_hard_fee_filter).
    """
    if not entities.categories:
        return None
    should_clauses: List[Dict[str, Any]] = [
        {"terms": {"metadata.category_tags": entities.categories}},
        {"terms": {"metadata.partner_category_tags": entities.categories}},
    ]
    return {"bool": {"should": should_clauses}}


def apply_hard_fee_filter(
    results: List[Any], max_annual_fee: Optional[float]
) -> List[Any]:
    """Hard constraint: keep only results where annual_fee <= max_annual_fee. Not optional when set."""
    if max_annual_fee is None:
        return results
    out: List[Any] = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        fee = _parse_annual_fee_from_meta(meta)
        if fee <= max_annual_fee:
            out.append(r)
    return out


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


def run_query_hybrid(client: VectorDBClient, query: str, top_k: int = 20) -> None:
    # Pipeline: Intent detection -> Entity extraction -> Constraint filter -> Synonym expansion
    # -> Hybrid search (top_k) -> Feature rerank -> Dynamic threshold -> Top results
    print(f"\nQuery: «{query}»")

    intents = detect_intents(query)
    static_threshold = get_relevance_threshold(intents)
    if intents:
        print(f"Detected intents: {intents}")
    else:
        print("Detected intents: []")
        print(f"Score (threshold): {static_threshold}")
        print("  No relevant credit card results found.")
        return
    print(f"Score (threshold): {static_threshold}")

    entities = extract_entities(query, intents)
    if entities.partners or entities.categories or entities.max_annual_fee is not None:
        print(f"Entities: partners={entities.partners}, categories={entities.categories}, max_annual_fee={entities.max_annual_fee}")

    es_filter = build_filter_from_entities(entities)
    if es_filter:
        print("Filter: applied (categories/partners)")

    expanded_query = expand_synonyms(query)
    if expanded_query != query:
        print(f"Expanded query: «{expanded_query}»")

    emb = query_embedding(expanded_query)

    try:
        results = client.search_hybrid(
            query_embedding=emb,
            query_text=expanded_query,
            index=INDEX_NAME,
            top_k=top_k,
            filters=es_filter,
            knn_weight=0.4,
            bm25_weight=0.6,
        )
    except Exception as e:
        print(f"  Hybrid search error: {e}")
        return

    if not results:
        print("  No results.")
        return

    # Hard constraint: only cards with annual_fee <= max_annual_fee (not optional)
    if entities.max_annual_fee is not None:
        results = apply_hard_fee_filter(results, entities.max_annual_fee)
        if not results:
            print("  No results after applying annual fee filter (max_annual_fee).")
            return
        print(f"Filter: annual_fee <= {entities.max_annual_fee} (hard constraint)")

    scored_results = rerank_results_with_scores(results, intents, entities=entities)
    if not scored_results:
        print("  No results after re-ranking.")
        return

    # Dynamic per-query threshold for intentful queries:
    # threshold = max(0.25, mean_score - 0.15)
    scores_only = [s for s, _ in scored_results]
    mean_score = sum(scores_only) / len(scores_only)
    dynamic_threshold = max(0.25, mean_score - 0.15)

    # Noise filter: use dynamic threshold so we adapt to each query's score distribution.
    ranked_results = [r for score, r in scored_results if score >= dynamic_threshold]
    passed_scores = [s for s, r in scored_results if s >= dynamic_threshold]

    if not ranked_results:
        print(f"  No relevant credit card results found (dynamic_thresh={dynamic_threshold:.3f}).")
        return

    score_max = max(passed_scores) if passed_scores else 0.0
    score_min = min(passed_scores) if passed_scores else 0.0
    print(
        f"Score (passed): {len(ranked_results)} results, "
        f"mean={mean_score:.3f}, dyn_thresh={dynamic_threshold:.3f}, "
        f"min={score_min:.3f}, max={score_max:.3f}"
    )

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
        run_query_hybrid(client, q, top_k=20)


if __name__ == "__main__":
    main()


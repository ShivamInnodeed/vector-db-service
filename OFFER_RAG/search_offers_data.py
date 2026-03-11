"""
Search indexed SBI Card partner offers (sbicard_offers_rag) using VectorDB hybrid search.

Flow:
- Take user queries (hardcoded list or CLI input)
- Embed queries using scripts.embedding_utils.encode
- Run hybrid search (semantic KNN + BM25) via VectorDBClient.search_hybrid
- Return top 5 most relevant offer chunks with basic partner/offer details
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient  # type: ignore
from scripts.embedding_utils import EMBEDDING_DIM, encode, using_real_embeddings  # type: ignore


INDEX_NAME = "sbicard_offers_rag"
TOP_K = 5
MIN_SCORE = 0.35


DEFAULT_QUERIES: List[str] = [
    "best offer in dyson",
    "i have to travel in bus so which bus should i take",
    "electronics emi offers with sbi credit card",
    "online shopping discount offers on sbicard",
     "what offers are available on samsung products with sbi credit card",
    "discount offers on flipkart using sbi card",
    "emi offers on dell laptops with sbi credit card",
    "travel booking offers on mmt make my trip with sbi credit card",
    "food dining discounts available on swiggy with sbi card",
    "amazon shopping offers using sbi credit card",
    "electronics discount offers on lg appliances with sbi card",
    "flight booking discount on cleartrip with sbi card",
    "mobile phone offers on oppo with sbi credit card",
    "which all cards are eligible for abhibus offers"
]


def query_embedding(query: str):
    """Encode query text to embedding vector."""
    return encode(query, EMBEDDING_DIM)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


BRAND_ALIASES: Dict[str, str] = {
    # Normalize common brand spellings to the stored entity_name
    "makemytrip": "Mmt",
    "mmt": "Mmt",
    "amazon.in": "Amazon",
    "flipkart.com": "Flipkart",
}


def detect_brand(query: str) -> List[str]:
    """Very lightweight brand detection based on known partner names in offers."""
    q = _normalize(query)
    # Extend this list over time as needed
    brands = [
        "dyson",
        "abhibus",
        "cleartrip",
        "mmt",
        "makemytrip",
        "flipkart",
        "amazon",
        "samsung",
        "oppo",
        "lg",
        "whirlpool",
        "haier",
        "dell",
        "acer",
        "asus",
        "ather",
        "khosla electronics",
    ]
    detected: List[str] = []
    for b in brands:
        if b in q:
            detected.append(b)
    return detected


def detect_category(query: str) -> List[str]:
    """Coarse category detection: travel, electronics, dining, shopping, mobiles, grocery."""
    q = _normalize(query)
    categories: List[str] = []

    if any(w in q for w in ["flight", "flights", "hotel", "bus", "travel", "ticket", "booking", "makemytrip", "cleartrip", "abhibus"]):
        categories.append("travel")
    if any(w in q for w in ["electronics", "appliance", "appliances", "tv", "ac", "fridge", "laptop", "mobile", "phone", "smartphone"]):
        categories.append("electronics")
    if any(w in q for w in ["dining", "restaurant", "food", "swiggy", "zomato", "meal", "buffet"]):
        categories.append("dining")
    if any(w in q for w in ["shopping", "online shopping", "offer", "offers", "discount"]):
        categories.append("shopping")
    if any(w in q for w in ["mobile", "phone", "smartphone"]):
        categories.append("mobiles")
    if any(w in q for w in ["grocery", "groceries", "bigbasket", "milk", "vegetable", "vegetables"]):
        categories.append("grocery")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def detect_offer_type(query: str) -> List[str]:
    """Detect whether query is about EMI, discount or cashback."""
    q = _normalize(query)
    types: List[str] = []
    if "emi" in q or "installment" in q or "installments" in q:
        types.append("emi")
    if "discount" in q or "instant discount" in q or "% off" in q:
        types.append("discount")
    if "cashback" in q or "cash back" in q:
        types.append("cashback")
    # Deduplicate
    seen = set()
    out: List[str] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_filters(brands: List[str], categories: List[str], offer_types: List[str]) -> Dict[str, Any]:
    """
    Build a simple Elasticsearch filter dict for hybrid search.

    We keep this conservative and compatible with ElasticsearchClient._build_filter_query,
    which expects a flat dict of {field: value} for simple term filters.

    - Prefer brand filter when a brand is clearly detected.
    - Categories/offer types are handled via query expansion (soft), not filters.
    """
    if brands:
        # Use first detected brand token; normalize via aliases to match indexed brand values
        detected = brands[0]
        canonical = BRAND_ALIASES.get(detected, detected)
        return {"metadata.brand": canonical}
    return {}


def expand_query_with_intent(query: str, categories: List[str], offer_types: List[str]) -> str:
    """
    Soften category / offer-type impact by expanding the BM25 query text
    instead of applying hard filters.
    """
    expanded = query
    if "travel" in categories:
        expanded += " travel flights hotels bus ticket booking"
    if "electronics" in categories:
        expanded += " electronics appliances tv ac laptop"
    if "dining" in categories:
        expanded += " dining restaurants food"
    if "shopping" in categories:
        expanded += " shopping offers discounts"
    if "mobiles" in categories:
        expanded += " mobiles smartphones phones"
    if "grocery" in categories:
        expanded += " grocery groceries supermarket"

    if "emi" in offer_types:
        expanded += " emi installment installments"
    if "discount" in offer_types:
        expanded += " discount instant discount percent off"
    if "cashback" in offer_types:
        expanded += " cashback cash back"

    return expanded


def format_result(r) -> str:
    meta = r.metadata or {}
    entity_name = meta.get("entity_name", "") or ""
    offer_title = meta.get("offer_title", "") or ""
    source_url = meta.get("source_url", "") or ""
    parent_snippet = (meta.get("parent_snippet") or "").strip()
    entity_summary = (meta.get("entity_summary") or "").strip()
    offer_summary_display = (meta.get("offer_summary_display") or "").strip()

    snippet_source = parent_snippet if parent_snippet else (r.text or "")
    snippet = " ".join(snippet_source.split())
    if len(snippet) > 260:
        snippet = snippet[:257] + "..."

    header_parts: List[str] = []
    if entity_name:
        header_parts.append(entity_name)
    if offer_title:
        header_parts.append(offer_title)
    header = " | ".join(header_parts) or r.id

    score_info = f"(score={r.score:.4f})" if getattr(r, "score", None) is not None else ""

    line = f"- {header} {score_info}".rstrip()
    if source_url:
        line += f"\n  URL: {source_url}"
    summary_text = offer_summary_display or entity_summary
    if summary_text:
        summary_clean = " ".join(summary_text.split())
        if len(summary_clean) > 260:
            summary_clean = summary_clean[:257] + "..."
        line += f"\n  Summary: {summary_clean}"
    if snippet:
        line += f"\n  Snippet: {snippet}"
    return line


def deduplicate_and_filter_results(results: List[Any], top_k: int) -> List[Any]:
    """
    Deduplicate results by URL / offer and drop low-score noise.

    - Key is primarily source_url; fallback to entity_id:offer_id; finally result id.
    - Apply MIN_SCORE threshold on r.score when available.
    """
    unique: Dict[str, Any] = {}
    ordered: List[Any] = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        source_url = (meta.get("source_url") or "").strip()
        entity_id = meta.get("entity_id") or ""
        offer_id = meta.get("offer_id") or ""
        key = source_url or f"{entity_id}:{offer_id}" or getattr(r, "id", "")
        if key and key not in unique:
            unique[key] = r
            ordered.append(r)
        if len(ordered) >= top_k * 3:
            # No need to keep too many extras once we have enough distinct offers
            break

    # Apply score threshold
    strong: List[Any] = []
    weak: List[Any] = []
    for r in ordered:
        score = getattr(r, "score", None)
        if score is None or score >= MIN_SCORE:
            strong.append(r)
        else:
            weak.append(r)

    if strong:
        return strong[:top_k]
    # Fallback: if everything is below threshold, return weakest-but-closest matches
    return weak[:top_k]


def run_query_hybrid(client: VectorDBClient, query: str, top_k: int = TOP_K) -> None:
    print(f"\nQuery: «{query}»")

    brands = detect_brand(query)
    categories = detect_category(query)
    offer_types = detect_offer_type(query)
    if brands or categories or offer_types:
        print(f"  Detected brands: {brands or []}")
        print(f"  Detected categories: {categories or []}")
        print(f"  Detected offer types: {offer_types or []}")

    filters = build_filters(brands, categories, offer_types)
    if filters:
        print("  Applying metadata filters for brand/category/type.")

    expanded_query = expand_query_with_intent(query, categories, offer_types)
    if expanded_query != query:
        print(f"  Expanded query: «{expanded_query}»")

    emb = query_embedding(expanded_query)
    try:
        results = client.search_hybrid(
            query_embedding=emb,
            query_text=expanded_query,
            index=INDEX_NAME,
            top_k=top_k,
            filters=filters or None,
            knn_weight=0.75,
            bm25_weight=0.25,
        )
    except Exception as e:
        print(f"  Hybrid search error: {e}")
        return

    if not results:
        print("  No results.")
        return

    final_results = deduplicate_and_filter_results(results, top_k=top_k)
    if not final_results:
        print("  No highly relevant offers found (after filtering).")
        return

    print(f"  Top {len(final_results)} results (after dedup + score filter):")
    for r in final_results:
        print(format_result(r))


def main() -> None:
    client = VectorDBClient()
    print(f"Searching index '{INDEX_NAME}' for SBI Card partner offers.")
    if using_real_embeddings():
        print("Using real embeddings for semantic search.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = list(DEFAULT_QUERIES)

    for q in queries:
        if not q:
            continue
        run_query_hybrid(client, q, top_k=TOP_K)


if __name__ == "__main__":
    main()


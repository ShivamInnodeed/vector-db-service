"""
Synonym expansion for credit card search queries.

Expands the query with synonyms for BM25 and embedding to improve recall.
Rule-based only; aligned with intent_detection INTENT_KEYWORDS.
"""

from __future__ import annotations

from typing import Dict, List

# Map phrase (lowercase) -> list of synonyms to append (bounded: 1–2 per term).
# Used for both BM25 and embedding input.
SYNONYM_MAP: Dict[str, List[str]] = {
    "cashback": ["cash back"],
    "cash back": ["cashback"],
    "rewards": ["reward points", "points"],
    "reward points": ["rewards", "points"],
    "points": ["rewards", "reward points"],
    "movie": ["movies", "cinema"],
    "movies": ["cinema", "movie"],
    "cinema": ["movies", "movie"],
    "dining": ["restaurant", "restaurants"],
    "restaurant": ["dining", "restaurants"],
    "travel": ["flight", "airline"],
    "flight": ["travel", "airline"],
    "airline": ["travel", "flight"],
    "shopping": ["online shopping", "ecommerce"],
    "ecommerce": ["shopping", "online shopping"],
    "fuel": ["petrol", "diesel"],
    "petrol": ["fuel"],
    "fee": ["annual fee", "joining fee"],
    "annual fee": ["fee"],
    "lounge": ["airport lounge", "lounge access"],
    "lounge access": ["lounge", "airport lounge"],
}


def expand_synonyms(query: str, max_extra_terms: int = 8) -> str:
    """
    Expand the query with synonyms for BM25 and embedding.

    For each recognized phrase that has synonyms, appends up to 1–2 synonyms
    so the expanded query is used for both BM25 and encoding. Caps total
    extra terms to avoid huge queries.

    Args:
        query: Original user query.
        max_extra_terms: Maximum number of synonym terms to append (default 8).

    Returns:
        Expanded query string (original + selected synonyms).
    """
    if not query or not query.strip():
        return query

    q = query.strip()
    extra: List[str] = []
    seen: set[str] = set()
    q_lower = q.lower()

    # Prefer longer phrases first
    for phrase in sorted(SYNONYM_MAP.keys(), key=len, reverse=True):
        if len(extra) >= max_extra_terms:
            break
        if phrase not in q_lower:
            continue
        for syn in SYNONYM_MAP[phrase]:
            if syn not in seen and syn not in q_lower:
                extra.append(syn)
                seen.add(syn)
                if len(extra) >= max_extra_terms:
                    break
        if len(extra) >= max_extra_terms:
            break

    if not extra:
        return q
    return q + " " + " ".join(extra[:max_extra_terms])

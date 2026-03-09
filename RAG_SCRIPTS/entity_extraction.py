"""
Rule-based entity extraction from the user query for credit card search.

Extracts: partners (merchant/brand names), categories (category_tags), and
max_annual_fee constraint. Uses only allowlists derived from indexed JSON
metadata; no LLM. Does not invent entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# Partners: names that appear in partner_category_tags or partner earnings in cards JSON.
PARTNER_ALLOWLIST = frozenset({
    "flipkart", "amazon", "myntra", "cleartrip", "indigo", "air india", "air india express",
    "bookmyshow", "pvr", "vi", "vodafone", "idea", "aditya birla", "fab india", "spar",
    "phonepe", "paytm", "apollo", "hpcl", "bpcl", "iocl", "etihad", "krisflyer",
    "amazon pay", "apollo pharmacy", "max", "shoppers stop",
})

# Categories: terms that map to category_tags / partner_category_tags in indexed JSON.
CATEGORY_ALLOWLIST = frozenset({
    "dining", "movies", "entertainment", "travel", "shopping", "fuel", "hotels",
    "general", "utility", "grocery", "groceries", "online shopping", "ecommerce",
})

# Query phrase -> category tag (normalize query terms to index tag values).
QUERY_TO_CATEGORY: dict[str, str] = {
    "movie": "movies",
    "cinema": "movies",
    "restaurant": "dining",
    "restaurants": "dining",
    "food": "dining",
    "flight": "travel",
    "flights": "travel",
    "airline": "travel",
    "airport": "travel",
    "petrol": "fuel",
    "diesel": "fuel",
    "hotel": "hotels",
    "stay": "travel",
    "ecommerce": "shopping",
    "online shopping": "shopping",
    "store": "shopping",
    "groceries": "grocery",
    "grocery": "grocery",
}


@dataclass
class ExtractedEntities:
    """Entities extracted from the query (rule-based, allowlist only)."""

    partners: List[str]
    categories: List[str]
    max_annual_fee: Optional[float]  # None = no constraint; 0 = no fee; >0 = max rupees


def extract_entities(query: str, detected_intents: Optional[List[str]] = None) -> ExtractedEntities:
    """
    Extract partners, categories, and max_annual_fee from the query.

    Uses only allowlists and regex; does not invent values. detected_intents
    is optional and can be used in the future to tune extraction.
    """
    q = query.lower().strip()
    partners: List[str] = []
    categories: List[str] = []
    max_annual_fee: Optional[float] = None

    # Partners: find allowlisted tokens/phrases in query
    words = re.findall(r"[a-z0-9]+", q)
    # Check multi-word phrases first (e.g. "air india", "aditya birla")
    for partner in sorted(PARTNER_ALLOWLIST, key=len, reverse=True):
        if partner in q and partner not in partners:
            partners.append(partner)
    # Single-word overlap with allowlist
    for w in words:
        if w in PARTNER_ALLOWLIST and w not in partners:
            partners.append(w)

    # Categories: map query terms to category_tags via allowlist and QUERY_TO_CATEGORY
    seen_cat: set[str] = set()
    for phrase, tag in QUERY_TO_CATEGORY.items():
        if phrase in q and tag not in seen_cat:
            categories.append(tag)
            seen_cat.add(tag)
    for cat in CATEGORY_ALLOWLIST:
        if cat in q and cat not in seen_cat:
            categories.append(cat)
            seen_cat.add(cat)

    # max_annual_fee: "below X", "under X", "low fee", "no fee", "zero fee"
    # "annual fee below 1000" -> 1000; "under 500" -> 500
    fee_below = re.search(r"(?:annual\s+)?fee\s+(?:below|under)\s+(\d+)", q)
    if fee_below:
        max_annual_fee = float(fee_below.group(1))
    if max_annual_fee is None:
        below = re.search(r"(?:below|under)\s+(\d+)\s*(?:and|&|,)?", q)
        if below:
            max_annual_fee = float(below.group(1))
    if max_annual_fee is None and re.search(r"\b(?:no|zero)\s*(?:annual\s+)?fee\b", q):
        max_annual_fee = 0.0
    if max_annual_fee is None and re.search(r"\blow\s*(?:annual\s+)?fee\b", q):
        # "low fee" -> cap at 1000 for filtering intent
        max_annual_fee = 1000.0

    return ExtractedEntities(
        partners=partners,
        categories=categories,
        max_annual_fee=max_annual_fee,
    )

from __future__ import annotations

from typing import Dict, List, Set


# NOTE: Intent labels are aligned with existing JSON / metadata fields.
# We do NOT invent new intents beyond what can be mapped to:
# - category_tags
# - partner_category_tags
# - user_intents
# - high-level card categories like cashback, travel, fuel, shopping, movies, dining, low fee.


INTENT_KEYWORDS: Dict[str, List[str]] = {
    # Maps to JSON/user_intents like "cashback"
    "cashback": [
        "cashback",
        "cash back",
        "cashbak",
        "cash back card",
        "cashback card",
    ],
    # Maps to category / partner tags "travel" and user_intent "travel_rewards"
    "travel": [
        "travel",
        "air miles",
        "airmile",
        "flight",
        "airline",
        "airport",
        "maharaja points",
    ],
    "travel_rewards": [
        "travel rewards",
        "air miles",
        "maharaja points",
        "air india",
        "flight rewards",
    ],
    # Maps to exclusions/tags "fuel"
    "fuel": [
        "fuel",
        "petrol",
        "diesel",
        "hpcl",
        "bpcl",
        "iocl",
    ],
    # Maps to category / partner tags "shopping"
    "shopping": [
        "shopping",
        "online shopping",
        "store",
        "ecommerce",
        "flipkart",
        "amazon",
        "myntra",
        "partner store",
        "partner offers",
    ],
    # Entertainment is present in category_tags
    "entertainment": [
        "entertainment",
        "events",
        "concerts",
        "shows",
    ],
    # Movies is a common specific query; some JSON also references movie benefits
    "movies": [
        "movie",
        "movies",
        "cinema",
        "bookmyshow",
        "pvr",
    ],
    # Dining is a primary JSON category tag
    "dining": [
        "dining",
        "restaurant",
        "restaurants",
        "food",
        "eat out",
        "dinner",
        "lunch",
    ],
    # Hotels appears in category_tags
    "hotels": [
        "hotel",
        "hotels",
        "stay",
        "staycation",
        "resort",
    ],
    # Maps to user_intent "lounge_access"
    "lounge_access": [
        "lounge access",
        "airport lounge",
        "free lounge",
        "complimentary lounge",
        "domestic lounge",
        "international lounge",
    ],
    # Fee sensitivity – not a JSON tag but useful for ranking using annual/joining fee
    "low_fee": [
        "low fee",
        "no fee",
        "zero fee",
        "zero annual fee",
        "low annual fee",
        "annual fee",
        "joining fee",
    ],
    # Maps to user_intent "category_rewards"
    "category_rewards": [
        "category rewards",
        "extra rewards on",
        "bonus points on",
        "higher rewards on",
    ],
    # Maps to user_intent "partner_rewards"
    "partner_rewards": [
        "partner rewards",
        "brand offers",
        "partner offers",
        "aditya birla",
        "vi offers",
    ],
    # General rewards intent when user asks broadly about rewards
    "rewards_general": [
        "rewards",
        "reward points",
        "points",
        "good rewards",
        "loyalty points",
        # Generic credit-card phrasing – treat as broad rewards/card intent
        "credit card",
        "credit cards",
        "sbi card",
        "sbi cards",
        "card",
    ],
}


def detect_intents(query: str) -> List[str]:
    """
    Lightweight, rule-based intent detection from the query text.

    This function ONLY relies on:
    - The raw query string.
    - A fixed mapping of keywords to intent labels that match our JSON metadata.

    It does NOT use any LLMs and does NOT invent intents beyond this mapping.
    """
    q = query.lower().strip()
    if not q:
        return []

    detected: Set[str] = set()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                detected.add(intent)
                break

    return sorted(detected)


from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple


# We operate ONLY on fields that are already present in the indexed metadata and
# we never invent values. All numeric scores are derived from:
# - Elasticsearch score
# - Exact metadata matches
# - Simple parsing of explicit numeric patterns in existing text (e.g. \"5% cashback\").

# Minimum combined score (ES + metadata + fees) for a result to be considered relevant.
# Below this, we treat as "no relevant results" and return a single message instead.
# When user query has detected intents (card-related): use lenient threshold to avoid losing valid card results.
MIN_RELEVANCE_SCORE = 0.20
# When no intents detected (e.g. "weather", "biryani recipe"): use stricter threshold to filter noise.
MIN_RELEVANCE_SCORE_NO_INTENT = 0.35


def _safe_metadata_str(meta: Mapping[str, object], key: str) -> str:
    value = meta.get(key)
    if isinstance(value, str):
        return value
    return ""


def _safe_metadata_list(meta: Mapping[str, object], key: str) -> List[str]:
    value = meta.get(key)
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def _metadata_intent_match_score(
    meta: Mapping[str, object], detected_intents: Sequence[str]
) -> float:
    """Score based on overlap between detected intents and card metadata tags."""
    if not detected_intents:
        return 0.0

    intents_set = set(detected_intents)
    score = 0.0

    # Lists of tags from metadata
    category_tags = set(_safe_metadata_list(meta, "category_tags"))
    partner_tags = set(_safe_metadata_list(meta, "partner_category_tags"))
    user_intents = set(_safe_metadata_list(meta, "user_intents"))

    # Exact matches between detected intents and user_intents/category_tags
    score += 1.0 * len(intents_set & user_intents)
    score += 0.5 * len(intents_set & category_tags)
    score += 0.25 * len(intents_set & partner_tags)

    # Very light-weight keyword presence inside reward_summary / parent_snippet
    reward_summary = _safe_metadata_str(meta, "reward_summary").lower()
    parent_snippet = _safe_metadata_str(meta, "parent_snippet").lower()
    for intent in intents_set:
        if intent in reward_summary or intent in parent_snippet:
            score += 0.25

    return score


def _parse_numeric_fee(text: str) -> float:
    """
    Extract a simple positive numeric fee from a string like '₹499', '499', 'Rs. 1499'.
    If parsing fails, returns 0.0. We do NOT guess; we only parse explicit digits.
    """
    import re

    if not text:
        return 0.0
    digits = re.findall(r"\\d+", text)
    if not digits:
        return 0.0
    try:
        return float(digits[0])
    except ValueError:
        return 0.0


def _numeric_features(meta: Mapping[str, object]) -> Tuple[float, float]:
    """
    Return (annual_fee_value, joining_fee_value) extracted from metadata strings.
    """
    annual_fee_str = _safe_metadata_str(meta, "annual_fee")
    joining_fee_str = _safe_metadata_str(meta, "joining_fee")
    return _parse_numeric_fee(annual_fee_str), _parse_numeric_fee(joining_fee_str)


def _parse_cashback_percent(meta: Mapping[str, object]) -> float:
    """
    Extract cashback percentage (0-10) from reward_summary and parent_snippet.
    Returns 0.0 if no explicit pattern like '5% cashback' is found.
    """
    reward_summary = _safe_metadata_str(meta, "reward_summary")
    parent_snippet = _safe_metadata_str(meta, "parent_snippet")
    text = (reward_summary + " " + parent_snippet).lower()
    cashback_m = re.findall(r"(\d+\.?\d*)\s*%\s*(?:cashback|cash\s*back)", text)
    if not cashback_m:
        return 0.0
    return min(max(float(x) for x in cashback_m), 10.0)


def compute_feature_score(
    meta: Mapping[str, object],
    query_intents: Sequence[str],
    entities: Any,
) -> float:
    """
    Compute a single feature-based score for ranking: cashback, merchant match (+0.35),
    category match, and low-fee preference. Combined as 0.65*semantic + 0.25*feature + 0.10*bm25.
    """
    score = 0.0
    intents_set = set(query_intents) if query_intents else set()

    # Cashback boost when user cares about cashback
    if "cashback" in intents_set:
        cashback_pct = _parse_cashback_percent(meta)
        score += cashback_pct * 0.05

    # Merchant/partner boost: +0.25 per matching partner
    partners = getattr(entities, "partners", []) or []
    category_tags = set(_safe_metadata_list(meta, "category_tags"))
    partner_tags = set(_safe_metadata_list(meta, "partner_category_tags"))
    reward_summary = _safe_metadata_str(meta, "reward_summary").lower()
    parent_snippet = _safe_metadata_str(meta, "parent_snippet").lower()
    text = (reward_summary + " " + parent_snippet).lower()

    for merchant in partners:
        if merchant in category_tags or merchant in partner_tags:
            score += 0.35
        elif merchant.replace(" ", "") in text or merchant in text:
            score += 0.35

    # Category boost: +0.15 per matching category
    categories = getattr(entities, "categories", []) or []
    reward_categories = category_tags | partner_tags
    for cat in categories:
        if cat in reward_categories:
            score += 0.15

    # Low-fee preference when user asks for low fee
    if "low_fee" in intents_set:
        annual_fee_val, _ = _numeric_features(meta)
        # annual_fee_val can be 0 or positive; treat "unknown" as high so no bonus
        if annual_fee_val <= 1000:
            score += max(0.0, (1000 - annual_fee_val) / 1000) * 0.1
        # else: fee > 1000 or unparseable, no low_fee bonus

    return score


def _reward_strength_score(meta: Mapping[str, object]) -> float:
    """
    Derive a scalar reward strength (0.0 to 1.0) from reward_summary and parent_snippet.
    Uses only explicit numeric patterns (e.g. X% cashback, points per 100); does not invent values.
    """
    reward_summary = _safe_metadata_str(meta, "reward_summary")
    parent_snippet = _safe_metadata_str(meta, "parent_snippet")
    text = (reward_summary + " " + parent_snippet).lower()
    if not text.strip():
        return 0.0

    score = 0.0

    # Percent cashback: e.g. "5% cashback", "2.5% cash back"
    cashback_m = re.findall(r"(\d+\.?\d*)\s*%\s*(?:cashback|cash\s*back)", text)
    if cashback_m:
        cashback_pcts = [float(x) for x in cashback_m]
        max_cashback = min(max(cashback_pcts), 10.0)  # cap at 10%
        score = max(score, min(1.0, max_cashback / 10.0))

    # Points per 100 / per Rs 100: e.g. "5 points per 100", "10 points per rs 100"
    points_m = re.findall(r"(\d+)\s*points?\s*per\s*(?:rs?\.?\s*)?(\d+)", text)
    if points_m and score < 0.5:
        # Approximate effective rate: points_per_100 with divisor 100 -> treat as low % equivalent
        for p, div in points_m:
            try:
                p_val, div_val = float(p), float(div)
                if div_val > 0:
                    effective = (p_val / div_val) * 10.0  # scale to ~percent-like
                    score = max(score, min(1.0, effective / 10.0))
            except ValueError:
                pass

    # Strong reward phrases: small bonus if no numeric found
    if score < 0.2:
        strong_phrases = ("strong rewards", "high rewards", "accelerated rewards", "premium rewards")
        if any(p in text for p in strong_phrases):
            score = 0.15

    return min(1.0, score)


# Intents for which we apply reward-based ranking boost (higher gamma).
_REWARD_INTENTS = frozenset({
    "cashback", "rewards_general", "category_rewards", "partner_rewards",
    "shopping", "movies", "dining", "entertainment", "hotels", "travel", "travel_rewards",
})
# Fee-focused intent: do not boost by reward strength so low-fee cards stay preferred.
_FEE_INTENT = "low_fee"


def _reward_weight_for_intents(detected_intents: Sequence[str]) -> float:
    """
    Return the weight (gamma) for reward strength in the final score.
    High when query is reward-related, zero or low when fee-focused or unrelated.
    """
    if not detected_intents:
        return 0.0
    intents_set = set(detected_intents)
    if _FEE_INTENT in intents_set and len(intents_set) == 1:
        return 0.0
    if intents_set & _REWARD_INTENTS:
        return 0.3
    return 0.0


def _constraint_match_bonus(meta: Mapping[str, object], entities: Any) -> float:
    """
    Bonus when card metadata matches extracted entities (partners/categories).
    entities must have .partners and .categories (List[str]); e.g. ExtractedEntities.
    """
    bonus = 0.0
    partners = getattr(entities, "partners", []) or []
    categories = getattr(entities, "categories", []) or []
    if not partners and not categories:
        return 0.0

    category_tags = set(_safe_metadata_list(meta, "category_tags"))
    partner_tags = set(_safe_metadata_list(meta, "partner_category_tags"))
    reward_summary = _safe_metadata_str(meta, "reward_summary").lower()
    parent_snippet = _safe_metadata_str(meta, "parent_snippet").lower()
    text = (reward_summary + " " + parent_snippet).lower()

    for p in partners:
        if p in category_tags or p in partner_tags or p.replace(" ", "") in text or p in text:
            bonus += 0.1
    for c in categories:
        if c in category_tags or c in partner_tags or c in text:
            bonus += 0.1
    return min(bonus, 0.5)


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max <= v_min:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def rerank_results_with_scores(
    results: Iterable[object],
    detected_intents: Sequence[str],
    entities: Optional[Any] = None,
) -> List[Tuple[float, object]]:
    """
    Re-rank using: 0.65 * semantic (KNN) + 0.25 * feature + 0.10 * BM25.
    When knn_score/bm25_score are present (from hybrid search), use them; else fallback to score/0.
    """
    results_list = list(results)
    if not results_list:
        return []

    semantic_scores: List[float] = []
    bm25_scores: List[float] = []
    feature_scores: List[float] = []

    for r in results_list:
        meta = getattr(r, "metadata", None) or {}
        knn_s = getattr(r, "knn_score", None)
        bm25_s = getattr(r, "bm25_score", None)
        semantic_scores.append(float(knn_s) if knn_s is not None else float(getattr(r, "score", 0.0) or 0.0))
        bm25_scores.append(float(bm25_s) if bm25_s is not None else 0.0)
        feature_scores.append(compute_feature_score(meta, detected_intents, entities or _empty_entities()))

    norm_semantic = _min_max_normalize(semantic_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)
    norm_feature = _min_max_normalize(feature_scores)

    scored: List[Tuple[float, object]] = []
    for idx, r in enumerate(results_list):
        final_score = (
            0.65 * norm_semantic[idx]
            + 0.25 * norm_feature[idx]
            + 0.10 * norm_bm25[idx]
        )
        scored.append((final_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


class _EmptyEntities:
    """Sentinel for compute_feature_score when entities is None (no partners/categories)."""
    def __init__(self) -> None:
        self.partners: List[str] = []
        self.categories: List[str] = []


def _empty_entities() -> _EmptyEntities:
    return _EmptyEntities()


def get_relevance_threshold(detected_intents: Sequence[str]) -> float:
    """
    Return the minimum score threshold for noise filtering.
    - No intents (off-topic query): stricter threshold to filter noise.
    - Has intents (card-related query): lenient threshold to avoid losing valid card results.
    """
    if not detected_intents:
        return MIN_RELEVANCE_SCORE_NO_INTENT
    return MIN_RELEVANCE_SCORE


def rerank_results(
    results: Iterable[object],
    detected_intents: Sequence[str],
    entities: Optional[Any] = None,
) -> List[object]:
    """
    Re-rank ES hybrid results using metadata, fees, and optional entity constraint boost.
    Returns only the result objects (no scores). For score-aware filtering use rerank_results_with_scores.
    """
    scored = rerank_results_with_scores(results, detected_intents, entities=entities)
    return [r for _, r in scored]


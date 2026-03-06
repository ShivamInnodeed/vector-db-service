from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple


# We operate ONLY on fields that are already present in the indexed metadata and
# we never invent values. All numeric scores are derived from:
# - Elasticsearch score
# - Exact metadata matches
# - Simple parsing of explicit numeric patterns in existing text (e.g. \"5% cashback\").


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


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max <= v_min:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def rerank_results(results: Iterable[object], detected_intents: Sequence[str]) -> List[object]:
    """
    Re-rank ES hybrid results using:
    - metadata overlap with detected intents
    - simple numeric features for fees

    We do NOT change the underlying ES scores, only add small adjustments.
    """
    results_list = list(results)
    if not results_list:
        return results_list

    # First: compute metadata-based scores
    meta_scores: List[float] = []
    annual_fees: List[float] = []
    joining_fees: List[float] = []

    for r in results_list:
        meta = getattr(r, "metadata", None) or {}
        meta_score = _metadata_intent_match_score(meta, detected_intents)
        meta_scores.append(meta_score)

        annual_fee_val, joining_fee_val = _numeric_features(meta)
        annual_fees.append(annual_fee_val)
        joining_fees.append(joining_fee_val)

    # Normalise numeric fees across the current result set
    annual_norm = _min_max_normalize(annual_fees)
    joining_norm = _min_max_normalize(joining_fees)

    # Combine into final scores
    alpha = 0.3  # weight for metadata intent matches
    beta_annual = 0.2  # penalty weight for higher annual fee
    beta_joining = 0.1  # penalty weight for higher joining fee

    scored: List[Tuple[float, object]] = []
    for idx, r in enumerate(results_list):
        es_score = float(getattr(r, "score", 0.0) or 0.0)
        meta_score = meta_scores[idx]
        annual_fee_penalty = annual_norm[idx]
        joining_fee_penalty = joining_norm[idx]

        final_score = (
            es_score
            + alpha * meta_score
            - beta_annual * annual_fee_penalty
            - beta_joining * joining_fee_penalty
        )

        scored.append((final_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CardJsonOffer:
    """
    Lightweight adapter over a single JSON file in data/cards_json_data.

    This is intentionally minimal and shaped to work well with our chunking:
    - string fields for simple display/filtering
    - list-of-string fields for bullets/snippets
    """

    card_id: str
    card_name: str
    issuer: str
    category: str
    network: str
    url: str
    status: str

    annual_fee: str
    joining_fee: str
    reward_summary: str

    key_benefits: List[str]
    joining_offers: List[str]
    milestone_offers: List[str]
    eligibility: List[str]
    important_terms: List[str]

    category_tags: List[str]
    partner_category_tags: List[str]
    user_intents: List[str]

    language: str = "en"

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "CardJsonOffer":
        """
        Build an adapter instance from a raw JSON dict produced in cards_json_data.

        The mapping is heuristic but stable:
        - fees.joining_fee / fees.renewal_fee → joining_fee / annual_fee (as strings)
        - reward_structure + cashback information → coarse reward_summary text
        - welcome_benefits / milestones → joining_offers / milestone_offers
        - tags and intents → key_benefits-style bullets
        """

        card_id = str(data.get("card_id") or "").strip()
        card_name = str(data.get("card_name") or "").strip() or card_id or "Unknown card"
        url = str(data.get("url") or "").strip()
        status = str(data.get("status") or "unknown").strip()

        if not card_id:
            card_id = card_name.replace(" ", "_").lower()

        # Fees
        fees = data.get("fees") or {}
        joining_fee_val = fees.get("joining_fee")
        renewal_fee_val = fees.get("renewal_fee")
        joining_fee = str(joining_fee_val) if joining_fee_val not in (None, "") else "unknown"
        annual_fee = str(renewal_fee_val) if renewal_fee_val not in (None, "") else "unknown"

        # Reward summary from reward_structure + cashback
        reward_structure = data.get("reward_structure") or {}
        cashback_section = data.get("cashback") or {}

        reward_bits: List[str] = []
        cat_earn = (reward_structure.get("category_earnings") or {}) if isinstance(reward_structure, dict) else {}
        if cat_earn:
            reward_bits.append(
                "Category earnings for "
                + ", ".join(sorted(k for k in cat_earn.keys() if k != "default"))
            )

        partner_earn = reward_structure.get("partner_earnings") or {}
        if partner_earn:
            reward_bits.append(
                "Extra rewards at partners: "
                + ", ".join(sorted(str(k) for k in partner_earn.keys()))
            )

        cb_cat = (cashback_section.get("category_specific") or {}) if isinstance(cashback_section, dict) else {}
        if cb_cat:
            reward_bits.append(
                "Cashback categories: "
                + ", ".join(
                    f"{k}:{v.get('percentage')}%"
                    for k, v in cb_cat.items()
                    if isinstance(v, dict) and v.get("percentage") is not None
                )
            )

        reward_summary = " | ".join(reward_bits) if reward_bits else "unknown"

        # Joining offers
        welcome = data.get("welcome_benefits") or {}
        joining_offers: List[str] = []
        if welcome:
            bonus_points = welcome.get("bonus_points")
            voucher = welcome.get("gift_voucher")
            conditions = welcome.get("conditions")
            if bonus_points:
                joining_offers.append(f"Welcome bonus: {bonus_points} points")
            if voucher:
                joining_offers.append(f"Welcome voucher: {voucher}")
            if conditions:
                joining_offers.append(f"Conditions: {conditions}")

        # Milestone offers
        milestone_offers: List[str] = []
        for m in data.get("milestones") or []:
            if not isinstance(m, dict):
                continue
            thr = m.get("spend_threshold")
            cat = m.get("category")
            bonus = m.get("bonus_points")
            parts: List[str] = []
            if thr is not None:
                parts.append(f"Spend ≥ {thr}")
            if cat:
                parts.append(f"on {cat}")
            if bonus is not None:
                parts.append(f"to earn {bonus} bonus points")
            desc = " ".join(parts).strip()
            if desc:
                milestone_offers.append(desc)

        # Key benefits from tags and intents
        category_tags = list(data.get("category_tags") or [])
        partner_category_tags = list(data.get("partner_category_tags") or [])
        user_intents = list(data.get("user_intents") or [])

        key_benefits: List[str] = []
        if category_tags:
            key_benefits.append("Good for: " + ", ".join(str(t) for t in category_tags))
        if partner_category_tags:
            key_benefits.append("Partner categories: " + ", ".join(str(t) for t in partner_category_tags))
        if user_intents:
            key_benefits.append("Optimised for intents: " + ", ".join(str(t) for t in user_intents))

        # Eligibility and important terms – placeholders for now (can be enriched later)
        eligibility: List[str] = []
        important_terms: List[str] = []

        return cls(
            card_id=card_id,
            card_name=card_name,
            issuer=str(data.get("issuer") or "SBI Card").strip() or "SBI Card",
            category=str(data.get("category") or "unknown").strip().lower(),
            network=str(data.get("network") or "unknown").strip(),
            url=url,
            status=status,
            annual_fee=annual_fee,
            joining_fee=joining_fee,
            reward_summary=reward_summary,
            key_benefits=key_benefits,
            joining_offers=joining_offers,
            milestone_offers=milestone_offers,
            eligibility=eligibility,
            important_terms=important_terms,
            category_tags=category_tags,
            partner_category_tags=partner_category_tags,
            user_intents=user_intents,
        )


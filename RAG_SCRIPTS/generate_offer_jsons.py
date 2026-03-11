from __future__ import annotations

"""
Parse `data/deepcrawlOffers.md` and emit per-brand offer JSON files under `data/offers_json/`.

The markdown is an exported scrape of sbicard.com offer pages. This script:
- scans for individual offer-detail sections
- extracts brand slug, summary, validity, eligibility, steps, terms, and key table rows
- groups offers by brand/entity
- writes one JSON per brand for downstream RAG + indexing
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATA_MD_DEFAULT = ROOT / "data" / "deepcrawlOffers.md"
OUTPUT_DIR_DEFAULT = ROOT / "data" / "offers_json"


OFFER_BREADCRUMB_RE = re.compile(
    r"\[Home\].*?\[Offers\].*?\[offer-detail\]\(https://www\.sbicard\.com/en/personal/offer/([^)\s]+)\.page\)",
    re.IGNORECASE,
)


DATE_RANGE_SEP_RE = re.compile(r"\s*[–-]\s*")  # en dash / hyphen as separator
DAY_RE = re.compile(r"(\d+)")
MONTH_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", re.IGNORECASE
)
YEAR_RE = re.compile(r"(20\d{2})")


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class OfferValidity:
    start_date: Optional[str] = None  # ISO yyyy-mm-dd
    end_date: Optional[str] = None
    raw_text: str = ""


@dataclass
class OfferDetails:
    minimum_transaction: Optional[str] = None
    cashback_percentages: Dict[str, str] = field(default_factory=dict)
    max_cashback_per_card: Optional[str] = None
    raw_table_rows: List[str] = field(default_factory=list)


@dataclass
class SingleOffer:
    offer_id: str
    title: str
    short_description: str
    summary_points: List[str]
    offer_type: str
    validity: OfferValidity
    eligible_cards: str
    channels: List[str]
    offer_details: OfferDetails
    terms_and_conditions: List[str]
    steps_to_avail: List[str]
    store_list_url: Optional[str]
    exclusions: List[str]
    source_url: str


@dataclass
class BrandOffers:
    entity_id: str
    entity_name: str
    summary: str = ""
    summary_points: List[str] = field(default_factory=list)
    entity_type: str = "offer_brand"
    category_tags: List[str] = field(default_factory=list)
    source_url: str = ""
    status: str = "unknown"
    user_intents: List[str] = field(
        default_factory=lambda: [
            "offer_discovery",
            "offer_validity",
            "eligibility_question",
        ]
    )
    offers: List[SingleOffer] = field(default_factory=list)


def read_markdown_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Keep line structure but strip trailing newlines
    return [line.rstrip("\n") for line in raw.splitlines()]


def iter_offer_blocks(lines: List[str]) -> Iterable[Tuple[int, int]]:
    """
    Yield (start_idx, end_idx) line indices for each offer-detail block.

    We detect the start by looking for the breadcrumb containing /personal/offer/<slug>.page.
    """
    starts: List[int] = []
    for i, line in enumerate(lines):
        if OFFER_BREADCRUMB_RE.search(line):
            starts.append(i)

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        yield start, end


def slug_to_entity_id_and_name(slug: str) -> Tuple[str, str]:
    """
    Normalize the offer slug into a brand-level entity id and human name.

    Many slugs look like `boyanika-23feb26`. We try to strip trailing date-like
    segments and keep the brand core.
    """
    core = slug

    # Strip .page if still present
    if core.endswith(".page"):
        core = core[: -len(".page")]

    # Heuristic: remove trailing -<day><mon><year> or -<mon><year> pieces
    # e.g. -23feb26, -feb26
    m = re.match(r"^(.*?)-\d{1,2}[a-z]{3}\d{2,4}$", core)
    if m:
        core = m.group(1)
    else:
        m2 = re.match(r"^(.*?)-[a-z]{3}\d{2,4}$", core)
        if m2:
            core = m2.group(1)

    entity_id = core.lower().replace(" ", "-")
    # Build entity_name by splitting on hyphens and capitalising
    parts = [p for p in re.split(r"[-_]", core) if p]
    entity_name = " ".join(p.capitalize() for p in parts) or core
    return entity_id, entity_name


def parse_date_token(token: str, fallback_year: Optional[int]) -> Optional[datetime]:
    token = token.strip()
    if not token:
        return None

    day_match = DAY_RE.search(token)
    month_match = MONTH_RE.search(token)
    year_match = YEAR_RE.search(token)

    if not month_match or not day_match:
        return None

    day = int(day_match.group(1))
    month_key = month_match.group(1).lower()
    month = MONTH_MAP.get(month_key)
    if not month:
        return None

    year: Optional[int]
    if year_match:
        year = int(year_match.group(1))
    else:
        year = fallback_year

    if year is None:
        return None

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def parse_validity_line(line: str) -> OfferValidity:
    """
    Parse lines like "21st Feb – 31st Mar 2026" into start/end ISO dates.
    """
    raw = line.strip()
    if not raw:
        return OfferValidity(raw_text=line)

    parts = DATE_RANGE_SEP_RE.split(raw)
    if len(parts) == 1:
        # Single date or unstructured text
        dt = parse_date_token(parts[0], None)
        iso = dt.date().isoformat() if dt else None
        return OfferValidity(start_date=iso, end_date=iso, raw_text=line)

    left, right = parts[0], parts[1]
    right_year_match = YEAR_RE.search(right)
    fallback_year = int(right_year_match.group(1)) if right_year_match else None

    dt_start = parse_date_token(left, fallback_year)
    dt_end = parse_date_token(right, fallback_year)

    return OfferValidity(
        start_date=dt_start.date().isoformat() if dt_start else None,
        end_date=dt_end.date().isoformat() if dt_end else None,
        raw_text=line,
    )


def detect_offer_type(text: str) -> str:
    t = text.lower()
    if "cashback" in t:
        return "cashback"
    if "discount" in t:
        return "discount"
    if "emi" in t:
        return "emi"
    if "bogo" in t or "buy one get one" in t:
        return "bogo"
    return "generic"


def parse_summary_and_table(block: List[str], start_idx: int) -> Tuple[str, OfferDetails, int]:
    """
    Parse the Summary section:
    - title/short description (first non-empty line)
    - markdown key/value table (Offer | X, etc.)
    Returns (title, OfferDetails, next_index_after_section)
    """
    i = start_idx
    # Skip possible 'Summary' heading itself if present
    if i < len(block) and block[i].strip().lower() == "summary":
        i += 1

    # Title
    title = ""
    while i < len(block) and not title:
        line = block[i].strip()
        if line:
            title = line
        i += 1

    details = OfferDetails()

    # Look for markdown table header followed by ---|---
    # We allow some noise between title and table, but usually it's immediate.
    while i < len(block):
        line = block[i].strip()
        if not line:
            i += 1
            continue
        if "|  " in line or " | " in line:
            # Candidate header; next non-empty line should be ---|---
            j = i + 1
            while j < len(block) and not block[j].strip():
                j += 1
            if j < len(block) and "---" in block[j]:
                # Table spans from i .. until blank line
                k = j + 1
                while k < len(block) and block[k].strip():
                    row = block[k].strip()
                    details.raw_table_rows.append(row)

                    # Parse key | value
                    if "|" in row:
                        key, val = [part.strip() for part in row.split("|", 1)]
                        key_l = key.lower()
                        if "minimum transaction" in key_l:
                            details.minimum_transaction = val
                        elif "cashback" in key_l or "trxn" in key_l:
                            details.cashback_percentages[key] = val
                        elif "max" in key_l and "cashback" in key_l:
                            details.max_cashback_per_card = val

                    k += 1
                i = k
                break
        i += 1

    return title, details, i


def extract_section_lines(
    block: List[str], heading: str, stop_headings: List[str]
) -> List[str]:
    """
    Extract consecutive non-empty lines under a given heading until another heading or blank block.
    """
    heading_l = heading.lower()
    stop_set = {s.lower() for s in stop_headings}
    out: List[str] = []

    i = 0
    while i < len(block):
        if block[i].strip().lower() == heading_l:
            i += 1
            # collect
            while i < len(block):
                line = block[i]
                stripped = line.strip()
                if not stripped:
                    break
                if stripped.lower() in stop_set:
                    break
                out.append(stripped.lstrip("*- ").strip())
                i += 1
            break
        i += 1

    return out


def extract_terms(block: List[str]) -> List[str]:
    out: List[str] = []
    capture = False
    for line in block:
        stripped = line.strip()
        if not stripped:
            continue
        if "OTHER TERMS AND CONDITIONS" in stripped.upper():
            capture = True
            continue
        if capture:
            if stripped.startswith(("*", "-")):
                out.append(stripped.lstrip("*- ").strip())
            else:
                out.append(stripped)
    return out


def extract_summary_points(block: List[str]) -> List[str]:
    """
    Extract summary lines under a 'Summary' heading.

    Handles:
    - bullet lists (lines starting with * / -)
    - markdown tables (header + ---|--- separator + rows), turning each row into a readable string
    - trailing plain-text summary sentences
    """
    points: List[str] = []
    in_summary = False
    i = 0

    def is_stop_heading(text: str) -> bool:
        lt = text.lower()
        return (
            lt.startswith("offer validity")
            or lt.startswith("eligible cards")
            or "terms & conditions" in lt
            or "terms and conditions" in lt
        )

    while i < len(block):
        stripped = block[i].strip()
        if not stripped:
            if in_summary and points:
                break
            i += 1
            continue

        lower = stripped.lower()

        if not in_summary:
            if lower == "summary":
                in_summary = True
            i += 1
            continue

        # In summary section
        if is_stop_heading(stripped):
            break

        # Bullet case
        if stripped.startswith(("*", "-")):
            points.append(stripped.lstrip("*- ").strip())
            i += 1
            continue

        # Markdown table case: header row + separator line
        if "|" in stripped:
            # Look ahead for separator line
            j = i + 1
            while j < len(block) and not block[j].strip():
                j += 1
            if j < len(block) and "---" in block[j]:
                # Parse header
                header_cells = [
                    c.strip(" *") for c in stripped.split("|") if c.strip()
                ]

                k = j + 1
                while k < len(block):
                    row = block[k].strip()
                    if not row:
                        break
                    if is_stop_heading(row):
                        break

                    # Plain text line after table rows (e.g. explanatory sentence)
                    if "|" not in row:
                        points.append(row.strip("_ ").strip())
                        k += 1
                        continue

                    cells = [c.strip(" *") for c in row.split("|") if c.strip()]
                    if len(cells) != len(header_cells):
                        k += 1
                        continue

                    row_dict = dict(zip(header_cells, cells))
                    # Build a readable summary like "Category: Domestic Flights; % Discount: 12%; ..."
                    components = [
                        f"{h}: {v}" for h, v in row_dict.items() if v
                    ]
                    if components:
                        points.append("; ".join(components))
                    k += 1

                i = k
                continue

        # Fallback: plain-text summary line
        points.append(stripped.strip("_ ").strip())
        i += 1

    return points


def extract_steps(block: List[str]) -> List[str]:
    out: List[str] = []
    capture = False
    for line in block:
        stripped = line.strip()
        if not stripped:
            if capture and out:
                break
            continue
        if stripped.lower().startswith("steps to avail"):
            capture = True
            continue
        if capture:
            if stripped.startswith(("*", "-")):
                out.append(stripped.lstrip("*- ").strip())
            else:
                out.append(stripped)
    return out


def extract_urls(block: List[str]) -> Tuple[str, Optional[str]]:
    """
    Return (offer_detail_url, store_list_url).
    """
    offer_url = ""
    store_url: Optional[str] = None

    for line in block:
        # Offer-detail URL (breadcrumb or Avail Now)
        m = re.search(
            r"\(https://www\.sbicard\.com/en/personal/offer/([^)\s]+)\.page\)",
            line,
            re.IGNORECASE,
        )
        if m and not offer_url:
            offer_url = f"https://www.sbicard.com/en/personal/offer/{m.group(1)}.page"

        # Store list
        if "store list" in line.lower() or "click here" in line.lower():
            m2 = re.search(r"\((https://[^\s)]+)\)", line)
            if m2:
                store_url = m2.group(1)

    return offer_url, store_url


def extract_exclusions(lines: Iterable[str]) -> List[str]:
    exclusions: List[str] = []
    for line in lines:
        lower = line.lower()
        if "offer not applicable on" in lower:
            # Take everything after this phrase, split by comma
            after = lower.split("offer not applicable on", 1)[1]
            for part in after.replace(".", " ").split(","):
                item = part.strip()
                if item:
                    exclusions.append(item)
    return exclusions


def build_single_offer(slug: str, block_lines: List[str]) -> Optional[SingleOffer]:
    offer_url, store_url = extract_urls(block_lines)

    # Offer Validity
    validity_lines = extract_section_lines(
        block_lines,
        heading="Offer Validity",
        stop_headings=["Eligible Cards", "Summary", "Offer Validity"],
    )
    validity = (
        parse_validity_line(validity_lines[0]) if validity_lines else OfferValidity()
    )

    # Eligible cards
    eligible_lines = extract_section_lines(
        block_lines,
        heading="Eligible Cards",
        stop_headings=["Summary", "Offer Validity"],
    )
    eligible_cards = " ".join(eligible_lines).strip()

    # Summary bullets + table
    summary_points = extract_summary_points(block_lines)

    title = ""
    details = OfferDetails()
    for idx, line in enumerate(block_lines):
        if line.strip().lower() == "summary":
            title, details, _ = parse_summary_and_table(block_lines, idx)
            break

    if not title:
        # fallback: look for "Summary" label and take next non-empty line
        for idx, line in enumerate(block_lines):
            if "summary" in line.lower():
                j = idx + 1
                while j < len(block_lines) and not block_lines[j].strip():
                    j += 1
                if j < len(block_lines):
                    title = block_lines[j].strip()
                break

    if not title and summary_points:
        title = summary_points[0]

    short_description = title
    offer_type = detect_offer_type(title)

    steps = extract_steps(block_lines)
    terms = extract_terms(block_lines)

    exclusions = extract_exclusions(terms + eligible_lines)

    channels: List[str] = []
    joined_text = " ".join(block_lines).lower()
    if "online" in joined_text:
        channels.append("online")
    if "store" in joined_text or "instore" in joined_text or "in-store" in joined_text:
        channels.append("instore")

    offer_id = slug
    if validity.start_date or validity.end_date:
        parts = [slug]
        if validity.start_date:
            parts.append(validity.start_date)
        if validity.end_date and validity.end_date != validity.start_date:
            parts.append(validity.end_date)
        offer_id = "-".join(parts)

    return SingleOffer(
        offer_id=offer_id,
        title=title or slug,
        short_description=short_description or title or slug,
        summary_points=summary_points,
        offer_type=offer_type,
        validity=validity,
        eligible_cards=eligible_cards,
        channels=channels,
        offer_details=details,
        terms_and_conditions=terms,
        steps_to_avail=steps,
        store_list_url=store_url,
        exclusions=exclusions,
        source_url=offer_url,
    )


def build_brand_offers(lines: List[str]) -> Dict[str, BrandOffers]:
    """
    High-level orchestrator: walk the markdown, build BrandOffers grouped by entity_id.
    """
    brands: Dict[str, BrandOffers] = {}

    for start, end in iter_offer_blocks(lines):
        block = lines[start:end]
        m = OFFER_BREADCRUMB_RE.search(block[0])
        if not m:
            continue
        slug = m.group(1)
        entity_id, entity_name = slug_to_entity_id_and_name(slug)

        offer = build_single_offer(slug, block)
        if not offer:
            continue

        brand = brands.get(entity_id)
        if not brand:
            brand = BrandOffers(
                entity_id=entity_id,
                entity_name=entity_name,
                source_url=offer.source_url or "",
            )
            brands[entity_id] = brand

        if not brand.source_url and offer.source_url:
            brand.source_url = offer.source_url

        brand.offers.append(offer)

    # Post-process brand-level summaries
    for brand in brands.values():
        if not brand.offers:
            continue

        # Simple human-readable summary
        titles = [o.title for o in brand.offers if o.title]
        n = len(brand.offers)
        if titles:
            brand.summary = (
                f"{brand.entity_name} has {n} SBI Card offer(s): "
                + "; ".join(titles[:5])
                + ("..." if len(titles) > 5 else "")
            )
        else:
            brand.summary = f"{brand.entity_name} has {n} SBI Card offer(s) with SBI Cards."

        # Bullet-style points for RAG display
        points: List[str] = []
        for offer in brand.offers:
            if offer.summary_points:
                points.extend(offer.summary_points)
                continue

            validity_bits: List[str] = []
            if offer.validity.start_date and offer.validity.end_date:
                if offer.validity.start_date == offer.validity.end_date:
                    validity_bits.append(f"on {offer.validity.start_date}")
                else:
                    validity_bits.append(
                        f"from {offer.validity.start_date} to {offer.validity.end_date}"
                    )
            elif offer.validity.raw_text:
                validity_bits.append(offer.validity.raw_text.strip())

            detail = offer.offer_details.minimum_transaction or ""
            parts: List[str] = [offer.title]
            if detail:
                parts.append(f"(Min txn: {detail})")
            if validity_bits:
                parts.append(f"({'; '.join(validity_bits)})")

            points.append(" ".join(p for p in parts if p).strip())

        brand.summary_points = points

    return brands


def dataclass_to_jsonable(obj):
    if isinstance(obj, list):
        return [dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return dataclass_to_jsonable(asdict(obj))
    return obj


def write_brand_files(brands: Dict[str, BrandOffers], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entity_id, brand in sorted(brands.items()):
        data = dataclass_to_jsonable(brand)
        out_path = output_dir / f"{entity_id}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse data/deepcrawlOffers.md and write per-brand offer JSON files "
            "under data/offers_json/."
        )
    )
    parser.add_argument(
        "--input-md",
        type=str,
        default=str(DATA_MD_DEFAULT),
        help="Path to deepcrawlOffers markdown file (default: data/deepcrawlOffers.md)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR_DEFAULT),
        help="Directory to write per-brand JSON files (default: data/offers_json/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing JSON files (default: overwrite).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    md_path = Path(args.input_md)
    output_dir = Path(args.output_dir)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    print(f"Reading markdown from {md_path}")
    lines = read_markdown_lines(md_path)

    print("Parsing offer blocks...")
    brands = build_brand_offers(lines)
    total_offers = sum(len(b.offers) for b in brands.values())

    print(f"Discovered {len(brands)} brands with {total_offers} offers.")

    if output_dir.exists() and not args.overwrite and any(
        p.suffix == ".json" for p in output_dir.glob("*.json")
    ):
        print(
            f"Output directory {output_dir} already has JSON files. "
            "Re-run with --overwrite to replace."
        )
        return

    print(f"Writing JSON files to {output_dir}")
    write_brand_files(brands, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()


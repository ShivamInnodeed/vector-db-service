"""
Index all SBI Card partner offers from JSON under data/offers_json into Elasticsearch.

Flow:
- Read *.json from data/offers_json (each file = one partner brand entity with nested offers)
- Build per-offer text blobs (title, description, key points, validity, eligibility, channels, etc.)
- Chunk text with fixed token window and overlap:
  - WINDOW_TOKENS = 50
  - OVERLAP_TOKENS = 25
- Generate embeddings for each chunk using scripts.embedding_utils.encode (MiniLM-based)
- Index chunks into Elasticsearch via VectorDBClient as Documents
  in index: sbicard_offers_rag
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient  # type: ignore
from vector_db.models import Document  # type: ignore
from scripts.embedding_utils import EMBEDDING_DIM, encode, using_real_embeddings  # type: ignore


DATA_DIR = ROOT / "data" / "offers_json"
INDEX_NAME = "sbicard_offers_rag"

# Hardcoded chunking configuration (token-based)
CHUNK_TOKEN_SIZE = 50
CHUNK_TOKEN_OVERLAP = 25
MIN_CHUNK_TOKENS = 10


def iter_offer_files() -> Iterable[Path]:
    """Yield all offer JSON files from DATA_DIR."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Offers JSON data directory not found: {DATA_DIR}")
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.is_file():
            yield path


def safe_get_list(obj: Dict[str, Any], key: str) -> List[str]:
    val = obj.get(key)
    if isinstance(val, list):
        return [str(x) for x in val]
    if val is None:
        return []
    return [str(val)]


def chunk_text_with_overlap(
    text: str, window: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_TOKEN_OVERLAP
) -> List[str]:
    """
    Chunk text into overlapping token windows.

    - Uses simple whitespace tokenization.
    - Window size = `window` tokens, with `overlap` tokens carried into the next chunk.
    - Very small trailing fragments (< MIN_CHUNK_TOKENS) are merged into the previous chunk.
    """
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) <= window:
        return [" ".join(tokens)]

    step = max(1, window - overlap)
    chunks: List[List[str]] = []

    start = 0
    while start < len(tokens):
        end = start + window
        window_tokens = tokens[start:end]
        if not window_tokens:
            break
        chunks.append(window_tokens)
        start += step

    # Merge tiny tail if last chunk is too small and we have more than one chunk
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_TOKENS:
        tail = chunks.pop()
        chunks[-1].extend(tail)

    return [" ".join(toks) for toks in chunks]


def build_entity_snippet(entity: Dict[str, Any]) -> str:
    name = str(entity.get("entity_name") or entity.get("entity_id") or "").strip()
    entity_type = str(entity.get("entity_type") or "").strip()
    summary = str(entity.get("summary") or "").strip()
    parts: List[str] = []
    if name:
        if entity_type:
            parts.append(f"{name} ({entity_type})")
        else:
            parts.append(name)
    if summary:
        parts.append(summary)
    return " | ".join(parts)


def build_offer_full_text(entity: Dict[str, Any], offer: Dict[str, Any]) -> str:
    """Assemble a rich textual representation for a single nested offer."""
    lines: List[str] = []

    entity_name = str(entity.get("entity_name") or entity.get("entity_id") or "").strip()
    if entity_name:
        lines.append(f"Partner: {entity_name}")

    title = str(offer.get("title") or "").strip()
    short_description = str(offer.get("short_description") or "").strip()
    if title:
        lines.append(f"Offer title: {title}")
    if short_description and short_description != title:
        lines.append(f"Short description: {short_description}")

    summary_points = safe_get_list(offer, "summary_points")
    if summary_points:
        lines.append("Summary points:")
        lines.extend(summary_points)

    validity = offer.get("validity") or {}
    if isinstance(validity, dict):
        raw_validity = str(validity.get("raw_text") or "").strip()
        if raw_validity:
            lines.append(f"Validity: {raw_validity}")

    eligible_cards = str(offer.get("eligible_cards") or "").strip()
    if eligible_cards:
        lines.append(f"Eligible cards: {eligible_cards}")

    channels = safe_get_list(offer, "channels")
    if channels:
        lines.append("Channels: " + ", ".join(channels))

    offer_details = offer.get("offer_details") or {}
    if isinstance(offer_details, dict):
        raw_rows = safe_get_list(offer_details, "raw_table_rows")
        if raw_rows:
            lines.append("Offer details:")
            lines.extend(raw_rows)

    terms = safe_get_list(offer, "terms_and_conditions")
    if terms:
        # Only keep the first few for indexing to avoid extremely long chunks
        lines.append("Key terms and conditions:")
        lines.extend(terms[:10])

    return "\n".join(line for line in lines if line.strip())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_brand(entity_name: str) -> str:
    """Normalize brand/merchant name for metadata."""
    return (entity_name or "").strip()


def infer_category(
    entity_name: str,
    category_tags: List[Any],
    offer_title: str,
    summary: str,
) -> str:
    """
    Infer a coarse category such as travel, electronics, dining, shopping, mobiles, grocery.
    Uses category_tags when present, else simple keyword heuristics.
    """
    text = " ".join(
        [
            str(entity_name or ""),
            " ".join(str(t) for t in category_tags or []),
            str(offer_title or ""),
            str(summary or ""),
        ]
    ).lower()

    if any(w in text for w in ["flight", "flights", "hotel", "bus", "travel", "ticket", "booking", "holiday"]):
        return "travel"
    if any(w in text for w in ["electronics", "appliance", "appliances", "tv", "ac", "air conditioner", "fridge", "refrigerator", "laptop"]):
        return "electronics"
    if any(w in text for w in ["dining", "restaurant", "food", "buffet", "cafe"]):
        return "dining"
    if any(w in text for w in ["shopping", "mall", "store", "bazaar", "baazar"]):
        return "shopping"
    if any(w in text for w in ["mobile", "phone", "smartphone"]):
        return "mobiles"
    if any(w in text for w in ["grocery", "groceries", "supermarket"]):
        return "grocery"
    return ""


def normalize_offer_type(offer_type: str, title: str, full_text: str) -> str:
    """Normalize offer type into discount, emi, cashback where possible."""
    t = (offer_type or "").strip().lower()
    text = " ".join([t, title or "", full_text or ""]).lower()

    if "emi" in text or "installment" in text or "installments" in text:
        return "emi"
    if "cashback" in text or "cash back" in text:
        return "cashback"
    if "discount" in text or "% off" in text or "instant discount" in text:
        return "discount"
    return t or ""


def main() -> None:
    print(f"Loading offers from {DATA_DIR} ...")
    files = list(iter_offer_files())
    print(f"Found {len(files)} offer JSON files.")

    documents: List[Document] = []
    ts = now_iso()

    for path in files:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            data: Dict[str, Any] = json.loads(raw_text)
        except Exception as e:
            print(f"Skipping {path.name}: failed to load JSON ({e})")
            continue

        entity_id = str(data.get("entity_id") or path.stem)
        entity_name = str(data.get("entity_name") or entity_id)
        entity_type = str(data.get("entity_type") or "").strip()
        status = str(data.get("status") or "").strip()
        source_url = str(data.get("source_url") or "").strip()
        category_tags = data.get("category_tags") or []
        user_intents = data.get("user_intents") or []
        summary = str(data.get("summary") or "").strip()

        offers = data.get("offers") or []
        if not isinstance(offers, list) or not offers:
            print(f"Skipping {path.name}: no 'offers' array found.")
            continue

        entity_snippet = build_entity_snippet(data)
        brand = normalize_brand(entity_name)
        entity_summary_points = safe_get_list(data, "summary_points")
        entity_summary_parts: List[str] = []
        if summary:
            entity_summary_parts.append(summary)
        if entity_summary_points:
            entity_summary_parts.extend(entity_summary_points[:3])
        entity_summary_display = " ".join(entity_summary_parts) if entity_summary_parts else ""

        for idx, offer in enumerate(offers):
            if not isinstance(offer, dict):
                continue
            offer_id = str(offer.get("offer_id") or f"{entity_id}-{idx}")
            offer_title = str(offer.get("title") or "").strip()
            offer_type = str(offer.get("offer_type") or "").strip()
            channels = safe_get_list(offer, "channels")
            offer_summary_points = safe_get_list(offer, "summary_points")
            offer_summary_display = " ".join(offer_summary_points[:3]) if offer_summary_points else ""

            parent_snippet_parts: List[str] = []
            if entity_name:
                parent_snippet_parts.append(entity_name)
            if offer_title:
                parent_snippet_parts.append(offer_title)
            if offer_type:
                parent_snippet_parts.append(f"Type: {offer_type}")
            parent_snippet = " | ".join(parent_snippet_parts) or entity_snippet

            full_text = build_offer_full_text(data, offer)
            if not full_text.strip():
                continue

            # Derive normalized fields that will be useful for filtering
            category = infer_category(entity_name, category_tags, offer_title, summary)
            offer_type_normalized = normalize_offer_type(offer_type, offer_title, full_text)

            # Prefer a single coherent chunk per offer when text is not too long
            token_count = len(full_text.split())
            if token_count <= 400:
                chunk_texts = [full_text]
            else:
                chunk_texts = chunk_text_with_overlap(full_text)

            for i, chunk_text in enumerate(chunk_texts):
                chunk_id = f"{entity_id}_{offer_id}_chunk{i}".replace(" ", "_")
                embedding_input = f"offer_chunk: {chunk_text}"
                embedding = encode(embedding_input, EMBEDDING_DIM)

                metadata: Dict[str, Any] = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "offer_id": offer_id,
                    "offer_title": offer_title,
                    "offer_type": offer_type,
                    "offer_type_normalized": offer_type_normalized,
                    "source_url": source_url,
                    "entity_summary": entity_summary_display,
                    "offer_summary_points": offer_summary_points,
                    "offer_summary_display": offer_summary_display,
                    "status": status,
                    "channels": channels,
                    "category_tags": category_tags,
                    "category": category,
                    "brand": brand,
                    "merchant": brand,
                    "user_intents": user_intents,
                    "parent_snippet": parent_snippet,
                    "chunk_index": i,
                    "chunk_token_size": CHUNK_TOKEN_SIZE,
                    "chunk_token_overlap": CHUNK_TOKEN_OVERLAP,
                    "source_type": "offers_json",
                    "timestamp": ts,
                }

                documents.append(
                    Document(
                        id=chunk_id,
                        embedding=embedding,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

    if not documents:
        print("No documents to index (no valid offers found).")
        return

    client = VectorDBClient()
    if using_real_embeddings():
        print("Using real embeddings (sentence-transformers) for offer chunks.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")

    print(f"Indexing {len(documents)} chunks into '{INDEX_NAME}' ...")
    count = client.index_documents(INDEX_NAME, documents, refresh=True)
    print(f"Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()


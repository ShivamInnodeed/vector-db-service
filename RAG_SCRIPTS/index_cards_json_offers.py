"""
Index credit card offers defined as structured JSON under data/cards_json_data.

Flow:
- Read *.json from data/cards_json_data
- Adapt each file into a CardJsonOffer
- Build logical chunks (overview, rewards, fees, eligibility/offers) with metadata
- Generate embeddings using MiniLM-L6-v2 via scripts.embedding_utils.encode
- Index chunks into Elasticsearch via VectorDBClient as Documents
  in index: credit_card_offers_rag_json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient  # type: ignore
from vector_db.models import Document  # type: ignore
from scripts.embedding_utils import EMBEDDING_DIM, encode, using_real_embeddings  # type: ignore
from RAG_SCRIPTS.cards_json_offer_adapter import CardJsonOffer  # type: ignore


DATA_DIR = ROOT / "data" / "cards_json_data"
INDEX_NAME = "credit_card_offers_rag_json"


def iter_offer_files() -> Iterable[Path]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Cards JSON data directory not found: {DATA_DIR}")
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.is_file() and path.name != "enrichment_log.json":
            yield path


def load_offers() -> List[CardJsonOffer]:
    offers: List[CardJsonOffer] = []
    for path in iter_offer_files():
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            data: Dict[str, object] = json.loads(raw_text)
            offer = CardJsonOffer.from_raw(data)  # type: ignore[arg-type]
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

        if not offer.url:
            print(f"Skipping {path.name}: missing url")
            continue
        if offer.status and offer.status.lower() not in ("active", "live", "enabled", ""):
            # Optionally skip inactive cards
            print(f"Skipping {path.name}: status={offer.status!r}")
            continue

        offers.append(offer)
    return offers


def build_chunks(offer: CardJsonOffer) -> List[Dict[str, str]]:
    """
    Build logical chunks for an offer.

    Each chunk dict has: chunk_id_suffix, chunk_type, text, parent_snippet.
    """
    chunks: List[Dict[str, str]] = []

    parent_snippet_parts: List[str] = [offer.card_name]
    if offer.category and offer.category != "unknown":
        parent_snippet_parts.append(f"Category: {offer.category}")
    if offer.reward_summary and offer.reward_summary != "unknown":
        parent_snippet_parts.append(f"Rewards: {offer.reward_summary}")
    if offer.annual_fee and offer.annual_fee != "unknown":
        parent_snippet_parts.append(f"Annual fee: {offer.annual_fee}")
    parent_snippet = " | ".join(parent_snippet_parts)

    # Overview
    overview_lines = [f"{offer.card_name} ({offer.category or 'card'})"]
    if offer.reward_summary and offer.reward_summary != "unknown":
        overview_lines.append(f"Reward summary: {offer.reward_summary}")
    if offer.annual_fee != "unknown" or offer.joining_fee != "unknown":
        overview_lines.append(f"Fees: joining {offer.joining_fee}, annual {offer.annual_fee}")
    if offer.key_benefits:
        overview_lines.append("Key benefits: " + "; ".join(offer.key_benefits[:5]))
    overview_text = ". ".join(overview_lines)
    if len(overview_text) >= 30:
        chunks.append(
            {
                "chunk_id_suffix": "overview",
                "chunk_type": "overview",
                "text": overview_text,
                "parent_snippet": parent_snippet,
            }
        )

    # Rewards & benefits
    if offer.key_benefits or (offer.reward_summary and offer.reward_summary != "unknown"):
        rewards_lines: List[str] = [f"{offer.card_name} rewards and benefits"]
        if offer.reward_summary and offer.reward_summary != "unknown":
            rewards_lines.append(f"Reward summary: {offer.reward_summary}")
        if offer.key_benefits:
            rewards_lines.append("Benefits: " + "; ".join(offer.key_benefits))
        rewards_text = ". ".join(rewards_lines)
        if len(rewards_text) >= 30:
            chunks.append(
                {
                    "chunk_id_suffix": "rewards",
                    "chunk_type": "rewards",
                    "text": rewards_text,
                    "parent_snippet": parent_snippet,
                }
            )

    # Fees & charges
    if offer.annual_fee != "unknown" or offer.joining_fee != "unknown" or offer.important_terms:
        fees_lines: List[str] = [f"{offer.card_name} fees and key terms"]
        fees_lines.append(f"Joining fee: {offer.joining_fee}")
        fees_lines.append(f"Annual fee: {offer.annual_fee}")
        if offer.important_terms:
            fees_lines.append("Important terms: " + "; ".join(offer.important_terms[:6]))
        fees_text = ". ".join(fees_lines)
        if len(fees_text) >= 30:
            chunks.append(
                {
                    "chunk_id_suffix": "fees",
                    "chunk_type": "fees",
                    "text": fees_text,
                    "parent_snippet": parent_snippet,
                }
            )

    # Eligibility – currently empty for most cards but keep the shape
    if offer.eligibility:
        elig_lines = [f"{offer.card_name} eligibility", "; ".join(offer.eligibility)]
        elig_text = ". ".join(elig_lines)
        if len(elig_text) >= 30:
            chunks.append(
                {
                    "chunk_id_suffix": "eligibility",
                    "chunk_type": "eligibility",
                    "text": elig_text,
                    "parent_snippet": parent_snippet,
                }
            )

    # Joining & milestone offers
    if offer.joining_offers or offer.milestone_offers:
        offers_lines: List[str] = [f"{offer.card_name} joining and milestone offers"]
        if offer.joining_offers:
            offers_lines.append("Joining offers: " + "; ".join(offer.joining_offers))
        if offer.milestone_offers:
            offers_lines.append("Milestone offers: " + "; ".join(offer.milestone_offers))
        offers_text = ". ".join(offers_lines)
        if len(offers_text) >= 30:
            chunks.append(
                {
                    "chunk_id_suffix": "offers",
                    "chunk_type": "offers",
                    "text": offers_text,
                    "parent_snippet": parent_snippet,
                }
            )

    return chunks


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    offers = load_offers()
    print(f"Loaded {len(offers)} card offers from {DATA_DIR}")

    client = VectorDBClient()
    if using_real_embeddings():
        print("Using real embeddings (sentence-transformers) for offer chunks.\n")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).\n")

    documents: List[Document] = []
    ts = now_iso()

    for offer in offers:
        chunks = build_chunks(offer)
        for chunk in chunks:
            chunk_id = f"{offer.card_id}_{chunk['chunk_id_suffix']}".replace(" ", "_")
            text = chunk["text"]
            chunk_type = chunk["chunk_type"]
            parent_snippet = chunk["parent_snippet"]

            embedding_input = f"{chunk_type}: {text}"
            embedding = encode(embedding_input, EMBEDDING_DIM)

            doc = Document(
                id=chunk_id,
                embedding=embedding,
                text=text,
                metadata={
                    "card_id": offer.card_id,
                    "card_name": offer.card_name,
                    "issuer": offer.issuer,
                    "category": offer.category,
                    "network": offer.network,
                    "chunk_type": chunk_type,
                    "source_url": offer.url,
                    "language": offer.language,
                    "parent_snippet": parent_snippet,
                    "reward_summary": offer.reward_summary,
                    "annual_fee": offer.annual_fee,
                    "joining_fee": offer.joining_fee,
                    "category_tags": offer.category_tags,
                    "partner_category_tags": offer.partner_category_tags,
                    "user_intents": offer.user_intents,
                    "source_type": "cards_json_data",
                    "timestamp": ts,
                },
            )
            documents.append(doc)

    if not documents:
        print("No documents to index (no valid offers found).")
        return

    print(f"Indexing {len(documents)} chunks into '{INDEX_NAME}' ...")
    count = client.index_documents(INDEX_NAME, documents, refresh=True)
    print(f"Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()


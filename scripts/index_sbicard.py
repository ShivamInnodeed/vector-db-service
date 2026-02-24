"""
Index sbicard_homepage.md into Elasticsearch (Docker) via VectorDB module.

Run after Elasticsearch is up (e.g. sudo docker-compose up -d elasticsearch).
Load env from .env (ELASTICSEARCH_URL etc.); then run:
  python scripts/index_sbicard.py [path_to_sbicard_homepage.md]
"""

import os
import sys
from pathlib import Path

# Add project root so vector_db is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient
from vector_db.models import Document
from scripts.parse_sbicard_md import parse_sbicard_md

# Default path to sbicard markdown (relative to repo root or cwd)
DEFAULT_MD_PATH = ROOT.parent / "Architecture Diagram" / "sbicard_homepage.md"
VECTOR_DIM = 384
INDEX_NAME = "sbicard_chunks"


def make_embedding_placeholder(text: str, dim: int = VECTOR_DIM):
    """Placeholder embedding (constant) so semantic search runs; BM25/hybrid use real text."""
    # Simple hash-based pseudo-embedding so different texts get slightly different vectors
    h = hash(text) % 10000
    return [0.1 + (h % 10) / 1000.0] * dim


def main():
    md_path = os.getenv("SBICARD_MD_PATH", str(DEFAULT_MD_PATH))
    if len(sys.argv) > 1:
        md_path = sys.argv[1]
    md_path = Path(md_path)
    if not md_path.is_absolute() and not md_path.exists():
        # Try relative to parent of project root (e.g. py/Architecture Diagram/...)
        alt = ROOT.parent / md_path
        if alt.exists():
            md_path = alt
    if not md_path.exists():
        print(f"Error: file not found: {md_path}")
        print("Usage: python scripts/index_sbicard.py [path_to_sbicard_homepage.md]")
        print("Default (no arg): ../Architecture Diagram/sbicard_homepage.md")
        sys.exit(1)
    print(f"Parsing {md_path} ...")
    chunks = list(parse_sbicard_md(md_path))
    print(f"Found {len(chunks)} link chunks.")
    client = VectorDBClient()
    documents = [
        Document(
            id=c.id,
            embedding=make_embedding_placeholder(c.text, VECTOR_DIM),
            text=c.text,
            metadata={"link": c.link, "header": c.header, "section": c.section},
        )
        for c in chunks
    ]
    count = client.index_documents(INDEX_NAME, documents, refresh=True)
    print(f"Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()

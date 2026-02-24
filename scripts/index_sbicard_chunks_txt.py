"""
Index Crawl4AI chunked file (sbicard_chunks.txt) into Elasticsearch via VectorDB.

Use this for data training of local Docker Elasticsearch with cosine-similarity chunks.
Run after ES is up; set ELASTICSEARCH_URL in .env.

  python scripts/index_sbicard_chunks_txt.py [path_to_sbicard_chunks.txt]
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db import VectorDBClient
from vector_db.models import Document
from scripts.parse_sbicard_chunks_txt import parse_sbicard_chunks_txt

DEFAULT_CHUNKS_PATH = ROOT.parent / "Architecture Diagram" / "sbicard_chunks.txt"
VECTOR_DIM = 384
INDEX_NAME = "sbicard_chunks"

# First https URL in text for metadata.link
URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)


def _first_link(text: str) -> str:
    m = URL_RE.search(text)
    return m.group(0).rstrip(".,;") if m else "https://www.sbicard.com/"


def _placeholder_embedding(text: str, dim: int = VECTOR_DIM):
    h = hash(text) % 10000
    return [0.1 + (h % 10) / 1000.0] * dim


def main():
    chunks_path = os.getenv("SBICARD_CHUNKS_PATH", str(DEFAULT_CHUNKS_PATH))
    if len(sys.argv) > 1:
        chunks_path = sys.argv[1]
    chunks_path = Path(chunks_path)
    if not chunks_path.is_absolute() and not chunks_path.exists():
        alt = ROOT.parent / chunks_path
        if alt.exists():
            chunks_path = alt
    if not chunks_path.exists():
        print(f"Error: file not found: {chunks_path}")
        print("Usage: python scripts/index_sbicard_chunks_txt.py [path_to_sbicard_chunks.txt]")
        sys.exit(1)

    print(f"Parsing {chunks_path} ...")
    chunks = list(parse_sbicard_chunks_txt(chunks_path))
    print(f"Found {len(chunks)} chunks.")

    client = VectorDBClient()
    documents = []
    for c in chunks:
        link = _first_link(c.text)
        doc_id = f"sbicard_cosine_{c.chunk_id}"
        documents.append(
            Document(
                id=doc_id,
                embedding=_placeholder_embedding(c.text, VECTOR_DIM),
                text=c.text,
                metadata={
                    "chunk_id": c.chunk_id,
                    "cosine_score": c.score,
                    "link": link,
                    "header": f"Chunk {c.chunk_id}",
                },
            )
        )

    count = client.index_documents(INDEX_NAME, documents, refresh=True)
    print(f"Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()

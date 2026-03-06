"""
Index sbicard_homepage.md into Elasticsearch (Docker) using
parent–child style chunks (children for search, parent snippet
for display).

This script is isolated under scripts_parent_child/ so we can
compare behaviour with the original scripts/ pipeline.
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
from scripts.embedding_utils import encode, using_real_embeddings, EMBEDDING_DIM
from scripts.data_cleaning import clean_markdown_content
from scripts_parent_child.parse_sbicard_parent_child import parse_sbicard_parent_child

DEFAULT_MD_PATH = ROOT / "data" / "sbicard_homepage.md"
INDEX_NAME = "sbicard_parent_child"


def main():
    md_path = os.getenv("SBICARD_MD_PATH", str(DEFAULT_MD_PATH))
    if len(sys.argv) > 1:
        md_path = sys.argv[1]
    md_path = Path(md_path)
    if not md_path.is_absolute() and not md_path.exists():
        alt = ROOT.parent / "Architecture Diagram" / "sbicard_homepage.md"
        if alt.exists():
            md_path = alt
    if not md_path.exists():
        print(f"Error: file not found: {md_path}")
        print("Usage: python scripts_parent_child/index_sbicard_parent_child.py [path_to_sbicard_homepage.md]")
        sys.exit(1)

    print(f"Reading and cleaning {md_path} ...")
    raw_content = md_path.read_text(encoding="utf-8", errors="replace")
    cleaned_content = clean_markdown_content(raw_content)
    print("Data cleaning done.")
    print("Parsing (parent–child style) ...")
    chunks = list(parse_sbicard_parent_child(md_path, content=cleaned_content))
    print(f"Found {len(chunks)} child chunks.")
    if using_real_embeddings():
        print("Using real embeddings (sentence-transformers) for semantic search.")
    else:
        print("Using placeholder embeddings (install sentence-transformers for better KNN).")

    client = VectorDBClient()
    documents = [
        Document(
            id=c.id,
            embedding=encode(c.child_text, EMBEDDING_DIM),
            text=c.child_text,
            metadata={
                "link": c.url,
                "header": c.header,
                "section": c.section,
                "parent_snippet": c.parent_snippet,
                "chunking_strategy": "parent_child_child_for_search",
            },
        )
        for c in chunks
    ]
    count = client.index_documents(INDEX_NAME, documents, refresh=True)
    print(f"Indexed {count} documents into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()


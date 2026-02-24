"""
Parse sbicard homepage markdown into chunks with link and header for indexing.

Extracts [text](url) patterns and builds one document per link with:
- text: searchable content (header/label + link)
- metadata: link, header
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SbicardChunk:
    """One chunk from the markdown: link + header for search."""
    id: str
    header: str
    link: str
    text: str
    section: str


# Regex for markdown links: [text](url)
LINK_PATTERN = re.compile(r'\[([^\]]*)\]\((https?://[^)\s]+)\)', re.IGNORECASE)


def _slug(id_num: int, link: str) -> str:
    """Generate a unique doc id."""
    safe = re.sub(r'[^a-zA-Z0-9-]', '_', link.strip())[:80]
    return f"sbicard_{id_num}_{safe}"


def parse_sbicard_md(md_path) -> Iterator[SbicardChunk]:
    """
    Parse markdown file and yield chunks (one per link) with header and link.
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    current_section = ""
    seen_links = set()
    chunk_id = 0
    for line in lines:
        # Track section from lines that look like headers (no link, short)
        stripped = line.strip()
        if stripped and not LINK_PATTERN.search(line) and len(stripped) < 80:
            if not stripped.startswith("![]") and not stripped.startswith("*"):
                current_section = stripped
        for m in LINK_PATTERN.finditer(line):
            label = (m.group(1) or "").strip()
            url = (m.group(2) or "").strip()
            if not url or url.startswith("javascript:"):
                continue
            # Dedupe by (url, label)
            key = (url, label)
            if key in seen_links:
                continue
            seen_links.add(key)
            header = label or current_section or "SBI Card"
            # Searchable text: header + link so both BM25 and semantic can match
            text = f"{header} {url}"
            if current_section and current_section != header:
                text = f"{current_section} {text}"
            chunk_id += 1
            yield SbicardChunk(
                id=_slug(chunk_id, url),
                header=header,
                link=url,
                text=text,
                section=current_section or "",
            )

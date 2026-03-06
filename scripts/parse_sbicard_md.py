"""
Parse sbicard homepage markdown into chunks with link and header for indexing.

Hybrid chunking: one chunk per link, with context-expanded text so that
- KNN (cosine/semantic) gets enough natural language for whole-sentence queries.
- BM25 (keyword) still matches label, URL, section, and surrounding words.

Extracts [text](url) and builds one document per link with:
- text: section + label + URL + surrounding context (next few prose lines, capped)
- metadata: link, header, section
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

# Hybrid chunking: max prose lines after a link to include, max words per chunk
MAX_CONTEXT_LINES = 3
MAX_CHUNK_WORDS = 400


@dataclass
class SbicardChunk:
    """One chunk from the markdown: link + header + context for search."""
    id: str
    header: str
    link: str
    text: str
    section: str


# Regex for markdown links: [text](url)
LINK_PATTERN = re.compile(r'\[([^\]]*)\]\((https?://[^)\s]+)\)', re.IGNORECASE)
# Page boundary in Crawl4AI .md
PAGE_MARKER = re.compile(r'^---\s*Page\s+\d+\s*---', re.IGNORECASE)


def _slug(id_num: int, link: str) -> str:
    """Generate a unique doc id."""
    safe = re.sub(r'[^a-zA-Z0-9-]', '_', link.strip())[:80]
    return f"sbicard_{id_num}_{safe}"


def _is_prose_line(line: str) -> bool:
    """True if line looks like body text (not only links/images)."""
    s = line.strip()
    if not s or len(s) < 10:
        return False
    stripped_links = re.sub(r'\[([^\]]*)\]\([^)]+\)', '', s)
    stripped_links = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', stripped_links)
    return len(stripped_links.strip()) >= 8


def _get_context(lines: List[str], start_index: int) -> str:
    """Gather up to MAX_CONTEXT_LINES of prose after start_index (same page)."""
    parts: List[str] = []
    for i in range(start_index + 1, min(start_index + 1 + MAX_CONTEXT_LINES, len(lines))):
        if PAGE_MARKER.match(lines[i].strip()):
            break
        if _is_prose_line(lines[i]):
            parts.append(lines[i].strip())
    return " ".join(parts) if parts else ""


def _trim_to_word_limit(text: str, limit: int = MAX_CHUNK_WORDS) -> str:
    """Keep first `limit` words."""
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


def parse_sbicard_md(md_path: Union[str, Path], content: Optional[str] = None) -> Iterator[SbicardChunk]:
    """
    Parse markdown file and yield chunks (one per link) with header, link, and
    surrounding context (hybrid chunking for semantic + keyword search).
    If content is provided, use it instead of reading from md_path (e.g. after data cleaning).
    """
    if content is not None:
        lines = content.splitlines()
    else:
        path = Path(md_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
    current_section = ""
    seen_links = set()
    chunk_id = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not LINK_PATTERN.search(line) and len(stripped) < 80:
            if not stripped.startswith("![]") and not stripped.startswith("*"):
                current_section = stripped
        for m in LINK_PATTERN.finditer(line):
            label = (m.group(1) or "").strip()
            url = (m.group(2) or "").strip()
            if not url or url.startswith("javascript:"):
                continue
            key = (url, label)
            if key in seen_links:
                continue
            seen_links.add(key)
            header = label or current_section or "SBI Card"
            base = f"{header} {url}"
            if current_section and current_section != header:
                base = f"{current_section} {base}"
            context = _get_context(lines, idx)
            text = f"{base} {context}".strip() if context else base
            text = _trim_to_word_limit(text, MAX_CHUNK_WORDS)
            chunk_id += 1
            yield SbicardChunk(
                id=_slug(chunk_id, url),
                header=header,
                link=url,
                text=text,
                section=current_section or "",
            )

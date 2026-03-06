"""
Parent–child style parsing for sbicard_homepage.md.

Children: small chunks used for search (short, focused text).
Parents: longer snippets stored in metadata (used for display as
\"long headers\" / short answer + URL).

This keeps retrieval precise (small child text) while letting the UI
show a richer parent_snippet to the user.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union


MAX_CHILD_WORDS = 80
MAX_PARENT_WORDS = 220


@dataclass
class SbicardChildChunk:
    """Child chunk for search, with parent snippet for display."""

    id: str
    header: str
    url: str
    child_text: str
    parent_snippet: str
    section: str


LINK_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)", re.IGNORECASE)
PAGE_MARKER = re.compile(r"^---\s*Page\s+(\d+)\s*---", re.IGNORECASE)


def _slug_child(page_num: int, child_idx: int, url: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9-]", "_", url.strip())[:60]
    return f"sbicard_p{page_num}_c{child_idx}_{safe}"


def _is_prose_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 10:
        return False
    stripped_links = re.sub(r"\[([^\]]*)\]\([^)]+\)", "", s)
    stripped_links = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", stripped_links)
    return len(stripped_links.strip()) >= 8


def _collect_context(lines: List[str], start: int, max_lines: int) -> List[str]:
    parts: List[str] = []
    for i in range(start + 1, min(start + 1 + max_lines, len(lines))):
        if PAGE_MARKER.match(lines[i].strip()):
            break
        if _is_prose_line(lines[i]):
            parts.append(lines[i].strip())
    return parts


def _trim_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


def parse_sbicard_parent_child(
    md_path: Union[str, Path], content: Optional[str] = None
) -> Iterator[SbicardChildChunk]:
    """
    Parse markdown file into child chunks + parent snippets.

    - Child text: short, focused string used for embedding / search.
    - Parent snippet: longer text from same page/section used as
      \"long header\" / short answer in the UI.
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

    page_num = 0
    current_section = ""
    seen = set()
    child_idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        m_page = PAGE_MARKER.match(stripped)
        if m_page:
            page_num = int(m_page.group(1))
            current_section = ""
            continue

        # Section tracking: short, non-link line
        if stripped and not LINK_PATTERN.search(line) and len(stripped) < 80:
            if not stripped.startswith("![]") and not stripped.startswith("*"):
                current_section = stripped

        for m in LINK_PATTERN.finditer(line):
            label = (m.group(1) or "").strip()
            url = (m.group(2) or "").strip()
            if not url or url.startswith("javascript:"):
                continue

            key = (page_num, url, label)
            if key in seen:
                continue
            seen.add(key)

            header = label or current_section or "SBI Card"

            # Child text: very focused string for search
            base_child = f"{header} {url}"
            if current_section and current_section != header:
                base_child = f"{current_section} {base_child}"
            child_text = _trim_words(base_child, MAX_CHILD_WORDS)

            # Parent snippet: richer context from same page/section
            context_lines = _collect_context(lines, idx, max_lines=6)
            parent_base = " ".join([header] + context_lines) if context_lines else header
            parent_snippet = _trim_words(parent_base, MAX_PARENT_WORDS)

            child_idx += 1
            yield SbicardChildChunk(
                id=_slug_child(page_num, child_idx, url),
                header=header,
                url=url,
                child_text=child_text,
                parent_snippet=parent_snippet,
                section=current_section or "",
            )


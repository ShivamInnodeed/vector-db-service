"""
Data cleaning for Crawl4AI-generated markdown before chunking.

Reduces boilerplate, dedupes repeated lines, and normalizes content
so chunking and embedding focus on meaningful text.
"""

import re
from typing import List

# Boilerplate lines that add noise when repeated (keep structure, dedupe consecutive)
BOILERPLATE_PHRASES = {
    "connect with us",
    "useful links",
    "quick actions",
    "©2026",
    "site best viewed in browsers",
}

# Consecutive blank lines above this become at most MAX_BLANK
MAX_CONSECUTIVE_BLANKS = 2

# Lines that are only markdown images with no meaningful alt: ![](url)
IMAGE_ONLY = re.compile(r"^!\[[^\]]*\]\s*\([^)]+\)\s*$")


def clean_markdown_content(content: str) -> str:
    """
    Clean raw Crawl4AI markdown: dedupe boilerplate, collapse blanks,
    drop image-only lines and javascript links, normalize whitespace.
    Returns cleaned markdown string (same structure, less noise).
    """
    if not content or not content.strip():
        return content

    lines = content.splitlines()
    out: List[str] = []
    prev_line: str = ""
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Collapse many blanks to MAX_CONSECUTIVE_BLANK
        if not stripped:
            blank_count += 1
            if blank_count <= MAX_CONSECUTIVE_BLANKS:
                out.append("")
            continue
        blank_count = 0

        # Drop image-only lines (no alt text / no semantic value for search)
        if IMAGE_ONLY.match(stripped):
            continue

        # Drop lines that are only javascript: or empty-link
        if stripped.startswith("(javascript:") or stripped == "[](javascript:void(0);)":
            continue

        # Dedupe consecutive identical boilerplate (e.g. "Useful Links" repeated 50x)
        lower = stripped.lower()
        if lower in BOILERPLATE_PHRASES and stripped == prev_line:
            continue
        if lower in BOILERPLATE_PHRASES:
            prev_line = stripped
        else:
            prev_line = ""

        # Normalize: single line, strip only (preserve one trailing space if needed for md)
        out.append(stripped)

    # Trim leading/trailing blanks from result
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()

    return "\n".join(out) if out else ""

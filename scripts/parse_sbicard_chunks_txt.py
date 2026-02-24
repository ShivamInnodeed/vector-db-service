"""
Parse Crawl4AI cosine-similarity chunked output (sbicard_chunks.txt).

Format:
  --- Chunk N ---
  Score: 0.XXXX
  <chunk text>
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Crawl4AIChunk:
    """One chunk from Crawl4AI chunked output."""
    chunk_id: int
    score: float
    text: str


# Pattern: --- Chunk N --- then Score: X.XXXX then newline then content
CHUNK_BLOCK = re.compile(
    r"--- Chunk (\d+) ---\s*\nScore:\s*([\d.]+)\s*\n(.*?)(?=\n--- Chunk \d+ ---|\Z)",
    re.DOTALL,
)


def parse_sbicard_chunks_txt(txt_path) -> Iterator[Crawl4AIChunk]:
    """
    Parse sbicard_chunks.txt and yield Crawl4AIChunk (chunk_id, score, text).
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    for m in CHUNK_BLOCK.finditer(content):
        chunk_id = int(m.group(1))
        score = float(m.group(2))
        text = (m.group(3) or "").strip()
        if not text:
            continue
        yield Crawl4AIChunk(chunk_id=chunk_id, score=score, text=text)

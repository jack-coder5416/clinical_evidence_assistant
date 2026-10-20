"""Segments -> retrievable chunks that carry citation metadata."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS
from src.ingestion.parser import Segment


@dataclass
class Chunk:
    id: str
    text: str                       # exactly what is shown / sent to the LLM
    metadata: dict = field(default_factory=dict)


def build_index_text(text: str, metadata: dict) -> str:
    """Text used for BOTH embedding and BM25.

    Prepending the section heading gives every chunk its context (a chunk that
    just says "the sample size should be justified" is ambiguous on its own),
    and lets keyword search match on heading words too. The displayed/cited
    text stays clean.
    """
    section = metadata.get("section", "")
    return f"{section}\n{text}" if section else text


def _split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",   # the ". " stays with the sentence it ends, not the next chunk
    )
    pieces = [p.lstrip(". ").strip() for p in splitter.split_text(text)]
    pieces = [p for p in pieces if p]

    # Fold a tiny trailing fragment into the previous chunk instead of
    # indexing a meaningless stub.
    merged: list[str] = []
    for p in pieces:
        if merged and len(p) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n{p}"
        else:
            merged.append(p)
    return merged


def chunk_segments(segments: list[Segment]) -> list[Chunk]:
    chunks: list[Chunk] = []
    per_page_counter: dict[tuple[str, int], int] = {}

    for seg in segments:
        for piece in _split(seg.text):
            if len(piece) < 60:          # drop stubs such as lone captions
                continue
            key = (seg.source, seg.page)
            idx = per_page_counter.get(key, 0)
            per_page_counter[key] = idx + 1

            # Stable, human-readable id: re-ingesting never creates duplicates.
            stem = hashlib.md5(seg.source.encode()).hexdigest()[:6]
            chunk_id = f"{stem}-p{seg.page:03d}-c{idx:02d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        "source": seg.source,
                        "title": seg.title,
                        "page": seg.page,
                        "page_label": seg.page_label,
                        "section": seg.section,
                        "section_path": seg.section_path,
                    },
                )
            )
    return chunks


# Backwards-compatible helper (old API: text -> list[str]).
def chunk_document(text: str) -> list[str]:
    return _split(text)

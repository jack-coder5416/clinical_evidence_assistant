"""Build the search index.

    python -m src.ingestion.ingest                 # every PDF in data/raw/
    python -m src.ingestion.ingest path/to/a.pdf   # specific file(s)

Re-running rebuilds the index from scratch (safe and idempotent).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import RAW_DATA_DIR
from src.ingestion.chunker import build_index_text, chunk_segments
from src.ingestion.parser import extract_segments
from src.vectordb.chroma_store import rebuild_collection


def build_chunks(pdf_paths: list[Path]):
    chunks = []
    for path in pdf_paths:
        segments = extract_segments(path)
        new = chunk_segments(segments)
        print(f"  {path.name}: {len(segments)} segments -> {len(new)} chunks")
        chunks.extend(new)
    return chunks


def ingest(pdf_paths: list[str | Path] | None = None) -> int:
    from src.embeddings.embedding_model import embed_documents

    paths = [Path(p) for p in pdf_paths] if pdf_paths else sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not paths:
        raise SystemExit(f"No PDFs found. Put files in {RAW_DATA_DIR} or pass paths.")

    print("Parsing...")
    chunks = build_chunks(paths)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_documents(
        [build_index_text(c.text, c.metadata) for c in chunks], show_progress=True
    )

    rebuild_collection(chunks, embeddings)
    print(f"Ingestion complete: {len(chunks)} chunks from {len(paths)} file(s).")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pdfs", nargs="*", help="PDF paths (default: all of data/raw/*.pdf)")
    ingest(ap.parse_args().pdfs or None)

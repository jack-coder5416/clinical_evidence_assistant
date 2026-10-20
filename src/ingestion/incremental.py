"""Idempotent incremental indexing for normalized external source records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import PROJECT_ROOT
from src.embeddings.embedding_model import embed_documents
from src.vectordb.chroma_store import get_or_create_collection

STATE_DB = PROJECT_ROOT / "data" / "sync_state.sqlite3"


def _connect():
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS source_state (
        source_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
        indexed_at TEXT NOT NULL, title TEXT NOT NULL, source_type TEXT NOT NULL,
        source_url TEXT NOT NULL)""")
    return conn


def sync_documents(documents) -> dict:
    """Upsert changed records and remove obsolete chunks for updated records."""
    collection = get_or_create_collection()
    conn = _connect()
    added = updated = unchanged = failed = chunks_written = 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                               separators=["\n\n", "\n", ". ", " ", ""])
    try:
        for doc in documents:
            digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            row = conn.execute("SELECT content_hash FROM source_state WHERE source_id=?", (doc.source_id,)).fetchone()
            if row and row[0] == digest:
                unchanged += 1
                continue
            try:
                pieces = [p.strip() for p in splitter.split_text(doc.text) if len(p.strip()) >= 40]
                if not pieces:
                    continue
                ids, metadatas = [], []
                now = datetime.now(timezone.utc).isoformat()
                for i, piece in enumerate(pieces):
                    chunk_id = hashlib.sha256(f"{doc.source_id}:{i}".encode()).hexdigest()[:32]
                    ids.append(chunk_id)
                    metadatas.append({
                        "source": doc.source_id, "source_id": doc.source_id,
                        "source_type": doc.source_type, "title": doc.title[:500],
                        "source_url": doc.url, "published_at": doc.published_at or "unknown",
                        "updated_at": doc.updated_at or "unknown", "ingested_at": now,
                        "license": (doc.license or "unknown")[:500],
                        "section": doc.source_type, "section_path": doc.source_type,
                        "page_label": "n/a", "page": 0,
                    })
                vectors = embed_documents(pieces, show_progress=False)
                # Remove old chunks for this source before upserting the new revision.
                existing = collection.get(where={"source_id": doc.source_id}, include=[])
                if existing.get("ids"):
                    collection.delete(ids=existing["ids"])
                collection.upsert(ids=ids, documents=pieces, metadatas=metadatas,
                                  embeddings=vectors.tolist())
                conn.execute("""INSERT INTO source_state(source_id,content_hash,indexed_at,title,source_type,source_url)
                    VALUES(?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET
                    content_hash=excluded.content_hash,indexed_at=excluded.indexed_at,
                    title=excluded.title,source_type=excluded.source_type,source_url=excluded.source_url""",
                    (doc.source_id, digest, now, doc.title, doc.source_type, doc.url))
                if row:
                    updated += 1
                else:
                    added += 1
                chunks_written += len(pieces)
            except Exception as exc:  # keep one bad record from stopping the batch
                failed += 1
                print(f"Failed to index {doc.source_id}: {type(exc).__name__}: {exc}")
        conn.commit()
    finally:
        conn.close()
    return {"fetched": len(documents), "added": added, "updated": updated,
            "unchanged": unchanged, "failed": failed, "chunks_written": chunks_written}


def get_sync_status() -> dict:
    if not STATE_DB.exists():
        return {"documents": 0, "last_indexed_at": None}
    conn = _connect()
    try:
        count, latest = conn.execute("SELECT COUNT(*), MAX(indexed_at) FROM source_state").fetchone()
        return {"documents": count, "last_indexed_at": latest}
    finally:
        conn.close()

"""ChromaDB access. One place that knows how the collection is configured."""
from __future__ import annotations

import chromadb

from src.config.settings import COLLECTION_NAME, VECTOR_DB_PATH


def get_client(path: str = VECTOR_DB_PATH):
    return chromadb.PersistentClient(path=path)


def get_collection(client=None, name: str = COLLECTION_NAME):
    """Existing collection; raises if the index hasn't been built."""
    client = client or get_client()
    return client.get_collection(name)


def get_or_create_collection(client=None, name: str = COLLECTION_NAME):
    """Get or create a cosine-distance collection for first-run deployment."""
    client = client or get_client()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name, configuration={"hnsw": {"space": "cosine"}})


def rebuild_collection(chunks, embeddings, client=None, name: str = COLLECTION_NAME):
    """Drop and recreate the collection so re-ingestion is idempotent.

    (The old `collection.add(ids=["0","1",...])` raised on the second run and
    silently mixed old/new chunks if the PDF changed.)
    """
    client = client or get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet
    # Cosine space, matching the normalized bge embeddings.
    collection = client.create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}}
    )

    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.metadata for c in part],
            embeddings=embeddings[i : i + batch].tolist(),
        )
    return collection


def load_all(collection) -> tuple[list[str], list[str], list[dict]]:
    """Everything in the collection (used to build the BM25 index)."""
    data = collection.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


# Backwards-compatible helper.
def index_docs(chunks, embeddings):
    raise NotImplementedError(
        "index_docs() was replaced by rebuild_collection(chunks, embeddings); "
        "chunks now carry metadata."
    )

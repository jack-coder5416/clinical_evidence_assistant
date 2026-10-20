"""Embedding model (loaded lazily, once)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config.settings import EMBEDDING_MODEL, QUERY_INSTRUCTION


@lru_cache(maxsize=1)
def get_model():
    # Imported here so importing this module (e.g. for tests) doesn't pull in torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=show_progress,
    )


def embed_query(query: str) -> np.ndarray:
    # bge v1.5 retrieval quality improves when short queries carry the
    # instruction prefix; passages are embedded WITHOUT it.
    return get_model().encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)


# Backwards-compatible alias for the old name.
generate_embeddings = embed_documents

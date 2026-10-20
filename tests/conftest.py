import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.chunker import Chunk  # noqa: E402
from src.vectordb.chroma_store import rebuild_collection  # noqa: E402

# A tiny corpus where we control the "embeddings" by hand, so tests can
# demonstrate exactly when dense fails and BM25 rescues (and vice-versa).
DOCS = {
    "d1": ("Blinding (masking) of participants reduces bias in outcome assessment.", "2.1.2 Blinding"),
    "d2": ("Informed consent must be freely given and participants may withdraw at any time.", "2.2 Consent"),
    "d3": ("A CTIMP is a clinical trial of an investigational medicinal product under regulation.", "Abbreviations"),
    "d4": ("Sample size should be justified so the trial can detect a meaningful effect.", "2.1.4 Adequate size"),
    "d5": ("Data monitoring committees review accumulating safety data independently.", "2.1.12 Monitoring"),
    "d6": ("Registries and open data sharing improve transparency of trial results.", "2.3 Transparency"),
}
DIM = len(DOCS)


def onehot(i: int, extra: dict | None = None) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    for j, w in (extra or {}).items():
        v[j] = w
    return v / np.linalg.norm(v)


@pytest.fixture
def collection(tmp_path):
    chunks, embs = [], []
    for i, (cid, (text, section)) in enumerate(DOCS.items()):
        chunks.append(Chunk(cid, text, {
            "source": "guide.pdf", "title": "Guide", "page": i + 1,
            "page_label": str(i + 1), "section": section, "section_path": section}))
        embs.append(onehot(i))
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    return rebuild_collection(chunks, np.array(embs), client=client)


class ScriptedReranker:
    """Scores by a fixed {chunk-text-substring: score} table."""
    def __init__(self, table): self.table = table
    def score(self, query, texts):
        return [next((s for k, s in self.table.items() if k in t), 0.01) for t in texts]


class BrokenReranker:
    def score(self, query, texts): raise OSError("model files not found")

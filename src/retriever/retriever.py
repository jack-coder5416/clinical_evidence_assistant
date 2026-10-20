"""Hybrid retrieval orchestrator: dense + BM25 -> RRF -> cross-encoder rerank."""
from __future__ import annotations

import time
from typing import Callable

from src.config.settings import (
    CANDIDATES_PER_RETRIEVER,
    DENSE_WEIGHT,
    FINAL_TOP_K,
    RERANK_CANDIDATES,
    RRF_K,
    SPARSE_WEIGHT,
)
from src.ingestion.chunker import build_index_text
from src.retriever.bm25 import BM25Index
from src.retriever.fusion import reciprocal_rank_fusion
from src.retriever.reranker import CrossEncoderReranker, Reranker
from src.retriever.types import Hit, RetrievalResult
from src.vectordb.chroma_store import get_collection, load_all

MODES = ("hybrid", "dense", "sparse")


class HybridRetriever:
    def __init__(
        self,
        collection=None,
        embed_query: Callable[[str], object] | None = None,
        reranker: Reranker | None = None,
    ):
        self.collection = collection if collection is not None else get_collection()
        ids, docs, metas = load_all(self.collection)
        if not ids:
            raise RuntimeError("The index is empty. Run: python -m src.ingestion.ingest")

        self._chunks = {i: (d, m) for i, d, m in zip(ids, docs, metas)}
        # BM25 is built in memory from the same chunks Chroma holds; for a
        # corpus of thousands of chunks this takes well under a second.
        self._index_text = {i: build_index_text(d, m) for i, (d, m) in self._chunks.items()}
        self.bm25 = BM25Index(ids, [self._index_text[i] for i in ids])

        self._embed_query = embed_query
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()

    # ------------------------------------------------------------------ utils
    @property
    def size(self) -> int:
        return len(self._chunks)

    def sources(self) -> list[str]:
        return sorted({m.get("source", "") for _, m in self._chunks.values()})

    def _embed(self, query: str):
        if self._embed_query is None:
            from src.embeddings.embedding_model import embed_query

            self._embed_query = embed_query
        return self._embed_query(query)

    def _hit(self, chunk_id: str) -> Hit:
        text, meta = self._chunks[chunk_id]
        return Hit(id=chunk_id, text=text, metadata=meta)

    def warmup(self) -> list[str]:
        """Load models up-front so the first user question isn't slow."""
        problems = []
        try:
            self._embed("warmup")
        except Exception as e:  # noqa: BLE001
            problems.append(f"Embedding model failed to load: {e}")
        try:
            self.reranker.score("warmup", ["warmup"])
        except Exception as e:  # noqa: BLE001
            problems.append(f"Reranker failed to load: {e}")
        return problems

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = FINAL_TOP_K,
        rerank: bool = True,
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        result = RetrievalResult(query=query, mode=mode, hits=[])
        hits: dict[str, Hit] = {}
        n = min(CANDIDATES_PER_RETRIEVER, self.size)

        # 1) Dense
        dense_ids: list[str] = []
        if mode in ("hybrid", "dense"):
            t = time.perf_counter()
            vec = self._embed(query)
            res = self.collection.query(
                query_embeddings=[list(map(float, vec))],
                n_results=n,
                include=["distances"],
            )
            for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), 1):
                h = hits.setdefault(cid, self._hit(cid))
                h.dense_rank, h.dense_score = rank, 1.0 - float(dist)  # cosine distance -> similarity
                dense_ids.append(cid)
            result.timings_ms["dense"] = (time.perf_counter() - t) * 1000

        # 2) Sparse (BM25)
        sparse_ids: list[str] = []
        if mode in ("hybrid", "sparse"):
            t = time.perf_counter()
            for rank, (cid, score) in enumerate(self.bm25.search(query, n), 1):
                h = hits.setdefault(cid, self._hit(cid))
                h.sparse_rank, h.sparse_score = rank, score
                sparse_ids.append(cid)
            result.timings_ms["sparse"] = (time.perf_counter() - t) * 1000

        # 3) Fuse
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                {"dense": dense_ids, "sparse": sparse_ids},
                weights={"dense": DENSE_WEIGHT, "sparse": SPARSE_WEIGHT},
                k=RRF_K,
            )
            for cid, score in fused:
                hits[cid].fused_score = score
            ordered = [cid for cid, _ in fused]
        else:
            ordered = dense_ids if mode == "dense" else sparse_ids

        candidates = [hits[c] for c in ordered[: RERANK_CANDIDATES if rerank else top_k]]

        # 4) Rerank
        if rerank and candidates:
            t = time.perf_counter()
            try:
                scores = self.reranker.score(
                    query, [self._index_text[h.id] for h in candidates]
                )
                for h, s in zip(candidates, scores):
                    h.rerank_score = s
                candidates.sort(key=lambda h: h.rerank_score, reverse=True)
                result.reranked = True
            except Exception as e:  # noqa: BLE001 - degrade gracefully, don't crash the chat
                result.warnings.append(
                    f"Reranker unavailable ({type(e).__name__}: {e}); using fused ranking."
                )
            result.timings_ms["rerank"] = (time.perf_counter() - t) * 1000

        result.hits = candidates[:top_k]
        return result

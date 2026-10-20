"""Cross-encoder reranking.

First-stage retrieval (dense + BM25) scores query and passage *independently*,
which is fast but coarse. A cross-encoder reads query and passage *together*,
so it is much better at judging "does this passage actually answer the
question?" - but too slow to run over the whole corpus. Hence: retrieve ~20
cheaply, rerank those, keep the best few.
"""
from __future__ import annotations

import math
from typing import Protocol

from src.config.settings import RERANKER_MODEL


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._load().predict([(query, t) for t in texts], batch_size=16)
        scores = [float(x) for x in raw]
        # Depending on library version/model, outputs are logits or already
        # sigmoid-ed. Normalise to 0-1 so the UI and the score gate are stable.
        if min(scores) < 0.0 or max(scores) > 1.0:
            scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        return scores

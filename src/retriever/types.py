from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    """One retrieved chunk plus the evidence of *how* it was retrieved."""
    id: str
    text: str
    metadata: dict
    dense_rank: int | None = None      # 1-based rank in dense results
    dense_score: float | None = None   # cosine similarity
    sparse_rank: int | None = None     # 1-based rank in BM25 results
    sparse_score: float | None = None  # raw BM25 score
    fused_score: float | None = None   # Reciprocal Rank Fusion score
    rerank_score: float | None = None  # cross-encoder relevance, 0-1

    # --- convenience accessors used by prompts / UI -------------------------
    @property
    def page_label(self) -> str:
        return str(self.metadata.get("page_label") or self.metadata.get("page", "?"))

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.source


@dataclass
class RetrievalResult:
    query: str
    mode: str
    hits: list[Hit]
    reranked: bool = False
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

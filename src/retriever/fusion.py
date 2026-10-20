"""Reciprocal Rank Fusion.

Dense scores (cosine, ~0-1) and BM25 scores (unbounded) are on different
scales, so adding them directly is meaningless. RRF only uses *ranks*:

    score(d) = sum over rankers of  weight / (k + rank_of_d)

It is robust, needs no tuning, and rewards documents that BOTH rankers like.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """rankings: {ranker_name: [doc_id best-first]} -> [(doc_id, score)] best-first."""
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

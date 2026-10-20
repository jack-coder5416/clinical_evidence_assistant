import numpy as np
from tests.conftest import BrokenReranker, ScriptedReranker, onehot
from src.retriever.retriever import HybridRetriever


def dense_pointing_at(i, second=None):
    """Fake query embedder whose nearest neighbour is doc index i."""
    return lambda q: onehot(i, {second: 0.5} if second is not None else None)


def test_bm25_rescues_what_dense_misses(collection):
    # Dense embedder "thinks" the query is about blinding (d1) - wrong.
    r = HybridRetriever(collection, embed_query=dense_pointing_at(0), reranker=ScriptedReranker({}))
    q = "What does CTIMP mean?"

    dense = r.search(q, mode="dense", top_k=1, rerank=False)
    assert dense.hits[0].id == "d1"                      # dense alone: wrong

    hybrid = r.search(q, mode="hybrid", top_k=6, rerank=False)
    ids = [h.id for h in hybrid.hits]
    assert "d3" in ids                                   # BM25 contributed the exact match
    assert hybrid.hits[[h.id for h in hybrid.hits].index("d3")].sparse_rank == 1


def test_dense_rescues_paraphrase_that_bm25_misses(collection):
    # No lexical overlap with d2, but the embedder knows it's about consent.
    r = HybridRetriever(collection, embed_query=dense_pointing_at(1), reranker=ScriptedReranker({}))
    q = "freedom to leave the study"
    assert r.search(q, mode="sparse", rerank=False).hits == [] or \
        r.search(q, mode="sparse", rerank=False).hits[0].id != "d2"
    assert r.search(q, mode="hybrid", top_k=1, rerank=False).hits[0].id == "d2"


def test_hit_carries_retrieval_trace_and_metadata(collection):
    r = HybridRetriever(collection, embed_query=dense_pointing_at(2), reranker=ScriptedReranker({}))
    h = r.search("CTIMP", top_k=1, rerank=False).hits[0]
    assert h.id == "d3" and h.dense_rank == 1 and h.sparse_rank == 1
    assert h.fused_score > 0 and 0.99 < h.dense_score <= 1.0
    assert (h.section, h.page_label, h.source) == ("Abbreviations", "3", "guide.pdf")


def test_reranker_reorders_candidates(collection):
    rr = ScriptedReranker({"Sample size": 0.95, "Blinding": 0.40})
    r = HybridRetriever(collection, embed_query=dense_pointing_at(0), reranker=rr)
    res = r.search("trial design bias", top_k=3, rerank=True)
    assert res.reranked and res.hits[0].id == "d4" and res.hits[0].rerank_score == 0.95
    scores = [h.rerank_score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_reranker_failure_degrades_gracefully(collection):
    r = HybridRetriever(collection, embed_query=dense_pointing_at(0), reranker=BrokenReranker())
    res = r.search("blinding", top_k=2, rerank=True)
    assert not res.reranked and res.hits and "Reranker unavailable" in res.warnings[0]


def test_top_k_respected_and_modes_validated(collection):
    r = HybridRetriever(collection, embed_query=dense_pointing_at(0), reranker=ScriptedReranker({}))
    assert len(r.search("trial", top_k=2, rerank=False).hits) == 2
    import pytest
    with pytest.raises(ValueError):
        r.search("x", mode="nope")

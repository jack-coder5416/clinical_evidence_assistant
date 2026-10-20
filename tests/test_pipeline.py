import pytest

from src.llm import openrouter
from src.llm.openrouter import LLMError
from src.llm.prompts import INSUFFICIENT, format_sources
from src.pipelines import rag_pipeline as rag
from src.retriever.retriever import HybridRetriever
from tests.conftest import ScriptedReranker, onehot


@pytest.fixture
def retriever(collection):
    return HybridRetriever(collection, embed_query=lambda q: onehot(1),
                           reranker=ScriptedReranker({"consent": 0.9}))


def test_sources_are_numbered_with_page_and_section(retriever):
    hits = retriever.search("informed consent withdraw", top_k=2).hits
    block = format_sources(hits)
    assert '<source id="1"' in block and 'id="2"' in block
    assert 'section="2.2 Consent"' in block and 'page="2"' in block


def test_full_answer_with_valid_and_hallucinated_citations(retriever, monkeypatch):
    monkeypatch.setattr(openrouter, "chat",
        lambda m, **k: "Participants may withdraw at any time without giving a reason [1]. Fake claim here [9].")
    ans = rag.ask("Can participants withdraw?", retriever, rewrite=False)
    assert ans.cited == [1] and ans.invalid_citations == [9]
    assert any("don't exist" in w for w in ans.warnings)
    assert ans.sources[0].id == "d2" and ":red-badge[?9]" in ans.display_text


def test_prompt_contains_only_retrieved_sources_and_history(retriever, monkeypatch):
    captured = {}
    monkeypatch.setattr(openrouter, "chat", lambda m, **k: captured.setdefault("m", m) and "ok [1].")
    history = [{"role": "user", "content": "Tell me about consent"},
               {"role": "assistant", "content": "Consent must be voluntary [1]."}]
    rag.ask("and withdrawal?", retriever, history, rewrite=False)
    roles = [m["role"] for m in captured["m"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "<sources>" in captured["m"][-1]["content"]


def test_no_hits_means_no_llm_call(retriever, monkeypatch):
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: pytest.fail("LLM must not be called"))
    ans = rag.ask("???", retriever, mode="sparse", rerank=False, rewrite=False)  # only stopwords/punct
    assert ans.insufficient and ans.text == INSUFFICIENT and ans.sources == []


def test_relevance_gate_blocks_weak_evidence(retriever, monkeypatch):
    monkeypatch.setattr(rag, "MIN_RERANK_SCORE", 0.5)
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: pytest.fail("LLM must not be called"))
    weak = HybridRetriever(retriever.collection, embed_query=lambda q: onehot(1),
                           reranker=ScriptedReranker({}))       # everything scores 0.01
    ans = rag.ask("informed consent", weak, rewrite=False)
    assert ans.insufficient


def test_model_saying_insufficient_is_recognised_and_not_flagged_uncited(retriever, monkeypatch):
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: INSUFFICIENT)
    ans = rag.ask("what is the moon made of", retriever, rewrite=False)
    assert ans.insufficient and ans.uncited_sentences == []


def test_rewrite_uses_history_and_falls_back_on_failure(monkeypatch):
    hist = [{"role": "user", "content": "What is blinding?"}, {"role": "assistant", "content": "It is..."}]
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: "importance of blinding in clinical trials")
    assert rag.rewrite_query("why is it important?", hist) == "importance of blinding in clinical trials"

    def boom(*a, **k): raise LLMError("429")
    monkeypatch.setattr(openrouter, "chat", boom)
    assert rag.rewrite_query("why is it important?", hist) == "why is it important?"
    assert rag.rewrite_query("first question", []) == "first question"      # no history -> no LLM call


def test_stream_then_finalize(retriever, monkeypatch):
    monkeypatch.setattr(openrouter, "stream_chat", lambda m, **k: iter(["Consent ", "is voluntary [1]."]))
    p = rag.prepare("consent?", retriever, rewrite=False)
    text = "".join(rag.stream_answer(p))
    ans = rag.finalize(p, text)
    assert ans.cited == [1] and ans.text == "Consent is voluntary. [1]"   # citation normalised

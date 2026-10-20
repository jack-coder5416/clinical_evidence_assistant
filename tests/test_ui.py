"""Drives the real Streamlit script headlessly with stubbed models + LLM."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.llm import openrouter
from src.llm.openrouter import LLMError
from src.retriever import retriever as retriever_mod
from tests.conftest import ScriptedReranker, onehot

APP = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


@pytest.fixture
def app(collection, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    Real = retriever_mod.HybridRetriever
    monkeypatch.setattr(retriever_mod, "HybridRetriever", lambda: Real(
        collection, embed_query=lambda q: onehot(1),
        reranker=ScriptedReranker({"consent": 0.93})))
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: "unused")
    at = AppTest.from_file(APP, default_timeout=30)
    return at.run()


def test_first_load_shows_examples_and_no_errors(app):
    assert not app.exception
    labels = [b.label for b in app.button]
    assert "Informed consent" in labels and len(labels) >= 4
    assert "6" in app.sidebar.caption[0].value          # 6 passages indexed


def test_question_produces_cited_answer_and_sources(app, monkeypatch):
    monkeypatch.setattr(openrouter, "stream_chat", lambda m, **k: iter(
        ["Participants may withdraw at any time [1]. ", "Consent must be free [1][2]."]))
    app.chat_input[0].set_value("Can participants withdraw?").run()
    assert not app.exception and not app.error
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    md = " ".join(m.value for m in app.markdown)
    assert ":blue-badge[1]" in md                        # citations rendered as badges
    labels = [e.label for e in app.expander]
    assert any(l.startswith("[1]") and "Consent" in l for l in labels)   # cited source card w/ section
    assert any("Retrieved but not cited" in l for l in labels)
    cap = " ".join(c.value for c in app.caption)
    assert "Hybrid + rerank" in cap and "2 cited" in cap


def test_llm_failure_keeps_chat_alive_and_shows_retrieved_passages(app, monkeypatch):
    def boom(*a, **k): raise LLMError("OpenRouter request failed after 3 attempts (429).")
    monkeypatch.setattr(openrouter, "stream_chat", boom)
    app.chat_input[0].set_value("consent?").run()
    assert not app.exception
    assert any("429" in e.value for e in app.error)
    assert any("Retrieved passages" in e.label for e in app.expander)
    # the app still works for the next question
    monkeypatch.setattr(openrouter, "stream_chat", lambda m, **k: iter(["Fine [1]."]))
    app.chat_input[0].set_value("again?").run()
    assert not app.exception and len([m for m in app.chat_message if m.name == "assistant"]) == 2


def test_sidebar_options_change_pipeline(app, monkeypatch):
    monkeypatch.setattr(openrouter, "stream_chat", lambda m, **k: iter(["Ok [1]."]))
    app.sidebar.selectbox[0].set_value("BM25 only")
    app.sidebar.toggle[0].set_value(False)               # reranking off
    app.run()
    app.chat_input[0].set_value("informed consent participants").run()
    cap = " ".join(c.value for c in app.caption)
    meta = next(c for c in cap.split("  ·  ") if "🔎" in c)
    assert "BM25" in meta and "rerank" not in meta


def test_missing_index_shows_setup_instructions(monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    def missing(): raise RuntimeError("Collection medical_guidelines does not exist")
    monkeypatch.setattr(retriever_mod, "HybridRetriever", missing)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "python -m src.ingestion.ingest" in at.error[0].value

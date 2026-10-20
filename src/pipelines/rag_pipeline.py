"""End-to-end RAG: rewrite -> retrieve -> ground -> generate -> verify citations.

Three steps are exposed separately so a UI can stream the generation:

    prepared = prepare(question, retriever, history)
    for token in stream_answer(prepared): ...
    answer = finalize(prepared, full_text)

`ask()` runs all three without streaming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from src import citations
from src.config.settings import FINAL_TOP_K, MIN_RERANK_SCORE
from src.llm import openrouter
from src.llm.openrouter import LLMError
from src.llm.prompts import INSUFFICIENT, REWRITE_PROMPT, build_messages
from src.retriever.retriever import HybridRetriever
from src.retriever.types import Hit, RetrievalResult


@dataclass
class Prepared:
    question: str
    search_query: str
    retrieval: RetrievalResult
    messages: list[dict] | None      # None -> no LLM call needed (gated)
    gated: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    search_query: str
    text: str
    sources: list[Hit]                       # numbered [1..n] in this order
    retrieval: RetrievalResult
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    insufficient: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def display_text(self) -> str:
        return citations.decorate(self.text, len(self.sources))


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Turn a follow-up ("what about in emergencies?") into a standalone query.

    Falls back to the raw question on any failure - retrieval must never break
    because the rewrite call hit a rate limit.
    """
    if not history:
        return question
    convo = "\n".join(
        f"{t['role'].capitalize()}: {t['content'][:400]}"
        for t in history[-4:]
        if t.get("content")
    )
    try:
        out = openrouter.chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest message: {question}"},
            ],
            temperature=0.0,
            max_tokens=400,
        ).strip().strip('"')
    except LLMError:
        return question
    return out if 0 < len(out) <= 300 else question


def prepare(
    question: str,
    retriever: HybridRetriever,
    history: list[dict] | None = None,
    *,
    mode: str = "hybrid",
    top_k: int = FINAL_TOP_K,
    rerank: bool = True,
    rewrite: bool = True,
) -> Prepared:
    search_query = rewrite_query(question, history) if rewrite else question
    retrieval = retriever.search(search_query, mode=mode, top_k=top_k, rerank=rerank)
    warnings = list(retrieval.warnings)

    hits = retrieval.hits
    if not hits:
        return Prepared(question, search_query, retrieval, None, gated=True, warnings=warnings)

    # Optional relevance gate: don't ask the LLM to answer from weak evidence.
    if (
        MIN_RERANK_SCORE > 0
        and retrieval.reranked
        and (hits[0].rerank_score or 0.0) < MIN_RERANK_SCORE
    ):
        return Prepared(question, search_query, retrieval, None, gated=True, warnings=warnings)

    return Prepared(
        question, search_query, retrieval, build_messages(question, hits, history), warnings=warnings
    )


def stream_answer(prepared: Prepared) -> Iterator[str]:
    if prepared.gated or prepared.messages is None:
        yield INSUFFICIENT
        return
    yield from openrouter.stream_chat(prepared.messages)


def finalize(prepared: Prepared, text: str) -> Answer:
    hits = prepared.retrieval.hits
    text = citations.tidy(text.strip())
    insufficient = prepared.gated or INSUFFICIENT.lower() in text.lower()
    report = citations.analyze(text, len(hits), ignore=(INSUFFICIENT,))

    warnings = list(prepared.warnings)
    if report.invalid:
        warnings.append(
            "The answer cites source number(s) that don't exist: "
            + ", ".join(map(str, report.invalid))
            + ". Treat those statements as unverified."
        )
    return Answer(
        question=prepared.question,
        search_query=prepared.search_query,
        text=text,
        sources=hits,
        retrieval=prepared.retrieval,
        cited=report.cited,
        invalid_citations=report.invalid,
        uncited_sentences=[] if insufficient else report.uncited_sentences,
        insufficient=insufficient,
        warnings=warnings,
    )


def ask(question: str, retriever: HybridRetriever, history: list[dict] | None = None, **kw) -> Answer:
    prepared = prepare(question, retriever, history, **kw)
    if prepared.gated or prepared.messages is None:
        return finalize(prepared, INSUFFICIENT)
    return finalize(prepared, openrouter.chat(prepared.messages))

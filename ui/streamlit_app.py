"""Clinical Evidence Assistant - Streamlit chat UI.

    streamlit run ui/streamlit_app.py
"""
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import streamlit as st

from src.config.settings import FINAL_TOP_K, MODEL_NAME, RAW_DATA_DIR
from src.llm.openrouter import LLMError
from src.pipelines.rag_pipeline import Answer, finalize, prepare, stream_answer
from src.retriever.bm25 import query_terms
from src.retriever.retriever import HybridRetriever
from src.vectordb.chroma_store import get_client, get_or_create_collection
from src.ingestion.ingest import ingest as ingest_pdfs
from src.ingestion.incremental import get_sync_status, sync_documents
from src.sources.pubmed import fetch_recent as fetch_pubmed
from src.sources.clinical_trials import fetch_recent as fetch_trials

st.set_page_config(page_title="Clinical Evidence Assistant", page_icon="🩺", layout="wide")

MODES = {"Hybrid (dense + BM25)": "hybrid", "Dense only": "dense", "BM25 only": "sparse"}
EXAMPLES = {   # button label -> question that is sent
    "Informed consent": "What are the key principles for obtaining informed consent in clinical trials?",
    "Blinding": "Why is blinding important, and what should be done when it isn't feasible?",
    "Monitoring committees": "What does the guidance say about data monitoring committees?",
    "Excluded populations": "Which populations are often excluded from trials, and why does it matter?",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 60rem;}
      mark {background: rgba(255, 200, 0, .35); color: inherit; padding: 0 2px; border-radius: 3px;}
      .src-text {white-space: pre-wrap; font-size: .92rem; line-height: 1.55;}
      div[data-testid="stChatMessage"] {padding-top: .4rem; padding-bottom: .4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Preparing evidence index and loading search models…")
def load_retriever():
    # On a fresh Streamlit Cloud container, the local Chroma directory may not
    # exist. Build the starter PDF index once, then try a bounded source sync.
    client = get_client()
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if "medical_guidelines" not in names or client.get_collection("medical_guidelines").count() == 0:
        pdfs = sorted(RAW_DATA_DIR.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError("No index or starter PDFs found. Add PDFs to data/raw or run ingestion.")
        ingest_pdfs(pdfs)

    # Keep startup ingestion deliberately small. Failures in public APIs should
    # not prevent users from querying the curated local corpus.
    try:
        from os import getenv
        if getenv("AUTO_SYNC_ON_STARTUP", "true").lower() in ("1", "true", "yes"):
            query = getenv("EVIDENCE_QUERY", "diabetes OR hypertension")
            days = int(getenv("SYNC_DAYS", "30"))
            limit = int(getenv("SYNC_LIMIT", "20"))
            docs = fetch_pubmed(query, days, limit) + fetch_trials(query, days, limit)
            sync_documents(list({d.source_id: d for d in docs}.values()))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Automatic source sync was unavailable; using indexed evidence. ({type(e).__name__})")

    retriever = HybridRetriever()
    return retriever, retriever.warmup()


try:
    retriever, load_problems = load_retriever()
except Exception as e:  # noqa: BLE001
    st.error(
        f"**The search index isn't ready** ({type(e).__name__}: {e}).\n\n"
        "Build it first, from the project root:\n\n"
        "```bash\npython -m src.ingestion.ingest\n```"
    )
    st.stop()


# -------------------------------------------------------------------- helpers
def highlight(text: str, terms: list[str]) -> str:
    safe = html.escape(text)
    if terms:
        pattern = r"\b(" + "|".join(re.escape(html.escape(t)) for t in terms) + r")\w*"
        safe = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", safe, flags=re.IGNORECASE)
    return f'<div class="src-text">{safe}</div>'


def short(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def trace_line(h) -> str:
    parts = []
    if h.dense_rank:
        parts.append(f"Dense #{h.dense_rank} (sim {h.dense_score:.2f})")
    if h.sparse_rank:
        parts.append(f"BM25 #{h.sparse_rank} ({h.sparse_score:.1f})")
    if h.fused_score is not None:
        parts.append(f"RRF {h.fused_score:.4f}")
    return " · ".join(parts)


def source_body(h, terms):
    source_type = h.metadata.get("source_type", "pdf")
    if source_type in ("research_abstract", "clinical_trial"):
        url = h.metadata.get("source_url", "")
        date = h.metadata.get("published_at", "unknown")
        updated = h.metadata.get("updated_at", "unknown")
        st.caption(f"{source_type.replace('_', ' ').title()} · Published/posted: {date} · Source update: {updated}")
        if url:
            st.markdown(f"[Open original source]({url})")
    else:
        st.caption(f"📄 {h.title}  ·  printed page {h.page_label}  ·  PDF page {h.page}")
    st.markdown(highlight(h.text, terms), unsafe_allow_html=True)
    if h.rerank_score is not None:
        st.progress(min(max(h.rerank_score, 0.0), 1.0), text=f"Reranker relevance {h.rerank_score:.2f}")
    st.caption(trace_line(h))


def render_sources(ans: Answer):
    if not ans.sources:
        return
    terms = query_terms(ans.search_query)
    numbered = list(enumerate(ans.sources, start=1))
    cited = [(n, h) for n, h in numbered if n in ans.cited]
    rest = [(n, h) for n, h in numbered if n not in ans.cited]

    if cited:
        st.markdown("**Sources cited**")
        for n, h in cited:
            with st.expander(f"[{n}]  {short(h.section)}  ·  p. {h.page_label}"):
                source_body(h, terms)
    if rest:
        label = "Retrieved but not cited" if cited else "Retrieved passages"
        with st.expander(f"{label} ({len(rest)})", expanded=not cited):
            for n, h in rest:
                with st.container(border=True):
                    st.markdown(f"**[{n}]  {short(h.section)}**  ·  p. {h.page_label}")
                    source_body(h, terms)


def render_meta(ans: Answer):
    r = ans.retrieval
    mode = {"hybrid": "Hybrid", "dense": "Dense", "sparse": "BM25"}[r.mode]
    bits = [f"🔎 {mode}" + (" + rerank" if r.reranked else ""),
            f"{len(ans.sources)} passages", f"{len(ans.cited)} cited",
            f"{fmt_ms(sum(r.timings_ms.values()))} retrieval"]
    st.caption("  ·  ".join(bits))
    if ans.search_query.strip() != ans.question.strip():
        st.caption(f"↪ Searched for: *{ans.search_query}*")
    for w in ans.warnings:
        st.warning(w, icon="⚠️")
    if ans.uncited_sentences:
        n = len(ans.uncited_sentences)
        with st.expander(f"⚠️ {n} statement{'s' if n > 1 else ''} without a citation — verify before relying on it"):
            for s in ans.uncited_sentences:
                st.markdown(f"- {s}")


def render_assistant(msg: dict):
    ans: Answer | None = msg.get("answer")
    if msg.get("error"):
        st.error(msg["error"])
    if ans:
        st.markdown(ans.display_text)
        render_meta(ans)
        render_sources(ans)


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🩺 Evidence Assistant")
    st.caption(f"Index: **{retriever.size}** passages from **{len(retriever.sources())}** document(s)")
    sync_status = get_sync_status()
    st.caption(f"External records indexed: **{sync_status['documents']}**")
    st.caption(f"Last external index update: **{sync_status['last_indexed_at'] or 'Not recorded'}**")
    sync_query = st.text_input("Sync topic query", value="diabetes OR hypertension", key="sync_query")
    if st.button("↻ Sync recent evidence now", use_container_width=True,
                 help="Fetch recent PubMed and ClinicalTrials.gov records, then update the index."):
        with st.spinner("Fetching recent records and updating the index…"):
            try:
                docs = fetch_pubmed(sync_query, 30, 25) + fetch_trials(sync_query, 30, 25)
                report = sync_documents(list({d.source_id: d for d in docs}.values()))
                st.success(f"Sync finished: {report['added']} new, {report['updated']} updated, {report['unchanged']} unchanged.")
                load_retriever.clear()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Sync failed: {type(e).__name__}: {e}")
    for src in retriever.sources()[:20]:
        st.caption(f"• {src}")

    st.subheader("Retrieval")
    mode_label = st.selectbox("Method", list(MODES), help="Hybrid fuses semantic (dense) and keyword (BM25) search.")
    do_rerank = st.toggle("Cross-encoder reranking", value=True,
                          help="Re-scores the top candidates by reading question and passage together.")
    top_k = st.slider("Passages sent to the model", 3, 10, FINAL_TOP_K)
    do_rewrite = st.toggle("Resolve follow-up questions", value=True,
                           help="Rewrites 'what about children?' into a standalone search query (one extra model call).")

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("♻️ Reload index", use_container_width=True, help="Use after re-running ingestion."):
        load_retriever.clear()
        st.rerun()
    st.caption(f"LLM: `{MODEL_NAME}`")
    for p in load_problems:
        st.warning(p, icon="⚠️")
    st.divider()
    st.caption("For research and reference only. Not medical advice and not a substitute for "
               "clinical judgement. Verify against the cited source.")

# ----------------------------------------------------------------------- main
st.title("Clinical Evidence Assistant")
st.caption("Ask questions about the indexed guidance documents. Every answer is grounded in "
           "retrieved passages and cited so you can check the source.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Resolve the prompt first so the example buttons vanish as soon as one is used.
prompt = st.chat_input("Ask a clinical evidence question…") or st.session_state.pop("pending", None)

if not st.session_state.messages and not prompt:
    st.markdown("##### Try asking")
    cols = st.columns(len(EXAMPLES))
    for col, (label, question) in zip(cols, EXAMPLES.items()):
        if col.button(label, help=question, use_container_width=True):
            st.session_state.pending = question
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🩺"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

if prompt:
    # LLM-facing history = plain text of past turns (errors excluded).
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages if m.get("content") and not m.get("error")]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        record = {"role": "assistant", "content": "", "answer": None, "error": None}
        prepared, waiting = None, None
        try:
            with st.status("Searching the evidence…", expanded=False) as status:
                prepared = prepare(prompt, retriever, history, mode=MODES[mode_label],
                                   top_k=top_k, rerank=do_rerank, rewrite=do_rewrite)
                r = prepared.retrieval
                status.update(
                    label=f"Found {len(r.hits)} passages" + (" · reranked" if r.reranked else ""),
                    state="complete",
                )

            slot = st.empty()
            waiting = st.caption("⏳ Writing answer…")

            def tokens():
                first = True
                for t in stream_answer(prepared):
                    if first:
                        waiting.empty()
                        first = False
                    yield t

            with slot.container():
                text = st.write_stream(tokens())
            answer = finalize(prepared, text)
            slot.markdown(answer.display_text)        # swap raw [n] for citation badges
            record.update(content=answer.text, answer=answer)
            render_meta(answer)
            render_sources(answer)
        except Exception as e:  # noqa: BLE001 - keep the chat alive, show a readable message
            if waiting is not None:
                waiting.empty()
            record["error"] = str(e) if isinstance(e, LLMError) else f"Something went wrong: {type(e).__name__}: {e}"
            if prepared is not None:   # the LLM failed, but still show what was retrieved
                record["answer"] = finalize(prepared, "")
            render_assistant(record)
        st.session_state.messages.append(record)

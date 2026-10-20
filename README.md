# Clinical Evidence Assistant

Citation-grounded RAG chatbot for guideline documents: **dense + BM25 hybrid retrieval → RRF fusion →
cross-encoder rerank → numbered-source prompt → validated `[n]` citations**, in a Streamlit chat UI.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env                      # add OPENROUTER_API_KEY (and ideally pin MODEL_NAME)
python scripts/fetch_documents.py         # optional: more PDFs (see data/raw/SOURCES.md)
python -m src.ingestion.ingest            # (re)builds the index from data/raw/*.pdf
streamlit run ui/streamlit_app.py
python -m pytest tests -q                 # no models or network needed
```
First run downloads `BAAI/bge-small-en-v1.5` (~130 MB) and `BAAI/bge-reranker-base` (~1.1 GB).
On a slow CPU set `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` in `.env`.

## Layout
`src/ingestion` parse (page/section aware) → chunk → ingest · `src/retriever` bm25, fusion, reranker, orchestrator ·
`src/llm` OpenRouter client + prompts · `src/citations.py` parse/validate citations · `src/pipelines` end-to-end ·
`ui/streamlit_app.py` chat UI. All tunables are in `src/config/settings.py`.

## Notes
- Heading detection in `parser.py` uses font size/weight, tuned on WHO PDFs. A new document style may need
  the thresholds adjusted; check the section labels shown on source cards.
- Chunks are per page, so a paragraph spanning a page break is split in two.
- `MIN_RERANK_SCORE` (off by default) skips the LLM on weak evidence. Calibrate it on your own questions.
- Not medical advice. Answers must be checked against the cited source.

## Freshness and incremental ingestion

The app can synchronize recent source records from PubMed and ClinicalTrials.gov.
It stores normalized source IDs, content hashes, source URLs, publication/update dates,
and ingestion timestamps in Chroma metadata. Repeated syncs skip unchanged records and
replace chunks when a source record changes.

```bash
# Build the curated PDF index (first run only)
python -m src.ingestion.ingest

# Fetch records added/updated recently for the selected topic and incrementally index them
python -m src.ingestion.sync --query "diabetes OR hypertension" --days 30 --limit 50

# Narrow to one source
python -m src.ingestion.sync --sources pubmed --query "cardiovascular disease" --days 14 --limit 50
```

Environment settings: `EVIDENCE_QUERY`, `SYNC_DAYS`, `SYNC_LIMIT`, `AUTO_SYNC_ON_STARTUP`.
The Streamlit UI also exposes a manual sync action. Automatic startup sync is bounded to a
small batch so public API failure does not prevent the curated index from being queried.
PubMed connector indexes abstracts, not necessarily full text. ClinicalTrials.gov records are
registry data, not proof that a treatment is effective. Always show the original source link
and distinguish publication date, source update date, and ingestion timestamp.

## Deploy to Streamlit Community Cloud

1. Rotate any API key that was ever included in a shared `.env` file. Do not commit `.env`.
2. Push this project to a GitHub repository. Keep `chroma_db/` and `data/sync_state.sqlite3`
   out of Git; the app builds a starter index from `data/raw/*.pdf` on first startup.
3. On Streamlit Community Cloud, create an app for the repository and choose
   `ui/streamlit_app.py` as the entrypoint.
4. In app Advanced settings, choose Python 3.11 or 3.12 and add secrets:

   ```toml
   OPENROUTER_API_KEY = "your-rotated-key"
   MODEL_NAME = "openrouter/free"
   AUTO_SYNC_ON_STARTUP = "true"
   EVIDENCE_QUERY = "diabetes OR hypertension"
   SYNC_DAYS = "30"
   SYNC_LIMIT = "20"
   RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
   ```

5. Deploy and check app logs. The first start downloads embedding/reranker models and builds
   the local PDF index, so it can be slower than later requests.
6. Test a question and verify that source cards link to PubMed or ClinicalTrials.gov. Use
   the sidebar sync button to test incremental ingestion.

### Important hosting limitation

Streamlit Community Cloud is suitable for a small portfolio demo, but its local filesystem is
not a durable database contract across restarts/redeploys. This app rebuilds its starter index
and attempts a bounded sync when a fresh container starts. For durable cross-restart state and
reliable scheduled updates while nobody is using the app, move Chroma plus sync state to a
persistent volume/managed vector database and run `python -m src.ingestion.sync` from an
external scheduler (for example GitHub Actions or a small worker). Do not claim guaranteed
24/7 scheduled refreshes from Streamlit Community Cloud alone.

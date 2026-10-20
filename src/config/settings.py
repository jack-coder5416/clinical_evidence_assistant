"""Central configuration. Everything tunable lives here (or in .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Streamlit Community Cloud stores values in st.secrets, not automatically in
# os.environ. Bridge known string secrets so the existing config/API clients
# work locally and in deployment without duplicating secret-handling logic.
try:
    import streamlit as _st
    for _key in ("OPENROUTER_API_KEY", "MODEL_NAME", "NCBI_API_KEY", "NCBI_EMAIL",
                 "AUTO_SYNC_ON_STARTUP", "EVIDENCE_QUERY", "SYNC_DAYS", "SYNC_LIMIT",
                 "RERANKER_MODEL", "EMBEDDING_MODEL"):
        if _key not in os.environ and _key in _st.secrets:
            os.environ[_key] = str(_st.secrets[_key])
except Exception:
    # No secrets file is expected during ordinary local CLI/test execution.
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- LLM (OpenRouter) -------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# "openrouter/free" routes to whichever free model is available, so answer
# quality and citation-following vary run to run. Pin a specific model in .env
# (MODEL_NAME=...) for consistent behaviour.
MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/free")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)   # low = more faithful to sources
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 2000)      # reasoning models need headroom
LLM_TIMEOUT = _int("LLM_TIMEOUT", 90)

# --- Paths ------------------------------------------------------------------
VECTOR_DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "medical_guidelines"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# --- Models -----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 was trained with this prefix on *queries only* (not passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Cross-encoder reranker. Lighter/faster alternative on CPU:
#   cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 120)

# --- Retrieval --------------------------------------------------------------
CANDIDATES_PER_RETRIEVER = _int("CANDIDATES_PER_RETRIEVER", 20)  # dense & BM25 each
RRF_K = _int("RRF_K", 60)                 # standard RRF constant
DENSE_WEIGHT = _float("DENSE_WEIGHT", 1.0)
SPARSE_WEIGHT = _float("SPARSE_WEIGHT", 1.0)
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 20)   # how many fused hits get reranked
FINAL_TOP_K = _int("FINAL_TOP_K", 5)                # chunks sent to the LLM
# If the best reranker score (0-1) is below this, skip the LLM and answer
# "insufficient evidence". 0 disables the gate. Calibrate on your own questions.
MIN_RERANK_SCORE = _float("MIN_RERANK_SCORE", 0.0)

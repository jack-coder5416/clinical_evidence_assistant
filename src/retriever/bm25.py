"""Sparse (keyword) retrieval with BM25.

Dense embeddings are good at meaning but weak on exact tokens - trial
acronyms, drug names, "CTIMP", "ICH E6", section numbers. BM25 is the
opposite, which is why the two are combined.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its may might of on or should so such than that the their
    them then there these they this those to was were what when where which who whom
    why will with would you your about between per via""".split()
)

try:  # nltk's Porter stemmer is pure-algorithmic; no corpus download needed.
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    @lru_cache(maxsize=50_000)
    def _stem(word: str) -> str:
        return _stemmer.stem(word)

except Exception:  # pragma: no cover - nltk is optional
    def _stem(word: str) -> str:
        return word


_OUR = re.compile(r"(?<=\w{3})our(s|ed|ing)?$")


def _normalize_spelling(word: str) -> str:
    """Map common British spellings onto American ones (applied to queries and
    documents alike, so it only needs to be *consistent*, not perfect):
    oedema/edema, paediatric/pediatric, haemoglobin/hemoglobin, behaviour/behavior."""
    word = word.replace("ae", "e").replace("oe", "e")
    if len(word) >= 6:
        word = _OUR.sub(r"or\1", word)
    return word


def _stem_word(word: str) -> str:
    stem = _stem(_normalize_spelling(word))
    # Porter turns randomized -> "random" but randomised -> "randomis"; strip that
    # residual "-is" so both spellings share one stem.
    if len(stem) >= 6 and stem.endswith("is"):
        stem = stem[:-2]
    return stem


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, normalise, stem.

    'randomised' / 'randomized' / 'randomization' all map to one stem, as do
    'oedema'/'edema' and 'behaviour'/'behavior', so spelling variants and
    plurals don't hurt recall.
    """
    words = _TOKEN.findall(text.lower())
    return [_stem_word(w) for w in words if w not in _STOPWORDS]


def query_terms(query: str) -> list[str]:
    """Unstemmed content words of the query - used for highlighting in the UI."""
    return [w for w in _TOKEN.findall(query.lower()) if w not in _STOPWORDS and len(w) > 2]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score), best first. Zero-score chunks are omitted."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

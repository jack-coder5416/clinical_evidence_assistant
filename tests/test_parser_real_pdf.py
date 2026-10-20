import re
from pathlib import Path

import pytest

PDF = Path(__file__).resolve().parents[1] / "data" / "raw" / "clinical_trials.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="sample PDF not present")


@pytest.fixture(scope="module")
def chunks():
    from src.ingestion.chunker import chunk_segments
    from src.ingestion.parser import extract_segments
    return chunk_segments(extract_segments(PDF))


def test_every_chunk_is_citable(chunks):
    for c in chunks:
        m = c.metadata
        assert m["source"] and m["page"] >= 1 and m["page_label"] and m["section"]
    assert len({c.id for c in chunks}) == len(chunks)


def test_no_toc_dot_leaders_or_bare_page_numbers(chunks):
    assert not any(re.search(r"(?:\s*\.){4,}", c.text) for c in chunks)
    assert not any(re.fullmatch(r"[ivx\d]{1,4}", c.text.strip()) for c in chunks)


def test_printed_page_label_differs_from_pdf_index(chunks):
    consent = [c for c in chunks if c.metadata["section"].startswith("2.2.3")]
    assert consent and consent[0].metadata["page"] - int(consent[0].metadata["page_label"]) == 14


def test_section_hierarchy_recovered(chunks):
    tops = {c.metadata["section_path"].split(" › ")[0] for c in chunks}
    assert "1. Introduction" in tops and any(t.startswith("2. Key scientific") for t in tops)
    blind = [c for c in chunks if "Blinding" in c.metadata["section"]]
    assert blind and blind[0].metadata["page_label"] in {"12", "13"}


def test_chunk_size_bounds(chunks):
    assert max(len(c.text) for c in chunks) <= 1000 and min(len(c.text) for c in chunks) >= 60


def test_no_chunk_starts_with_stray_punctuation(chunks):
    assert not [c.text[:20] for c in chunks if c.text[0] in ".,;:)"]

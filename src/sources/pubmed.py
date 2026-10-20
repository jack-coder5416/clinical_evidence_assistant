"""Fetch recent PubMed records via NCBI E-utilities (abstracts, not full text)."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import requests

from src.sources.models import SourceDocument

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT = 30


def fetch_recent(query: str, days: int = 30, limit: int = 100) -> list[SourceDocument]:
    """Fetch records newly added to PubMed within the last `days` days.

    `edat` targets Entrez entry date; this catches records newly added/indexed,
    which is more useful for synchronization than publication date alone.
    """
    common = {"tool": "clinical_evidence_assistant", "email": os.getenv("NCBI_EMAIL", "")}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        common["api_key"] = api_key
    response = requests.get(
        f"{BASE}/esearch.fcgi",
        params={**common, "db": "pubmed", "term": query, "datetype": "edat",
                "reldate": max(1, int(days)), "retmax": min(max(1, int(limit)), 200),
                "sort": "pub_date", "retmode": "json"}, timeout=TIMEOUT)
    response.raise_for_status()
    ids = response.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    response = requests.get(
        f"{BASE}/efetch.fcgi", params={**common, "db": "pubmed", "id": ",".join(ids),
                                       "retmode": "xml"}, timeout=TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    results: list[SourceDocument] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="").strip()
        title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else "Untitled PubMed record"
        abstract_parts = [" ".join("".join(node.itertext()).split()) for node in article.findall(".//Abstract/AbstractText")]
        abstract = "\n\n".join(p for p in abstract_parts if p)
        journal = article.findtext(".//Journal/Title", default="")
        pubdate = article.find(".//JournalIssue/PubDate")
        published = ""
        if pubdate is not None:
            published = "-".join(
                part
                for part in (
                    pubdate.findtext("Year", default=""),
                    pubdate.findtext("Month", default=""),
                    pubdate.findtext("Day", default=""),
                )
                if part
            )
            continue
        text = f"Title: {title}\nJournal: {journal}\nPublication date: {published}\n\nAbstract:\n{abstract}"
        results.append(SourceDocument(
            source_id=f"pubmed:{pmid}", title=title, text=text,
            source_type="research_abstract", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            published_at=published, updated_at="", license="metadata/abstract; full-text reuse varies"))
    return results

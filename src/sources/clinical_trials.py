"""Fetch recently updated ClinicalTrials.gov study records via API v2."""
from __future__ import annotations

from datetime import date, timedelta
import requests

from src.sources.models import SourceDocument

API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_recent(query: str, days: int = 30, limit: int = 100) -> list[SourceDocument]:
    since = (date.today() - timedelta(days=max(1, int(days)))).isoformat()
    # LastUpdatePostDate is the source's posted-update date, not the original
    # study start date. The API returns structured records; no PDF is needed.
    params = {
        "format": "json", "pageSize": min(max(1, int(limit)), 100),
        "query.cond": query,
        "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{since},MAX]",
        "sort": "LastUpdatePostDate:desc",
    }
    response = requests.get(API, params=params, timeout=30)
    response.raise_for_status()
    studies = response.json().get("studies", [])
    output: list[SourceDocument] = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        design = protocol.get("designModule", {})
        desc = protocol.get("descriptionModule", {})
        status = protocol.get("statusModule", {})
        arms = protocol.get("armsInterventionsModule", {})
        outcomes = protocol.get("outcomesModule", {})
        nct = ident.get("nctId", "")
        title = ident.get("briefTitle") or ident.get("officialTitle") or nct
        if not nct:
            continue
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        interventions = [x.get("name", "") for x in arms.get("interventions", []) if x.get("name")]
        primary = [x.get("measure", "") for x in outcomes.get("primaryOutcomes", []) if x.get("measure")]
        summary = desc.get("briefSummary", "")
        text = "\n".join([
            f"Title: {title}", f"Study ID: {nct}",
            f"Overall status: {status.get('overallStatus', '')}",
            f"Study type: {design.get('studyType', '')}",
            f"Conditions: {', '.join(conditions)}",
            f"Interventions: {', '.join(interventions)}",
            f"Primary outcomes: {'; '.join(primary)}",
            f"Brief summary: {summary}",
        ])
        output.append(SourceDocument(
            source_id=f"clinicaltrials:{nct}", title=title, text=text,
            source_type="clinical_trial", url=f"https://clinicaltrials.gov/study/{nct}",
            published_at=status.get("studyFirstPostDateStruct", {}).get("date", ""),
            updated_at=status.get("studyLastUpdatePostDateStruct", {}).get("date", ""),
            license="ClinicalTrials.gov public API data"))
    return output

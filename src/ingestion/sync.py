"""Fetch recent evidence and incrementally sync it into ChromaDB.

Run: python -m src.ingestion.sync --query "diabetes OR hypertension" --days 30 --limit 100
"""
import argparse
import os

from src.config.settings import PROJECT_ROOT
from src.ingestion.incremental import sync_documents
from src.sources.pubmed import fetch_recent as fetch_pubmed
from src.sources.clinical_trials import fetch_recent as fetch_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=os.getenv("EVIDENCE_QUERY", "diabetes OR hypertension"))
    parser.add_argument("--days", type=int, default=int(os.getenv("SYNC_DAYS", "30")))
    parser.add_argument("--limit", type=int, default=int(os.getenv("SYNC_LIMIT", "100")))
    parser.add_argument("--sources", choices=["all", "pubmed", "trials"], default="all")
    args = parser.parse_args()
    docs = []
    if args.sources in ("all", "pubmed"):
        docs.extend(fetch_pubmed(args.query, args.days, args.limit))
    if args.sources in ("all", "trials"):
        docs.extend(fetch_trials(args.query, args.days, args.limit))
    # Stable source IDs let the indexer deduplicate records from repeated syncs.
    unique = {d.source_id: d for d in docs}
    result = sync_documents(list(unique.values()))
    print(f"Sync complete: {result}")


if __name__ == "__main__":
    main()

"""Download open-access guideline PDFs into data/raw/.

    python scripts/fetch_documents.py                  # the verified list below
    python scripts/fetch_documents.py --url URL [--name file.pdf]   # any other PDF

Only direct-PDF links I could confirm are listed. For everything else see
data/raw/SOURCES.md and download manually. Each download is checked to be a
real PDF (not an HTML landing/redirect page) before it is saved.
"""
import argparse
import sys
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

VERIFIED = {
    "who_hypertension_2021.pdf":
        "https://apps.who.int/iris/bitstream/handle/10665/344424/9789240033986-eng.pdf",
}


def download(url: str, name: str, force: bool = False) -> bool:
    dest = RAW / name
    if dest.exists() and not force:
        print(f"= {name}: already present")
        return True
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 clinical-evidence-tracker"})
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"! {name}: download failed ({e}). Download it manually from {url}")
        return False
    if not r.content.startswith(b"%PDF"):
        print(f"! {name}: {url} did not return a PDF (landing page or block). Download manually.")
        return False
    RAW.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"+ {name}: {len(r.content) / 1e6:.1f} MB")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url"); ap.add_argument("--name"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.url:
        ok = download(a.url, a.name or a.url.rstrip("/").split("/")[-1], a.force)
    else:
        ok = all([download(u, n, a.force) for n, u in VERIFIED.items()])
    print("\nNext: python -m src.ingestion.ingest")
    sys.exit(0 if ok else 1)

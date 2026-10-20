# Documents to ingest

Put PDFs in this folder, then run `python -m src.ingestion.ingest`. Every PDF here is indexed.

## Already included
- **clinical_trials.pdf**: WHO, *Guidance for best practices for clinical trials* (2024).

## Auto-download (`python scripts/fetch_documents.py`)
- **WHO, Guideline for the pharmacological treatment of hypertension in adults (2021)**
  Direct PDF verified. Landing page: https://www.who.int/publications/i/item/9789240033986

## Download manually (landing pages confirmed; I did not confirm direct PDF links)
- **CDC Clinical Practice Guideline for Prescribing Opioids for Pain, 2022** (MMWR Recomm Rep 71(3), ~100 pp).
  US government work, listed as public domain on CDC Stacks.
  https://stacks.cdc.gov/view/cdc/122248  ·  full text: https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm

## Good next candidates (search by title; check each licence before redistributing)
Open-access guidelines work best for this kind of index:
- WHO guidelines on iris.who.int (malaria, TB, HIV, diabetes, antenatal/postnatal care, antibiotic use / AWaRe).
  Most are CC BY-NC-SA 3.0 IGO.
- CDC/MMWR Recommendations and Reports (STI treatment guidelines, immunization recommendations). US public domain.
- ICH E6(R3) Good Clinical Practice and ICH E9(R1) estimands (ich.org), relevant to the researcher use case.
- Cochrane Handbook for Systematic Reviews of Interventions (free online chapters).
- CONSORT 2010 / PRISMA 2020 reporting statements (open-access articles).

Note: NICE, ESC, AHA/ACC and similar guidelines have their own reuse terms. Fine for private use,
but check before sharing an index built from them.

"""PDF -> page/section-aware text segments.

Why this replaces the old `extract_text`:
  * The old version concatenated the whole PDF into one string, so every chunk
    lost its page number and section -> impossible to cite properly.
  * It also embedded running headers, page numbers and table-of-contents dot
    leaders as if they were content.

Each `Segment` is a run of body text on ONE page under ONE section heading, so
any chunk cut from it can be cited as "Section X, p. N".
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Dot leaders as used in tables of contents: "Foreword . . . . . . iv"
_LEADER = re.compile(r"(?:\s*\.){4,}")
_WS = re.compile(r"\s+")
_PAGE_NUM = re.compile(r"^(?:[ivxlcdm]{1,6}|\d{1,3})$", re.IGNORECASE)

# Header/footer margins, as a fraction of page height.
_TOP_MARGIN = 0.075
_BOTTOM_MARGIN = 0.94
_HEADER_FOOTER_MAX_CHARS = 120

# Heading detection: bold text at least this many points.
_HEADING_MIN_SIZE = 12.0
_CHAPTER_MIN_SIZE = 20.0   # chapter titles are large but not always flagged bold
_HEAVY_FONT_HINTS = ("bold", "medium", "semibold", "black", "heavy")
_HEADING_MAX_CHARS = 160
_HEADING_MAX_LINES = 4


@dataclass
class Segment:
    source: str        # file name, e.g. "clinical_trials.pdf"
    title: str         # human-readable document title
    page: int          # 1-based physical PDF page
    page_label: str    # printed page label ("iv", "13"); falls back to `page`
    section: str       # leaf heading, e.g. "2.1.3 Appropriate trial population"
    section_path: str  # breadcrumb of headings
    text: str


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")  # ligatures, soft hyphens
    text = _LEADER.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _document_title(doc: pymupdf.Document, pdf_path: Path) -> str:
    meta = (doc.metadata or {}).get("title") or ""
    # "9 Sept_24049_GCTF_Guidance for best ..." -> "Guidance for best ..."
    meta = re.sub(r"^(?:[\w ]*?_)+", "", meta).strip()
    return meta or pdf_path.stem.replace("_", " ")


def _block_info(block: dict) -> tuple[str, float, bool, int]:
    """Return (text, first_span_size, first_span_bold, n_lines) for a text block."""
    lines: list[str] = []
    size, bold = 0.0, False
    for line in block["lines"]:
        s = "".join(sp["text"] for sp in line["spans"])
        if not s.strip():
            continue
        if not lines:
            first = next(sp for sp in line["spans"] if sp["text"].strip())
            size = first["size"]
            bold = bool(first["flags"] & 16) or any(h in first["font"].lower() for h in _HEAVY_FONT_HINTS)
        lines.append(s.strip())

    text = ""
    for s in lines:
        # Re-join words hyphenated across a line break ("randomi-" + "zed").
        if text.endswith("-") and s[:1].islower() and text[-2:-1].isalpha():
            text = text[:-1] + s
        else:
            text = f"{text} {s}" if text else s
    return _normalize(text), size, bold, len(lines)


def _heading_level(size: float) -> int:
    return 1 if size >= _CHAPTER_MIN_SIZE else 2 if size >= 13.5 else 3


def _repeated_running_text(doc: pymupdf.Document) -> set[str]:
    """Short lines that repeat on many pages (running headers/footers)."""
    counts: Counter[str] = Counter()
    for page in doc:
        seen = set()
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            t = re.sub(r"\d+", "#", _normalize(b[4]))
            if 0 < len(t) <= _HEADER_FOOTER_MAX_CHARS:
                seen.add(t)
        counts.update(seen)
    threshold = max(4, int(0.3 * len(doc)))
    return {t for t, n in counts.items() if n >= threshold}


def extract_segments(pdf_path: str | Path) -> list[Segment]:
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        title = _document_title(doc, pdf_path)
        running = _repeated_running_text(doc)

        stack: dict[int, str] = {}       # heading level -> heading text
        leaf = "Front matter"
        segments: list[Segment] = []

        for page in doc:
            # Table-of-contents pages are pure noise for retrieval.
            if len(_LEADER.findall(page.get_text())) >= 5:
                continue

            height = page.rect.height
            label = page.get_label() or str(page.number + 1)
            buffer: list[str] = []

            def flush() -> None:
                if buffer:
                    segments.append(
                        Segment(
                            source=pdf_path.name,
                            title=title,
                            page=page.number + 1,
                            page_label=label,
                            section=leaf,
                            section_path=" › ".join(stack[k] for k in sorted(stack)) or leaf,
                            text="\n\n".join(buffer),
                        )
                    )
                    buffer.clear()

            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue
                text, size, bold, n_lines = _block_info(block)
                if not text:
                    continue

                y0, y1 = block["bbox"][1], block["bbox"][3]
                in_margin = y1 < _TOP_MARGIN * height or y0 > _BOTTOM_MARGIN * height
                if in_margin and len(text) <= _HEADER_FOOTER_MAX_CHARS:
                    continue
                if _PAGE_NUM.match(text):
                    continue
                if re.sub(r"\d+", "#", text) in running:
                    continue

                is_heading = (
                    (bold and size >= _HEADING_MIN_SIZE) or size >= _CHAPTER_MIN_SIZE
                ) and len(text) <= _HEADING_MAX_CHARS and n_lines <= _HEADING_MAX_LINES
                if is_heading:
                    flush()  # text so far belongs to the previous heading
                    level = _heading_level(size)
                    stack[level] = text
                    for deeper in [k for k in stack if k > level]:
                        del stack[deeper]
                    leaf = text
                else:
                    buffer.append(text)

            flush()

        return segments
    finally:
        doc.close()


# Backwards-compatible helper (the old API returned one big string).
def extract_text(pdf_path: str | Path) -> str:
    return "\n\n".join(s.text for s in extract_segments(pdf_path))

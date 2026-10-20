"""Parse and validate [n] citations in a model answer.

LLMs cite imperfectly: they may reference a number that doesn't exist, cite
in odd formats ([Source 2], [1-3]), or state facts with no citation at all.
This module turns the raw answer into something the UI can trust and display,
and flags what it can't verify.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# [1]  [1,2]  [1, 2]  [1-3]  [Source 2]  [Sources 1 and 3]
_GROUP = re.compile(
    r"\[\s*(?:sources?\s*)?(\d+(?:\s*(?:,|and|&|–|-)\s*\d+)*)\s*\]", re.IGNORECASE
)
_INNER = r"\[\s*(?:sources?\s*)?\d+(?:\s*(?:,|and|&|–|-)\s*\d+)*\s*\]"
_RUN = rf"{_INNER}(?:[ \t]*{_INNER})*"          # one or more citation groups: [1][3]
_DOUBLE_PUNCT = re.compile(rf"([.!?])[ \t]*({_RUN})[ \t]*[.!?]")   # "text. [1]."
_BEFORE_PUNCT = re.compile(rf"[ \t]*({_RUN})[ \t]*([.!?])")        # "text [1]."
# Split into sentences, but never separate a sentence from the citation that follows it.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?!\[)|(?<=\])\s+(?=[A-Z])|\n+")


def _expand(group: str) -> list[int]:
    nums: list[int] = []
    for part in re.split(r"\s*(?:,|and|&)\s*", group.strip(), flags=re.IGNORECASE):
        if re.search(r"[–-]", part):
            a, b = re.split(r"\s*[–-]\s*", part, maxsplit=1)
            lo, hi = int(a), int(b)
            if 0 < lo <= hi and hi - lo < 20:
                nums.extend(range(lo, hi + 1))
        elif part.strip().isdigit():
            nums.append(int(part))
    return nums


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)      # valid, first-appearance order
    invalid: list[int] = field(default_factory=list)    # numbers with no matching source
    uncited_sentences: list[str] = field(default_factory=list)


def tidy(answer: str) -> str:
    """Canonical citation placement: AFTER the sentence's punctuation.

    "...at any time [1]."  and  "...at any time. [1]."  both become
    "...at any time. [1]", so a citation is never orphaned from its sentence
    and the UI badge doesn't leave a stray period floating after it.
    """
    answer = _DOUBLE_PUNCT.sub(r"\1 \2", answer)
    return _BEFORE_PUNCT.sub(r"\2 \1", answer)


def analyze(answer: str, n_sources: int, ignore: tuple[str, ...] = ()) -> CitationReport:
    answer = tidy(answer)
    report = CitationReport()
    for m in _GROUP.finditer(answer):
        for n in _expand(m.group(1)):
            if 1 <= n <= n_sources:
                if n not in report.cited:
                    report.cited.append(n)
            elif n not in report.invalid:
                report.invalid.append(n)

    # Heuristic: substantive sentences that carry no citation at all.
    for raw in _SENTENCE_SPLIT.split(answer):
        s = raw.strip().lstrip("-•*0123456789. ").strip()
        if len(s) < 50 or s.endswith(":") or _GROUP.search(s):
            continue
        if any(marker and marker in s for marker in ignore):
            continue
        report.uncited_sentences.append(s)
    return report


def decorate(answer: str, n_sources: int) -> str:
    """Markdown for display: valid citations become blue badges (1), unknown ones
    a red badge (?9) so they can't be mistaken for real, verifiable sources."""

    def repl(m: re.Match) -> str:
        out = []
        for n in _expand(m.group(1)):
            out.append(f":blue-badge[{n}]" if 1 <= n <= n_sources else f":red-badge[?{n}]")
        return "".join(out)

    return _GROUP.sub(repl, tidy(answer))

"""Pathology 3 — selective deafness (see README.md "Communication Pathologies").

Deterministic, non-LLM sentence-boundary truncation: an agent that only
"hears" the last sentence of whatever it's told, dropping every earlier
sentence — including any caveats, hedges, or corrections they contained.
Pure function, safe to unit-test with zero live LLM calls, and idempotent:
`deafen(deafen(x).last_sentence).last_sentence == deafen(x).last_sentence`,
since the last sentence of a string contains no further sentence boundary
of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common abbreviations whose trailing "." is not a real sentence boundary
# even though it's followed by whitespace and an uppercase word (e.g.
# "Dr. Smith"). Decimals (e.g. "3.14") never need special-casing here: the
# split regex below only fires on a period followed by whitespace, and a
# decimal point is never followed by whitespace.
_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "e.g.", "i.e.", "no.", "fig.", "eq.", "approx.",
}

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DeafenResult:
    original: str
    last_sentence: str
    n_dropped_sentences: int
    n_dropped_chars: int


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SPLIT_RE.split(text.strip())
    sentences: list[str] = []
    for part in parts:
        if sentences:
            last_word = sentences[-1].rsplit(maxsplit=1)[-1].lower() if sentences[-1] else ""
            if last_word in _ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return [s for s in sentences if s]


def deafen(context: str) -> DeafenResult:
    """Truncate `context` down to only its last sentence."""
    text = context or ""
    sentences = _split_sentences(text)
    if not sentences:
        return DeafenResult(original=text, last_sentence="", n_dropped_sentences=0, n_dropped_chars=0)

    last_sentence = sentences[-1]
    dropped_prefix = " ".join(sentences[:-1])
    return DeafenResult(
        original=text,
        last_sentence=last_sentence,
        n_dropped_sentences=len(sentences) - 1,
        n_dropped_chars=len(dropped_prefix),
    )

"""IMMUTABLE — deterministic quote verification. NOT an LLM tool.

Run automatically by the controller after every Extractor call (not exposed
via `tools=[...]`). Exact normalized-substring match first, fuzzy fallback for
whitespace/punctuation drift. Feeds HopResult.quote_verified: a cheap, free
signal distinct from (and complementary to) the Concluder's own semantic
`llm_grounded` judgment, which is what actually drives the bounded retry.
"""

import difflib
import re

import config
from mas_state import Paragraph


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def verify_quote(quote: str, context_paragraphs: list[Paragraph]) -> dict:
    """Best-effort verification that `quote` actually appears in the question's
    context. Returns {"verified": bool, "title": str|None, "sent_id": int|None}."""
    if not quote or not quote.strip():
        return {"verified": False, "title": None, "sent_id": None}

    nq = _normalize(quote)
    best_ratio = 0.0
    best_para = None
    for p in context_paragraphs:
        nt = _normalize(p.text)
        if nq in nt:
            return {"verified": True, "title": p.title, "sent_id": p.sent_id}
        ratio = difflib.SequenceMatcher(None, nq, nt).ratio()
        if ratio > best_ratio:
            best_ratio, best_para = ratio, p

    if best_para is not None and best_ratio >= config.QUOTE_FUZZY_MATCH_THRESHOLD:
        return {"verified": True, "title": best_para.title, "sent_id": best_para.sent_id}
    return {"verified": False, "title": None, "sent_id": None}

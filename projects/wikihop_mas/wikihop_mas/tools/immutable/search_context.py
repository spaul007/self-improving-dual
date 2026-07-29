"""IMMUTABLE — closed-book BM25 search over one question's own context.

This is the Retriever's only tool. It is closed-book by design (confirmed with
the user): the corpus is exactly the ~10 context paragraphs shipped with the
current question, never an external/open-domain index. Hand-rolled pure-Python
Okapi BM25 -- the shared environment doesn't have rank_bm25/sklearn, and a
corpus of a few dozen sentences doesn't need an embeddings/index library.

Not part of the benchmark's answer contract like answer_extraction.py, but
still "do not tune" in the sense that changing the ranking algorithm changes
what the Retriever can find, not how well the agents reason -- so it lives in
tools/immutable/, matching math_mas's convention for environment-defining code.
"""

import math
import re
from dataclasses import dataclass

import config
from mas_state import Paragraph

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 over one question's sentences. One "document" = one sentence."""

    def __init__(self, paragraphs: list[Paragraph]):
        self.docs = paragraphs
        self.tokenized = [_tokenize(p.text) for p in paragraphs]
        self.doc_len = [len(t) for t in self.tokenized]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n = len(self.docs)
        self.df: dict[str, int] = {}
        for toks in self.tokenized:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int) -> list[tuple[Paragraph, float]]:
        if self.n == 0:
            return []
        q_terms = _tokenize(query)
        scores = [0.0] * self.n
        for i, toks in enumerate(self.tokenized):
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = self.doc_len[i]
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf(term)
                num = tf[term] * (config.BM25_K1 + 1)
                den = tf[term] + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * num / den
        ranked = sorted(range(self.n), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], scores[i]) for i in ranked]


def build_index(paragraphs: list[Paragraph]) -> BM25Index:
    """Build a fresh, in-memory index for one question. Never share across
    questions -- see mas_state.py's per-question isolation note."""
    return BM25Index(paragraphs)


SEARCH_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_context",
        "description": (
            "Search the ~10 context paragraphs provided with this question using BM25 "
            "lexical search. Returns the top-k most relevant sentences with their "
            "(title, sent_id). CLOSED-BOOK: no internet, no full Wikipedia access -- "
            "only the sentences shipped with this question are searchable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Entity name, relation, or question fragment."},
                "k": {
                    "type": "integer", "minimum": 1, "maximum": config.SEARCH_MAX_K,
                    "description": f"Number of top sentences to return (default {config.SEARCH_DEFAULT_K}).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def make_search_context_handler(index: BM25Index, collected: list[Paragraph]):
    """Return a closure bound to THIS question's index. `collected` accumulates
    deduped hits across tool-call rounds so the controller can read
    HopResult.retrieved_paragraphs straight off it once the tool loop ends."""
    seen: set[tuple[str, int]] = set()

    def _handler(args: dict) -> str:
        query = str(args.get("query", ""))
        try:
            k = int(args.get("k", config.SEARCH_DEFAULT_K))
        except (TypeError, ValueError):
            k = config.SEARCH_DEFAULT_K
        k = min(max(k, 1), config.SEARCH_MAX_K)

        hits = index.search(query, k)
        lines = []
        for para, score in hits:
            key = (para.title, para.sent_id)
            if key not in seen:
                seen.add(key)
                collected.append(para)
            lines.append(f'[title="{para.title}", sent_id={para.sent_id}, score={score:.2f}] "{para.text}"')
        return "\n".join(lines) if lines else "No matching sentences found."

    return _handler

"""One-phrase summaries of a node's measured outcome, shared by every reader
of the records: the belief-mode steering lines, the planning pass, the
ledger and the calibration report. Pure text — no imports, no I/O — so any
module can use it without an import cycle.

Two facts about the same node are summarised:

* the SCORE evidence — the paired Δ over shared cases when enough are
  shared, else the unpaired Δ (each side's mean over its OWN cases) with its
  standard error, else "unmeasured";
* the JUDGE verdict — the per-node analysis LLM's effect verdict with its
  evidence grade (``improved (strong)``), when the node has one.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

EFFECTS = ("improved", "no_effect", "regressed", "unclear")
EVIDENCE_LEVELS = ("strong", "moderate", "weak")
EVIDENCE_RANK = {"weak": 0, "moderate": 1, "strong": 2}


def classify_delta(delta: Optional[float], n_shared: int, threshold: float,
                   min_shared: int) -> str:
    if delta is None or n_shared == 0:
        return "unmeasured"
    if n_shared < min_shared:
        return "inconclusive"
    if delta >= threshold - 1e-9:
        return "helped"
    if delta <= -threshold + 1e-9:
        return "hurt"
    return "neutral"


def score_context(*, delta: Optional[float], n_shared: int,
                  delta_all: Optional[float] = None,
                  se_all: Optional[float] = None,
                  n_child: int = 0, n_parent: int = 0,
                  threshold: float = 0.02, min_shared: int = 8) -> str:
    """The score evidence in one phrase. Paired first (it is the cleaner
    estimate), unpaired with its SE when the shared set is too thin."""
    if delta is not None and n_shared >= min_shared:
        v = classify_delta(delta, n_shared, threshold, min_shared)
        return f"{v} Δ{delta:+.4f}/{n_shared} shared"
    if delta_all is not None:
        se = f" ± {se_all:.3f}" if se_all is not None else ""
        sizes = (f" ({n_child} vs {n_parent} own cases)"
                 if n_child and n_parent else "")
        extra = f"; only {n_shared} shared" if n_shared else ""
        return f"Δ{delta_all:+.4f}{se} unpaired{sizes}{extra}"
    if delta is not None:
        return (f"Δ{delta:+.4f}/{n_shared} shared (below the {min_shared}-shared "
                "minimum)")
    return "unmeasured"


def perf_summary(rec: Mapping[str, Any], *, threshold: float = 0.02,
                 min_shared: int = 8) -> str:
    """:func:`score_context` over a record dict from
    ``edit_memory_render._load_records``."""
    return score_context(
        delta=rec.get("delta"), n_shared=int(rec.get("n_shared") or 0),
        delta_all=rec.get("delta_all"), se_all=rec.get("se_all"),
        n_child=int(rec.get("n_abs") or 0),
        n_parent=int(rec.get("parent_n_all") or 0),
        threshold=threshold, min_shared=min_shared)


def judge_summary(rec: Mapping[str, Any]) -> str:
    """``improved (strong; targets: opening_hours)`` from the record's
    node-level effect verdict — the first judged sub-edit's — with the checks
    it targeted when the record carries them; ``""`` when the node has no
    verdict yet."""
    effect = rec.get("effect")
    if not effect:
        return ""
    inner: list[str] = []
    ev = rec.get("evidence")
    if ev:
        inner.append(str(ev))
    by_edit = rec.get("effect_by_edit") or {}
    targets: list[str] = []
    if by_edit:
        try:
            first = min(int(k) for k in by_edit)
            tb = rec.get("targets_by_edit") or {}
            targets = [str(t) for t in (tb.get(first) or tb.get(str(first)) or [])][:4]
        except (TypeError, ValueError):
            targets = []
    if targets:
        inner.append("targets: " + ", ".join(targets))
    return f"{effect} ({'; '.join(inner)})" if inner else str(effect)


def regressions_summary(rec: Mapping[str, Any]) -> str:
    """The judge's node-level regressions line, ``""`` when empty or
    "none observed"; capped so it fits a one-line summary."""
    reg = " ".join(str(rec.get("regressions") or "").split())
    if not reg or reg.lower().rstrip(".") == "none observed":
        return ""
    return reg[:120]


def score_well_measured(*, delta: Optional[float], n_shared: int,
                        delta_all: Optional[float] = None,
                        se_all: Optional[float] = None,
                        min_shared_strong: int = 16) -> Optional[str]:
    """Is the score evidence for a node good enough to be worth quoting next
    to the judge's verdict? ``"paired"`` for a Δ over ≥ ``min_shared_strong``
    shared cases, ``"unpaired"`` when the own-case Δ exceeds twice its SE,
    else ``None``. Parent-independent: a node measured against a thin overlap
    can still be well measured on its own cases."""
    if delta is not None and n_shared >= min_shared_strong:
        return "paired"
    if delta_all is not None and se_all:
        if abs(delta_all) > 2 * se_all:
            return "unpaired"
    return None

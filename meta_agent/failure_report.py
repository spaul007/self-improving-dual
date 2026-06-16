"""Generic, example-driven failure report for the meta-agent editor.

This module is **project-agnostic** on purpose: it reads only the per-case
*contract* the framework guarantees — ``score`` / ``passed`` / ``error`` plus
``details["query"]`` (task input, injected by the evaluator) and
``details["raw_result"]`` (the agent's plan, also injected by the evaluator) —
together with the *output* of a project's ``categorize_errors`` function (a
``failure_rate``-sorted list of categories, each carrying representative
``sample_id`` + ``messages``). It never parses domain-specific ``details`` keys
(``failed_checks``, ``missing_products``, …); all such parsing lives in the
project's categorizer. This keeps the optimization core generic.

It produces, for the under-performing parent node:
  * a ranked failure-category table,
  * per top category a *diverse* pair of representative cases (a high-scoring
    near-miss + a low-scoring severe failure), each shown as query + plan +
    what-failed,
  * a separate set of the hardest (lowest-scoring) cases.

A ``focus_category_id`` mode returns just one category's examples — used to
steer an error-conditioned Stage B specialist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FailureReportConfig:
    top_error_categories: int = 4
    examples_per_category: int = 2
    n_hard_cases: int = 3
    query_char_cap: int = 600
    plan_char_cap: int = 1000
    failure_char_cap: int = 500
    # A case counts as "failing" if it did not pass OR scored below this.
    pass_threshold: float = 1.0


# ---------------------------------------------------------------------- #
# Small generic helpers
# ---------------------------------------------------------------------- #

def _truncate(text: str, cap: int) -> str:
    text = (text or "").strip()
    if cap <= 0 or len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "…"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _spread_indices(m: int, n: int) -> list[int]:
    """``n`` indices spread across ``range(m)`` including both endpoints."""
    if n <= 0 or m <= 0:
        return []
    if n == 1:
        return [0]  # the lowest-scoring (severe) end
    if m <= n:
        return list(range(m))
    idxs = sorted({round(i * (m - 1) / (n - 1)) for i in range(n)})
    # Backfill if rounding collapsed duplicates so we still return ~n picks.
    i = 0
    while len(idxs) < n and i < m:
        if i not in idxs:
            idxs.append(i)
        i += 1
    return sorted(idxs)[:n]


def _is_failing(case: Any, pass_threshold: float) -> bool:
    return (not getattr(case, "passed", False)) or (
        float(getattr(case, "score", 0.0)) < pass_threshold
    )


def _details(case: Any) -> dict[str, Any]:
    d = getattr(case, "details", None)
    return d if isinstance(d, dict) else {}


# ---------------------------------------------------------------------- #
# Example assembly
# ---------------------------------------------------------------------- #

def _failed_text_for(
    case_id: str, category: Optional[dict[str, Any]], case: Any, cfg: FailureReportConfig
) -> str:
    """What-failed text for one case, drawn from the categorizer's messages
    (generic contract) and, as a fallback, the case's own error string."""
    parts: list[str] = []
    if category:
        for ex in category.get("representative_errors") or []:
            if str(ex.get("sample_id")) != str(case_id):
                continue
            checks = [str(c) for c in (ex.get("checks_failed") or [])]
            if checks:
                parts.append("failed: " + ", ".join(checks[:8]))
            for msg in (ex.get("messages") or [])[:3]:
                parts.append(str(msg))
            break
    err = getattr(case, "error", None)
    if not parts and err:
        parts.append(f"error: {err}")
    return _truncate("\n".join(parts), cfg.failure_char_cap)


def _example_record(
    case: Any, category: Optional[dict[str, Any]], cfg: FailureReportConfig
) -> dict[str, Any]:
    det = _details(case)
    return {
        "category_id": (category or {}).get("category_id"),
        "case_id": str(getattr(case, "case_id", "?")),
        "score": round(float(getattr(case, "score", 0.0)), 4),
        "query": _truncate(_as_text(det.get("query")), cfg.query_char_cap),
        "plan": _truncate(_as_text(det.get("raw_result")), cfg.plan_char_cap),
        "failed": _failed_text_for(
            str(getattr(case, "case_id", "?")), category, case, cfg
        ),
    }


def _select_for_category(
    category: dict[str, Any],
    case_by_id: dict[str, Any],
    cfg: FailureReportConfig,
    used: set[str],
) -> list[Any]:
    """Pick a diverse (near-miss + severe) set of cases for one category from
    its representative sample_ids, sorted by score; dedupe against ``used``."""
    cands = []
    seen: set[str] = set()
    for ex in category.get("representative_errors") or []:
        sid = str(ex.get("sample_id"))
        if sid in seen or sid not in case_by_id:
            continue
        seen.add(sid)
        cands.append(case_by_id[sid])
    if not cands:
        return []
    cands.sort(key=lambda c: (float(getattr(c, "score", 0.0)), str(getattr(c, "case_id", ""))))
    picks_idx = _spread_indices(len(cands), cfg.examples_per_category)
    chosen: list[Any] = []
    for idx in picks_idx:
        cand = cands[idx]
        cid = str(getattr(cand, "case_id", ""))
        if cid in used:
            # Prefer an unused alternative if one exists.
            alt = next(
                (c for c in cands if str(getattr(c, "case_id", "")) not in used
                 and c not in chosen),
                None,
            )
            if alt is not None:
                cand = alt
                cid = str(getattr(cand, "case_id", ""))
        if cand in chosen:
            continue
        chosen.append(cand)
        used.add(cid)
    return chosen


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #

def build_failure_report(
    per_case: list[Any],
    categories: list[dict[str, Any]],
    *,
    cfg: Optional[FailureReportConfig] = None,
    focus_category_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the generic failure report. ``categories`` is the output of a
    project ``categorize_errors`` (may be empty). Returns ``{}`` when there is
    nothing useful to report."""
    cfg = cfg or FailureReportConfig()
    cases = list(per_case or [])
    if not cases:
        return {}
    case_by_id = {str(getattr(c, "case_id", "")): c for c in cases}
    categories = list(categories or [])

    # ---- Focused (single-category) mode for Stage B specialists. ---- #
    if focus_category_id is not None:
        category = next(
            (c for c in categories if str(c.get("category_id")) == str(focus_category_id)),
            None,
        )
        if category is None:
            return {}
        chosen = _select_for_category(category, case_by_id, cfg, used=set())
        return {
            "focus_category": {
                "category_id": category.get("category_id"),
                "category_name": category.get("category_name"),
                "category_type": category.get("category_type"),
                "num_failing_samples": category.get("num_failing_samples"),
                "total_samples": category.get("total_samples"),
                "failure_rate": category.get("failure_rate"),
            },
            "examples": [_example_record(c, category, cfg) for c in chosen],
        }

    # ---- Full report. ---- #
    failing = [c for c in cases if _is_failing(c, cfg.pass_threshold)]
    scores = [float(getattr(c, "score", 0.0)) for c in cases]
    summary = {
        "total_cases": len(cases),
        "n_failing": len(failing),
        "n_passing": len(cases) - len(failing),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "n_error_cases": sum(1 for c in cases if getattr(c, "error", None)),
    }

    top_categories = categories[: max(0, cfg.top_error_categories)]
    cat_table = [
        {
            "category_id": c.get("category_id"),
            "category_name": c.get("category_name"),
            "category_type": c.get("category_type"),
            "num_failing_samples": c.get("num_failing_samples"),
            "total_samples": c.get("total_samples"),
            "failure_rate": c.get("failure_rate"),
        }
        for c in top_categories
    ]

    used: set[str] = set()
    examples: list[dict[str, Any]] = []
    for category in top_categories:
        for case in _select_for_category(category, case_by_id, cfg, used):
            examples.append(_example_record(case, category, cfg))

    # Hardest cases: lowest-scoring failing cases not already shown.
    hard_sorted = sorted(
        failing,
        key=lambda c: (float(getattr(c, "score", 0.0)), str(getattr(c, "case_id", ""))),
    )
    hard_cases: list[dict[str, Any]] = []
    for case in hard_sorted:
        cid = str(getattr(case, "case_id", ""))
        if cid in used:
            continue
        # Attach any category messages that mention this case (generic).
        cat = next(
            (
                c for c in categories
                if any(str(e.get("sample_id")) == cid
                       for e in (c.get("representative_errors") or []))
            ),
            None,
        )
        hard_cases.append(_example_record(case, cat, cfg))
        used.add(cid)
        if len(hard_cases) >= cfg.n_hard_cases:
            break

    if not (cat_table or examples or hard_cases):
        # No categorizer and no failures worth surfacing.
        if summary["n_failing"] == 0:
            return {}

    return {
        "summary": summary,
        "categories": cat_table,
        "examples": examples,
        "hard_cases": hard_cases,
    }


def render_failure_report(report: dict[str, Any]) -> str:
    """Render a report (full or focused) into a compact prompt section."""
    if not report:
        return ""
    lines: list[str] = []

    def _render_example(ex: dict[str, Any], header: str) -> None:
        lines.append(header)
        if ex.get("query"):
            lines.append(f"  query: {ex['query']}")
        if ex.get("plan"):
            lines.append(f"  agent plan: {ex['plan']}")
        if ex.get("failed"):
            failed = ex["failed"].replace("\n", "\n    ")
            lines.append(f"  what failed: {failed}")

    # Focused (Stage B) shape.
    if "focus_category" in report:
        fc = report["focus_category"]
        rate = fc.get("failure_rate")
        rate_s = f"{rate:.2f}" if isinstance(rate, (int, float)) else "?"
        lines.append(
            f"## Assigned failure category: {fc.get('category_name')} "
            f"({fc.get('category_type')}) — failing "
            f"{fc.get('num_failing_samples')}/{fc.get('total_samples')} "
            f"(rate {rate_s})"
        )
        for ex in report.get("examples", []):
            _render_example(ex, f"\n- case {ex.get('case_id')} (score {ex.get('score')})")
        return "\n".join(lines) + "\n"

    # Full shape.
    s = report.get("summary", {})
    lines.append(
        f"## Failure analysis — {s.get('n_failing')}/{s.get('total_cases')} cases "
        f"failing, mean score {s.get('mean_score')}"
    )
    cats = report.get("categories", [])
    if cats:
        lines.append("Top failure categories (by failure rate):")
        for c in cats:
            rate = c.get("failure_rate")
            rate_s = f"{rate:.2f}" if isinstance(rate, (int, float)) else "?"
            lines.append(
                f"  - {c.get('category_name')} ({c.get('category_type')}): "
                f"{c.get('num_failing_samples')}/{c.get('total_samples')} (rate {rate_s})"
            )
    examples = report.get("examples", [])
    if examples:
        lines.append("\n### Representative failures (query → plan → what failed)")
        for ex in examples:
            cat = ex.get("category_id") or "?"
            _render_example(
                ex, f"\n[{cat}] case {ex.get('case_id')} (score {ex.get('score')})"
            )
    hard = report.get("hard_cases", [])
    if hard:
        lines.append("\n### Hardest cases (lowest score)")
        for ex in hard:
            _render_example(ex, f"\ncase {ex.get('case_id')} (score {ex.get('score')})")
    return "\n".join(lines) + "\n"

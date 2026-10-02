"""Shared ``list_cases``/``show_case`` agentic tools.

Both ``AgentEditor`` and ``BlockSuggester``'s agentic loops receive the same
``AgentFeedback.eval_result.per_case`` (a ``list[CaseResult]``, generic
across every project -- see ``models.py``) and want the same
overview/detail views over it, so the tool schemas and rendering live here
once rather than duplicated per caller.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .models import CaseResult

LIST_CASES_TOOL: dict[str, Any] = {
    "name": "list_cases",
    "description": (
        "Overview of this node's own evaluated cases: each case's id, "
        "pass/fail, and score. `failed_check` filters to failing cases "
        "whose details mention that substring (e.g. a specific "
        "constraint/check label); omit it to see every case. `limit` "
        "caps how many rows are returned (default 30)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "failed_check": {"type": "string"},
            "limit": {"type": "integer"},
        },
    },
}

SHOW_CASE_TOOL: dict[str, Any] = {
    "name": "show_case",
    "description": (
        "One case in full: its id, pass/fail, score, error (if any), and "
        "the complete `details` dict the project's scorer attached for it "
        "(e.g. the raw plan, dimension scores, failed checks with grader "
        "messages) -- use after `list_cases` to inspect a specific case."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"case_id": {"type": "string"}},
        "required": ["case_id"],
    },
}


def render_list_cases(
    per_case: list[CaseResult],
    *,
    failed_check: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    rows = per_case
    if failed_check:
        needle = failed_check.lower()
        rows = [
            c for c in rows
            if not c.passed and needle in json.dumps(c.details, default=str).lower()
        ]
    try:
        cap = int(limit) if limit else 30
    except (TypeError, ValueError):
        cap = 30
    cap = max(0, cap)
    total = len(rows)
    rows = rows[:cap]
    if not rows:
        return "(no matching cases)"
    lines = [
        f"{c.case_id}: {'PASS' if c.passed else 'FAIL'} score={c.score:.3f}"
        + (f" error={c.error!r}" if c.error else "")
        for c in rows
    ]
    if total > len(rows):
        lines.append(f"... ({total - len(rows)} more not shown; raise `limit` to see them)")
    return "\n".join(lines)


def render_show_case(per_case: list[CaseResult], case_id: str) -> str:
    for c in per_case:
        if str(c.case_id) == str(case_id):
            return json.dumps(
                {
                    "case_id": c.case_id,
                    "passed": c.passed,
                    "score": c.score,
                    "error": c.error,
                    "details": c.details,
                },
                indent=2,
                default=str,
            )
    return f"(case not found: {case_id})"

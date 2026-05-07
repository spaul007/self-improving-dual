"""Math-easy scorer: extract the first integer in the agent's output and
compare to the expected integer. Pass = exact match. Score = 1.0 / 0.0.
"""
from __future__ import annotations

import re
from typing import Any


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """``agent_output`` is an :class:`AgentOutput` (from
    ``platform_core.runner``). We read its ``.result`` field, which for
    this benchmark should be a string."""
    raw = str(getattr(agent_output, "result", agent_output) or "")
    expected = str(case["expected"]).strip()
    match = _NUM.search(raw)
    found = match.group(0) if match else ""
    try:
        passed = float(found) == float(expected)
    except ValueError:
        passed = False
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "details": {"expected": expected, "found": found, "raw_output": raw[:200]},
    }

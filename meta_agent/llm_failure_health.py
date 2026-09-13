"""Shared detection logic for the OpenRouter Responses-API "status=failed"
empty-response condition (see platform_core/llm_wrapper.py::call_llm --
a response that comes back without raising an exception, but whose own
status/stop_reason field says the generation failed).

Extracted from the standalone CLI tool analyze_llm_call_failures.py (repo
root) so it can also be imported by meta_agent.managers.hgm -- which
needs the exact same trace.jsonl-derived signal, per round, to (a) write
a small monitoring artifact every round for the dashboard and (b)
optionally exclude terminal-failure-corrupted cases from a node's reward
(see HGMManager's `exclude_llm_call_failures`/`llm_call_failure_threshold_pct`).
analyze_llm_call_failures.py now imports these two functions rather than
defining them inline; its own CLI output is unchanged.

Two things are counted, per trace.jsonl file:
  1. llm_call_retry events caused specifically by status=failed (as
     opposed to a genuine thrown exception) -- these were retried and
     (unless this was also the terminal attempt) recovered.
  2. llm_response events whose own stop_reason ended up "failed" -- every
     retry for that call was exhausted and the failure reached the
     caller anyway. This is what actually corrupts a case's score.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Shared default so HGMManager's own loud-warning threshold
# (llm_call_failure_threshold_pct) and the dashboard's Diagnostics-panel
# flag (meta_agent/run_inspect.py::extract_diagnostics) can't drift apart
# -- a healthy round was observed this session at <1% incidence; a real
# provider outage was seen at 70%+.
DEFAULT_INCIDENCE_THRESHOLD_PCT = 3.0


def iter_trace_files(path: Path) -> list[Path]:
    """A single trace.jsonl path, or every round_*/logs/trace.jsonl under
    a run directory."""
    if path.is_file():
        return [path]
    return sorted(path.glob("round_*/logs/trace.jsonl"))


def analyze_trace_file(path: Path) -> dict:
    """Parse one trace.jsonl file and return its failure-health summary.

    Returns an all-zero/empty summary (never raises) when ``path``
    doesn't exist -- callers that unconditionally analyze a round's own
    trace.jsonl (which may not exist yet, e.g. mid-EXPAND) should treat
    that as "nothing to report" rather than an error."""
    n_llm_calls = 0
    n_llm_responses = 0
    n_status_failed_retries = 0
    n_exception_retries = 0
    n_terminal_failed_responses = 0
    cases_with_terminal_failure: Counter = Counter()
    cases_with_status_failed_retry: Counter = Counter()
    error_codes: Counter = Counter()

    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            payload = event.get("payload", {}) or {}

            if kind == "llm_call":
                n_llm_calls += 1
            elif kind == "llm_call_retry":
                err = str(payload.get("error", ""))
                case_id = payload.get("case_id")
                if "status/stop_reason == 'failed'" in err:
                    n_status_failed_retries += 1
                    if case_id is not None:
                        cases_with_status_failed_retry[case_id] += 1
                    code = payload.get("response_error_code")
                    error_codes[code or "(none given by API)"] += 1
                else:
                    n_exception_retries += 1
            elif kind == "llm_response":
                n_llm_responses += 1
                if payload.get("stop_reason") == "failed":
                    n_terminal_failed_responses += 1
                    case_id = payload.get("case_id")
                    if case_id is not None:
                        cases_with_terminal_failure[case_id] += 1
                    code = payload.get("response_error_code")
                    error_codes[code or "(none given by API)"] += 1

    return {
        "path": str(path),
        "n_llm_calls": n_llm_calls,
        "n_llm_responses": n_llm_responses,
        "n_status_failed_retries": n_status_failed_retries,
        "n_exception_retries": n_exception_retries,
        "n_terminal_failed_responses": n_terminal_failed_responses,
        "cases_with_status_failed_retry": dict(cases_with_status_failed_retry),
        "cases_with_terminal_failure": dict(cases_with_terminal_failure),
        "error_codes": dict(error_codes),
    }


def incidence_rate_pct(health: dict) -> float:
    """(status=failed retries + terminal failures) / total responses, as
    a percentage -- 0.0 when there were no responses at all (nothing to
    divide by, and nothing to flag)."""
    responses = health.get("n_llm_responses", 0)
    if not responses:
        return 0.0
    return (
        100.0
        * (health.get("n_status_failed_retries", 0) + health.get("n_terminal_failed_responses", 0))
        / responses
    )

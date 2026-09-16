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

Three things are counted, per trace.jsonl file:
  1. llm_call_retry events caused specifically by status=failed (the
     OpenRouter/DeepInfra empty-response condition) -- these were
     retried and (unless this was also the terminal attempt) recovered.
  2. llm_call_retry events caused by a genuine thrown exception instead
     (network errors, rate limits, etc. -- e.g. OpenRouter/CoreWeave's
     `429 rate_limit_exceeded` on a saturated shared model pool) --
     also retried/recovered, just via a different code path in
     platform_core/llm_wrapper.py::call_llm. Counted separately
     (n_exception_retries) since the two conditions have different root
     causes, but incidence_rate_pct() below folds both into the same
     reported rate -- from an operator's perspective, either one means
     "this round needed extra attempts to get a clean response."
  3. llm_response events whose own stop_reason ended up "failed" -- every
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
    # Which upstream provider (e.g. "CoreWeave", "DeepInfra") an exception
    # retry's error body named, when present -- see
    # platform_core/llm_wrapper.py::_exception_provider_name's own
    # docstring for why this is only ever populated on an error (never a
    # successful call, which OpenRouter's Responses API doesn't expose
    # provider identity for at all). Absent on any trace predating the
    # 2026-09-16 logging addition -- every caller here treats a key
    # simply not existing in a given event the same as it being None.
    exception_retry_providers: Counter = Counter()

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
                    provider_name = payload.get("provider_name")
                    if provider_name:
                        exception_retry_providers[provider_name] += 1
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
        "exception_retry_providers": dict(exception_retry_providers),
    }


def incidence_rate_pct(health: dict) -> float:
    """(status=failed retries + exception retries + terminal failures) /
    total responses, as a percentage -- 0.0 when there were no responses
    at all (nothing to divide by, and nothing to flag).

    FIX (2026-09-16): originally omitted n_exception_retries entirely --
    a real reliability signal (e.g. OpenRouter/CoreWeave 429
    rate_limit_exceeded storms on the shared Gemma-4-31B pool, observed
    this session: 161+ exception retries in a single 120-case pass, 0%
    reported incidence rate under the old formula) was silently
    invisible to every caller of this function -- HGMManager's own
    >3%-threshold warning, the dashboard's Diagnostics alert, and every
    eval_node*_*.py standalone verification script's printed failure
    rate. All three now reflect it automatically via this shared
    function. Callers that also want the raw counts should read
    n_status_failed_retries/n_exception_retries/n_terminal_failed_responses
    directly rather than trying to back them out of this percentage."""
    responses = health.get("n_llm_responses", 0)
    if not responses:
        return 0.0
    return (
        100.0
        * (
            health.get("n_status_failed_retries", 0)
            + health.get("n_exception_retries", 0)
            + health.get("n_terminal_failed_responses", 0)
        )
        / responses
    )

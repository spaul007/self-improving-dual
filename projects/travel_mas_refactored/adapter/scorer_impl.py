"""Travel-MAS benchmark scorer (real logic; see benchmark/scorer.py for the
framework-mandated thin shim that imports this module).

Two stages, both running in the parent (evaluator) process:

1. Plan conversion. The agent's free-text plan is converted to structured
   JSON by `gpt-5-2025-08-07` (per project mandate; mirrors the special
   branch in the reference's evaluation/convert_report.py:102 that omits
   ``max_tokens`` for reasoning models). The model wraps its JSON in
   ``<JSON>...</JSON>`` tags; the scorer extracts and parses it.

2. Constraint scoring → composite. The structured plan is run through the
   reference's commonsense and hard-constraint evaluators, then composited:

       composite_score = (commonsense_weighted + hard_pass) / 2

   The composite score is the *primary* score returned to the meta-agent so
   the hill-climber optimises directly for it. The component scores are
   preserved in ``details`` so the gatherer can surface failed checks to the
   editor on the next round.

Registered under the ``"scorer"`` registry bucket as ``travel_mas_refactored_default``
(identical logic to ``projects/travel_mas/adapter/scorer_impl.py``.s
``travel_mas_default`` -- kept as distinct names so both projects can be imported in
the same process without a registry collision). A top-level
``score(case, agent_output)`` function is exposed for the alternate
"load scorer.py and call score()" fallback path used by simpler benchmarks
like math_mas.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Make _eval importable as a top-level package. _eval lives under
# benchmark/_eval/ (a sibling of adapter/), not under adapter/ itself, so
# this bootstraps benchmark/ onto sys.path -- unlike a project.scorer_impl
# module loaded via a normal package import, this also has to work when
# benchmark/scorer.py loads this file via importlib.util.spec_from_file_location
# (no parent package context).
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from _eval.constraints_commonsense import eval_commonsense  # noqa: E402
from _eval.constraints_hard import eval_hard  # noqa: E402
from _eval.eval_converted import (  # noqa: E402
    calculate_hard_score,
    calculate_weighted_score,
)
from _eval.prompts import FORMAT_CONVERT_PROMPT_EN  # noqa: E402

from meta_agent.registry import register  # noqa: E402

CONVERT_MODEL = "gpt-5-2025-08-07"
# Reference travel_agent (evaluation/convert_report.py) uses max_retries=30,
# giving 31 attempts. Match that so transient JSON-parse failures don't drop
# cases to score=0 on the first miss.
DEFAULT_RETRIES = 31
# Per-attempt HTTP timeout was the only guard before (see the comment in
# _convert_plan_to_json below) -- 31 retries x a large per-attempt timeout
# can still legitimately sum to hours even when every individual timeout
# fires correctly. CONVERT_OVERALL_TIMEOUT_S is a hard wall-clock ceiling on
# the WHOLE retry loop, checked every iteration, independent of how many of
# the 31 attempts have run -- confirmed live: a real run stalled 6.6+ hours
# on this exact loop despite the existing 300s per-attempt timeout.
CONVERT_OVERALL_TIMEOUT_S = 600.0
# Smaller than the overall ceiling on purpose, so more than one attempt can
# actually fit inside CONVERT_OVERALL_TIMEOUT_S -- a 300s per-attempt
# timeout with a 300s overall ceiling would only ever allow a single try.
#
# 150s, not 60s: measured live (2026-08-31, node-6, real vLLM/Qwen3.5-35B-A3B
# traffic) that a genuine, eventually-successful conversion call for a short
# (~600-token) plan took 96s wall-clock under realistic concurrent load --
# 60s was tight enough to manufacture a timeout on nearly every real call
# (confirmed live: a full 120-case reval of a known-good agent scored 0.0 on
# effectively every case, all via "conversion timed out after ~366s", with
# node-6 completely uncontended -- i.e. a self-inflicted false failure, not
# a real quality signal). 150s x 2 attempts still fits the 300s ceiling
# above, so the "5 min" hard cap this was built to guarantee is unchanged.
CONVERT_PER_ATTEMPT_TIMEOUT_S = 300.0
JSON_BLOCK_RE = re.compile(r"<JSON>(.*?)</JSON>", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Plan conversion (LLM call)
# --------------------------------------------------------------------------- #


def _resolve_database_root() -> Optional[Path]:
    val = os.environ.get("TRAVEL_DATABASE_ROOT")
    if val:
        return Path(val)
    # Project-relative default — mirrors this project's own
    # tools/_csv.py::database_root() so the scorer reads from the same
    # database the agent's tools do when TRAVEL_DATABASE_ROOT is unset (the
    # common case). parents[1] from adapter/scorer_impl.py resolves to
    # projects/travel_mas_refactored/, same as it did from
    # benchmark/scorer.py (both are direct children of
    # travel_mas_refactored/). That directory's data/database_en is a
    # symlink to projects/travel/data/database_en (avoids duplicating
    # ~431MB), not a live import/path dependency on projects/travel.
    candidate = Path(__file__).resolve().parents[1] / "data" / "database_en"
    return candidate if candidate.exists() else None


def _convert_plan_to_json(plan_text: str, *, retries: int = DEFAULT_RETRIES) -> tuple[Optional[dict], Optional[str]]:
    """Call gpt-5-2025-08-07 to convert the agent's text plan into the
    structured JSON the constraint evaluators expect.

    Returns (parsed_json_or_None, error_or_None). On unrecoverable failure,
    parsed_json is None and error contains a short description.
    """
    if not plan_text or not plan_text.strip():
        return None, "agent produced no plan"

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - openai is a hard dep
        return None, f"openai sdk not importable: {exc}"

    # Route conversion at a local OpenAI-compatible server (e.g. vLLM) when
    # one is configured, using its model id. Falls back to OpenAI's
    # CONVERT_MODEL when no base_url is set — the original behaviour is
    # unchanged for the OpenAI path. base_url defaults to the same
    # LLM_BASE_URL the task-agent uses (set from task_agent.base_url by
    # meta_agent/runtime_env.py); TRAVEL_CONVERT_MODEL / TRAVEL_CONVERT_BASE_URL
    # override it independently if conversion should use a different endpoint.
    base_url = os.environ.get("TRAVEL_CONVERT_BASE_URL") or os.environ.get("LLM_BASE_URL")
    convert_model = (
        os.environ.get("TRAVEL_CONVERT_MODEL")
        or (os.environ.get("LLM_MODEL") if base_url else None)
        or CONVERT_MODEL
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if base_url:
            # Local servers ignore auth, but the SDK constructor needs a string.
            api_key = "EMPTY"
        else:
            return None, "OPENAI_API_KEY not set"

    # Per-attempt timeout (SDK default is ~600s): without this, a single
    # slow local-model completion blocks that one attempt indefinitely --
    # see platform_core/llm_wrapper.py's DEFAULT_REQUEST_TIMEOUT_S for the
    # same fix on the task-agent side. On its own this is NOT sufficient --
    # see the overall deadline enforced in the loop below.
    # max_retries=0: the SDK's own default (2) retries silently *inside* a
    # single call, so one manual "attempt" here could actually cost up to
    # 3x CONVERT_PER_ATTEMPT_TIMEOUT_S before raising -- confirmed live as
    # part of the same 2026-08-31 investigation above (2 manual attempts
    # summing to ~366s, not the expected ~2x60s=120s). The manual loop
    # below already provides its own retry/backoff, deliberately timed
    # against CONVERT_OVERALL_TIMEOUT_S; the SDK's internal retries just
    # fight it for the same budget.
    client = (
        OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=CONVERT_PER_ATTEMPT_TIMEOUT_S, max_retries=0,
        )
        if base_url
        else OpenAI(timeout=CONVERT_PER_ATTEMPT_TIMEOUT_S, max_retries=0)
    )
    messages = [
        {"role": "system", "content": FORMAT_CONVERT_PROMPT_EN},
        {"role": "user", "content": plan_text},
    ]

    last_err: Optional[str] = None
    started = time.monotonic()
    for attempt in range(retries):
        elapsed = time.monotonic() - started
        if elapsed >= CONVERT_OVERALL_TIMEOUT_S:
            # The hard ceiling this function exists to guarantee: give up
            # after ~CONVERT_OVERALL_TIMEOUT_S total, no matter how many of
            # `retries` attempts have actually run or how long any single
            # per-attempt timeout takes to fire.
            last_err = (
                f"conversion timed out after {elapsed:.0f}s "
                f"({attempt} attempt(s) made, last error: {last_err})"
            )
            break
        try:
            # gpt-5-2025-08-07 is a reasoning model — do not pass max_tokens.
            resp = client.chat.completions.create(
                model=convert_model,
                messages=messages,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surface SDK errors as scorer detail
            last_err = f"openai call failed (attempt {attempt + 1}): {exc!r}"
            time.sleep(min(2 ** attempt, 8))
            continue

        match = JSON_BLOCK_RE.search(content)
        raw = match.group(1) if match else content
        try:
            return json.loads(raw), None
        except json.JSONDecodeError as exc:
            last_err = f"could not parse JSON on attempt {attempt + 1}: {exc.msg}"
            continue

    return None, last_err or "conversion failed"


# --------------------------------------------------------------------------- #
# Constraint evaluation
# --------------------------------------------------------------------------- #


def _evaluate(plan: dict, meta: dict, sample_id: str) -> dict[str, Any]:
    """Run commonsense + hard checks against a structured plan, return the
    composite score breakdown."""
    db_root = _resolve_database_root()
    sample_db: Optional[Path] = None
    if db_root is not None:
        candidate = db_root / f"id_{sample_id}"
        if candidate.exists():
            sample_db = candidate
        else:
            alt = db_root / sample_id
            if alt.exists():
                sample_db = alt

    commonsense_results = eval_commonsense(plan, meta, database_dir=sample_db)
    hard_results = eval_hard(plan, meta)

    weighted = calculate_weighted_score(commonsense_results)
    hard = calculate_hard_score(hard_results)

    commonsense_score = float(weighted["total_weighted_score"])
    hard_score = float(hard["score"])
    composite_score = (commonsense_score + hard_score) / 2

    failed_checks: list[str] = []
    for dim_name, dim in weighted["dimension_details"].items():
        for check in dim["checks"]:
            if not check["passed"]:
                failed_checks.append(f"commonsense:{dim_name}:{check['name']}")
    for cname, cinfo in hard["constraints"].items():
        if not cinfo["passed"]:
            failed_checks.append(f"hard:{cname}")

    return {
        "commonsense_score": commonsense_score,
        "hard_score": hard_score,
        "composite_score": composite_score,
        "passed": commonsense_score == 1.0 and hard_score == 1.0,
        "dimension_scores": weighted["dimension_scores"],
        "dimension_details": weighted["dimension_details"],
        "hard_constraints": hard["constraints"],
        "failed_checks": failed_checks,
    }


# --------------------------------------------------------------------------- #
# Public scorer surface
# --------------------------------------------------------------------------- #


def _zero_result(reason: str, *, raw_output: str = "") -> dict[str, Any]:
    return {
        "score": 0.0,
        "passed": False,
        "details": {
            "error": reason,
            "commonsense_score": 0.0,
            "hard_score": 0.0,
            "composite_score": 0.0,
            "raw_output_preview": (raw_output or "")[:500],
            # Full, untruncated agent output -- the preview above is kept
            # for existing consumers, but debugging a failed conversion
            # (or comparing agents qualitatively) needs the whole plan
            # text, not just its first 500 chars.
            "raw_plan_text": raw_output or "",
        },
    }


@register("scorer", "travel_mas_refactored_default")
class TravelCompositeScorer:
    """Travel scorer + round-level aggregator.

    Two methods:

    - ``score(case, agent_output)`` — per-case judge. Called once per
      case by the evaluator (in parallel). Converts the agent's plan
      to JSON via gpt-5-2025-08-07, runs commonsense + hard
      constraints, returns ``{score, passed, details}``.
    - ``aggregate(per_case, trace_events)`` — round-level summarizer.
      Called once per round by the framework gatherer. Reads the
      ``details`` shapes the per-case ``score()`` produced and
      returns the project_metrics dict that lands on
      ``AgentFeedback.project_metrics``.

    Co-locating the two methods makes the scorer-emit shape and the
    aggregator-consume shape explicit (they share field names like
    ``failed_checks`` and ``dimension_scores``).
    """

    def __init__(self, *, retries: int = DEFAULT_RETRIES) -> None:
        self.retries = retries

    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        # ``agent_output`` is an AgentOutput from platform_core.runner; we
        # consume its ``.result`` field. ``getattr`` keeps the scorer usable
        # by tests and tools that hand it a bare string.
        plan_text = str(getattr(agent_output, "result", agent_output) or "")
        meta = case.get("meta_info") or {}
        sample_id = str(case.get("id") or case.get("env", {}).get("TRAVEL_SAMPLE_ID") or "")
        if not meta:
            return _zero_result("case is missing meta_info", raw_output=plan_text)

        parsed, err = _convert_plan_to_json(plan_text, retries=self.retries)
        if parsed is None:
            return _zero_result(f"plan conversion failed: {err}", raw_output=plan_text)

        try:
            breakdown = _evaluate(parsed, meta, sample_id)
        except Exception as exc:  # noqa: BLE001
            return _zero_result(f"constraint evaluation raised: {exc!r}", raw_output=plan_text)

        details: dict[str, Any] = {
            "commonsense_score": breakdown["commonsense_score"],
            "hard_score": breakdown["hard_score"],
            "composite_score": breakdown["composite_score"],
            "dimension_scores": breakdown["dimension_scores"],
            "dimension_details": breakdown["dimension_details"],
            "hard_constraints": breakdown["hard_constraints"],
            "failed_checks": breakdown["failed_checks"],
            "converted_plan": parsed,
            # The agent's original free-text plan, before JSON conversion --
            # needed to inspect what was actually generated, not just the
            # structured form the constraint evaluators consumed.
            "raw_plan_text": plan_text,
        }
        return {
            "score": breakdown["composite_score"],
            "passed": breakdown["passed"],
            "details": details,
        }

    def aggregate(
        self,
        per_case: list[Any],
        trace_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Round-level roll-up over ``per_case`` (list of CaseResult).
        Returns the project_metrics dict the framework gatherer puts on
        ``AgentFeedback.project_metrics``. Trace events are unused for
        travel today but the parameter is present for future use (e.g.
        per-case tool-error breakdowns)."""
        if not per_case:
            return {}

        no_plan_count = 0
        check_counts: dict[str, int] = {}
        dim_sums: dict[str, float] = {}
        dim_counts: dict[str, int] = {}

        for case in per_case:
            details = getattr(case, "details", None) or {}
            err = details.get("error")
            if isinstance(err, str) and _NO_PLAN_RE.search(err):
                no_plan_count += 1
                # Cases with no plan have no failed_checks/dimension_scores.
                continue

            for check in details.get("failed_checks") or []:
                if isinstance(check, str):
                    check_counts[check] = check_counts.get(check, 0) + 1

            dims = details.get("dimension_scores") or {}
            if isinstance(dims, dict):
                for name, score in dims.items():
                    if not isinstance(name, str):
                        continue
                    try:
                        score_f = float(score)
                    except (TypeError, ValueError):
                        continue
                    dim_sums[name] = dim_sums.get(name, 0.0) + score_f
                    dim_counts[name] = dim_counts.get(name, 0) + 1

        no_plan_rate = no_plan_count / len(per_case)
        top_failed_checks: list[tuple[str, int]] = sorted(
            check_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[:15]
        dimension_means: dict[str, float] = {
            name: dim_sums[name] / dim_counts[name]
            for name in dim_sums
            if dim_counts.get(name, 0) > 0
        }
        return {
            "no_plan_rate": no_plan_rate,
            "top_failed_checks": top_failed_checks,
            "dimension_means": dimension_means,
        }


_NO_PLAN_RE = re.compile(r"^\s*plan conversion failed", re.IGNORECASE)


# Default scorer instance for the evaluator's "load scorer.py" path.
_DEFAULT_SCORER = TravelCompositeScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level scorer entry point used by ``SubprocessEvaluator`` when
    no registered scorer is named in YAML."""
    return _DEFAULT_SCORER.score(case, agent_output)

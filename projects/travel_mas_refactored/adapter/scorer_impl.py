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

import functools
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
# Check semantics
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _load_check_semantics() -> dict[str, str]:
    """Merge ``error_semantics.json`` (task-level checks, from the
    immutable scorer) and ``harness_error_semantics.json`` (harness-level
    crash-cause flags, from the seed's own mutable workflow) -- both live
    alongside this file -- into one ``name -> human-readable description``
    dict, loaded once and cached for the process's lifetime.

    This is deliberately independent of ``Curriculum``'s own descriptions
    loading (``hgm.py::_load_curriculum_check_descriptions``): that path
    only ever runs when curriculum is enabled, so a curriculum-off config
    (like a production run with the behavior summarizer turned off) would
    otherwise leave the improvement proposer seeing bare check names and
    counts with no explanation of what a check actually verifies. Loading
    directly here means every ``aggregate()`` call gets rich semantics
    regardless of whether curriculum is on.

    Missing/malformed files, or a non-dict/non-string entry, degrade
    silently to being omitted -- same "nice-to-have, never a reason to
    fail" convention as block_suggester.py's strategies_path and hgm.py's
    curriculum descriptions loader."""
    base = Path(__file__).resolve().parent
    merged: dict[str, str] = {}
    for filename in ("error_semantics.json", "harness_error_semantics.json"):
        try:
            data = json.loads((base / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            if isinstance(key, str) and isinstance(val, str) and key != "_comment":
                merged[key] = val
    return merged


# --------------------------------------------------------------------------- #
# Tool-input traceability
# --------------------------------------------------------------------------- #

_MIN_TRACEABLE_VALUE_LEN = 4


def _tool_input_traceability(
    per_case: list[Any], trace_events: list[dict[str, Any]],
) -> tuple[float, list[tuple[str, int]]]:
    """Directional signal for tool-call argument values that look invented
    rather than derived from something the agent actually saw: for each
    ``tool_call`` argument value (case-insensitive substring match, values
    under 4 chars skipped as too short to mean anything), check whether it
    appears in that SAME case's own request text (``details["query"]``), an
    EARLIER ``tool_call``'s own arguments, or an EARLIER ``tool_result``'s
    ``result_preview`` -- never a later one, and never across cases.

    Deliberately named "untraced," not "hallucinated": ``result_preview``
    in trace.jsonl is truncated to ~200 chars, so a value genuinely copied
    from later in a longer real tool result will incorrectly read as
    untraced here. This is a real, unavoidable false-positive source given
    what trace.jsonl actually stores, not a metric bug -- treat this as a
    rough, directional signal (is this getting better or worse across
    rounds?), not proof that any single flagged value was fabricated.

    Returns ``(untraced_rate, ranked_untraced_counts_by_tool_name)`` -- the
    latter in the same ``[(name, count), ...]`` shape/sort convention as
    ``top_failed_checks``/``harness_checks`` (count descending, ties
    alphabetical, capped at 15). ``(0.0, [])`` when there's nothing to
    check. Never raises: a malformed event is skipped, not fatal."""
    case_query_by_id: dict[str, str] = {}
    for case in per_case:
        details = getattr(case, "details", None) or {}
        query = details.get("query")
        case_query_by_id[str(getattr(case, "case_id", ""))] = (
            query if isinstance(query, str) else ""
        )

    events_by_case: dict[str, list[dict[str, Any]]] = {}
    for event in trace_events:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        case_id = str(payload.get("case_id", ""))
        events_by_case.setdefault(case_id, []).append(event)

    checked = 0
    untraced = 0
    untraced_by_tool: dict[str, int] = {}

    for case_id, events in events_by_case.items():
        blobs: list[str] = [case_query_by_id.get(case_id, "").lower()]
        results_by_id: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.get("kind") == "tool_result":
                payload = event.get("payload") or {}
                call_id = payload.get("id")
                if isinstance(call_id, str):
                    results_by_id[call_id] = payload

        for event in events:
            if event.get("kind") != "tool_call":
                continue
            payload = event.get("payload") or {}
            name = payload.get("name")
            if not isinstance(name, str):
                name = "(unknown tool)"
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}

            for value in arguments.values():
                text = value if isinstance(value, str) else str(value)
                if len(text) < _MIN_TRACEABLE_VALUE_LEN:
                    continue
                checked += 1
                needle = text.lower()
                if not any(needle in blob for blob in blobs if blob):
                    untraced += 1
                    untraced_by_tool[name] = untraced_by_tool.get(name, 0) + 1

            # Only AFTER checking: this call's own arguments and its
            # result become traceable evidence for later calls in the
            # same case (never for itself, never for earlier calls).
            blobs.append(
                " ".join(str(v) for v in arguments.values()).lower()
            )
            call_id = payload.get("id")
            if isinstance(call_id, str) and call_id in results_by_id:
                preview = results_by_id[call_id].get("result_preview")
                if isinstance(preview, str):
                    blobs.append(preview.lower())

    untraced_rate = (untraced / checked) if checked else 0.0
    untraced_ranked = sorted(
        untraced_by_tool.items(), key=lambda kv: (-kv[1], kv[0])
    )[:15]
    return untraced_rate, untraced_ranked


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
            reason = f"plan conversion failed: {err}"
            if not plan_text.strip():
                # plan_text was empty -- _convert_plan_to_json short-
                # circuited without ever attempting a conversion, so this
                # prefix ("plan conversion failed") is itself misleading:
                # nothing was generated to convert, OR something WAS
                # generated but the agent's own workflow discarded it
                # before conversion (a self-inflicted rejection of
                # otherwise-good output -- a fundamentally different,
                # more actionable failure than a genuine non-generation).
                # ``agent_output.metadata`` already distinguishes these
                # (whatever the evolving workflow chooses to record --
                # e.g. which stage failed, budget_exhausted, a validation
                # gate's own rejection reason) but was previously
                # discarded entirely here, leaving every empty-output
                # case looking identical to the meta-agent's diagnosis
                # LLM. Appended generically (not keyed on specific field
                # names) since agent_metadata is free-form and owned by
                # the mutable workflow HGM evolves -- this scorer must
                # stay correct regardless of what keys a future edit adds.
                # The "plan conversion failed" PREFIX is left unchanged
                # so _NO_PLAN_RE's no_plan_rate counting (and anything
                # else matching on it) is unaffected.
                agent_meta = dict(getattr(agent_output, "metadata", None) or {})
                if agent_meta:
                    reason += f" (agent_metadata: {agent_meta})"
            return _zero_result(reason, raw_output=plan_text)

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
        ``AgentFeedback.project_metrics``. ``trace_events`` feeds
        ``_tool_input_traceability`` (see its own docstring) for a
        directional tool-call-input traceability signal."""
        if not per_case:
            return {}

        no_plan_count = 0
        check_counts: dict[str, int] = {}
        dim_sums: dict[str, float] = {}
        dim_counts: dict[str, int] = {}
        # Tally of *why* no-plan cases crashed, built generically from
        # whichever boolean flags the agent's own (mutable, HGM-evolved)
        # workflow happens to set on AgentOutput.metadata (e.g.
        # sightseeing_failed, budget_exhausted, meal_validation_failed --
        # see meta_agent/evaluator.py's agent_artifact, which always
        # copies agent_output.metadata onto details["agent_metadata"]
        # regardless of scorer outcome). Deliberately NOT keyed on
        # hardcoded field names: agent_metadata's shape is owned by the
        # workflow code HGM keeps rewriting, so a scorer that hardcodes
        # specific flag names would silently go stale the moment an edit
        # renames or restructures them. A generic count-of-True-flags
        # stays correct across any future generation of the workflow.
        harness_checks: dict[str, int] = {}

        for case in per_case:
            details = getattr(case, "details", None) or {}
            err = details.get("error")
            if isinstance(err, str) and _NO_PLAN_RE.search(err):
                no_plan_count += 1
                for key, val in (details.get("agent_metadata") or {}).items():
                    if val is True:
                        harness_checks[key] = harness_checks.get(key, 0) + 1
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
        # Same shape/sort convention as top_failed_checks (count descending,
        # ties broken alphabetically) so it renders identically via
        # render_metrics's list-of-(name, count) branch.
        harness_checks_ranked: list[tuple[str, int]] = sorted(
            harness_checks.items(), key=lambda kv: (-kv[1], kv[0])
        )[:15]
        # Pairs each check/flag actually appearing above with its
        # human-readable semantics (what it verifies, when it fires) --
        # independent of Curriculum, so the improvement proposer gets rich
        # explanations of bare check names even when curriculum is
        # disabled (see _load_check_semantics's own docstring). Checks
        # with no known description (a name error_semantics.json/
        # harness_error_semantics.json doesn't cover) are simply omitted,
        # never fabricated.
        semantics = _load_check_semantics()
        check_semantics: list[tuple[str, str]] = [
            (name, semantics[name])
            for name, _ in (top_failed_checks + harness_checks_ranked)
            if name in semantics
        ]
        # See _tool_input_traceability's own docstring for what this does
        # and does not prove (directional signal, not a hallucination
        # proof -- result_preview's ~200-char truncation biases it toward
        # false positives).
        untraced_rate, untraced_ranked = _tool_input_traceability(
            per_case, trace_events
        )
        return {
            "no_plan_rate": no_plan_rate,
            "top_failed_checks": top_failed_checks,
            "dimension_means": dimension_means,
            "harness_checks": harness_checks_ranked,
            "check_semantics": check_semantics,
            "tool_input_untraced_rate": untraced_rate,
            "untraced_tool_inputs": untraced_ranked,
        }


_NO_PLAN_RE = re.compile(r"^\s*plan conversion failed", re.IGNORECASE)


# Default scorer instance for the evaluator's "load scorer.py" path.
_DEFAULT_SCORER = TravelCompositeScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level scorer entry point used by ``SubprocessEvaluator`` when
    no registered scorer is named in YAML."""
    return _DEFAULT_SCORER.score(case, agent_output)

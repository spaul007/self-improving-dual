"""Cross-project error-bucket prevalence analysis over a completed eval's
logs/ directory (trace.jsonl + case_*.json), companion to
analyze_error_buckets.py (repo root CLI wrapper).

Classifies each FAILING case into a small, fixed, domain-agnostic taxonomy
of agent-capability failure modes (see BUCKETS below) -- the same taxonomy
strategies.md's per-block guidance is organized around -- and reports how
prevalent each one is. Two cases can share a bucket, and one case can carry
more than one label (a case can be both tool_omission and
long_horizon_state_loss at once; that co-occurrence is itself a signal, not
noise to resolve).

Design, same philosophy as failure_summarizer.py/behavior_summarizer.py:
deterministic extraction in code (which cases failed, what each one's
query/error/tool-call trace actually says) wherever that's cheap and
exact, one narrow-purpose LLM call per batch of cases for the part that
genuinely needs judgment (which bucket(s) a given failure's evidence
supports). Aggregation/counting is done in code from the LLM's
STRUCTURED per-case output, never by asking the LLM to also do the
arithmetic -- that would just reintroduce the same "should have been
code" failure mode this whole taxonomy exists to name.

The LLM used for classification is intended to be the SAME backbone the
project already uses for its other meta-agent roles (block_suggester,
failure_summarizer, editor) -- callers pass in model/base_url/
reasoning_effort explicitly (see analyze_error_buckets.py's CLI), or leave
them None to fall back to the LLM_MODEL/LLM_BASE_URL/LLM_REASONING_EFFORT
env vars, exactly like every other call_llm call in this codebase.

One failure class -- tool_calling_budget_exceeded (never finished at all)
-- is cheap enough to detect deterministically (empty output + an error
string) that it's pre-classified in code and never sent to the LLM at
all; see _classify_deterministic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

# ------------------------------------------------------------------ #
# Fixed taxonomy -- see strategies.md's per-block guidance, which this
# mirrors exactly. Keys are stable aggregation labels (independent of
# whatever wording the LLM uses); values are the definition shown to it.
# ------------------------------------------------------------------ #

BUCKETS: dict[str, str] = {
    "tool_omission": (
        "Skips a tool call it needed and states/guesses a fact instead of "
        "looking it up, even though a tool that could have answered it "
        "exactly was available."
    ),
    "wrong_tool_or_arg": (
        "Calls the wrong tool for what it needed, or calls the right tool "
        "with a fabricated/incorrect argument not grounded in any prior "
        "real result (as opposed to omitting the call entirely)."
    ),
    "constraint_misreading": (
        "Misreads, drops, or misapplies an explicit requirement/constraint "
        "stated in the original task input."
    ),
    "wrong_tool_calling_order": (
        "Performs tool calls or steps in an order that violates a real "
        "dependency -- e.g. uses a value before anything actually produced "
        "it, or acts on information before it was available."
    ),
    "long_horizon_state_loss": (
        "Loses track of information or a rule across a long/repeated "
        "structure -- drift, inconsistency, or contradiction across many "
        "turns/units/items that were each individually fine in isolation "
        "(e.g. repeats something already used earlier, or a rule followed "
        "early on stops being followed later)."
    ),
    "apply_info_incorrectly": (
        "Had the correct information available (from a tool result or "
        "given context) but still combined or applied it incorrectly -- "
        "the fact was right there and still got it wrong."
    ),
    "tool_calling_budget_exceeded": (
        "Never finished -- hit its tool-calling iteration cap or output-token "
        "budget without producing a valid final result at all."
    ),
    "other": (
        "A real failure that doesn't fit any bucket above -- give a short "
        "free-text reason in `evidence` when using this."
    ),
}

_BUCKET_KEYS = set(BUCKETS)

_QUERY_CAP = 600
_OUTPUT_CAP = 1500
_TRACE_MAX_CALLS = 25
_TRACE_PREVIEW_CHARS = 200


# ------------------------------------------------------------------ #
# Deterministic loading (no LLM)
# ------------------------------------------------------------------ #

def load_cases(logs_dir: Path) -> list[dict[str, Any]]:
    """Every ``case_*.json`` directly under ``logs_dir``, sorted by
    case_id. Tolerates malformed files (skipped, not fatal)."""
    cases: list[dict[str, Any]] = []
    for fp in sorted(logs_dir.glob("case_*.json")):
        try:
            cases.append(json.loads(fp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    cases.sort(key=lambda c: str(c.get("case_id", "")))
    return cases


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Parse a trace.jsonl file into event dicts. Tolerates malformed
    lines and a missing file -- same convention as
    failure_summarizer.py's own copy of this helper."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def group_tool_events_by_case(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("kind") not in ("tool_call", "tool_result"):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        case_id = payload.get("case_id")
        if not isinstance(case_id, str):
            continue
        grouped.setdefault(case_id, []).append(event)
    return grouped


def render_case_tool_trace(events_for_case: list[dict[str, Any]]) -> str:
    """One case's ordered tool_call -> tool_result narrative. Project-
    agnostic: relies only on the tool_call/tool_result shape every
    ToolWrapper-based project emits identically, never on a specific
    tool/field name."""
    results_by_id: dict[str, dict[str, Any]] = {}
    for event in events_for_case:
        if event.get("kind") == "tool_result":
            payload = event.get("payload") or {}
            call_id = payload.get("id")
            if isinstance(call_id, str):
                results_by_id[call_id] = payload

    calls = [
        event.get("payload") or {}
        for event in events_for_case
        if event.get("kind") == "tool_call"
    ]
    if not calls:
        return "(no tool calls)"
    lines: list[str] = []
    for i, call in enumerate(calls[:_TRACE_MAX_CALLS], 1):
        name = call.get("name", "(unknown tool)")
        arguments = call.get("arguments", {})
        lines.append(f"{i}. {name}({arguments})")
        call_id = call.get("id")
        result = results_by_id.get(call_id) if isinstance(call_id, str) else None
        if result is None:
            lines.append("   -> (no result)")
            continue
        preview = result.get("result_preview")
        preview = preview if isinstance(preview, str) else ""
        lines.append(f"   -> {preview[:_TRACE_PREVIEW_CHARS]}")
    if len(calls) > _TRACE_MAX_CALLS:
        lines.append(f"   ... ({len(calls) - _TRACE_MAX_CALLS} more calls omitted)")
    return "\n".join(lines)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _truncate_middle(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = text[: cap // 2]
    tail = text[-(cap // 2):]
    return f"{head}\n...<{len(text) - cap} chars elided>...\n{tail}"


_NO_OUTPUT_ERROR_RE = re.compile(
    r"produced no plan|no plan|budget.?exhausted|no <\w+> tag|"
    r"output_failure|incomplete", re.IGNORECASE,
)


def _classify_deterministic(case: dict[str, Any]) -> Optional[str]:
    """Cheap, exact pre-classification for tool_calling_budget_exceeded --
    never worth spending an LLM call on. Returns the bucket key, or None
    if this case needs real judgment (send it to the LLM instead)."""
    details = case.get("details") or {}
    error_text = _as_text(details.get("error") or case.get("error"))
    raw_output = _as_text(
        details.get("raw_result") or details.get("raw_output_preview")
    ).strip()
    if error_text and not raw_output and _NO_OUTPUT_ERROR_RE.search(error_text):
        return "tool_calling_budget_exceeded"
    return None


_DIGEST_SHOWN_KEYS = {"error", "query", "raw_result", "raw_output_preview", "raw_plan_text"}
_SCORER_VERDICT_CAP = 2500


def _scorer_verdict_text(details: dict[str, Any]) -> str:
    """Whatever the scorer's own per-check verdict looks like for this
    project (travel_mas_refactored's own dimension_details/
    hard_constraints, or some other project's own shape) -- everything in
    `details` NOT already shown separately as error/query/raw_output. Kept
    generic (no project-specific key names) so this works for any
    project's scorer without per-project code. Empty string if `details`
    has nothing beyond the already-shown fields (a bare pass/fail scorer
    with no structured breakdown) -- the LLM falls back to judging from
    the query/output/trace alone in that case, exactly as before this
    field existed."""
    extra = {k: v for k, v in details.items() if k not in _DIGEST_SHOWN_KEYS}
    if not extra:
        return ""
    return _truncate_middle(_as_text(extra), _SCORER_VERDICT_CAP)


def build_case_digest(case: dict[str, Any], events_for_case: list[dict[str, Any]]) -> dict[str, Any]:
    details = case.get("details") or {}
    return {
        "case_id": str(case.get("case_id")),
        "score": round(float(case.get("score", 0.0)), 4),
        "error": _truncate_middle(_as_text(details.get("error") or case.get("error")), _OUTPUT_CAP),
        "query": _truncate_middle(_as_text(details.get("query")), _QUERY_CAP),
        "raw_output": _truncate_middle(
            _as_text(details.get("raw_result") or details.get("raw_output_preview")), _OUTPUT_CAP
        ),
        "scorer_verdict": _scorer_verdict_text(details),
        "tool_trace": render_case_tool_trace(events_for_case) if events_for_case else "(no tool-call trace found for this case)",
    }


# ------------------------------------------------------------------ #
# LLM classification
# ------------------------------------------------------------------ #

_TAXONOMY_BLOCK = "\n".join(f"- {k}: {v}" for k, v in BUCKETS.items())

_SYSTEM_PROMPT = f"""You are diagnosing WHY an AI agent's outputs failed, for
a fixed, general taxonomy of agent-capability failure modes -- not for this
one project's specific rules. You are given several failing cases: each
one's query, its error/final-output text, and (when available) the literal
ordered sequence of tool calls it made and what each tool actually
returned.

For EACH case, decide which of the following buckets its evidence
supports. A case can have zero, one, or several buckets -- most failures
support exactly one or two; do not force a fit.

{_TAXONOMY_BLOCK}

Ground every label in something actually shown for THAT case -- a real
absence of an expected tool call in its trace, a real quote from its
error/output text, a real contradiction between two things it did. Never
invent a bucket for a case just because it failed; if the evidence doesn't
clearly support one of the buckets above, return an empty list for that
case rather than guessing. Never use `other` without a specific, concrete
`evidence` string explaining what actually went wrong.

Respond with ONLY a single JSON object, no prose before or after, no
markdown code fence, mapping each case_id (as given) to a list of
{{"bucket": "<one of the keys above>", "evidence": "<short, specific,
quoted-or-closely-paraphrased reason, referencing this case's own
trace/error/output>"}} objects. Every case_id you were shown must appear
as a key, even if its list is empty. Example shape:
{{"3": [{{"bucket": "tool_omission", "evidence": "..."}}], "7": []}}"""


def _build_batch_user_prompt(digests: list[dict[str, Any]]) -> str:
    parts = []
    for d in digests:
        segment = (
            f"\n### case {d['case_id']} (score={d['score']})\n"
            f"query: {d['query']}\n"
            f"error: {d['error'] or '(none)'}\n"
            f"final output / raw_result: {d['raw_output'] or '(empty)'}\n"
        )
        if d.get("scorer_verdict"):
            segment += (
                f"scorer's own per-check verdict for this case (ground your "
                f"labels in this when present -- it is the exact, "
                f"machine-computed reason grading failed, more reliable "
                f"than reconstructing the failure by eye from the output "
                f"alone): {d['scorer_verdict']}\n"
            )
        segment += f"tool call trace:\n{d['tool_trace']}"
        parts.append(segment)
    return "\n".join(parts)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


_CASE_KEY_PREFIX_RE = re.compile(r"^\s*case\s+", re.IGNORECASE)


def _normalize_case_key(raw_key: str, expected_case_ids: set[str]) -> Optional[str]:
    """Recovers a case_id the model got slightly wrong in a predictable
    way, rather than either silently accepting a phantom key or silently
    dropping real classification data. Confirmed live (2026-09-17):
    Qwen3.5-122B sometimes writes JSON keys like "case 4_r00" instead of
    the bare "4_r00" it was actually shown (echoing its own "### case X"
    prompt framing back as part of the key) -- these are NOT real
    case_ids, and without this normalization they were silently accepted
    as extra phantom entries, inflating n_llm_classified/n_checked_cases
    beyond the real case count and attributing examples to a case_id that
    doesn't exist. Tries an exact match first, then the same string with
    a leading "case "/"Case "/"CASE " prefix stripped; returns None
    (caller drops it, with a warning) if neither matches."""
    if raw_key in expected_case_ids:
        return raw_key
    stripped = _CASE_KEY_PREFIX_RE.sub("", raw_key).strip()
    if stripped in expected_case_ids:
        return stripped
    return None


def _parse_batch_response(text: str, expected_case_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    """Best-effort JSON extraction -- tolerates a model wrapping the JSON
    in prose or a code fence despite being asked not to. Returns {} (not
    a raise) on total failure, and fills in any case_id the model dropped
    with an empty list rather than silently omitting it from the count.
    Never accepts a JSON key that isn't (after normalization) one of
    ``expected_case_ids`` -- see ``_normalize_case_key``."""
    result: dict[str, list[dict[str, str]]] = {}
    expected_set = set(expected_case_ids)
    match = _JSON_OBJECT_RE.search(text or "")
    found_valid_json = False
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            found_valid_json = True
            for raw_case_id, labels in parsed.items():
                if not isinstance(labels, list):
                    continue
                case_id = _normalize_case_key(str(raw_case_id), expected_set)
                if case_id is None:
                    print(
                        f"[error_bucket_analyzer] WARNING: dropping "
                        f"classification for unrecognized case key "
                        f"{raw_case_id!r} (not one of the case_ids shown "
                        f"in this batch, even after stripping a 'case ' "
                        f"prefix) -- likely the model echoing its own "
                        f"prompt framing back as a key.",
                        flush=True,
                    )
                    continue
                cleaned = []
                for item in labels:
                    if not isinstance(item, dict):
                        continue
                    bucket = item.get("bucket")
                    if bucket not in _BUCKET_KEYS:
                        continue
                    cleaned.append({
                        "bucket": bucket,
                        "evidence": str(item.get("evidence", ""))[:400],
                    })
                result[case_id] = cleaned
    if not found_valid_json:
        # Distinguish "the model answered but genuinely found nothing" from
        # "the batch silently failed" -- confirmed live (2026-09-16): a
        # Qwen3.5-122B classification call reasoning out loud in visible
        # content (not a hidden reasoning channel) burned an entire
        # 8192-token budget on a case-by-case internal debate and never
        # reached the JSON at all, and every case_id still ended up with an
        # empty list via the fallback below -- indistinguishable from a
        # real "no findings" result unless this is surfaced explicitly.
        print(
            f"[error_bucket_analyzer] WARNING: no valid JSON object found in "
            f"this batch's response ({len(text or '')} chars) -- every case "
            f"in it will be reported as having NO buckets, which likely "
            f"means the response was truncated before reaching the answer, "
            f"not that nothing was found. Consider a lower reasoning_effort, "
            f"a higher max_output_tokens, or a smaller --batch-size.",
            flush=True,
        )
    for cid in expected_case_ids:
        result.setdefault(cid, [])
    return result


def classify_batches(
    digests: list[dict[str, Any]],
    *,
    llm_caller: Callable[..., Any],
    model: Optional[str],
    base_url: Optional[str],
    reasoning_effort: Optional[str],
    batch_size: int,
    max_output_tokens: Optional[int] = 16384,
    on_batch: Optional[Callable[[str, str, str], None]] = None,
    api_key_env: Optional[str] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, list[dict[str, str]]]:
    """Classifies every digest, batch_size cases at a time. ``on_batch``,
    if given, is called with (system_prompt, user_prompt, raw_response)
    for each batch -- use it to persist forensics artifacts.

    ``reasoning_effort`` defaults to "low" (not None/implicit) when not
    given -- confirmed live (2026-09-16) that "medium"/implicit thinking
    on this classification prompt makes a Qwen3.5 backbone reason out loud
    in VISIBLE content for every one of several cases in a batch at once,
    exhausting max_output_tokens before ever reaching the requested JSON.
    This is a pure transcription/categorization task per case, same as
    the travel_mas_refactored Phase-3 composition call that motivated this
    default originally -- it doesn't need open-ended deliberation.

    ``api_key_env`` / ``extra_body`` are passed to ``call_llm`` only when set:
    the env var holding this call's key (e.g. an OpenRouter key while the
    run's OPENAI_API_KEY is another provider's) and request-body extras
    such as an OpenRouter provider pin."""
    effective_reasoning_effort = reasoning_effort or "low"
    all_labels: dict[str, list[dict[str, str]]] = {}
    for i in range(0, len(digests), batch_size):
        batch = digests[i : i + batch_size]
        case_ids = [d["case_id"] for d in batch]
        user_prompt = _build_batch_user_prompt(batch)

        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": effective_reasoning_effort,
        }
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if api_key_env:
            kwargs["api_key_env"] = api_key_env
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = llm_caller(**kwargs)
            raw_text = getattr(response, "content", None) or ""
        except Exception as exc:  # noqa: BLE001 - one bad batch must not sink the run
            print(f"[error_bucket_analyzer] batch {i}-{i+len(batch)} LLM call failed: {exc!r}", flush=True)
            raw_text = ""

        if on_batch:
            on_batch(_SYSTEM_PROMPT, user_prompt, raw_text)

        batch_labels = _parse_batch_response(raw_text, case_ids)
        all_labels.update(batch_labels)
    return all_labels


# ------------------------------------------------------------------ #
# Aggregation + report
# ------------------------------------------------------------------ #

def aggregate(
    cases: list[dict[str, Any]],
    labels_by_case: dict[str, list[dict[str, str]]],
    deterministic_by_case: dict[str, str],
) -> dict[str, Any]:
    n_total = len(cases)
    failing = [c for c in cases if not c.get("passed") or float(c.get("score", 0.0)) < 1.0]
    n_failing = len(failing)
    # Percentages are computed against how many failing cases were actually
    # CHECKED (deterministic + LLM-classified), not n_failing -- when
    # max_cases caps classification below n_failing (the common case for a
    # large batch: e.g. a 60-case eval with error_bucket_max_cases=15), a
    # bucket's true prevalence can never show as more than
    # n_checked/n_failing regardless of how many of the checked cases share
    # it, silently understating prevalence in exactly the batches large
    # enough to need the cap. Equals n_failing exactly (a no-op) whenever
    # nothing was capped.
    n_checked = len(deterministic_by_case) + len(labels_by_case)

    bucket_instances: dict[str, int] = {k: 0 for k in BUCKETS}
    bucket_cases: dict[str, set] = {k: set() for k in BUCKETS}
    examples: dict[str, list[dict[str, str]]] = {k: [] for k in BUCKETS}

    for case_id, bucket in deterministic_by_case.items():
        bucket_instances[bucket] += 1
        bucket_cases[bucket].add(case_id)

    for case_id, labels in labels_by_case.items():
        for item in labels:
            bucket = item["bucket"]
            bucket_instances[bucket] += 1
            bucket_cases[bucket].add(case_id)
            if len(examples[bucket]) < 3:
                examples[bucket].append({"case_id": case_id, "evidence": item["evidence"]})

    per_bucket = []
    for key in BUCKETS:
        n_cases = len(bucket_cases[key])
        per_bucket.append({
            "bucket": key,
            "description": BUCKETS[key],
            "instances": bucket_instances[key],
            "distinct_cases": n_cases,
            "pct_of_checked_cases": round(100.0 * n_cases / n_checked, 1) if n_checked else 0.0,
            "examples": examples[key],
        })
    per_bucket.sort(key=lambda r: r["distinct_cases"], reverse=True)

    return {
        "n_total_cases": n_total,
        "n_failing_cases": n_failing,
        "n_checked_cases": n_checked,
        "n_deterministic_budget_exceeded": len(deterministic_by_case),
        "n_llm_classified": len(labels_by_case),
        "buckets": per_bucket,
    }


def render_error_bucket_prevalence_for_prompt(agg: dict[str, Any]) -> str:
    """Compact prompt-section rendering of an ``analyze_cases()``/
    ``analyze()`` result, for ``AgentFeedback.error_bucket_prevalence`` --
    consumed by both block_suggester.py's ``_format_feedback_digest`` and
    agent_editor.py's ``_format_feedback``, so every block suggester and
    the editor sees the same capability-failure breakdown. Returns "" for
    an empty/missing aggregate (e.g. the gatherer had this disabled, or
    there were no failing cases at all) -- callers should skip the
    section entirely rather than showing an empty header."""
    if not agg or not agg.get("buckets"):
        return ""
    lines = [
        "## Error-bucket prevalence (LLM-classified root cause of each "
        "failing case, against the fixed taxonomy in strategies.md)",
        render_report(agg),
    ]
    return "\n".join(lines)


def render_report(agg: dict[str, Any]) -> str:
    n_failing = agg["n_failing_cases"]
    n_checked = agg.get("n_checked_cases", n_failing)
    header = (
        f"Cases: {agg['n_total_cases']} total, {n_failing} failing "
        f"({agg['n_deterministic_budget_exceeded']} pre-classified tool_calling_budget_exceeded, "
        f"{agg['n_llm_classified']} sent to the LLM for classification)"
    )
    if n_checked < n_failing:
        header += (
            f"\n(NOTE: only {n_checked} of {n_failing} failing cases were "
            f"checked -- error_bucket_max_cases capped classification. "
            f"Percentages below are of the {n_checked} CHECKED cases, not "
            f"all {n_failing} failing ones -- true prevalence across all "
            f"failing cases may be higher.)"
        )
    lines = [
        header,
        "",
        f"{'bucket':<30} {'cases':>7} {'% of checked':>13} {'instances':>10}",
    ]
    for row in agg["buckets"]:
        lines.append(
            f"{row['bucket']:<30} {row['distinct_cases']:>7} "
            f"{row['pct_of_checked_cases']:>12.1f}% {row['instances']:>10}"
        )
    lines.append("")
    for row in agg["buckets"]:
        if not row["examples"]:
            continue
        lines.append(f"--- {row['bucket']} examples ---")
        for ex in row["examples"]:
            lines.append(f"  case {ex['case_id']}: {ex['evidence']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Orchestration
# ------------------------------------------------------------------ #

def analyze_cases(
    cases: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    llm_caller: Callable[..., Any],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    batch_size: int = 4,
    max_cases: Optional[int] = None,
    artifacts_dir: Optional[Path] = None,
    source_label: str = "",
    api_key_env: Optional[str] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The core pipeline, over already-loaded ``cases`` (each shaped like
    one ``case_*.json`` -- case_id/passed/score/error/details) and
    ``events`` (already-parsed trace.jsonl events): pre-classify
    tool_calling_budget_exceeded deterministically, batch-classify the
    rest via the LLM, aggregate. ``max_cases`` caps how many failing cases
    are sent to the LLM (worst-scoring first) for cost control -- None
    (default) sends every failing case.

    Takes plain data rather than a directory so it can be called either
    from disk (see ``analyze()`` below, the standalone-CLI path) or
    directly from an in-memory ``EvaluationResult.per_case`` (see
    ``feedback_gatherer.py``'s round-level use, which has no on-disk
    case_*.json files to read in an HGM round -- only the cases already
    held in memory and that round's own trace.jsonl)."""
    events_by_case = group_tool_events_by_case(events)

    failing = [c for c in cases if not c.get("passed") or float(c.get("score", 0.0)) < 1.0]
    failing.sort(key=lambda c: float(c.get("score", 0.0)))

    deterministic_by_case: dict[str, str] = {}
    needs_llm: list[dict[str, Any]] = []
    for case in failing:
        case_id = str(case.get("case_id"))
        bucket = _classify_deterministic(case)
        if bucket:
            deterministic_by_case[case_id] = bucket
        else:
            needs_llm.append(case)

    if max_cases is not None:
        needs_llm = needs_llm[:max_cases]

    digests = [
        build_case_digest(c, events_by_case.get(str(c.get("case_id")), []))
        for c in needs_llm
    ]

    on_batch = None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        batch_counter = {"i": 0}

        def on_batch(system: str, user: str, response: str) -> None:  # noqa: ANN001
            batch_counter["i"] += 1
            (artifacts_dir / f"batch_{batch_counter['i']:03d}.txt").write_text(
                f"### SYSTEM\n{system}\n\n### USER\n{user}\n\n### RESPONSE\n{response}",
                encoding="utf-8",
            )

    labels_by_case = classify_batches(
        digests,
        llm_caller=llm_caller,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        batch_size=batch_size,
        on_batch=on_batch,
        api_key_env=api_key_env,
        extra_body=extra_body,
    ) if digests else {}

    agg = aggregate(cases, labels_by_case, deterministic_by_case)

    if artifacts_dir is not None:
        (artifacts_dir / "error_bucket_analysis.json").write_text(
            json.dumps(
                {
                    "source": source_label,
                    "deterministic": deterministic_by_case,
                    "llm_labels": labels_by_case,
                    "aggregate": agg,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    return agg


def analyze(
    logs_dir: Path,
    *,
    llm_caller: Callable[..., Any],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    batch_size: int = 4,
    max_cases: Optional[int] = None,
    artifacts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Standalone-CLI entry point: load every case_*.json + trace.jsonl
    directly from ``logs_dir`` on disk, then delegate to
    ``analyze_cases()``. See that function for the actual pipeline."""
    cases = load_cases(logs_dir)
    events = read_trace(logs_dir / "trace.jsonl")
    return analyze_cases(
        cases,
        events,
        llm_caller=llm_caller,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        batch_size=batch_size,
        max_cases=max_cases,
        artifacts_dir=artifacts_dir,
        source_label=str(logs_dir),
    )

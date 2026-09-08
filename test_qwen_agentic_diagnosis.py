"""Independent test: can Qwen3.5-122B-A10B (the same local vLLM endpoint
already used as block_suggester/failure_summarizer in the production runs)
independently derive the same class of root-cause diagnosis a human
manually derived this session for the curriculum's top-5 goals, if given
AGENTIC read-only access to exactly the same information used manually:
the harness's own current source, the frozen tool source, real execution
traces, and real per-case evaluation results (including violation
messages and structured/converted plans) -- but NEVER the scorer/grading
source code itself.

Tools given: list_dir, read_file (paginated), grep (for the 29MB
trace.jsonl) -- generic read primitives, not pre-aggregated helpers, so
this tests the model's OWN exploration/reasoning, not scripted shortcuts.
The system prompt also includes strategies.md verbatim (general + every
block section), matching what the real pipeline shows this same model as
block_suggester.

Usage: python3 test_qwen_agentic_diagnosis.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
from openai import OpenAI

REPO_ROOT = Path("/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
RUN_DIR = Path(
    "/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual/runs/"
    "20260906_225402_travel_mas_refactored_full_scale_openrouter_deepseek_editor_curriculum_X100Y180"
)
NODE_DIR = RUN_DIR / "round_027"

MODEL = "Qwen/Qwen3.5-122B-A10B"
BASE_URL = "http://gpu-aic-mv-02-st-p5-node-5:8000/v1"
MAX_TURNS = 100
# Configurable: how many times to retry the agentic run (each a full,
# independent MAX_TURNS-budget attempt) before falling back to the
# non-agentic single-shot baseline. See run_one_check_with_retries.
MAX_RETRIES = 5

# Read-only, ALIAS-based virtual filesystem -- harness + frozen tool
# source + this node's real eval logs/trace/eval_result. Aliases (not real
# paths) so the model never has to guess/construct the (long, opaque) real
# run directory name -- explicitly EXCLUDES
# projects/travel_mas_refactored/benchmark/ (the scorer/grading source).
ALIAS_ROOTS: dict[str, Path] = {
    "harness": NODE_DIR / "task_agent",
    "tools": REPO_ROOT / "projects/travel_mas_refactored/tools",
    "logs": NODE_DIR / "logs",
    "eval_result.json": NODE_DIR / "eval_result.json",
    "error_semantics.json": REPO_ROOT / "projects/travel_mas_refactored/adapter/error_semantics.json",
    "harness_error_semantics.json": REPO_ROOT / "projects/travel_mas_refactored/adapter/harness_error_semantics.json",
}


def _resolve_alias(path: str) -> Path | None:
    """Map an alias-rooted path (e.g. "harness/agents/sightseeing.py",
    "logs/trace.jsonl", "eval_result.json") to the real file, staying
    inside that alias's own root. Returns None for anything else."""
    path = path.strip().lstrip("/")
    for alias, root in ALIAS_ROOTS.items():
        if path == alias:
            return root
        prefix = alias + "/"
        if path.startswith(prefix):
            rel = path[len(prefix):]
            candidate = (root / rel).resolve()
            root_r = root.resolve()
            if candidate == root_r or root_r in candidate.parents:
                return candidate
    return None


def tool_list_dir(path: str) -> str:
    if path.strip().strip("/") in ("", "."):
        return "\n".join(sorted(ALIAS_ROOTS)) + "\n\n(these are the top-level aliases -- list_dir('harness') etc. to go deeper)"
    p = _resolve_alias(path)
    if p is None:
        return f"ERROR: {path!r} is not under a known alias root: {sorted(ALIAS_ROOTS)}"
    if not p.exists():
        return f"(not found: {path})"
    if p.is_file():
        return f"(this is a file, not a directory: {path})"
    entries = sorted(os.listdir(p))
    return "\n".join(entries) if entries else "(empty directory)"


def tool_read_file(path: str, offset: int = 0, limit: int = 200) -> str:
    """LINE-based offset/limit (offset = starting line number, 0-indexed;
    limit = max lines) -- deliberately matches grep's own line-number
    output so a match at "L241" can be read directly via offset=241,
    unlike a character-offset scheme which would land somewhere else
    entirely."""
    p = _resolve_alias(path)
    if p is None:
        return f"ERROR: {path!r} is not under a known alias root: {sorted(ALIAS_ROOTS)}"
    if not p.exists() or not p.is_file():
        return f"(file not found: {path})"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"ERROR reading {path}: {exc!r}"
    chunk_lines = lines[offset : offset + limit]
    chunk = "\n".join(f"L{offset + i}: {line}" for i, line in enumerate(chunk_lines))
    remaining = len(lines) - (offset + limit)
    if remaining > 0:
        chunk += f"\n\n[... {remaining} more lines -- call read_file again with offset={offset + limit} ...]"
    return chunk if chunk else "(empty file or offset past end)"


def tool_grep(path: str, pattern: str, max_matches: int = 12) -> str:
    p = _resolve_alias(path)
    if p is None:
        return f"ERROR: {path!r} is not under a known alias root: {sorted(ALIAS_ROOTS)}"
    if not p.exists() or not p.is_file():
        return f"(file not found: {path})"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex {pattern!r}: {exc!r}"
    matches = []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                m = rx.search(line)
                if m:
                    # Window CENTERED on the match, not the line's start --
                    # a line can be thousands of chars long (e.g. a JSON
                    # string value with escaped \n's holding an entire
                    # multi-day itinerary on one physical line), and the
                    # match can be far from its beginning.
                    start = max(0, m.start() - 150)
                    end = min(len(line), m.end() + 350)
                    prefix = "..." if start > 0 else ""
                    suffix = "..." if end < len(line) else ""
                    snippet = f"{prefix}{line[start:end].strip()}{suffix}"
                    matches.append(f"L{i} (char {m.start()}): {snippet}")
                    if len(matches) >= max_matches:
                        break
    except OSError as exc:
        return f"ERROR reading {path}: {exc!r}"
    if not matches:
        return "(no matches)"
    return "\n".join(matches)


TOOLS = [
    {
        "type": "function", "name": "list_dir",
        "description": "List a directory's contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "type": "function", "name": "read_file",
        "description": "Read a range of LINES from a text file (offset = starting line number, limit = max lines) -- offset uses the same line numbering grep reports.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "type": "function", "name": "grep",
        "description": "Search a (possibly large) file line-by-line for a regex pattern, returning up to max_matches matching lines with their line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "pattern": {"type": "string"},
                "max_matches": {"type": "integer"},
            },
            "required": ["path", "pattern"],
        },
    },
    {
        "type": "function", "name": "submit_diagnosis",
        "description": "Submit your final root-cause diagnosis for this check.",
        "parameters": {
            "type": "object",
            "properties": {
                "concrete_example": {"type": "string", "description": "A specific case_id and exact evidence you found for it."},
                "root_cause": {"type": "string", "description": "The precise mechanism causing this check to fail."},
                "correct_answer": {"type": "string", "description": "What the correct output should have been for the concrete example."},
                "attribution": {
                    "type": "string",
                    "description": "One of: harness_prompt_gap, model_noncompliance_with_correct_rule, tool_calling_error, harness_code_bug, missing_information",
                },
            },
            "required": ["concrete_example", "root_cause", "correct_answer", "attribution"],
        },
    },
]

TOOL_FNS = {"list_dir": tool_list_dir, "read_file": tool_read_file, "grep": tool_grep}

CHECKS = [
    ("commonsense:Time Feasibility:reasonable_transfer_time",
     "Fires when the time gap between two consecutive stops (hotel/attraction/meal/intercity-leg) doesn't match how long that trip would actually take by taxi -- either impossibly fast or implausibly slow with no explanation.",
     811),
    ("commonsense:Itinerary Structure:essential_meal_coverage",
     "Fires when a day doesn't have the right number of meals scheduled for its type -- a full sightseeing day needs both lunch and dinner with a real gap between them; a travel/transfer day's meal requirement depends on when the traveler arrives or departs.",
     571),
    ("commonsense:Activity Diversity:diverse_meal_options",
     "Fires as soon as the exact same restaurant is used for a meal more than once anywhere in the trip -- meals should come from different restaurants.",
     442),
    ("commonsense:Business Hours:attraction_visit_within_opening_hours",
     "Fires when an attraction is visited outside its real opening or closing hours, unless that attraction is open 24 hours.",
     428),
    ("commonsense:Activity Diversity:diverse_attraction_options",
     "Fires as soon as the exact same attraction is visited more than once anywhere in the trip.",
     430),
]

STRATEGIES_MD = (REPO_ROOT / "strategies.md").read_text(encoding="utf-8")


def build_system_prompt(check_name: str, description: str, count: int) -> str:
    return (
        "You are diagnosing why a travel-itinerary-planning multi-agent system's "
        "generated plans fail one specific automated check, using real data from "
        "an actual production run.\n\n"
        f"Check: `{check_name}`\n"
        f"Description: {description}\n"
        f"This check failed in {count} logged (case, round) instances across this run.\n\n"
        "You have READ-ONLY tools to explore:\n"
        "  - list_dir(path) / read_file(path, offset, limit) / grep(path, pattern)\n"
        "Paths are ALIAS-rooted (call list_dir('.') first to see the top-level "
        "aliases) -- there are exactly six:\n"
        "  - 'harness/...' -- the task agent's own current source (e.g. "
        "'harness/agents/sightseeing.py'); use list_dir('harness') and "
        "list_dir('harness/agents') to see the real file names.\n"
        "  - 'tools/...' -- frozen tool implementations the harness calls "
        "(e.g. 'tools/attraction.py').\n"
        "  - 'logs/...' -- per-case JSON files (e.g. 'logs/case_42.json'), "
        "each with a 'converted_plan' field (the exact structured plan that "
        "was scored, one day/activity/field per line -- PREFER this for "
        "inspecting day-by-day structure, it greps and pages cleanly) and a "
        "'raw_plan_text' field (the model's literal text output, stored as "
        "ONE long JSON-escaped line -- much harder to grep/page through; "
        "only use it when you specifically need the literal formatting, "
        "e.g. checking an exact prompt-required phrase); and "
        "'logs/trace.jsonl', real tool_call/tool_result/llm_call events per "
        "case -- it's 29MB, use grep on it, never read_file.\n"
        "  - 'eval_result.json' -- every case's score and, per dimension, "
        "each check's pass/fail and exact violation message.\n"
        "  - 'error_semantics.json' -- human-written descriptions of every "
        "TASK-level check the scorer evaluates (a {check_name: description} "
        "map): for each one, what it actually verifies and the specific "
        "condition that makes it fire, e.g. 'fires when a day (other than "
        "the last) doesn't clearly show where the traveler stayed that "
        "night'. Use this to understand a check's intended semantics before "
        "you look for evidence of it failing.\n"
        "  - 'harness_error_semantics.json' -- the same, but for HARNESS-"
        "level crash flags (boolean metadata the seed workflow itself sets "
        "when a case fails outright before any task-level check can even "
        "run, e.g. a stage producing no plan at all) rather than scored "
        "constraints.\n\n"
        "You do NOT have access to the scorer/grading source code -- only its "
        "already-computed textual outputs (scores, messages, converted plans). "
        "Never assume you know its exact internal logic; infer only from what "
        "you can observe.\n\n"
        "Diagnose the root cause as precisely as you can. Find and cite a "
        "CONCRETE real example (a specific case_id, with the exact evidence you "
        "found for it). State exactly what the correct output should have been "
        "for that example. Classify the root cause as one of: "
        "(a) harness_prompt_gap -- the harness's own rule is missing/ambiguous, "
        "(b) model_noncompliance_with_correct_rule -- the rule is already "
        "correct and unconditional but wasn't followed, "
        "(c) tool_calling_error -- a tool was called wrong or its result was "
        "misused, (d) harness_code_bug -- deterministic harness code (not the "
        "LLM) computed something wrong, or (e) missing_information -- the "
        "needed information genuinely wasn't available anywhere.\n\n"
        "Use your tools to gather REAL evidence before concluding -- do not "
        "guess. You have up to 100 tool-call turns -- 2-3 well-chosen concrete "
        "examples are enough; you do not need to exhaustively read every "
        "failing case. When ready, call submit_diagnosis.\n\n"
        "For reference, here is this project's curated strategy guidance "
        "(the same text shown to this model in its other role as this "
        "system's own improvement proposer):\n\n"
        f"{STRATEGIES_MD}"
    )


def run_one_check(check_name: str, description: str, count: int) -> dict:
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=0)
    system = build_system_prompt(check_name, description, count)
    history: list = [{"role": "user", "content": system + "\n\nBegin your investigation."}]
    turn_log = []

    for turn in range(MAX_TURNS):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=MODEL, input=history, tools=TOOLS,
                reasoning={"effort": "medium"}, max_output_tokens=8192,
            )
        except Exception as exc:
            turn_log.append(f"turn {turn}: api_error {exc!r}")
            break
        elapsed = time.time() - t0

        calls = [item for item in (response.output or []) if getattr(item, "type", None) == "function_call"]
        if not calls:
            # Capture what it said instead of calling a tool -- this is a
            # real, observed (non-deterministic) failure mode for this
            # agentic setup: the model sometimes stops with a free-text
            # answer instead of formally closing out via submit_diagnosis.
            turn_log.append(f"turn {turn}: no tool call ({elapsed:.1f}s) -- stopping")
            return {
                "turns": turn + 1, "diagnosis": None, "turn_log": turn_log,
                "free_text_stop": response.content,
            }

        for fc in calls:
            history.append({"type": "function_call", "call_id": fc.call_id, "name": fc.name, "arguments": fc.arguments})
        for fc in calls:
            try:
                args = json.loads(fc.arguments)
            except json.JSONDecodeError:
                args = {}
            if fc.name == "submit_diagnosis":
                print(f"    [{check_name}] turn {turn}: submit_diagnosis ({elapsed:.1f}s)", flush=True)
                return {"turns": turn + 1, "diagnosis": args, "turn_log": turn_log}
            fn = TOOL_FNS.get(fc.name)
            output = fn(**args) if fn else f"ERROR: unknown tool {fc.name!r}"
            print(f"    [{check_name}] turn {turn}: {fc.name}({args}) -> {output[:100]!r} ({elapsed:.1f}s)", flush=True)
            history.append({"type": "function_call_output", "call_id": fc.call_id, "output": output[:16000]})

    return {"turns": MAX_TURNS, "diagnosis": None, "turn_log": turn_log}


def _sample_failing_cases(check_name: str, max_examples: int = 5) -> list[dict]:
    """Plain-script curation (NOT a tool call) of a handful of real failing
    (case_id, message) pairs for this check, read directly from
    eval_result.json -- mirrors what the REAL production pipeline's
    project_metrics/failure-report digest already curates for
    block_suggester/failure_summarizer today (a few representative
    examples bundled directly into one prompt), since this is the
    NON-agentic baseline this check name is compared against."""
    p = NODE_DIR / "eval_result.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    examples = []
    for c in data.get("per_case", []):
        dd = (c.get("details") or {}).get("dimension_details") or {}
        for dim, info in dd.items():
            for chk in info.get("checks", []):
                if chk.get("name") == check_name.rsplit(":", 1)[-1] and chk.get("passed") is False:
                    examples.append({"case_id": c.get("case_id"), "message": chk.get("message")})
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


def run_one_check_nonagentic(check_name: str, description: str, count: int) -> dict:
    """The NON-agentic baseline: one single-shot call, no read_file/grep/
    list_dir tools -- only the curated check description + a handful of
    real failing-case messages bundled directly into the prompt (the same
    shape of context block_suggester/failure_summarizer actually get in
    production today), plus submit_diagnosis. Exactly one attempt, no
    internal retry -- this mirrors how those real components are actually
    called (one shot; retrying belongs one layer up, at the EXPAND level),
    so it's a fair "what would today's non-agentic setup produce" baseline."""
    examples = _sample_failing_cases(check_name)
    examples_text = "\n".join(
        f"- case {e['case_id']}: {e['message']}" for e in examples
    ) or "(no example messages available)"
    system = (
        "You are diagnosing why a travel-itinerary-planning multi-agent system's "
        "generated plans fail one specific automated check, using real data from "
        "an actual production run.\n\n"
        f"Check: `{check_name}`\n"
        f"Description: {description}\n"
        f"This check failed in {count} logged (case, round) instances across this run.\n\n"
        "Here are a few real examples of this check failing (case_id and the exact "
        "violation message):\n"
        f"{examples_text}\n\n"
        "You do NOT have any tools to explore further -- this is everything you get. "
        "Diagnose the root cause as precisely as you can from just this. State what "
        "the correct answer should have been, and classify the root cause as one of: "
        "harness_prompt_gap, model_noncompliance_with_correct_rule, "
        "tool_calling_error, harness_code_bug, missing_information. Call "
        "submit_diagnosis with your answer.\n\n"
        f"For reference, here is this project's curated strategy guidance:\n\n{STRATEGIES_MD}"
    )
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=0)
    try:
        response = client.responses.create(
            model=MODEL, input=[{"role": "user", "content": system}],
            tools=[TOOLS[-1]],  # submit_diagnosis only -- no exploration tools
            reasoning={"effort": "medium"}, max_output_tokens=8192,
        )
    except Exception as exc:
        return {"diagnosis": None, "error": repr(exc)}
    calls = [item for item in (response.output or []) if getattr(item, "type", None) == "function_call"]
    if not calls:
        return {"diagnosis": None, "free_text_stop": response.content}
    try:
        args = json.loads(calls[0].arguments)
    except json.JSONDecodeError:
        args = None
    return {"diagnosis": args}


def run_one_check_with_retries(
    check_name: str, description: str, count: int, max_retries: int = 5,
) -> dict:
    """Retry the agentic run up to max_retries times (the observed
    early-stop/turn-budget failures are non-deterministic sampling
    variance, not a systematic bug -- see this session's own retry
    investigation). If every agentic attempt fails to produce a
    diagnosis, fall back to the NON-agentic single-shot baseline rather
    than giving up entirely."""
    attempts = []
    for attempt in range(1, max_retries + 1):
        print(f"  [{check_name}] agentic attempt {attempt}/{max_retries}", flush=True)
        result = run_one_check(check_name, description, count)
        attempts.append({"attempt": attempt, "turns": result["turns"], "turn_log": result.get("turn_log", [])})
        if result.get("diagnosis") is not None:
            result["mode"] = "agentic"
            result["attempts"] = attempts
            return result

    print(f"  [{check_name}] all {max_retries} agentic attempts failed -- "
          "falling back to non-agentic single-shot", flush=True)
    fallback = run_one_check_nonagentic(check_name, description, count)
    fallback["mode"] = "nonagentic_fallback"
    fallback["attempts"] = attempts
    return fallback


if __name__ == "__main__":
    results = {}
    for check_name, description, count in CHECKS:
        print(f"=== {check_name} ===", flush=True)
        result = run_one_check(check_name, description, count)
        results[check_name] = result
        print(json.dumps(result, indent=2), flush=True)
        print(flush=True)

    out_path = REPO_ROOT / "qwen_agentic_diagnosis_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")

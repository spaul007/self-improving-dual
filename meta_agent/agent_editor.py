"""LLM-driven self-improvement step — mutates a task agent in one call.

The editor copies the base round into the out dir, then makes a SINGLE LLM
call that diagnoses what to change (from the current source + the previous
round's feedback + an optional manager-supplied steering ``context``) and
emits both a short strategy summary and the full file edits. It writes the
new files and runs validators; failed validation surfaces as
``EditResult.success=False`` with a list of error strings — the manager
decides what to do with that.

This replaced an older two-call design (a separate manager "strategy
proposal" call feeding the editor). Collapsing to one call removes a lossy
text hand-off: the same LLM context that diagnoses also writes the code.
``EvolutionStrategy`` is now an *output* (carried on ``EditResult.strategy``
for logging), not an input.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from . import source_context, verbose_log
from .case_tools import (
    LIST_CASES_TOOL,
    SHOW_CASE_TOOL,
    render_list_cases,
    render_show_case,
)
from .editor_validators import MUTABLE_DIRS, MUTABLE_FILES, is_excluded
from .error_bucket_analyzer import render_error_bucket_prevalence_for_prompt
from .failure_report import render_failure_report
from .feedback_gatherer import render_metrics
from .models import AgentFeedback, EditResult, EvolutionStrategy
from .registry import register
from .workspace import reset_workspace


class Validator(Protocol):
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]: ...


# Allowed values for an ``EvolutionStrategy.target_files`` entry. Mirrors the
# Literal in models.py. Lives here (the single home) so the coercion helpers
# below — and the managers, via import — share one definition.
_ALLOWED_TARGET_FILES = ("workflow.py", "tool_wrapper.py", "tools_schema.json")


def _coerce_target_files(
    value: Any, valid: Optional[Callable[[str], bool]] = None
) -> list[str]:
    """Coerce a model's ``target_files``-shaped value into a clean list[str].

    Models without strict schema enforcement (notably local vLLM-hosted
    open-weights models — caught gpt-oss-120b returning the bare string
    ``"workflow.py"`` on 2026-05-12) sometimes return a single string
    instead of a list. Unknown values are dropped; empty/None falls back to
    ``["workflow.py"]`` so a strategy summary always validates.

    ``valid`` overrides what counts as a "known" path. Default (``None``)
    preserves the exact legacy behavior — membership in
    ``_ALLOWED_TARGET_FILES`` — for projects using the include-list mutable
    surface. Projects using an exclude-list surface (see
    ``config.FrameworkConfig.mutable_exclude``) pass a predicate instead,
    since their editable paths aren't a fixed 3-name set.
    """
    if value is None or value == "":
        return ["workflow.py"]
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(v) for v in value if v]
    else:
        return ["workflow.py"]
    check = valid if valid is not None else (lambda c: c in _ALLOWED_TARGET_FILES)
    cleaned = [c for c in candidates if check(c)]
    return cleaned or ["workflow.py"]


def _coerce_str(value: Any) -> str:
    """Coerce a response field into a string — guards against a model
    returning the wrong scalar type (number/bool/None) for a text field."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


_MALFORMED_JSON_GOAL = "(malformed JSON: tool call arguments failed to parse)"


def fallback_strategy() -> EvolutionStrategy:
    """A constant placeholder ``EvolutionStrategy`` — used by managers when
    an ``EditResult`` carries no summary (defensive; ``apply`` always sets
    one, so this normally never fires)."""
    return EvolutionStrategy(
        target_files=["workflow.py"],
        optimization_goal="(editor produced no strategy summary)",
        proposed_changes="",
        rationale="",
    )


# Tool schema for the single self-improvement call. The model states what it
# is doing (optimization_goal / proposed_changes / rationale) AND does it
# (files) in one structured call.
SELF_IMPROVEMENT_TOOL: dict[str, Any] = {
    "name": "submit_self_improvement",
    "description": (
        "Diagnose the task agent from its current code and last round's "
        "feedback, then submit a targeted self-improvement: a short "
        "strategy summary plus the full file edits that implement it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "optimization_goal": {"type": "string"},
            "proposed_changes": {"type": "string"},
            "rationale": {"type": "string"},
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        "required": ["optimization_goal", "proposed_changes", "files"],
    },
}


# Tool schemas for the opt-in agentic self-improvement flow (see
# ``agentic_editing`` on ``AgentEditor.__init__`` and
# ``_self_improve_agentic``). One file's content per ``write_file`` call —
# never bundled — is the entire point: confirmed live this session that
# bundling every changed file's full content into one large tool call
# (SELF_IMPROVEMENT_TOOL's ``files`` array above) causes a real,
# reproducible malformed-JSON failure rate on large multi-file edits
# (~65% over 81 EXPANDs with DeepSeek v4), root-caused to that shape
# specifically and eliminated by switching to one-file-per-call. Kept
# deliberately model-agnostic (no diagnosis content, no provider-specific
# wording) so this works the same for GPT, Qwen, or DeepSeek editors.
# Line-based default for read_file, mirroring BlockSuggester's own
# proven pagination (meta_agent/block_suggester.py::_agentic_read_file).
# Bigger than that component's 200-line default because AgentEditor's
# read_file is the primary way the editor reads a WHOLE source file
# before editing it, and real post-edit files in this project routinely
# exceed 200 lines (confirmed: flight.py grew to 643 lines after one
# EXPAND) -- 2000 avoids forcing an extra paginated call for ordinary
# files while still bounding the common case.
_READ_FILE_DEFAULT_LINE_LIMIT = 2000
# Hard ceiling in CHARACTERS, applied on top of the line slice above --
# see _paginated_read's docstring for why the line cap alone isn't
# enough. 100_000 comfortably covers any real source file at the
# 2000-line default above.
_READ_FILE_MAX_CHARS = 100_000

AGENTIC_READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read one file's current content from the task agent workspace. "
        "Capped per call (see offset/limit) -- for a large file (e.g. "
        "internal_runs/trace.jsonl), use `grep` to search for a specific "
        "pattern instead of reading it whole."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "Starting line number (0-indexed)."},
            "limit": {"type": "integer", "description": "Max lines to return."},
        },
        "required": ["path"],
    },
}

AGENTIC_WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Submit ONE file's full new content. Call once per changed file — "
        "never bundle multiple files into one call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}

AGENTIC_RUN_VALIDATORS_TOOL: dict[str, Any] = {
    "name": "run_code_validators",
    "description": (
        "Run the project's real validator suite against your changes so "
        "far (syntax, imports, signatures, etc.)."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

AGENTIC_SUBMIT_SUMMARY_TOOL: dict[str, Any] = {
    "name": "submit_self_improvement_summary",
    "description": (
        "Finish: state what you changed and why, once all edits are "
        "written and validators pass."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "optimization_goal": {"type": "string"},
            "proposed_changes": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["optimization_goal", "proposed_changes", "rationale"],
    },
}

AGENTIC_GREP_TOOL: dict[str, Any] = {
    "name": "grep",
    "description": (
        "Search one file for a regex pattern (case-insensitive), returning "
        "a window centered on each match, not just the line's start -- "
        "safe for a very long single line. Same path rules as read_file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "max_matches": {"type": "integer"},
        },
        "required": ["path", "pattern"],
    },
}

AGENTIC_STR_REPLACE_FILE_TOOL: dict[str, Any] = {
    "name": "str_replace_file",
    "description": (
        "Targeted edit of a file you can write to: replace `old_str` "
        "(must occur EXACTLY ONCE in the file's current content) with "
        "`new_str`. Use this for a small, precise change instead of "
        "resending a whole file's content via write_file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
        "required": ["path", "old_str", "new_str"],
    },
}

# run_python: sandboxed offline execution against this node's own evaluated
# cases (AgentFeedback.eval_result.per_case -- generic across every project,
# see models.py::CaseResult). Adapted from experiment_harness_redesign.py's
# run_python (same AST import/name/attribute allow-list convention, same
# subprocess+rlimit isolation as meta_agent/evaluator.py::Evaluator._preexec),
# but sources its `train_data` from the feedback this call already has
# instead of a project-specific corpus of stored case files -- keeps this
# generic, no per-project wiring needed.
RUN_PYTHON_ALLOWED_IMPORTS = {
    "re", "json", "math", "datetime", "itertools", "collections", "statistics",
    "typing", "dataclasses", "functools", "textwrap", "string", "copy", "heapq",
    "bisect", "random", "decimal", "fractions", "operator", "enum", "abc",
    "__future__", "time", "unicodedata",
}
RUN_PYTHON_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "help",
    "exit", "quit", "memoryview",
}
RUN_PYTHON_BLOCKED_ATTRS = {
    "os", "sys", "subprocess", "builtins", "importlib", "shutil", "pathlib",
    "io", "socket", "system", "popen", "environ", "getenv", "putenv",
    "listdir", "walk", "scandir", "remove", "unlink", "rmdir", "rename",
    "execv", "fork", "spawn", "kill", "open", "read_text", "write_text",
    "read_bytes", "write_bytes", "modules", "load_module", "import_module",
}
# Defense in depth: the framework's own scorer/adapter/harness internals are
# never a legitimate run_python import even if somehow spelled around the
# AST import check above (e.g. via a dynamically-built module name).
RUN_PYTHON_LEAK_PATTERNS = [r"\bscorer\b", r"\badapter\b", r"\bmeta_agent\b", r"\bbenchmark\b"]

RUN_PYTHON_PRELUDE = r'''
import sys, json, types
_ws, _data, _codefile = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [_ws]
_d = json.load(open(_data, encoding="utf-8"))
_cases = _d["cases"]
_m = types.ModuleType("train_data")
_m.CASES = _cases
def _by_id(cid):
    for c in _cases:
        if str(c["case_id"]) == str(cid):
            return c
    return None
_m.by_id = _by_id
def _failing(label=None):
    return [c for c in _cases if not c["passed"]
            and (label is None or label in json.dumps(c.get("details", {}), default=str))]
_m.failing = _failing
sys.modules["train_data"] = _m
_code = open(_codefile, encoding="utf-8").read()
exec(compile(_code, "<run_python>", "exec"), {"__name__": "__main__"})
'''


def _run_python_problems(code: str, ws_modules: set) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    allowed = RUN_PYTHON_ALLOWED_IMPORTS | ws_modules | {"train_data"}
    probs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    probs.append(
                        f"import {a.name!r} not allowed (allowed: stdlib "
                        "data modules, train_data, your workspace modules)"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in allowed:
                probs.append(f"from {node.module!r} import ... not allowed")
        elif isinstance(node, ast.Name) and node.id in RUN_PYTHON_BLOCKED_NAMES:
            if node.id == "open":
                # Confirmed live: the model reached for open() to read a
                # file's content (e.g. internal_runs/case_N.json) from
                # inside run_python -- a reasonable instinct, but
                # run_python has no file I/O at all. Bare "name 'open'
                # not allowed" gives no path forward; say what to use
                # instead rather than just what's blocked.
                probs.append(
                    "name 'open' not allowed -- run_python has no file "
                    "I/O. Use the read_file tool to read a file's "
                    "content instead (e.g. read_file('internal_runs/"
                    "case_<id>.json')), or train_data.by_id(cid)/"
                    "train_data.failing(label) for the case data already "
                    "loaded here."
                )
            else:
                probs.append(f"name {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in RUN_PYTHON_BLOCKED_ATTRS
        ):
            probs.append(f"attribute {node.attr!r} not allowed")
    for pat in RUN_PYTHON_LEAK_PATTERNS:
        m = re.search(pat, code, re.IGNORECASE)
        if m:
            probs.append(f"forbidden token {m.group(0)!r}")
    return sorted(set(probs))[:6]


def _run_python_preexec() -> None:
    # POSIX-only resource caps, same convention as
    # evaluator.py::Evaluator._preexec.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1GB
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except Exception:
        pass


AGENTIC_RUN_PYTHON_TOOL: dict[str, Any] = {
    "name": "run_python",
    "description": (
        "Run your OWN Python code offline (no LLM, no network) and see its "
        "stdout/stderr. `import train_data` gives CASES (this node's own "
        "evaluated cases: case_id, passed, score, error, details -- the "
        "same per-case data list_cases/show_case expose), by_id(cid), and "
        "failing(label_substring). You may also import your own workspace "
        "modules (e.g. `from agents.flight import ...`) and stdlib data "
        "modules (re, json, math, itertools, collections, statistics, "
        "...). No file/env/network access, and no importing this "
        "framework's own scorer/adapter/meta_agent internals. Use it to "
        "test a hypothesis against real evaluated cases before acting on it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "timeout_s": {"type": "integer"},
        },
        "required": ["code"],
    },
}

# Opt-in (only offered when the editor was constructed with an `evaluator`
# -- see AgentEditor.__init__ and config.py's editor_injections, which only
# sets this when AgentEditor is the one declaring the param, matching the
# "only inject params the constructor actually declares" convention used
# for validators' evaluator/benchmark_dir injection). Mirrors
# experiment_harness_redesign.py's own evaluate_variant: runs the CURRENT
# workspace through the real evaluator, full TRAIN set by default (the real
# accept/reject-quality signal) or a cheaper case_ids subset. Rate-limited
# per EXPAND (see `evaluate_variant_max_calls`) since each call is a real
# subprocess run of every case through the task agent -- genuine wall-clock
# and API cost, independent of whatever evaluation the calling HGM manager
# does of its own accord after this EXPAND returns.
AGENTIC_EVALUATE_VARIANT_TOOL: dict[str, Any] = {
    "name": "evaluate_variant",
    "description": (
        "Run your CURRENT workspace (everything written so far via "
        "write_file/str_replace_file) through the REAL evaluator and get "
        "back each case's score and pass/fail. Without case_ids: "
        "evaluates the FULL TRAIN set (a genuine run of the task agent "
        "on every case -- not instant). Pass case_ids (specific TRAIN "
        "case ids) to evaluate just that subset instead. Calls are "
        "limited per EXPAND."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "case_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "optional: specific TRAIN case ids to evaluate instead "
                    "of the full set"
                ),
            },
        },
    },
}

_AGENTIC_TOOLS: list[dict[str, Any]] = [
    AGENTIC_READ_FILE_TOOL,
    AGENTIC_WRITE_FILE_TOOL,
    AGENTIC_STR_REPLACE_FILE_TOOL,
    AGENTIC_GREP_TOOL,
    AGENTIC_RUN_VALIDATORS_TOOL,
    AGENTIC_RUN_PYTHON_TOOL,
    LIST_CASES_TOOL,
    SHOW_CASE_TOOL,
    AGENTIC_SUBMIT_SUMMARY_TOOL,
]

# Every agentic tool AgentEditor can offer, by name -- used to resolve the
# configurable `agentic_tools` constructor param (see __init__) into the
# actual schema list sent to the model. `evaluate_variant` lives here too
# even though it's also gated separately by `self.evaluator is not None`
# (see _self_improve_agentic) -- both conditions must hold for it to
# actually be offered.
_AGENTIC_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {
    "read_file": AGENTIC_READ_FILE_TOOL,
    "write_file": AGENTIC_WRITE_FILE_TOOL,
    "str_replace_file": AGENTIC_STR_REPLACE_FILE_TOOL,
    "grep": AGENTIC_GREP_TOOL,
    "run_code_validators": AGENTIC_RUN_VALIDATORS_TOOL,
    "run_python": AGENTIC_RUN_PYTHON_TOOL,
    "list_cases": LIST_CASES_TOOL,
    "show_case": SHOW_CASE_TOOL,
    "submit_self_improvement_summary": AGENTIC_SUBMIT_SUMMARY_TOOL,
    "evaluate_variant": AGENTIC_EVALUATE_VARIANT_TOOL,
}

# The ORIGINAL tool set, from before grep/str_replace_file/run_python/
# list_cases/show_case/evaluate_variant were added. This is the default
# (see __init__'s `agentic_tools` param) so every existing config's
# behavior is unchanged unless it explicitly opts into the newer tools --
# pass e.g. agentic_tools: ["read_file", "write_file", "run_code_validators",
# "submit_self_improvement_summary", "run_python", "evaluate_variant"] in
# editor.config to add specific ones back.
_DEFAULT_AGENTIC_TOOL_NAMES: tuple[str, ...] = (
    "read_file", "write_file", "run_code_validators", "submit_self_improvement_summary",
)

_AGENTIC_TURN_BUDGET_GOAL = "(editor exceeded agentic turn budget without submitting a summary)"
# Distinct from the above: the model itself stopped emitting tool calls
# (a totally empty turn -- no content, no tool_calls) well before
# agentic_max_turns was reached, rather than genuinely running out of
# budget. See `stopped_early` in _self_improve_agentic.
_AGENTIC_EMPTY_RESPONSE_GOAL = "(editor's model stopped responding -- no tool call -- before submitting a summary)"
_AGENTIC_MALFORMED_SUMMARY_GOAL = "(summary call had malformed JSON; edits below were still applied)"
_AGENTIC_LLM_CALL_FAILED_GOAL_PREFIX = "(editor's LLM call failed mid-conversation"


def run_python_sandboxed(
    agent_dir: Path,
    out_dir: Path,
    per_case: list[Any],
    code: str,
    timeout_s: Optional[int],
) -> str:
    """Shared backing implementation for the ``run_python`` agentic tool,
    used by both ``AgentEditor`` (its own in-progress edit) and
    ``BlockSuggester`` (the parent node's already-evaluated cases, for
    falsifying a candidate diagnosis before reporting it — see
    ``BlockSuggester``'s own agentic closing prompt). Runs ``code`` in a
    subprocess with ``train_data`` (built from ``per_case``) injected via
    ``RUN_PYTHON_PRELUDE``, ``agent_dir`` on ``sys.path`` so the model's
    own workspace modules are importable, and no network/file access
    beyond that (the AST checks in ``_run_python_problems`` already ran
    before this is called). Scratch files live under
    ``out_dir/_run_python_scratch``, a sibling of ``task_agent/`` --
    outside every path ``AgentEditor._is_path_allowed`` would ever show or
    allow writing to, so it can never collide with a real edit (and
    ``BlockSuggester`` never writes at all, so it's simply a scratch dir
    there)."""
    scratch = out_dir / "_run_python_scratch"
    scratch.mkdir(exist_ok=True)
    cases = []
    for c in per_case:
        cases.append({
            "case_id": c.case_id,
            "passed": c.passed,
            "score": c.score,
            "error": c.error,
            "details": c.details,
        })
    data_path = scratch / "cases.json"
    data_path.write_text(json.dumps({"cases": cases}, default=str), encoding="utf-8")
    prelude_path = scratch / "prelude.py"
    prelude_path.write_text(RUN_PYTHON_PRELUDE, encoding="utf-8")
    codefile = scratch / f"user_{uuid.uuid4().hex}.py"
    codefile.write_text(code, encoding="utf-8")
    try:
        timeout = max(5, min(int(timeout_s or 60), 120))
    except (TypeError, ValueError):
        timeout = 60
    env = {
        "PATH": os.environ.get("PATH", ""),
        # Offline: any accidental LLM call fails fast rather than hanging
        # or reaching a real endpoint.
        "LLM_BASE_URL": "http://127.0.0.1:9/v1",
        "OPENAI_API_KEY": "EMPTY",
    }
    try:
        r = subprocess.run(
            [sys.executable, "-B", str(prelude_path), str(agent_dir), str(data_path), str(codefile)],
            cwd=scratch, env=env, capture_output=True, text=True, timeout=timeout,
            preexec_fn=_run_python_preexec if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        return f"run_python TIMED OUT after {timeout}s"
    out = r.stdout[-9000:] if len(r.stdout) > 9000 else r.stdout
    err = r.stderr[-2500:]
    return f"exit code {r.returncode}\n--- stdout ---\n{out}" + (
        f"\n--- stderr ---\n{err}" if err.strip() else ""
    )


@register("editor", "default")
class AgentEditor:
    MUTABLE_FILES = MUTABLE_FILES
    MUTABLE_DIRS = MUTABLE_DIRS

    def __init__(
        self,
        llm_caller: Callable[..., object],
        validators: Iterable[Validator],
        *,
        max_attempts: int = 2,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        # Static project context injected by build_components (read from the
        # project folder by convention). tools_source + db_schema are shown
        # in BOTH modes (the agent's own tooling/data shape, not ground
        # truth); scorer_source is shown only when eval_visibility == "whitebox".
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
        # Directory backing read_file/grep's "tools/<rel>" alias (the
        # project's own tool implementation, e.g.
        # projects/<p>/tools/roadroute.py) -- populated by build_components
        # whenever the project has a tools/ dir, same convention as
        # tools_source below, independent of whether the alias is actually
        # enabled (see tools_dir_alias_enabled). None when the project has
        # no tools/ dir at all.
        tools_dir: Optional[Path] = None,
        # Opt-in gate for the alias above -- False (default) is ZERO
        # behavior change for every existing config. Confirmed live this
        # is a real, legitimate gap worth A/B testing, not just adding
        # unconditionally: tools_source above is a single upfront bundle
        # capped at config._SOURCE_BUNDLE_CAP (24000 chars) and
        # concatenated alphabetically, so a project with enough tool files
        # (travel_mas_refactored: 34279 chars across 9 files) silently
        # omits whichever ones sort last -- the editor correctly reasoning
        # that a tool it's trying to optimize against is part of its own
        # harness, and reaching for it, found it simply wasn't there
        # (round_005: roadroute.py was the one omitted; the editor then
        # tried importing it directly inside run_python, which rejected
        # it, since that sandbox is for the task agent's OWN workspace
        # modules, not project-level tool implementations). Set True to
        # give read_file/grep on-demand, paginated access to any tools/
        # file actually present -- kept behind this flag (rather than
        # always-on whenever tools_dir is set) specifically so a config
        # can run the same project both ways for an ablation.
        tools_dir_alias_enabled: bool = False,
        # `None` (default) = legacy include-list mode (MUTABLE_FILES/
        # MUTABLE_DIRS). A list = exclude-list mode: everything under the
        # seed dir is editable except these paths. See
        # config.FrameworkConfig.mutable_exclude.
        mutable_exclude: Optional[list[str]] = None,
        # Caps runaway generation (e.g. a reasoning-model repetition loop) --
        # 16384 matches the task_agent default. None disables the cap.
        max_output_tokens: Optional[int] = 16384,
        # Opt-in: replace the single bundled submit_self_improvement call
        # (all changed files' full content in one `files` array) with a
        # multi-turn loop of read_file/write_file/run_code_validators calls,
        # one file's content per call. False (default) reproduces today's
        # exact behavior for every existing config. See
        # ``_self_improve_agentic`` and the module-level AGENTIC_*_TOOL
        # schemas above.
        agentic_editing: bool = False,
        # Bounds the agentic loop (one LLM round-trip per turn). 20 is
        # generous relative to what real trials needed (typically 3-6 turns
        # even for a genuine 4-file fix, confirmed live this session).
        agentic_max_turns: int = 20,
        # Retries, WITHOUT consuming a turn slot, when the model returns a
        # genuinely empty response -- no content AND no tool_calls, and no
        # exception either (that's the pre-existing, separately-handled
        # LLM-call-failure path just above). Confirmed live: this can
        # happen mid-session after a long, substantive, clearly-not-done
        # turn (the model's own text ended with "let me test ... now" --
        # it was mid-flow, not winding down) -- a bare API/generation
        # glitch, not reasoning-stripped-to-nothing (that turn's own text
        # proves ordinary content comes through fine in the same
        # session). Retrying the IDENTICAL call relies on ordinary
        # sampling randomness to recover, the same "repeating a run is a
        # legitimate way to tell signal from noise" principle this file's
        # own evaluate_variant guidance already states. 2 is small enough
        # that a persistently-broken connection still gives up quickly and
        # falls through to the existing _AGENTIC_EMPTY_RESPONSE_GOAL path
        # unchanged.
        max_empty_response_retries: int = 2,
        # Backs the opt-in evaluate_variant agentic tool. All three are
        # injected by config.py's editor_injections (evaluator/benchmark_dir
        # already exist there for validators; train_case_ids is the same
        # split the manager's own EVALUATE step and the gatherer use).
        # evaluator=None (default -- unchanged for every config that
        # doesn't wire it, including every non-agentic one) disables the
        # tool entirely: it's never added to the agentic tool list.
        evaluator: Optional[Any] = None,
        benchmark_dir: Optional[Path] = None,
        train_case_ids: Optional[list[str]] = None,
        # Hard cap on evaluate_variant calls PER EXPAND, independent of
        # agentic_max_turns -- each call is a real subprocess run of every
        # requested case through the task agent (genuine wall-clock + API
        # cost), so the turn budget alone is not a cost control.
        evaluate_variant_max_calls: int = 3,
        # Which agentic tools to offer, by name (see _AGENTIC_TOOLS_BY_NAME
        # for the full set). None (default) resolves to
        # _DEFAULT_AGENTIC_TOOL_NAMES -- the ORIGINAL tool set, from before
        # grep/str_replace_file/run_python/list_cases/show_case/
        # evaluate_variant were added -- so every existing config's
        # behavior is unchanged unless it explicitly opts into more.
        # evaluate_variant is additionally gated by `evaluator is not
        # None` regardless of whether it's listed here.
        agentic_tools: Optional[list[str]] = None,
    ) -> None:
        self.llm = llm_caller
        self.validators = list(validators)
        self.max_attempts = max_attempts
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.tools_source = tools_source
        self.db_schema = db_schema
        self.scorer_source = scorer_source
        self.tools_dir = Path(tools_dir) if tools_dir is not None else None
        self.tools_dir_alias_enabled = tools_dir_alias_enabled
        self.mutable_exclude = mutable_exclude
        self.max_output_tokens = max_output_tokens
        self.agentic_editing = agentic_editing
        self.agentic_max_turns = agentic_max_turns
        self.max_empty_response_retries = max_empty_response_retries
        self.evaluator = evaluator
        self.benchmark_dir = benchmark_dir
        self.train_case_ids = train_case_ids
        self.evaluate_variant_max_calls = evaluate_variant_max_calls
        if agentic_tools is None:
            self.agentic_tool_names = list(_DEFAULT_AGENTIC_TOOL_NAMES)
        else:
            unknown = [n for n in agentic_tools if n not in _AGENTIC_TOOLS_BY_NAME]
            if unknown:
                raise ValueError(
                    f"agentic_tools: unknown tool name(s) {unknown!r} -- "
                    f"must be a subset of {sorted(_AGENTIC_TOOLS_BY_NAME)}"
                )
            self.agentic_tool_names = list(agentic_tools)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(
        self,
        feedback: Optional[AgentFeedback],
        base_dir: Path,
        out_dir: Path,
        *,
        context: Optional[str] = None,
        # True only when the caller's manager actually produced a
        # block-scoped suggestion (see block_suggester.py) for THIS
        # apply() call -- see hgm.py::HGMManager._last_suggestion_produced.
        # Default False reproduces today's exact behavior for every
        # caller that omits it: HGMDualManager's Stage B and
        # HillClimbingManager never pass this at all, since neither path
        # ever computes a suggestion.
        has_suggestion: bool = False,
        # Path prefixes (relative to task_agent/) this EXPAND may write, e.g.
        # ["skills/"] for the ``skills`` block (hgm.py block_edit_scopes).
        # None (default): unchanged -- any editable path. Enforced three ways:
        # write/edit tools refuse out-of-scope paths, the prompt states the
        # scope, and the finished diff is re-checked before validators run.
        edit_scope: Optional[list[str]] = None,
    ) -> EditResult:
        """Produce one self-improvement of the agent in ``base_dir``.

        Copies ``base_dir/task_agent`` into ``out_dir``, then runs the
        single-call self-improvement LLM step (up to ``max_attempts`` times,
        re-diagnosing against validator errors on each retry). ``context`` is
        optional manager-supplied steering text (history, lineage, scores) —
        the editor reads the actual code itself, so ``context`` carries only
        cheap signal, never source.

        ``has_suggestion`` (default False) tells the editor's own feedback
        digest that a block-scoped diagnosis (see block_suggester.py) was
        already produced for this call -- when True, ``project_metrics`` is
        omitted from ``_format_feedback`` (the suggester already showed it
        the same raw numbers, and the suggestion text itself is already in
        ``context``); ``failure_report`` is always shown regardless.

        Returns ``EditResult``; ``.strategy`` carries the editor's emitted
        summary (the last attempt's, on failure).
        """
        self._edit_scope = list(edit_scope) if edit_scope else None
        try:
            return self._apply(feedback, base_dir, out_dir, context=context,
                               has_suggestion=has_suggestion)
        finally:
            self._edit_scope = None

    def _apply(
        self,
        feedback: Optional[AgentFeedback],
        base_dir: Path,
        out_dir: Path,
        *,
        context: Optional[str],
        has_suggestion: bool,
    ) -> EditResult:
        self._copy_workspace(base_dir, out_dir)

        attempt_errors: list[str] = []
        last_strategy: Optional[EvolutionStrategy] = None
        self_improve = (
            self._self_improve_agentic if self.agentic_editing else self._self_improve
        )
        for attempt in range(1, self.max_attempts + 1):
            strategy, files = self_improve(
                out_dir=out_dir,
                base_dir=base_dir,
                feedback=feedback,
                context=context,
                has_suggestion=has_suggestion,
                prior_errors=attempt_errors,
                attempt=attempt,
            )
            last_strategy = strategy
            if not files:
                if strategy.optimization_goal == _MALFORMED_JSON_GOAL:
                    # Give the retry something actionable instead of a
                    # generic "no edits" message -- the model's own JSON
                    # was broken, not its diagnosis or code.
                    attempt_errors = [
                        "Your previous response's tool call arguments were "
                        "not valid JSON and could not be parsed at all, so "
                        "none of it (not even the diagnosis) could be used. "
                        "Make sure you return a valid JSON object: "
                        "double-check that every string value — especially "
                        "each file's `content` — has its quotes, "
                        "backslashes, and newlines properly escaped."
                    ]
                else:
                    attempt_errors = ["editor returned no file edits"]
                continue

            written, write_errors = self._write_edits(out_dir, files)
            if write_errors:
                attempt_errors = write_errors
                continue

            errors = self._scope_violations(out_dir, base_dir) + self._run_validators(
                out_dir, base_dir
            )
            if not errors:
                return EditResult(
                    success=True, edited_files=written, strategy=strategy
                )
            attempt_errors = errors
            # Reset the workspace and try again with the validator feedback.
            self._copy_workspace(base_dir, out_dir)

        return EditResult(
            success=False, errors=attempt_errors, strategy=last_strategy
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _copy_workspace(self, base_dir: Path, out_dir: Path) -> None:
        reset_workspace(base_dir, out_dir)

    def _self_improve(
        self,
        *,
        out_dir: Path,
        # Unused here; shared signature with _self_improve_agentic (see
        # apply()'s dispatch) -- defaulted so direct-call test sites that
        # predate this parameter keep working unchanged.
        base_dir: Optional[Path] = None,
        feedback: Optional[AgentFeedback],
        context: Optional[str],
        prior_errors: list[str],
        attempt: int = 1,
        has_suggestion: bool = False,
    ) -> tuple[EvolutionStrategy, list[dict]]:
        """One self-improvement LLM call: diagnose + edit.

        Builds the prompt from the hard rules, the optional steering
        ``context``, the previous round's ``feedback`` digest, the agent's
        current mutable sources, and any ``prior_errors`` from a failed
        validation attempt. Returns ``(EvolutionStrategy, files)`` where
        ``files`` is the raw ``[{path, content}]`` payload to write.
        """
        agent_dir = out_dir / "task_agent"
        current = self._read_mutable_sources(agent_dir)

        if self.mutable_exclude is not None:
            excl = ", ".join(sorted(self.mutable_exclude)) or "(nothing)"
            system = (
                "You are the self-improvement module of a self-evolving agent. "
                "Diagnose what to change from the feedback and the current code, "
                "then call `submit_self_improvement`.\n"
                "First understand the task: read the agent's own code, and "
                "(when provided below) the tool implementations, database "
                "schema, and evaluation scoring code — together they show what "
                "the system does, what the data looks like, and how output is "
                "graded. Target edits at the failures that most affect the score.\n"
                "You may edit ANY file in the task_agent workspace EXCEPT:\n"
                f"  - {excl}\n"
                "This includes prompt/config text AND the actual orchestration "
                "code (workflow logic, tool implementations, retry/decision "
                "branches, control flow) — not just one or the other. If the "
                "failure pattern points to a structural or logical problem "
                "(e.g. how evidence is gathered, when a retry fires, how a "
                "decision is made), change the code that implements it; don't "
                "default to a prompt-only wording tweak just because it is the "
                "easiest edit to make. Pick whichever kind of change actually "
                "fixes the diagnosed failure, or both together.\n\n"
                "Hard rules:\n"
                "  1. The workspace MUST keep exposing "
                "`def run_task(task: Task) -> AgentOutput` from its top-level "
                "`workflow.py` (the single arg must be named `task` — the "
                "validator enforces this); it may delegate to any other file "
                "you're allowed to edit.\n\n"
                "If this workspace uses, or you introduce, a `tool_wrapper.py` + "
                "`tools_schema.json` + `mutable_tools/` pattern (the framework's "
                "generic tool-calling convention), these additional rules apply "
                "to that pattern specifically — irrelevant otherwise:\n"
                "  2. workflow.py may import only: "
                "platform_core.llm_wrapper.call_llm, platform_core.runner "
                "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
                "  3. tool_wrapper.py may import only: platform_core.tools, "
                "mutable_tools.*, plus stdlib.\n"
                "  4. Mutable tools (mutable_tools/*.py) may import only: "
                "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
                "  5. Reach capabilities only via `platform_core.tools`: "
                "`call_tool(name, **kwargs)` for immutable tools, and "
                "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
                "invoke a mutable tool's `run()` directly — routing through "
                "`call_mutable_tool` keeps its calls recorded in the trace.\n"
                "  6. tools_schema.json: every entry's `name` must be backed by "
                "either an immutable tool OR a `mutable_tools/<name>.py` file. "
                "No collisions between immutable and mutable names.\n\n"
                "  7. Instrument your edits for the behavior summarizer. When "
                "you add a verifier, helper, or decision branch, call "
                "`platform_core.trace.log(label='your_label', "
                "verdict='pass'|'fail'|'skip', name='specific_check_name', "
                "**context)` at the decision point. Conventions: pick a stable "
                "`label` (e.g. 'verifier_fired', 'decision_branch'); use `name` "
                "to disambiguate within a label; set `verdict` to `pass`, "
                "`fail`, or `skip`. The summarizer cross-tabs these logs "
                "against case outcomes so the next editor knows which of your "
                "additions helped, which didn't, and why. Skip instrumentation "
                "only for trivial edits (renames, docstrings).\n\n"
                "  8. Double-check your own rationale before submitting. Every "
                "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
                "is malformed\", \"case X failed because of Y\") must be verified "
                "against the actual current source shown above or the "
                "feedback/metrics below — re-read the specific line or field "
                "you are citing and confirm it really says what you're about "
                "to claim. Do not invent a plausible-sounding diagnosis you "
                "have not checked. If a claim doesn't hold up on re-reading, "
                "drop it or soften it (e.g. \"possibly\", \"this may "
                "contribute\") rather than stating it as settled fact — a "
                "correct edit with an honest, hedged rationale is better than "
                "a confident but unverified one.\n\n"
                "  9. If the steering context above includes a specific "
                "suggestion or recommendation for what to change, state in "
                "your `rationale` whether your edit follows it. If your "
                "edit departs from it in any way — a different target, a "
                "different mechanism, or scope beyond what it proposed — "
                "say so explicitly: what you changed instead and why. "
                "Silently doing something else is not acceptable; an "
                "honest \"I deviated from the suggestion because X\" is.\n\n"
                '  10. If the feedback below includes an error-bucket prevalence table, use it before deciding HOW to implement your fix: which bucket(s) dominate the failing cases is itself evidence of this backbone\'s actual capability tier (low/medium/high), not just a pointer to which case to cite. See strategies.md\'s "Reading the error-bucket prevalence table" section for how to infer the tier from the table\'s shape, and its per-(bucket, tier) strategy table for what KIND of fix that combination calls for -- a code-level fix, a prompt/instruction change, or a lightweight verifier. A high tool_omission/tool_calling_budget_exceeded rate is evidence this backbone needs the more aggressive, code-level fix regardless of how clean a prompt-only edit might look; a low rate with mostly constraint_misreading/apply_info_incorrectly is evidence a lighter prompt/verifier-level fix is more proportionate.'
                "\n\n"
                "Call `submit_self_improvement` with a one-line optimization_goal, "
                "a proposed_changes summary, a rationale, and the `files` payload. "
                "Each file is the FULL replacement content — do not produce diffs. "
                "Omit files you do not change."
            )
        else:
            system = (
                "You are the self-improvement module of a self-evolving agent. "
                "Diagnose what to change from the feedback and the current code, "
                "then call `submit_self_improvement`.\n"
                "First understand the task: read the agent's system prompt in "
                "workflow.py, and (when provided below) the tool implementations, "
                "database schema, and evaluation scoring code — together they show "
                "what each tool does, what the data looks like, and how output is "
                "graded. Target edits at the failures that most affect the score.\n"
                "You may only modify these "
                "files in the task_agent workspace:\n"
                f"  - {', '.join(sorted(MUTABLE_FILES))}\n"
                f"  - any *.py file under mutable_tools/\n\n"
                "Hard rules:\n"
                "  1. workflow.py MUST define "
                "`def run_task(task: Task) -> AgentOutput`. The single arg must "
                "be named `task` (the validator enforces this).\n"
                "  2. workflow.py may import only: "
                "platform_core.llm_wrapper.call_llm, platform_core.runner "
                "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
                "  3. tool_wrapper.py may import only: platform_core.tools, "
                "mutable_tools.*, plus stdlib.\n"
                "  4. Mutable tools (mutable_tools/*.py) may import only: "
                "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
                "  5. Reach capabilities only via `platform_core.tools`: "
                "`call_tool(name, **kwargs)` for immutable tools, and "
                "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
                "invoke a mutable tool's `run()` directly — routing through "
                "`call_mutable_tool` keeps its calls recorded in the trace.\n"
                "  6. tools_schema.json: every entry's `name` must be backed by "
                "either an immutable tool OR a `mutable_tools/<name>.py` file. "
                "No collisions between immutable and mutable names.\n"
                "  7. Instrument your edits for the behavior summarizer. When you "
                "add a verifier, helper, or decision branch in workflow.py or a "
                "mutable_tools/*.py file, call "
                "`platform_core.trace.log(label='your_label', verdict='pass'|'fail'|'skip', "
                "name='specific_check_name', **context)` at the decision point. "
                "Conventions: pick a stable `label` (e.g. 'verifier_fired', "
                "'decision_branch'); use `name` to disambiguate within a label; "
                "set `verdict` to `pass`, `fail`, or `skip`. The summarizer "
                "cross-tabs these logs against case outcomes so the next editor "
                "knows which of your additions helped, which didn't, and why. "
                "Skip instrumentation only for trivial edits (renames, docstrings).\n\n"
                "  8. Double-check your own rationale before submitting. Every "
                "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
                "is malformed\", \"case X failed because of Y\") must be verified "
                "against the actual current source shown above or the "
                "feedback/metrics below — re-read the specific line or field "
                "you are citing and confirm it really says what you're about "
                "to claim. Do not invent a plausible-sounding diagnosis you "
                "have not checked. If a claim doesn't hold up on re-reading, "
                "drop it or soften it (e.g. \"possibly\", \"this may "
                "contribute\") rather than stating it as settled fact — a "
                "correct edit with an honest, hedged rationale is better than "
                "a confident but unverified one.\n\n"
                "  9. If the steering context above includes a specific "
                "suggestion or recommendation for what to change, state in "
                "your `rationale` whether your edit follows it. If your "
                "edit departs from it in any way — a different target, a "
                "different mechanism, or scope beyond what it proposed — "
                "say so explicitly: what you changed instead and why. "
                "Silently doing something else is not acceptable; an "
                "honest \"I deviated from the suggestion because X\" is.\n\n"
                '  10. If the feedback below includes an error-bucket prevalence table, use it before deciding HOW to implement your fix: which bucket(s) dominate the failing cases is itself evidence of this backbone\'s actual capability tier (low/medium/high), not just a pointer to which case to cite. See strategies.md\'s "Reading the error-bucket prevalence table" section for how to infer the tier from the table\'s shape, and its per-(bucket, tier) strategy table for what KIND of fix that combination calls for -- a code-level fix, a prompt/instruction change, or a lightweight verifier. A high tool_omission/tool_calling_budget_exceeded rate is evidence this backbone needs the more aggressive, code-level fix regardless of how clean a prompt-only edit might look; a low rate with mostly constraint_misreading/apply_info_incorrectly is evidence a lighter prompt/verifier-level fix is more proportionate.'
                "\n\n"
                "Call `submit_self_improvement` with a one-line optimization_goal, "
                "a proposed_changes summary, a rationale, and the `files` payload. "
                "Each file is the FULL replacement content — do not produce diffs. "
                "Omit files you do not change."
            )
        system += self._skills_section()

        user_parts: list[str] = []
        if context:
            user_parts.append(f"## Steering context\n{context}\n")
        if feedback is not None:
            user_parts.append(
                self._format_feedback(feedback, has_suggestion=has_suggestion)
            )
        user_parts.extend(self._format_project_context())
        user_parts.append(self._format_current_sources(current))
        user_parts.extend(self._format_edit_scope())
        if prior_errors:
            joined = "\n".join(f"  - {e}" for e in prior_errors)
            user_parts.append(
                "## Previous attempt failed validation. Fix these errors:\n"
                f"{joined}\n"
            )

        llm_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            "tools": [SELF_IMPROVEMENT_TOOL],
        }
        if self.model:
            llm_kwargs["model"] = self.model
        if self.reasoning_effort:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            llm_kwargs["temperature"] = 0.2
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.max_output_tokens is not None:
            llm_kwargs["max_output_tokens"] = self.max_output_tokens
        user_body = "\n".join(user_parts)
        if verbose_log.is_enabled():
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_system.txt", system
            )
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_user.txt", user_body
            )
        try:
            response = self.llm(**llm_kwargs)
        except Exception as exc:  # noqa: BLE001 -- an LLM-call failure must
            # degrade this ONE EXPAND to a failed edit, never crash the
            # entire multi-day HGM process. Confirmed live this session: a
            # 75-generation-deep lineage's accumulated mutable source
            # pushed a single-shot prompt to 245K+ input tokens, exceeding
            # Qwen3.5-122B-A10B's 262144-token context window and raising
            # an uncaught openai.BadRequestError all the way up through
            # apply() -> hgm.py::_expand() -> evolve() -> main_loop.py.
            # No exception type is assumed here deliberately -- a context
            # overflow, a connection error surviving call_llm's own retry
            # loop, or any other LLM-call failure all get the same
            # graceful-failure treatment as a malformed tool call already
            # does (see the _raw_arguments handling below).
            print(
                f"[editor] warning: LLM call failed ({exc!r}) -- treating "
                "as a failed self-improvement attempt",
                flush=True,
            )
            if verbose_log.is_enabled():
                verbose_log.write_text(
                    out_dir, f"editor_attempt_{attempt}_llm_error.txt", repr(exc)
                )
            return EvolutionStrategy(
                target_files=[],
                optimization_goal=f"(editor's LLM call failed: {exc!r})"[:300],
                proposed_changes="",
                rationale="",
            ), []

        if verbose_log.is_enabled():
            verbose_log.write_json(
                out_dir,
                f"editor_attempt_{attempt}_response.json",
                {
                    "content": getattr(response, "content", None),
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments}
                        for c in (getattr(response, "tool_calls", []) or [])
                    ],
                },
            )

        for call in getattr(response, "tool_calls", []) or []:
            if call.name == "submit_self_improvement":
                # Two distinct ways a tool call's arguments can fail to be
                # a usable JSON object, both handled the same way here:
                #   1. platform_core.llm_wrapper.call_llm wraps them as
                #      {"_raw_arguments": <raw string>} when they fail to
                #      json.loads at all -- confirmed live: a model
                #      (DeepSeek v4, via OpenRouter) can produce a
                #      complete, well-reasoned fix and still have this
                #      happen from a single mis-escaped character deep in
                #      a large embedded-code string.
                #   2. The arguments DO parse as valid JSON, but the
                #      top-level value isn't an object (e.g. a bare list,
                #      string, or null) -- json.loads succeeds so (1)
                #      never fires, but `_parse_self_improvement`'s
                #      `args.get("files")` would raise AttributeError on
                #      anything without a `.get` method. Previously
                #      unguarded: this crashed the whole HGM run (no
                #      try/except wraps editor.apply() anywhere up to
                #      main_loop.py's fw.manager.evolve() call), not just
                #      one EXPAND.
                # In both cases the real content (if any) is unusable
                # without a hand-rolled JSON repair (risking silently
                # corrupted code), so this is surfaced as a distinct,
                # actionable error for the retry loop below rather than
                # crashing or silently falling through to the generic
                # empty-files path.
                if not isinstance(call.arguments, dict) or "_raw_arguments" in call.arguments:
                    raw_len = (
                        len(call.arguments.get("_raw_arguments") or "")
                        if isinstance(call.arguments, dict)
                        else len(repr(call.arguments))
                    )
                    print(
                        "[editor] warning: submit_self_improvement's "
                        f"arguments were not a valid JSON object ({raw_len} "
                        "raw chars) -- treating as a malformed-JSON attempt",
                        flush=True,
                    )
                    return EvolutionStrategy(
                        target_files=[],
                        optimization_goal=_MALFORMED_JSON_GOAL,
                        proposed_changes="",
                        rationale="",
                    ), []
                return self._parse_self_improvement(call.arguments)

        # Fallback: the model didn't tool-call. Try to recover a files
        # payload from a fenced JSON block; warn so the failure is visible.
        text = getattr(response, "content", None) or ""
        files: list[dict] = []
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                files = json.loads(match.group(1)).get("files") or []
            except (json.JSONDecodeError, AttributeError):
                files = []
        print(
            "[editor] warning: model did not call submit_self_improvement; "
            f"recovered {len(files)} file(s) from fenced JSON",
            flush=True,
        )
        # Reflect whatever was ACTUALLY recovered -- a hardcoded
        # "workflow.py" placeholder would misrepresent real recovered
        # paths when the fenced-JSON recovery above did find files, and
        # falsely claim a (frozen, normally-excluded) target when it
        # found none at all.
        recovered_paths = [
            f.get("path", "") for f in files if isinstance(f, dict) and f.get("path")
        ]
        fallback = EvolutionStrategy(
            target_files=recovered_paths,
            optimization_goal="(editor produced no structured proposal)",
            proposed_changes=text[:500],
            rationale="",
        )
        return fallback, files

    # ------------------------------------------------------------------ #
    # Agentic self-improvement (opt-in, see AgentEditor.agentic_editing)
    # ------------------------------------------------------------------ #

    def _diagnosis_rules(self) -> str:
        """The shared 'how to diagnose and what you may edit' rules,
        independent of *how* the model submits its edits. Kept separate
        from ``_self_improve``'s own inline system-prompt strings (which
        are left completely untouched by this method's existence) so the
        single-shot path carries zero refactor risk; this text is used
        only by ``_self_improve_agentic``. Mirrors the same two
        ``mutable_exclude`` vs. legacy-``MUTABLE_FILES`` branching
        ``_self_improve`` already does, and the same numbered hard rules
        1-9, minus anything naming a specific submission tool (each mode
        states its own tool-usage instructions separately)."""
        if self.mutable_exclude is not None:
            excl = ", ".join(sorted(self.mutable_exclude)) or "(nothing)"
            return (
                "You are the self-improvement module of a self-evolving agent. "
                "Diagnose what to change from the feedback and the current code.\n"
                "First understand the task: read the agent's own code, and "
                "(when provided below) the tool implementations, database "
                "schema, and evaluation scoring code — together they show what "
                "the system does, what the data looks like, and how output is "
                "graded. Target edits at the failures that most affect the score.\n"
                "You may edit ANY file in the task_agent workspace EXCEPT:\n"
                f"  - {excl}\n"
                "This includes prompt/config text AND the actual orchestration "
                "code (workflow logic, tool implementations, retry/decision "
                "branches, control flow) — not just one or the other. If the "
                "failure pattern points to a structural or logical problem "
                "(e.g. how evidence is gathered, when a retry fires, how a "
                "decision is made), change the code that implements it; don't "
                "default to a prompt-only wording tweak just because it is the "
                "easiest edit to make. Pick whichever kind of change actually "
                "fixes the diagnosed failure, or both together.\n\n"
                "Hard rules:\n"
                "  1. The workspace MUST keep exposing "
                "`def run_task(task: Task) -> AgentOutput` from its top-level "
                "`workflow.py` (the single arg must be named `task` — the "
                "validator enforces this); it may delegate to any other file "
                "you're allowed to edit.\n\n"
                "If this workspace uses, or you introduce, a `tool_wrapper.py` + "
                "`tools_schema.json` + `mutable_tools/` pattern (the framework's "
                "generic tool-calling convention), these additional rules apply "
                "to that pattern specifically — irrelevant otherwise:\n"
                "  2. workflow.py may import only: "
                "platform_core.llm_wrapper.call_llm, platform_core.runner "
                "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
                "  3. tool_wrapper.py may import only: platform_core.tools, "
                "mutable_tools.*, plus stdlib.\n"
                "  4. Mutable tools (mutable_tools/*.py) may import only: "
                "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
                "  5. Reach capabilities only via `platform_core.tools`: "
                "`call_tool(name, **kwargs)` for immutable tools, and "
                "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
                "invoke a mutable tool's `run()` directly — routing through "
                "`call_mutable_tool` keeps its calls recorded in the trace.\n"
                "  6. tools_schema.json: every entry's `name` must be backed by "
                "either an immutable tool OR a `mutable_tools/<name>.py` file. "
                "No collisions between immutable and mutable names.\n\n"
                "  7. Instrument your edits for the behavior summarizer. When "
                "you add a verifier, helper, or decision branch, call "
                "`platform_core.trace.log(label='your_label', "
                "verdict='pass'|'fail'|'skip', name='specific_check_name', "
                "**context)` at the decision point. Conventions: pick a stable "
                "`label` (e.g. 'verifier_fired', 'decision_branch'); use `name` "
                "to disambiguate within a label; set `verdict` to `pass`, "
                "`fail`, or `skip`. The summarizer cross-tabs these logs "
                "against case outcomes so the next editor knows which of your "
                "additions helped, which didn't, and why. Skip instrumentation "
                "only for trivial edits (renames, docstrings).\n\n"
                "  8. Double-check your own rationale before submitting. Every "
                "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
                "is malformed\", \"case X failed because of Y\") must be verified "
                "against the actual current source shown above or the "
                "feedback/metrics below — re-read the specific line or field "
                "you are citing and confirm it really says what you're about "
                "to claim. Do not invent a plausible-sounding diagnosis you "
                "have not checked. If a claim doesn't hold up on re-reading, "
                "drop it or soften it (e.g. \"possibly\", \"this may "
                "contribute\") rather than stating it as settled fact — a "
                "correct edit with an honest, hedged rationale is better than "
                "a confident but unverified one.\n\n"
                "  9. If the steering context above includes a specific "
                "suggestion or recommendation for what to change, state in "
                "your `rationale` whether your edit follows it. If your "
                "edit departs from it in any way — a different target, a "
                "different mechanism, or scope beyond what it proposed — "
                "say so explicitly: what you changed instead and why. "
                "Silently doing something else is not acceptable; an "
                "honest \"I deviated from the suggestion because X\" is.\n\n"
            '  10. If the feedback below includes an error-bucket prevalence table, use it before deciding HOW to implement your fix: which bucket(s) dominate the failing cases is itself evidence of this backbone\'s actual capability tier (low/medium/high), not just a pointer to which case to cite. See strategies.md\'s "Reading the error-bucket prevalence table" section for how to infer the tier from the table\'s shape, and its per-(bucket, tier) strategy table for what KIND of fix that combination calls for -- a code-level fix, a prompt/instruction change, or a lightweight verifier. A high tool_omission/tool_calling_budget_exceeded rate is evidence this backbone needs the more aggressive, code-level fix regardless of how clean a prompt-only edit might look; a low rate with mostly constraint_misreading/apply_info_incorrectly is evidence a lighter prompt/verifier-level fix is more proportionate.'
            "\n"
            )
        return (
            "You are the self-improvement module of a self-evolving agent. "
            "Diagnose what to change from the feedback and the current code.\n"
            "First understand the task: read the agent's system prompt in "
            "workflow.py, and (when provided below) the tool implementations, "
            "database schema, and evaluation scoring code — together they show "
            "what each tool does, what the data looks like, and how output is "
            "graded. Target edits at the failures that most affect the score.\n"
            "You may only modify these "
            "files in the task_agent workspace:\n"
            f"  - {', '.join(sorted(MUTABLE_FILES))}\n"
            f"  - any *.py file under mutable_tools/\n\n"
            "Hard rules:\n"
            "  1. workflow.py MUST define "
            "`def run_task(task: Task) -> AgentOutput`. The single arg must "
            "be named `task` (the validator enforces this).\n"
            "  2. workflow.py may import only: "
            "platform_core.llm_wrapper.call_llm, platform_core.runner "
            "(Task, AgentOutput), tool_wrapper, plus stdlib.\n"
            "  3. tool_wrapper.py may import only: platform_core.tools, "
            "mutable_tools.*, plus stdlib.\n"
            "  4. Mutable tools (mutable_tools/*.py) may import only: "
            "platform_core.tools, sibling mutable_tools.*, plus stdlib.\n"
            "  5. Reach capabilities only via `platform_core.tools`: "
            "`call_tool(name, **kwargs)` for immutable tools, and "
            "`call_mutable_tool(name, **kwargs)` for `mutable_tools/*`. Never "
            "invoke a mutable tool's `run()` directly — routing through "
            "`call_mutable_tool` keeps its calls recorded in the trace.\n"
            "  6. tools_schema.json: every entry's `name` must be backed by "
            "either an immutable tool OR a `mutable_tools/<name>.py` file. "
            "No collisions between immutable and mutable names.\n"
            "  7. Instrument your edits for the behavior summarizer. When you "
            "add a verifier, helper, or decision branch in workflow.py or a "
            "mutable_tools/*.py file, call "
            "`platform_core.trace.log(label='your_label', verdict='pass'|'fail'|'skip', "
            "name='specific_check_name', **context)` at the decision point. "
            "Conventions: pick a stable `label` (e.g. 'verifier_fired', "
            "'decision_branch'); use `name` to disambiguate within a label; "
            "set `verdict` to `pass`, `fail`, or `skip`. The summarizer "
            "cross-tabs these logs against case outcomes so the next editor "
            "knows which of your additions helped, which didn't, and why. "
            "Skip instrumentation only for trivial edits (renames, docstrings).\n\n"
            "  8. Double-check your own rationale before submitting. Every "
            "factual claim in `rationale`/`proposed_changes` (e.g. \"the tag "
            "is malformed\", \"case X failed because of Y\") must be verified "
            "against the actual current source shown above or the "
            "feedback/metrics below — re-read the specific line or field "
            "you are citing and confirm it really says what you're about "
            "to claim. Do not invent a plausible-sounding diagnosis you "
            "have not checked. If a claim doesn't hold up on re-reading, "
            "drop it or soften it (e.g. \"possibly\", \"this may "
            "contribute\") rather than stating it as settled fact — a "
            "correct edit with an honest, hedged rationale is better than "
            "a confident but unverified one.\n\n"
            "  9. If the steering context above includes a specific "
            "suggestion or recommendation for what to change, state in "
            "your `rationale` whether your edit follows it. If your "
            "edit departs from it in any way — a different target, a "
            "different mechanism, or scope beyond what it proposed — "
            "say so explicitly: what you changed instead and why. "
            "Silently doing something else is not acceptable; an "
            "honest \"I deviated from the suggestion because X\" is.\n\n"
            '  10. If the feedback below includes an error-bucket prevalence table, use it before deciding HOW to implement your fix: which bucket(s) dominate the failing cases is itself evidence of this backbone\'s actual capability tier (low/medium/high), not just a pointer to which case to cite. See strategies.md\'s "Reading the error-bucket prevalence table" section for how to infer the tier from the table\'s shape, and its per-(bucket, tier) strategy table for what KIND of fix that combination calls for -- a code-level fix, a prompt/instruction change, or a lightweight verifier. A high tool_omission/tool_calling_budget_exceeded rate is evidence this backbone needs the more aggressive, code-level fix regardless of how clean a prompt-only edit might look; a low rate with mostly constraint_misreading/apply_info_incorrectly is evidence a lighter prompt/verifier-level fix is more proportionate.'
            "\n"
        )

    def _agentic_closing(self, *, base_dir: Optional[Path] = None) -> str:
        """Describes only the tools actually enabled (self.agentic_tool_names,
        plus the evaluate_variant/evaluator gate) -- telling the model about
        a tool it can't call would be actively misleading, not just unused
        filler, so every sentence below is conditional on the tool it
        describes actually being offered. ``base_dir`` (the parent round's
        dir) gates the 'full_metrics.json' mention the same way -- checked
        by existence, not a static flag, since whether it exists depends
        on the PROJECT's scorer (full_metrics() is opt-in per scorer), not
        on anything AgentEditor itself configures."""
        has = lambda name: (  # noqa: E731
            name in self.agentic_tool_names
            and (name != "evaluate_variant" or self.evaluator is not None)
        )
        full_metrics_available = bool(
            base_dir is not None and (base_dir / "full_metrics.json").exists()
        )

        descriptions = {
            "read_file": (
                "`read_file` to inspect any of the files listed below "
                "before editing it (capped per call with offset/limit "
                "to page through a big one -- for a large file like "
                "internal_runs/trace.jsonl, `grep` it for a specific "
                "pattern instead of reading it whole)"
                + (
                    ". You can also read the project's own tool "
                    "implementation (read-only) at 'tools/<filename>.py' "
                    "-- the actual code behind the tools you call, e.g. "
                    "'tools/roadroute.py' for query_road_route_info"
                    if self.tools_dir_alias_enabled else ""
                )
                + (
                    ". 'full_metrics.json' has the PARENT round's real "
                    "fail_rate for every single commonsense check and "
                    "hard constraint it saw (not just the top few in the "
                    "project metrics above) -- read it if you want a "
                    "specific constraint's actual rate instead of "
                    "guessing from a handful of cases"
                    if full_metrics_available else ""
                )
            ),
            "grep": "`grep` to search one file for a regex pattern without reading it whole",
            "write_file": (
                "`write_file` to submit ONE file's FULL new content (call "
                "once per changed file — never bundle multiple files' "
                "content into one call)"
            ),
            "str_replace_file": (
                "`str_replace_file` for a small, targeted edit instead "
                "(replace one exact-match snippet in a file you can "
                "already write to)"
            ),
            # list_cases/show_case are described together when both are
            # enabled, but each must still get its OWN sentence when the
            # other is absent -- a config enabling only one of the two is
            # unusual but valid, and the prose must not silently omit it.
            "list_cases": (
                "`list_cases` and `show_case` to inspect the PARENT "
                "node's own evaluated cases -- the results from BEFORE "
                "any edit you make this round, not a live view of your "
                "current work (pass/fail, score, and — per case — the "
                "full details the project's scorer attached, e.g. the raw "
                "plan and failed checks)"
                if has("show_case") else
                "`list_cases` for a pass/fail/score overview of the "
                "PARENT node's own evaluated cases -- the results from "
                "BEFORE any edit you make this round, not a live view of "
                "your current work"
            ),
            "show_case": (
                None  # merged into list_cases's sentence above when both present
                if has("list_cases") else
                "`show_case` to inspect one of the PARENT node's own "
                "evaluated cases in full -- the results from BEFORE any "
                "edit you make this round, not a live view of your "
                "current work (pass/fail, score, and the full details the "
                "project's scorer attached)"
            ),
            "run_python": (
                "`run_python` to run your own offline Python against those "
                "same cases (no LLM, no network) to test a hypothesis or a "
                "snippet of logic before committing to a file edit"
            ),
            "evaluate_variant": (
                "`evaluate_variant` to run your CURRENT in-progress "
                "workspace through the REAL evaluator and see each case's "
                "actual score/pass-fail for THIS round's own edit -- "
                "unlike list_cases/show_case, which only ever show the "
                "parent's older results. Pass case_ids (specific TRAIN "
                "case ids) to evaluate just that subset, or omit case_ids "
                "to evaluate the full TRAIN set. You get "
                f"{self.evaluate_variant_max_calls} calls per EXPAND -- "
                "use them as you see fit."
                + (
                    " Each call's own logs are then readable via "
                    "read_file/grep at 'internal_runs/trace.jsonl' and "
                    "'internal_runs/case_<id>.json'"
                    if has("read_file") or has("grep") else ""
                )
            ),
            "run_code_validators": (
                "`run_code_validators` to check your changes so far "
                "(syntax, imports, signatures, etc.)"
            ),
            "submit_self_improvement_summary": "`submit_self_improvement_summary` to finish",
        }
        parts = [
            descriptions[name] for name in
            ("read_file", "grep", "write_file", "str_replace_file", "list_cases",
             "show_case", "run_python", "evaluate_variant", "run_code_validators",
             "submit_self_improvement_summary")
            if has(name) and descriptions.get(name)
        ]
        tool_list_sentence = "\nYou have these tools: " + "; ".join(parts) + ".\n\n"

        grounding_tools = [n for n in ("grep", "list_cases", "show_case", "run_python") if has(n)]
        order_parts = []
        if has("list_cases"):
            order_parts.append(
                "Before deciding what to fix, call `list_cases` to see "
                "the full set of the parent's currently-failing cases, "
                "and look at more than one of them"
                + (" (with `show_case`)" if has("show_case") else "")
                + " before committing to a diagnosis -- picking a fix "
                "based on only the first failing case you happen to "
                "notice risks missing a different failure that actually "
                "affects more of the score. "
            )
        if has("read_file"):
            order_parts.append(
                "Work in this order: call `read_file` on each file you "
                "plan to change (you don't need to read files you won't "
                "touch)"
                + (f" — use {'/'.join('`'+g+'`' for g in grounding_tools)} "
                   "as needed to ground your diagnosis in real evidence first"
                   if grounding_tools else "")
                + ". "
            )
        if has("write_file"):
            order_parts.append(
                "Then call `write_file`"
                + (" (or `str_replace_file` for a small change)" if has("str_replace_file") else "")
                + " once per changed file. "
            )
        if has("run_code_validators"):
            order_parts.append(
                "After writing your changes, call `run_code_validators`; if "
                "it reports problems, fix them with another `write_file`"
                + ("/`str_replace_file`" if has("str_replace_file") else "")
                + " call to the relevant file(s) and check again."
            )
        if has("evaluate_variant"):
            order_parts.append(
                " `evaluate_variant` runs your current workspace through "
                "the real evaluator and returns each case's actual score "
                "and pass/fail. Pass case_ids (specific TRAIN case ids) to "
                "evaluate just that subset, or omit case_ids to evaluate "
                f"the full TRAIN set. You get {self.evaluate_variant_max_calls} "
                "calls per EXPAND -- use them as you see fit. Expect "
                "some run-to-run variance in the score even with no code "
                "change (the task agent's own LLM calls are stochastic) -- "
                "don't read a small difference between two evaluate_variant "
                "calls as proof your edit helped or hurt; weigh it against "
                "the per-case pass/fail pattern, not the aggregate score "
                "alone. If you want more confidence on a borderline result, "
                "you can call evaluate_variant again on the same case_ids "
                "and compare -- within your remaining call budget, "
                "repeating a run is a legitimate way to tell signal from "
                "noise."
            )
        if has("submit_self_improvement_summary"):
            order_parts.append(
                " When everything is written"
                + (" and validators pass" if has("run_code_validators") else "")
                + ", call `submit_self_improvement_summary` with a one-line "
                "optimization_goal, a proposed_changes summary, and a "
                "rationale to finish."
            )
        return tool_list_sentence + "".join(order_parts)

    def _run_python(
        self,
        agent_dir: Path,
        out_dir: Path,
        feedback: Optional[AgentFeedback],
        code: str,
        timeout_s: Optional[int],
    ) -> str:
        """Backing implementation for the ``run_python`` agentic tool.
        Thin wrapper over ``run_python_sandboxed`` (module-level, shared
        with ``BlockSuggester`` — see that function's own docstring for
        what it actually does); only reshapes ``feedback`` into the plain
        ``per_case`` list that function takes, so neither caller needs to
        know about ``AgentFeedback``'s shape."""
        return run_python_sandboxed(
            agent_dir, out_dir,
            feedback.eval_result.per_case if feedback is not None else [],
            code, timeout_s,
        )


    def _self_improve_agentic(
        self,
        *,
        out_dir: Path,
        base_dir: Path,
        feedback: Optional[AgentFeedback],
        context: Optional[str],
        prior_errors: list[str],
        attempt: int = 1,
        has_suggestion: bool = False,
    ) -> tuple[EvolutionStrategy, list[dict]]:
        """Multi-turn analogue of ``_self_improve``: instead of one call
        bundling every changed file's full content into a single
        ``submit_self_improvement`` tool call (confirmed live this session
        to cause a real, reproducible malformed-JSON failure rate on large
        multi-file edits), the model reads/writes one file at a time across
        several turns, with the real validator suite available as a tool so
        it can self-correct before finishing. Returns the exact same
        ``(EvolutionStrategy, files)`` shape ``_self_improve`` does, so
        every line of ``apply()`` after the dispatch is unchanged."""
        agent_dir = out_dir / "task_agent"
        available_paths = sorted(self._read_mutable_sources(agent_dir).keys())

        system = self._diagnosis_rules() + self._agentic_closing(base_dir=base_dir) + self._skills_section()

        user_parts: list[str] = []
        if context:
            user_parts.append(f"## Steering context\n{context}\n")
        if feedback is not None:
            user_parts.append(
                self._format_feedback(feedback, has_suggestion=has_suggestion)
            )
        user_parts.extend(self._format_project_context())
        listing = "\n".join(f"  - {p}" for p in available_paths) or "  (none)"
        user_parts.append(
            "## Files you may read/edit\n"
            f"{listing}\n\n"
            "Use `read_file` to see any of these before editing it -- their "
            "content isn't shown here.\n"
        )
        user_parts.extend(self._format_edit_scope())
        if prior_errors:
            joined = "\n".join(f"  - {e}" for e in prior_errors)
            user_parts.append(
                "## Previous attempt failed validation. Fix these errors:\n"
                f"{joined}\n"
            )

        history: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        if verbose_log.is_enabled():
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_system.txt", system
            )
            verbose_log.write_text(
                out_dir, f"editor_attempt_{attempt}_user.txt", "\n".join(user_parts)
            )

        written: dict[str, str] = {}
        evaluate_variant_calls = 0
        # Set after each successful evaluate_variant call to that call's
        # OWN isolated out_dir/internal_runs/call_<n>/logs/ dir -- never
        # out_dir/logs/ itself, which is reserved for the real, framework-
        # triggered evaluation that happens later, after this EXPAND
        # finishes. Keeps every self-triggered run fully separate from
        # both that real evaluation AND from each other (each call gets
        # its own call_<n> dir, so a 2nd call can never clobber the 1st's
        # trace the way writing straight into out_dir/logs/ used to).
        last_variant_logs_dir: Optional[Path] = None
        # Set when the loop exits via "model returned no tool calls" (see
        # `if not calls: break` below) on a turn well before the real
        # budget ran out -- distinguishes that from true exhaustion of
        # self.agentic_max_turns for the fallback label below. Confirmed
        # live: DeepSeek v4 can return a totally empty turn (no content,
        # no tool_calls -- plausibly an all-reasoning response left empty
        # by META_AGENT_STRIP_REASONING) at turn 38 of a 100-turn budget,
        # after already writing a real edit and spending 18 more turns
        # verifying it -- mislabeling that as "(editor exceeded agentic
        # turn budget...)" is simply false and corrupts the lineage
        # memory text later EXPANDs read as this round's own history.
        stopped_early = False
        agentic_tools = [
            _AGENTIC_TOOLS_BY_NAME[name] for name in self.agentic_tool_names
            if name != "evaluate_variant" or self.evaluator is not None
        ]
        offered_tool_names = {t["name"] for t in agentic_tools}

        for turn in range(self.agentic_max_turns):
            llm_kwargs: dict[str, Any] = {
                "messages": history,
                "tools": agentic_tools,
            }
            if self.model:
                llm_kwargs["model"] = self.model
            if self.reasoning_effort:
                llm_kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                llm_kwargs["temperature"] = 0.2
            if self.base_url:
                llm_kwargs["base_url"] = self.base_url
            if self.max_output_tokens is not None:
                llm_kwargs["max_output_tokens"] = self.max_output_tokens
            try:
                response = self.llm(**llm_kwargs)
            except Exception as exc:  # noqa: BLE001 -- same principle as
                # _self_improve's own guard: an LLM-call failure (context
                # overflow from accumulated multi-turn history, a
                # connection error surviving call_llm's retries, etc.)
                # must never crash the whole HGM process. Unlike
                # _self_improve, files already written via write_file in
                # earlier turns are real and independently validated, so
                # they're preserved here rather than discarded.
                print(
                    f"[editor] warning: LLM call failed mid-conversation "
                    f"({exc!r}) -- returning {len(written)} file(s) "
                    "written so far",
                    flush=True,
                )
                if verbose_log.is_enabled():
                    verbose_log.write_text(
                        out_dir,
                        f"editor_attempt_{attempt}_turn_{turn}_llm_error.txt",
                        repr(exc),
                    )
                strategy = EvolutionStrategy(
                    target_files=sorted(written),
                    optimization_goal=f"{_AGENTIC_LLM_CALL_FAILED_GOAL_PREFIX}: {exc!r})"[:300],
                    proposed_changes="",
                    rationale="",
                )
                files = [{"path": p, "content": c} for p, c in written.items()]
                return strategy, files

            # Retry (without consuming a turn slot) when the model comes
            # back with literally nothing -- no content, no tool_calls,
            # and no exception (that's the separately-handled path just
            # above). See max_empty_response_retries' own docstring for
            # why a bare retry on the identical call is the right fix
            # here, not a different prompt or a turn-budget change.
            empty_retries = 0
            while (
                not (getattr(response, "tool_calls", None) or [])
                and not getattr(response, "content", None)
                and empty_retries < self.max_empty_response_retries
            ):
                empty_retries += 1
                print(
                    f"[editor] warning: turn {turn} got a completely empty "
                    f"response (no content, no tool_calls) -- retrying "
                    f"({empty_retries}/{self.max_empty_response_retries})",
                    flush=True,
                )
                if verbose_log.is_enabled():
                    verbose_log.write_json(
                        out_dir,
                        f"editor_attempt_{attempt}_turn_{turn}_empty_retry_{empty_retries}.json",
                        {"content": None, "tool_calls": []},
                    )
                try:
                    response = self.llm(**llm_kwargs)
                except Exception as exc:  # noqa: BLE001 -- same as above;
                    # a retry attempt can fail too, and must be handled
                    # identically rather than propagating uncaught.
                    print(
                        f"[editor] warning: LLM call failed mid-conversation "
                        f"({exc!r}) -- returning {len(written)} file(s) "
                        "written so far",
                        flush=True,
                    )
                    if verbose_log.is_enabled():
                        verbose_log.write_text(
                            out_dir,
                            f"editor_attempt_{attempt}_turn_{turn}_llm_error.txt",
                            repr(exc),
                        )
                    strategy = EvolutionStrategy(
                        target_files=sorted(written),
                        optimization_goal=f"{_AGENTIC_LLM_CALL_FAILED_GOAL_PREFIX}: {exc!r})"[:300],
                        proposed_changes="",
                        rationale="",
                    )
                    files = [{"path": p, "content": c} for p, c in written.items()]
                    return strategy, files

            calls = getattr(response, "tool_calls", None) or []
            if verbose_log.is_enabled():
                verbose_log.write_json(
                    out_dir,
                    f"editor_attempt_{attempt}_turn_{turn}_response.json",
                    {
                        "content": getattr(response, "content", None),
                        "tool_calls": [
                            {"name": c.name, "arguments": c.arguments} for c in calls
                        ],
                    },
                )
            if not calls:
                stopped_early = turn < self.agentic_max_turns - 1
                break

            for idx, call in enumerate(calls):
                call_id = (
                    getattr(call, "id", None)
                    or getattr(call, "call_id", None)
                    or f"call_{turn}_{idx}"
                )
                args = call.arguments
                # Same two malformed shapes _self_improve already guards
                # against: platform_core.llm_wrapper.call_llm wraps a
                # genuine JSON-parse failure as {"_raw_arguments": raw},
                # and a non-dict `args` (bare list/string/None) would
                # otherwise crash a `.get()` call below.
                malformed = not isinstance(args, dict) or "_raw_arguments" in args
                history.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": call.name,
                    "arguments": json.dumps(args if isinstance(args, dict) else {}),
                })

                if malformed and call.name == "submit_self_improvement_summary":
                    # Unlike the bundled single-shot mode, a malformed
                    # closing summary here is a small loss, not a total
                    # one: every write_file call already succeeded and was
                    # validated independently. Return the real recovered
                    # files immediately rather than burning turns retrying
                    # a narrative-only payload.
                    strategy = EvolutionStrategy(
                        target_files=sorted(written),
                        optimization_goal=_AGENTIC_MALFORMED_SUMMARY_GOAL,
                        proposed_changes="",
                        rationale="",
                    )
                    files = [{"path": p, "content": c} for p, c in written.items()]
                    return strategy, files
                try:
                    if malformed:
                        output = (
                            "ERROR: your arguments were not valid JSON and could "
                            "not be parsed. Make sure you return a valid JSON "
                            "object: double-check that every string value -- "
                            "especially `content` -- has its quotes, "
                            "backslashes, and newlines properly escaped. Retry "
                            f"this {call.name} call."
                        )
                    elif call.name not in offered_tool_names:
                        output = (
                            f"ERROR: tool {call.name!r} is not enabled for this "
                            "run (not in the configured agentic_tools)."
                        )
                    elif call.name == "read_file":
                        fpath, path_err = self._resolve_agentic_read_path(
                            args.get("path") or "", agent_dir=agent_dir,
                            internal_run_logs_dir=last_variant_logs_dir,
                            base_dir=base_dir,
                        )
                        if fpath is None:
                            output = path_err
                        else:
                            path = args.get("path") or ""
                            if fpath.is_dir():
                                # A real, reproducible crash otherwise: the
                                # model can ask for a bare directory name
                                # (e.g. "agents") since _is_path_allowed
                                # only checks the exclude list, not whether
                                # the path is actually a file -- read_text()
                                # on a directory raises IsADirectoryError,
                                # which would kill the whole HGM process
                                # uncaught. Confirmed live in production.
                                try:
                                    entries = sorted(
                                        e.name + ("/" if e.is_dir() else "")
                                        for e in fpath.iterdir()
                                    )
                                except OSError:
                                    entries = []
                                output = (
                                    f"ERROR: {path!r} is a directory, not a "
                                    "file -- read one of its contents "
                                    "instead: " + (", ".join(entries) or "(empty)")
                                )
                            elif fpath.exists():
                                output = self._paginated_read(fpath, args)
                            else:
                                output = f"(file not found: {path})"
                    elif call.name == "grep":
                        fpath, path_err = self._resolve_agentic_read_path(
                            args.get("path") or "", agent_dir=agent_dir,
                            internal_run_logs_dir=last_variant_logs_dir,
                            base_dir=base_dir,
                        )
                        if fpath is None:
                            output = path_err
                        else:
                            path = args.get("path") or ""
                            if not fpath.exists() or not fpath.is_file():
                                output = f"(file not found: {path})"
                            else:
                                pattern = args.get("pattern") or ""
                                try:
                                    rx = re.compile(pattern, re.IGNORECASE)
                                except re.error as exc:
                                    output = f"ERROR: invalid regex {pattern!r}: {exc!r}"
                                else:
                                    try:
                                        max_matches = int(args.get("max_matches") or 12)
                                    except (TypeError, ValueError):
                                        max_matches = 12
                                    matches: list[str] = []
                                    text = fpath.read_text(encoding="utf-8", errors="replace")
                                    for i, line in enumerate(text.splitlines()):
                                        m = rx.search(line)
                                        if not m:
                                            continue
                                        start = max(0, m.start() - 150)
                                        end = min(len(line), m.end() + 350)
                                        prefix = "..." if start > 0 else ""
                                        suffix = "..." if end < len(line) else ""
                                        matches.append(
                                            f"L{i} (char {m.start()}): "
                                            f"{prefix}{line[start:end].strip()}{suffix}"
                                        )
                                        if len(matches) >= max_matches:
                                            break
                                    output = "\n".join(matches) if matches else "(no matches)"
                    elif call.name == "write_file":
                        path = (args.get("path") or "").lstrip("/")
                        content = args.get("content")
                        if content is None or not path:
                            output = (
                                "ERROR: write_file requires both a non-empty "
                                "`path` and a `content` string."
                            )
                        elif not self._is_path_allowed(path):
                            output = (
                                f"ERROR: forbidden edit path {path!r} -- allowed "
                                f"paths are: {', '.join(available_paths) or '(none)'}"
                            )
                        elif not self._in_scope(path):
                            output = self._scope_refusal(path)
                        else:
                            target = agent_dir / path
                            if target.is_dir():
                                output = (
                                    f"ERROR: {path!r} is a directory, not a "
                                    "file -- specify a file path inside it."
                                )
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_text(content, encoding="utf-8")
                                written[path] = content
                                output = f"written {path} ({len(content)} chars)"
                    elif call.name == "str_replace_file":
                        path = (args.get("path") or "").lstrip("/")
                        old_str = args.get("old_str")
                        new_str = args.get("new_str")
                        if not path or old_str is None or new_str is None:
                            output = (
                                "ERROR: str_replace_file requires `path`, "
                                "`old_str`, and `new_str`."
                            )
                        elif not self._is_path_allowed(path):
                            output = (
                                f"ERROR: forbidden edit path {path!r} -- allowed "
                                f"paths are: {', '.join(available_paths) or '(none)'}"
                            )
                        elif not self._in_scope(path):
                            output = self._scope_refusal(path)
                        else:
                            target = agent_dir / path
                            if path in written:
                                current_content: Optional[str] = written[path]
                            elif target.exists() and target.is_file():
                                current_content = target.read_text(encoding="utf-8")
                            else:
                                current_content = None
                            if current_content is None:
                                output = (
                                    f"ERROR: {path!r} does not exist yet -- "
                                    "use write_file to create it first."
                                )
                            else:
                                count = current_content.count(old_str)
                                if count != 1:
                                    output = (
                                        f"ERROR: old_str occurs {count} time(s) "
                                        f"in {path!r} -- must occur exactly "
                                        "once. Provide more surrounding "
                                        "context to make it unique, or use "
                                        "write_file for a full rewrite."
                                    )
                                else:
                                    new_content = current_content.replace(old_str, new_str)
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    target.write_text(new_content, encoding="utf-8")
                                    written[path] = new_content
                                    output = (
                                        f"replaced 1 occurrence in {path} "
                                        f"({len(new_content)} chars)"
                                    )
                    elif call.name == "list_cases":
                        if feedback is None:
                            output = "(no evaluated cases available for this node)"
                        else:
                            output = render_list_cases(
                                feedback.eval_result.per_case,
                                failed_check=args.get("failed_check"),
                                limit=args.get("limit"),
                            ) + (
                                "\n\n[PARENT's evaluation, from BEFORE any edit "
                                "you've made this round -- this does NOT "
                                "reflect your write_file/str_replace_file "
                                "changes. To see your current code's real "
                                "effect, use evaluate_variant.]"
                            )
                    elif call.name == "show_case":
                        if feedback is None:
                            output = "(no evaluated cases available for this node)"
                        else:
                            output = render_show_case(
                                feedback.eval_result.per_case, args.get("case_id") or ""
                            ) + (
                                "\n\n[PARENT's evaluation, from BEFORE any edit "
                                "you've made this round -- this does NOT "
                                "reflect your write_file/str_replace_file "
                                "changes. To see your current code's real "
                                "effect, use evaluate_variant.]"
                            )
                    elif call.name == "run_python":
                        code = args.get("code")
                        if not code:
                            output = "ERROR: run_python requires `code`."
                        else:
                            ws_modules: set = set()
                            for p in available_paths:
                                parts = Path(p).parts
                                first = parts[0]
                                ws_modules.add(
                                    first[:-3] if len(parts) == 1 and first.endswith(".py")
                                    else first
                                )
                            probs = _run_python_problems(code, ws_modules)
                            if probs:
                                output = (
                                    "run_python REJECTED (nothing executed): "
                                    + "; ".join(probs)
                                )
                            else:
                                output = self._run_python(
                                    agent_dir, out_dir, feedback, code, args.get("timeout_s")
                                )
                    elif call.name == "evaluate_variant":
                        if self.evaluator is None:
                            output = "ERROR: evaluate_variant is not available in this run."
                        elif evaluate_variant_calls >= self.evaluate_variant_max_calls:
                            output = (
                                "ERROR: evaluate_variant call limit reached "
                                f"({self.evaluate_variant_max_calls} per EXPAND) -- "
                                "no more evaluation calls left; finish with what "
                                "you've already confirmed."
                            )
                        else:
                            raw_ids = args.get("case_ids")
                            if raw_ids is not None and not isinstance(raw_ids, list):
                                output = (
                                    "ERROR: case_ids must be a list of strings, "
                                    "or omit it entirely to evaluate the full "
                                    "TRAIN set."
                                )
                            else:
                                train_set = {str(c) for c in (self.train_case_ids or [])}
                                if raw_ids:
                                    requested = [str(c).strip() for c in raw_ids if str(c).strip()]
                                    bad = [c for c in requested if c not in train_set]
                                    ids: Optional[list[str]] = None
                                    if bad:
                                        output = (
                                            "ERROR: case_ids not in the TRAIN "
                                            f"split: {bad[:10]}"
                                        )
                                    else:
                                        ids = requested
                                else:
                                    ids = list(self.train_case_ids or [])
                                if ids is not None:
                                    evaluate_variant_calls += 1
                                    # Isolated per-call dir, NEVER out_dir
                                    # itself -- out_dir/logs/ is reserved for
                                    # the real, framework-triggered
                                    # evaluation that happens later (after
                                    # this EXPAND finishes); writing straight
                                    # into it here would both collide with
                                    # that later run AND (since evaluator.run
                                    # truncates trace.jsonl on every call)
                                    # clobber an earlier evaluate_variant
                                    # call's own logs within this same turn
                                    # loop. A fresh task_agent/ copy is needed
                                    # per call since evaluator.run executes
                                    # from <round_dir>/task_agent.
                                    variant_dir = (
                                        out_dir / "internal_runs"
                                        / f"call_{evaluate_variant_calls}"
                                    )
                                    try:
                                        if variant_dir.exists():
                                            shutil.rmtree(variant_dir)
                                        shutil.copytree(
                                            out_dir / "task_agent",
                                            variant_dir / "task_agent",
                                        )
                                        result = self.evaluator.run(
                                            variant_dir, self.benchmark_dir, case_ids=ids
                                        )
                                    except Exception as exc:  # noqa: BLE001 -- an
                                        # infrastructure failure (not a real
                                        # quality signal) must not cost the
                                        # editor one of its limited calls.
                                        evaluate_variant_calls -= 1
                                        output = (
                                            f"evaluate_variant FAILED for an "
                                            f"infrastructure reason ({exc!r}); "
                                            "your call was NOT counted. Try again."
                                        )
                                    else:
                                        last_variant_logs_dir = variant_dir / "logs"
                                        header = (
                                            f"score={result.score:.4f} "
                                            f"passed={result.passed} "
                                            f"failed={result.failed} "
                                            f"crashed={result.crashed} "
                                            f"(n={len(result.per_case)}, "
                                            f"{evaluate_variant_calls}/"
                                            f"{self.evaluate_variant_max_calls} "
                                            "calls used)"
                                        )
                                        output = header + "\n" + render_list_cases(
                                            result.per_case,
                                            limit=len(result.per_case) or 1,
                                        )
                                        if "read_file" in offered_tool_names or "grep" in offered_tool_names:
                                            output += (
                                                "\n\nThis call's own logs are now "
                                                "readable via read_file/grep at "
                                                "'internal_runs/trace.jsonl' "
                                                "(tool_call/tool_result/llm_call "
                                                "events) and "
                                                "'internal_runs/case_<id>.json' "
                                                "(one per case above) -- NOT the "
                                                "parent's logs (see list_cases/"
                                                "show_case for those), and NOT "
                                                "the real scored evaluation "
                                                "(that happens separately, after "
                                                "you finish this EXPAND). A "
                                                "later evaluate_variant call "
                                                "replaces what 'internal_runs/' "
                                                "points at with ITS OWN run."
                                            )
                    elif call.name == "run_code_validators":
                        errors = self._run_validators(out_dir, base_dir)
                        output = (
                            "All validators passed." if not errors
                            else "Validator errors:\n" + "\n".join(f"- {e}" for e in errors)
                        )
                    elif call.name == "submit_self_improvement_summary":
                        strategy = EvolutionStrategy(
                            target_files=sorted(written),
                            optimization_goal=_coerce_str(args.get("optimization_goal")),
                            proposed_changes=_coerce_str(args.get("proposed_changes")),
                            rationale=_coerce_str(args.get("rationale")),
                        )
                        files = [{"path": p, "content": c} for p, c in written.items()]
                        return strategy, files
                    else:
                        output = f"ERROR: unknown tool {call.name!r}."
                except Exception as exc:  # noqa: BLE001 -- any other
                    # unexpected OS/filesystem error from a single tool
                    # call (permissions, encoding, etc.) must degrade to an
                    # in-conversation error the model can react to, never
                    # crash the whole HGM process -- same principle as the
                    # outer LLM-call-failure guard above.
                    output = (
                        f"ERROR: {call.name} raised {exc!r} on this call -- "
                        "try different arguments."
                    )

                if verbose_log.is_enabled():
                    verbose_log.write_json(
                        out_dir,
                        f"editor_attempt_{attempt}_turn_{turn}_call_{idx}_{call.name}.json",
                        {"arguments": args, "output": output},
                    )

                history.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                })

        # Turn budget exhausted (or the model stopped calling tools) without
        # ever calling submit_self_improvement_summary -- reflect whatever
        # was ACTUALLY written via write_file, same "never claim more than
        # really happened" principle as _self_improve's own fallbacks.
        # Naturally degrades to target_files=[]/files=[] (handled by
        # apply()'s existing "editor returned no file edits" branch) when
        # nothing was ever written. Label distinguishes a true budget
        # exhaustion from the model going silent early (see
        # `stopped_early` above) -- both keep whatever was written, only
        # the recorded explanation differs.
        files = [{"path": p, "content": c} for p, c in written.items()]
        strategy = EvolutionStrategy(
            target_files=sorted(written),
            optimization_goal=(
                _AGENTIC_EMPTY_RESPONSE_GOAL if stopped_early
                else _AGENTIC_TURN_BUDGET_GOAL
            ),
            proposed_changes="",
            rationale="",
        )
        return strategy, files

    def _parse_self_improvement(
        self, args: dict[str, Any]
    ) -> tuple[EvolutionStrategy, list[dict]]:
        """Split a ``submit_self_improvement`` tool call into a validated
        ``EvolutionStrategy`` summary and the raw ``files`` payload.
        ``target_files`` is derived from the emitted file paths."""
        files = args.get("files") or []
        edited = [
            f.get("path", "") for f in files if isinstance(f, dict)
        ]
        if self.mutable_exclude is not None:
            valid = lambda p: not is_excluded(p, self.mutable_exclude)  # noqa: E731
        else:
            valid = None
            edited = [p for p in edited if p in _ALLOWED_TARGET_FILES]
        strategy = EvolutionStrategy(
            target_files=_coerce_target_files(edited, valid=valid),
            optimization_goal=_coerce_str(args.get("optimization_goal")),
            proposed_changes=_coerce_str(args.get("proposed_changes")),
            rationale=_coerce_str(args.get("rationale")),
        )
        return strategy, files

    # Noise directories skipped regardless of mode -- generated/scratch
    # output (e.g. db-mas's own `results/raw/*.json` writes) and Python's
    # own cache, never source the editor should read or be judged against.
    _ALWAYS_IGNORE_DIRS = {"__pycache__", "results"}

    def _read_mutable_sources(self, agent_dir: Path) -> dict[str, str]:
        return source_context.read_mutable_sources(
            agent_dir, mutable_exclude=self.mutable_exclude
        )

    def _format_project_context(self) -> list[str]:
        return source_context.format_project_context(
            tools_source=self.tools_source,
            db_schema=self.db_schema,
            scorer_source=self.scorer_source,
        )

    def _format_current_sources(self, sources: dict[str, str]) -> str:
        return source_context.format_current_sources(sources)

    def _format_feedback(
        self, feedback: AgentFeedback, *, has_suggestion: bool = False
    ) -> str:
        """Render the previous round's ``AgentFeedback`` into a compact
        prompt section — score, tool usage/errors, project metrics (unless
        ``has_suggestion``), exceptions, validator complaints, and a trace
        excerpt."""
        ev = feedback.eval_result
        lines = [
            "## Last round's feedback",
            f"score={ev.score:.3f}  passed={ev.passed}  failed={ev.failed}  "
            f"crashed={ev.crashed}",
        ]
        # Data-driven scope note: the trace-derived stats below cover only the
        # cases present in the parsed trace (typically the latest evaluation
        # batch), which can be fewer than the cumulative evaluated set behind
        # the score / project metrics / failure analysis. Stated as actual
        # counts (not a hardcoded "last batch" claim) so it stays correct
        # regardless of how the trace was produced.
        n_eval = len(ev.per_case)
        if feedback.trace_n_cases and n_eval and feedback.trace_n_cases < n_eval:
            lines.append(
                f"(scope: tool_usage / tool error rates / llm_calls / log excerpt "
                f"below are over {feedback.trace_n_cases} traced case(s); score, "
                f"project metrics, and failure analysis cover {n_eval} evaluated "
                f"case(s))"
            )
        lines += [
            f"llm_calls={feedback.llm_calls}",
            f"tool_usage={feedback.tool_usage}",
        ]
        if feedback.tool_error_rate:
            ranked = sorted(
                feedback.tool_error_rate.items(), key=lambda kv: -kv[1]
            )
            err_lines = [f"{n}={r:.2f}" for n, r in ranked[:5] if r > 0]
            if err_lines:
                lines.append("tool error rates: " + ", ".join(err_lines))
        # Trimmed only when a block-scoped suggestion (block_suggester.py)
        # was actually produced for THIS apply() call -- that module now
        # owns diagnosis grounded in this same project_metrics data (see
        # its own _format_feedback_digest), so repeating it here would be
        # redundant with the "## Block-scoped suggestion" section already
        # in `context` (see hgm.py::_render_expand_context). Defaults to
        # False (today's exact behavior) for every caller that never
        # computes a suggestion -- HGMDualManager's Stage B and
        # HillClimbingManager -- and for any round where no block_suggester
        # is configured, or a configured one's call failed/returned empty.
        # failure_report (below) is NEVER trimmed, regardless of
        # has_suggestion.
        if feedback.project_metrics and not has_suggestion:
            lines.append("project metrics:")
            lines.extend(render_metrics(feedback.project_metrics, cap=5, indent="  "))
        if feedback.runtime_exceptions:
            lines.append("runtime_exceptions:")
            for exc in feedback.runtime_exceptions[:5]:
                lines.append(f"  - {exc}")
        if feedback.edit_errors:
            lines.append("edit_errors (previous round did not run — these are validator complaints):")
            for err in feedback.edit_errors[:5]:
                lines.append(f"  - {err}")
        rendered = "\n".join(lines) + "\n"
        # Example-driven failure analysis (query → plan → what failed +
        # hardest cases) replaces the old generic trace tail, which bloated
        # the prompt with low-signal events. Error events still surface above
        # via runtime_exceptions.
        report = render_failure_report(feedback.failure_report)
        if report:
            rendered += "\n" + report
        # Same "never trimmed regardless of has_suggestion" policy as
        # failure_report above -- block_suggester.py's own feedback digest
        # shows the identical section (see its _format_feedback_digest), so
        # both the editor and every block suggester see the same breakdown.
        bucket_section = render_error_bucket_prevalence_for_prompt(
            feedback.error_bucket_prevalence
        )
        if bucket_section:
            rendered += "\n" + bucket_section
        return rendered

    def _write_edits(
        self, out_dir: Path, files: list[dict]
    ) -> tuple[list[str], list[str]]:
        """Write each ``{path, content}`` entry into ``out_dir/task_agent``.
        Returns ``(written_paths, errors)`` — an entry whose path is outside
        the mutable surface is rejected into ``errors``, not written."""
        agent_dir = out_dir / "task_agent"
        written: list[str] = []
        errors: list[str] = []
        for entry in files:
            # A non-compliant model can return a bare string / junk instead
            # of a {path, content} object — skip it rather than crashing.
            if not isinstance(entry, dict):
                errors.append(f"malformed edit entry (not an object): {entry!r}")
                continue
            path = (entry.get("path") or "").lstrip("/")
            content = entry.get("content")
            if not path or content is None:
                errors.append(f"malformed edit entry: {entry!r}")
                continue
            if self._is_path_allowed(path) and not self._in_scope(path):
                errors.append(self._scope_refusal(path))
                continue
            if not self._is_path_allowed(path):
                if self.mutable_exclude is not None:
                    errors.append(
                        f"forbidden edit path: {path!r} "
                        f"(excluded: {sorted(self.mutable_exclude)})"
                    )
                else:
                    errors.append(
                        f"forbidden edit path: {path!r} "
                        f"(must be one of {sorted(MUTABLE_FILES)} or under mutable_tools/)"
                    )
                continue
            target = agent_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(path)
        return written, errors

    _edit_scope: Optional[list[str]] = None
    # Set by meta_agent.config.build_components when `skills: {enabled: true}`; None
    # (default) leaves every prompt byte-identical.
    skills_guide: Optional[str] = None

    def _skills_section(self) -> str:
        return f"\n\n{self.skills_guide}" if self.skills_guide else ""

    def _in_scope(self, rel_path: str) -> bool:
        """True when no edit scope is active, or ``rel_path`` falls under one of its
        prefixes (a prefix ending in '/' is a directory; otherwise an exact file)."""
        if not self._edit_scope:
            return True
        rel = Path(rel_path).as_posix().lstrip("/")
        for p in self._edit_scope:
            if p.endswith("/"):
                if rel.startswith(p):
                    return True
            elif rel == p:
                return True
        return False

    def _scope_refusal(self, rel_path: str) -> str:
        return (
            f"ERROR: {rel_path!r} is outside this EXPAND's edit scope "
            f"({', '.join(self._edit_scope or [])}). This block may only change "
            "files under those paths -- make the change there, or leave it for "
            "another block."
        )

    def _format_edit_scope(self) -> list[str]:
        if not self._edit_scope:
            return []
        return [
            "## Edit scope for this EXPAND (enforced)\n"
            + "\n".join(f"  - {p}" for p in self._edit_scope)
            + "\n\nYou may READ any file, but writes/edits outside these paths are "
            "refused and a change set touching anything else fails validation.\n"
        ]

    def _scope_violations(self, out_dir: Path, base_dir: Path) -> list[str]:
        """Files the attempt created/changed/deleted outside the active scope."""
        if not self._edit_scope:
            return []
        new_root, old_root = out_dir / "task_agent", base_dir / "task_agent"

        def files(root: Path) -> dict[str, Path]:
            if not root.exists():
                return {}
            return {
                p.relative_to(root).as_posix(): p
                for p in root.rglob("*")
                if p.is_file() and not (set(p.relative_to(root).parts) & self._ALWAYS_IGNORE_DIRS)
            }

        new, old = files(new_root), files(old_root)
        changed = [
            r for r in sorted(set(new) | set(old))
            if r not in new or r not in old or new[r].read_bytes() != old[r].read_bytes()
        ]
        return [
            f"edit scope violation: {r} changed outside {', '.join(self._edit_scope)}"
            for r in changed if not self._in_scope(r)
        ]

    def _resolve_agentic_read_path(
        self, raw_path: str, *, agent_dir: Path,
        internal_run_logs_dir: Optional[Path],
        base_dir: Optional[Path] = None,
    ) -> tuple[Optional[Path], Optional[str]]:
        """Resolve a read_file/grep path in agentic mode.
        ``'internal_runs/<rel>'`` reads from the MOST RECENT
        evaluate_variant call's own, isolated logs dir
        (``internal_run_logs_dir``, set by the dispatch loop after each
        successful call to ``out_dir/internal_runs/call_<n>/logs``) --
        never ``out_dir/logs/`` itself, which is reserved for the real,
        framework-triggered evaluation that happens later, after this
        EXPAND finishes, and never the parent's own logs either (see
        ``list_cases``/``show_case`` for those). ``None`` when no
        evaluate_variant call has been made yet this round.

        ``'tools/<rel>'`` reads from ``self.tools_dir`` (the project's own
        tool implementation) when ``tools_dir_alias_enabled`` is True --
        off by default (an ablation-testable opt-in; see that flag's own
        docstring in ``__init__``). Every other path resolves against
        ``agent_dir`` exactly as before, still gated by
        ``_is_path_allowed`` -- this is purely additive, read-only, and
        does not change what ``write_file``/``str_replace_file`` can
        touch. Returns ``(path, None)`` on success or
        ``(None, error_message)``."""
        path = (raw_path or "").lstrip("/")
        if path.startswith("internal_runs/"):
            if internal_run_logs_dir is None:
                return None, (
                    "ERROR: no evaluate_variant call has been made yet "
                    "this round -- call it first, then "
                    "'internal_runs/...' paths become readable."
                )
            rel = path[len("internal_runs/"):]
            logs_root = internal_run_logs_dir.resolve()
            target = (internal_run_logs_dir / rel).resolve()
            if target != logs_root and logs_root not in target.parents:
                return None, f"ERROR: {raw_path!r} escapes the internal_runs/ root."
            return target, None
        if path.startswith("tools/"):
            if not self.tools_dir_alias_enabled:
                return None, (
                    "ERROR: 'tools/...' paths are not enabled for this "
                    "run (tools_dir_alias_enabled is off)."
                )
            if self.tools_dir is None:
                return None, (
                    "ERROR: no tools/ directory is configured for this "
                    "project -- 'tools/...' paths aren't readable here."
                )
            rel = path[len("tools/"):]
            tools_root = self.tools_dir.resolve()
            target = (self.tools_dir / rel).resolve()
            if target != tools_root and tools_root not in target.parents:
                return None, f"ERROR: {raw_path!r} escapes the tools/ root."
            return target, None
        if path == "full_metrics.json":
            # Bare-filename alias (same convention BlockSuggester already
            # uses for its own "eval_result.json") -- the PARENT round's
            # uncapped, denominator-aware per-constraint breakdown (see
            # scorer_impl.py::full_metrics / feedback_gatherer.py::
            # _write_full_metrics). base_dir is the parent's own round
            # dir (apply()'s own param), not agent_dir (agent_dir/task_agent
            # has no such file). None when no project scorer opts into
            # full_metrics() -- not an error, just nothing to read yet.
            if base_dir is None or not (base_dir / "full_metrics.json").exists():
                return None, (
                    "ERROR: 'full_metrics.json' is not available for the "
                    "parent round -- this project's scorer may not "
                    "define full_metrics(), or the parent was never "
                    "evaluated."
                )
            return base_dir / "full_metrics.json", None
        if not self._is_path_allowed(path):
            return None, (
                f"ERROR: {path!r} is not readable/editable here -- see "
                "the '## Files you may read/edit' list above for what's "
                "available."
            )
        return agent_dir / path, None

    def _paginated_read(self, fpath: Path, args: dict[str, Any]) -> str:
        """Backing implementation for the ``read_file`` tool: line-sliced
        by offset/limit (mirrors BlockSuggester's own
        ``_agentic_read_file``), plus a hard character ceiling that no
        line-count limit alone can guarantee -- a single line can be
        enormous (confirmed live: a ``read_file`` call on
        ``internal_runs/trace.jsonl`` -- one JSON line per LLM call --
        returned 47,258,272 characters and blew past OpenRouter's 8MB
        total-request-size limit on the very next turn, ending that
        EXPAND early). ``_READ_FILE_MAX_CHARS`` makes that impossible
        regardless of line count.

        Deliberately returns plain text with NO line-number prefix
        (unlike BlockSuggester's version, which is diagnostic-only) --
        this output can end up verbatim inside a later
        ``str_replace_file`` call, and an "L123: " prefix would corrupt
        that exact-match."""
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR reading file: {exc!r}"
        # keepends=True + "".join(...) (not "\n".join(lines.splitlines()))
        # so an untruncated read reconstructs the file's EXACT original
        # bytes, trailing newline included -- str_replace_file does
        # exact-text matching downstream, so silently dropping a
        # trailing "\n" here would be a real (if small) fidelity bug.
        lines = text.splitlines(keepends=True)
        offset = max(0, int(args.get("offset") or 0))
        limit = args.get("limit")
        limit = int(limit) if limit else _READ_FILE_DEFAULT_LINE_LIMIT
        chunk_lines = lines[offset:offset + limit]
        chunk = "".join(chunk_lines)
        remaining = len(lines) - (offset + limit)
        note = None
        if len(chunk) > _READ_FILE_MAX_CHARS:
            chunk = chunk[:_READ_FILE_MAX_CHARS]
            note = (
                f"\n\n[... cut at {_READ_FILE_MAX_CHARS} characters -- "
                "at least one of these lines is very long (e.g. a "
                "trace.jsonl event). For a large file like this, use "
                "`grep` to search for a specific pattern instead of "
                "reading it line-by-line.]"
            )
        elif remaining > 0:
            note = (
                f"\n\n[... {remaining} more line(s) -- call read_file "
                f"again with offset={offset + limit} to continue, or "
                "use `grep` to search a large file (e.g. trace.jsonl) "
                "for a specific pattern instead.]"
            )
        if not chunk:
            return "(empty file or offset past end)"
        return chunk + (note or "")

    def _is_path_allowed(self, rel_path: str) -> bool:
        parts = Path(rel_path).parts
        if ".." in parts or Path(rel_path).is_absolute():
            return False
        if set(parts) & self._ALWAYS_IGNORE_DIRS:
            # Generated/scratch output (e.g. db-mas's own results/raw/*.json
            # writes) and __pycache__ -- never a legitimate edit target,
            # mode-independent. Matches what _read_mutable_sources already
            # never shows the editor in the first place.
            return False
        if self.mutable_exclude is not None:
            return not is_excluded(Path(rel_path).as_posix(), self.mutable_exclude)
        if rel_path in MUTABLE_FILES:
            return True
        if len(parts) == 2 and parts[0] in MUTABLE_DIRS and parts[1].endswith(".py"):
            return True
        return False

    def _run_validators(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        for validator in self.validators:
            errors.extend(validator.validate(out_dir, base_dir))
        return errors

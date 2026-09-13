"""The agentic editor's tools: ``bash``, ``editor``, ``validate`` and
``submit_self_improvement``.

Schemas use the Anthropic ``input_schema`` shape (what ``call_llm``'s
normaliser and the other editors use); each tool is a callable taking the
model's arguments as kwargs and returning a string, exactly like HGM's
``tool_function``. Every failure is returned as a string starting with
``Error:`` so the loop never breaks and the model can react.

``bash`` mirrors HGM's tool (fresh shell per call) plus exit code and
head+tail truncation. ``editor`` replaces HGM's whole-file ``edit`` with the
Anthropic text-editor operations — ``view`` (with line ranges), ``create``,
``str_replace`` (unique match) and ``insert`` — so the model makes targeted
edits instead of rewriting files. ``validate`` runs the round's validators on
demand (the fast "unit test"; running the task agent on cases is impossible
inside the sandbox and deliberately not offered).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .policy import PathPolicy, resolve
from .sandbox import Sandbox

SUBMIT_TOOL_NAME = "submit_self_improvement"
VALIDATE_TOOL_NAME = "validate"

SUBMIT_TOOL: dict[str, Any] = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Finish the session. Your edits are already on disk — this call "
        "submits only a short summary of the self-improvement you made. The "
        "validators run on this call; if they report errors your workspace is "
        "kept as-is: fix them and call this again. When a belief document "
        "exists, include `prediction` against one of its belief ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "optimization_goal": {
                "type": "string",
                "description": "One line: what this edit is meant to improve.",
            },
            "proposed_changes": {
                "type": "string",
                "description": "What you changed, file by file, in a few lines.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this change should raise the score (the evidence).",
            },
            "prediction": {
                "type": "object",
                "description": (
                    "Optional. Which belief this edit tests and what you expect."
                ),
                "properties": {
                    "belief_id": {"type": "string"},
                    "expected_direction": {
                        "type": "string", "enum": ["up", "down", "neutral"],
                    },
                    "expected_delta": {"type": "number"},
                    "why": {"type": "string"},
                },
            },
        },
        "required": ["optimization_goal", "proposed_changes"],
    },
}

VALIDATE_TOOL: dict[str, Any] = {
    "name": VALIDATE_TOOL_NAME,
    "description": (
        "Run the framework's validators on the current workspace (syntax, "
        "run_task signature, import rules, schema/wrapper consistency, "
        "immutable files unchanged, and a real import of workflow.py and "
        "every mutable tool in a subprocess). Fast, no model calls, does not "
        "count as a submission. Returns the same error list a submission "
        "would — use it before you submit."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def bash_tool_info(*, bash_timeout_s: float, max_output_chars: int) -> dict[str, Any]:
    return {
        "name": "bash",
        "description": (
            "Run a bash command in a fresh, sandboxed shell.\n"
            "* No internet access. Only the workspace paths listed in the task "
            "message exist; everything else is absent or read-only.\n"
            "* Each call starts a NEW shell in the task_agent directory — cwd, "
            "variables and background processes do NOT persist between calls "
            "(chain with && or ;).\n"
            "* `python3` is available with PYTHONPATH preset, so "
            "`python3 -c \"import workflow\"` works from the default cwd. The "
            "task agent cannot be run on cases here (no model access, no "
            "database) — use `validate`, import checks and small scratch "
            "scripts instead.\n"
            "* To inspect a line range of a file, use `sed -n 10,25p /abs/path`.\n"
            f"* Avoid commands with very large output; output is truncated to "
            f"{max_output_chars} chars (head and tail kept). Commands are killed "
            f"after {bash_timeout_s:g}s — no servers or long-lived processes.\n"
            "* The task_agent directory is read-only except the mutable files "
            "themselves: `sed -i`, `mv`, `rm` or writing new files there fail "
            "(\"Read-only file system\" / \"Device or resource busy\"). Make "
            "edits with the `editor` tool; put scratch files in the scratch dir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
            },
            "required": ["command"],
        },
    }


def editor_tool_info(*, max_view_chars: int) -> dict[str, Any]:
    return {
        "name": "editor",
        "description": (
            "View and edit files with targeted operations. Absolute paths only.\n"
            "* `view`: a file is shown with line numbers (`cat -n` style); pass "
            "`view_range` [start, end] (end -1 = EOF) for a slice. A directory "
            "is listed up to 2 levels deep, hidden entries excluded.\n"
            "* `create`: write a NEW file (`file_text`); fails if the path "
            "exists. Only allowed for new mutable tools and scratch files.\n"
            "* `str_replace`: replace `old_str` with `new_str` in a writable "
            "file. `old_str` must match EXACTLY ONCE (copy it verbatim from a "
            "`view`, including indentation); add surrounding lines to make it "
            "unique. This is how you edit — there is no whole-file overwrite.\n"
            "* `insert`: insert `new_str` after line `insert_line` (0 = top).\n"
            f"* Long output is truncated at {max_view_chars} chars and marked "
            "`<response clipped>`; use `view_range` to see the rest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                },
                "path": {"type": "string", "description": "Absolute path."},
                "file_text": {
                    "type": "string", "description": "create: full content of the new file.",
                },
                "old_str": {
                    "type": "string", "description": "str_replace: exact text to replace (must be unique).",
                },
                "new_str": {
                    "type": "string", "description": "str_replace / insert: replacement or inserted text.",
                },
                "insert_line": {
                    "type": "integer", "description": "insert: line number to insert after (0 = top).",
                },
                "view_range": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "view: [start_line, end_line], 1-based, end -1 = EOF.",
                },
            },
            "required": ["command", "path"],
        },
    }


# ---------------------------------------------------------------------- #
# Text helpers
# ---------------------------------------------------------------------- #

def truncate_head_tail(text: str, limit: int) -> str:
    """Keep ~60% head and ~40% tail of an over-long tool result."""
    if limit <= 0 or len(text) <= limit:
        return text
    head_n = int(limit * 0.6)
    tail_n = limit - head_n
    dropped = len(text) - head_n - tail_n
    return (text[:head_n] + f"\n…[truncated {dropped} chars]…\n"
            + (text[-tail_n:] if tail_n > 0 else ""))


def format_numbered(content: str, *, start: int = 1) -> str:
    lines = content.expandtabs().split("\n")
    return "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(lines))


def _snippet(lines: list[str], center_start: int, center_end: int, radius: int = 4) -> str:
    """Numbered lines around an edited region (0-based, end exclusive)."""
    lo = max(0, center_start - radius)
    hi = min(len(lines), center_end + radius)
    return format_numbered("\n".join(lines[lo:hi]), start=lo + 1)


# ---------------------------------------------------------------------- #
# Tool implementations
# ---------------------------------------------------------------------- #

class BashTool:
    def __init__(self, sandbox: Sandbox, *, max_output_chars: int = 20000) -> None:
        self.sandbox = sandbox
        self.max_output_chars = int(max_output_chars)

    def __call__(self, command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string"
        res = self.sandbox.run(command)
        parts: list[str] = []
        if res.stdout:
            parts.append(res.stdout.rstrip("\n"))
        if res.stderr.strip():
            parts.append("\nError:\n" + res.stderr.rstrip("\n"))
        if res.timed_out:
            parts.append(f"\n[killed: exceeded {self.sandbox.bash_timeout_s:g}s]")
        elif res.returncode not in (0, None):
            parts.append(f"\n[exit code {res.returncode}]")
        text = "".join(parts).strip()
        return truncate_head_tail(text, self.max_output_chars) or "(no output)"


class EditorTool:
    def __init__(self, policy: PathPolicy, *, max_view_chars: int = 40000) -> None:
        self.policy = policy
        self.max_view_chars = int(max_view_chars)

    def __call__(
        self,
        command: str,
        path: str,
        file_text: Any = None,
        old_str: Any = None,
        new_str: Any = None,
        insert_line: Any = None,
        view_range: Any = None,
    ) -> str:
        try:
            target = resolve(path)
        except ValueError as exc:
            return str(exc)
        if command == "view":
            return self._view(target, view_range)
        if command == "create":
            return self._create(target, file_text)
        if command == "str_replace":
            return self._str_replace(target, old_str, new_str)
        if command == "insert":
            return self._insert(target, insert_line, new_str)
        return (f"Error: unknown command {command!r}; expected one of "
                "view, create, str_replace, insert")

    # -- view ------------------------------------------------------------ #

    def _view(self, target: Path, view_range: Any) -> str:
        if not self.policy.can_read(target):
            return (f"Error: {target} is not readable; readable roots are "
                    "listed in the workspace map")
        if not target.exists():
            return f"Error: {target} does not exist"
        if target.is_dir():
            body = self.policy.list_dir(target, depth=2)
            return self._clip(
                f"Here's the files and directories up to 2 levels deep in "
                f"{target}, excluding hidden items:\n{body}", n_lines=None,
            )
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error: could not read {target}: {exc}"
        lines = content.split("\n")
        n = len(lines)
        start = 1
        if view_range is not None:
            rng = self._coerce_range(view_range, n)
            if isinstance(rng, str):
                return rng
            a, b = rng
            lines = lines[a - 1:b]
            start = a
        numbered = format_numbered("\n".join(lines), start=start)
        return self._clip(
            f"Here's the result of running `cat -n` on {target}:\n{numbered}\n",
            n_lines=n,
        )

    @staticmethod
    def _coerce_range(view_range: Any, n_lines: int) -> tuple[int, int] | str:
        try:
            if isinstance(view_range, str):
                parts = [int(float(x)) for x in view_range.strip("[]() ").split(",")]
            else:
                parts = [int(float(x)) for x in view_range]
        except (TypeError, ValueError):
            return f"Error: invalid view_range {view_range!r}; expected [start, end]"
        if len(parts) != 2:
            return f"Error: invalid view_range {view_range!r}; expected [start, end]"
        a, b = parts
        if b == -1:
            b = n_lines
        if a < 1 or a > n_lines or b < a:
            return (f"Error: invalid view_range [{a}, {b}]; file has {n_lines} "
                    f"lines (start >= 1, end >= start or -1)")
        return a, min(b, n_lines)

    def _clip(self, text: str, *, n_lines: int | None) -> str:
        if len(text) <= self.max_view_chars:
            return text
        hint = ("use view_range=[start,end] to see the rest"
                + (f"; file has {n_lines} lines" if n_lines is not None else ""))
        return text[: self.max_view_chars] + f"\n<response clipped — {hint}>"

    # -- create ------------------------------------------------------------ #

    def _create(self, target: Path, file_text: Any) -> str:
        if file_text is None:
            return "Error: missing required 'file_text' for create"
        if target.exists():
            return (f"Error: cannot create, {target} already exists "
                    "(use str_replace/insert to edit it)")
        if not self.policy.can_write(target):
            return (f"Error: {target} is not writable; new files may only be "
                    f"created under {self._mutable_dirs()} (*.py) or "
                    f"{self.policy.scratch}/")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(file_text), encoding="utf-8")
        except OSError as exc:
            return f"Error: failed to write {target}: {exc}"
        return f"File created successfully at: {target}"

    # -- str_replace ------------------------------------------------------- #

    def _str_replace(self, target: Path, old_str: Any, new_str: Any) -> str:
        if old_str is None or old_str == "":
            return "Error: missing required 'old_str' for str_replace"
        err = self._writable_existing(target)
        if err:
            return err
        content = target.read_text(encoding="utf-8")
        old = str(old_str)
        new = "" if new_str is None else str(new_str)
        count = content.count(old)
        if count == 0:
            return (f"Error: no replacement performed — old_str did not appear "
                    f"verbatim in {target} (check whitespace/indentation; use "
                    "view to copy the exact text)")
        if count > 1:
            at = [content[:i].count("\n") + 1
                  for i in _find_all(content, old)]
            return (f"Error: no replacement performed — old_str occurs {count} "
                    f"times in {target} at lines {at}; include more surrounding "
                    "context so it is unique")
        idx = content.index(old)
        updated = content[:idx] + new + content[idx + len(old):]
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"Error: failed to write {target}: {exc}"
        first = content[:idx].count("\n")
        last = first + new.count("\n") + 1
        return (f"The file {target} has been edited. Snippet of the edited "
                f"region:\n{_snippet(updated.split(chr(10)), first, last)}")

    # -- insert ------------------------------------------------------------ #

    def _insert(self, target: Path, insert_line: Any, new_str: Any) -> str:
        if new_str is None:
            return "Error: missing required 'new_str' for insert"
        err = self._writable_existing(target)
        if err:
            return err
        content = target.read_text(encoding="utf-8")
        lines = content.split("\n")
        n = len(lines) - (1 if content.endswith("\n") else 0)
        try:
            at = int(insert_line)
        except (TypeError, ValueError):
            return f"Error: insert_line must be an integer between 0 and {n}"
        if at < 0 or at > n:
            return f"Error: insert_line must be between 0 and {n}"
        inserted = str(new_str).split("\n")
        if inserted and inserted[-1] == "":
            inserted = inserted[:-1]
        lines[at:at] = inserted
        try:
            target.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            return f"Error: failed to write {target}: {exc}"
        return (f"The file {target} has been edited. Snippet of the edited "
                f"region:\n{_snippet(lines, at, at + len(inserted))}")

    # -- helpers ----------------------------------------------------------- #

    def _writable_existing(self, target: Path) -> str | None:
        if not self.policy.can_write(target):
            return (f"Error: {target} is not writable; writable paths: "
                    f"{', '.join(str(p) for p in self.policy.write_files)}, "
                    f"{self._mutable_dirs()}/*.py, {self.policy.scratch}/")
        if not target.exists():
            return f"Error: {target} does not exist (use create for a new file)"
        if target.is_dir():
            return f"Error: {target} is a directory"
        return None

    def _mutable_dirs(self) -> str:
        return ", ".join(str(d) for d in self.policy.write_dirs if d != self.policy.scratch)


class ValidateTool:
    def __init__(self, run_validators: Callable[[], list[str]]) -> None:
        self.run_validators = run_validators

    def __call__(self) -> str:
        errors = self.run_validators()
        if not errors:
            return "All validators passed."
        return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)


def _find_all(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    i = haystack.find(needle)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


# ---------------------------------------------------------------------- #
# Dispatch
# ---------------------------------------------------------------------- #

class ToolSet:
    """Name → (schema, callable). ``call`` has HGM ``process_tool_call``
    semantics: never raises, every failure is an ``Error: …`` string."""

    def __init__(self, tools: list[tuple[dict[str, Any], Callable[..., str]]]) -> None:
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., str]]] = {
            info["name"]: (info, fn) for info, fn in tools
        }

    def infos(self) -> list[dict[str, Any]]:
        return [info for info, _ in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: Any) -> str:
        if name not in self._tools:
            return (f"Error: Tool {name!r} not found. Available: "
                    f"{', '.join([*self._tools, SUBMIT_TOOL_NAME])}")
        if not isinstance(args, dict):
            args = {}
        if "_raw_arguments" in args:
            raw = str(args.get("_raw_arguments"))[:300]
            return (f"Error: could not parse tool arguments as JSON: {raw}. "
                    "Re-issue the call with valid JSON arguments")
        fn = self._tools[name][1]
        try:
            return fn(**args)
        except TypeError as exc:
            return f"Error: bad arguments for tool {name!r}: {exc}"
        except Exception as exc:  # noqa: BLE001 - tool failures go back to the model
            return f"Error executing tool {name!r}: {exc}"

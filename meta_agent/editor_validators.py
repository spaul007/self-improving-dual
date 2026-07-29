"""Static and dynamic validators applied to a round folder before it is
evaluated.

Each validator is a class with ``validate(out_dir, base_dir) -> list[str]`` and
self-registers under ``"validator"``. An empty error list means the validator
passed; any non-empty list short-circuits the round.

The first six validators are AST-based (parse, signature, imports, schema,
mutable-tool imports, immutable-files). The seventh, ``LoadTestValidator``,
spawns a subprocess that actually imports the agent's mutable modules — so
that ``NameError``, ``ImportError`` on missing dependencies, exceptions
raised at module top-level, etc. surface here instead of crashing every
per-case subprocess at evaluation time.
"""
from __future__ import annotations

import ast
import filecmp
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

from .registry import register

MUTABLE_FILES = {"workflow.py", "tool_wrapper.py", "tools_schema.json"}
MUTABLE_DIRS = {"mutable_tools"}


def is_excluded(rel_path: str, excludes: list[str]) -> bool:
    """Prefix-match ``rel_path`` (POSIX-style, relative to the seed dir)
    against an exclude list (see ``config.FrameworkConfig.mutable_exclude``).
    An entry matches either the exact path or anything under it as a
    directory (``"environment/"`` or ``"environment"`` both exclude
    ``environment/anomaly_injection.py``, but not a sibling file that merely
    shares the prefix like ``environment_notes.md``). Shared by
    ``agent_editor.py`` and ``ImmutableFilesValidator`` below so the editor's
    idea of "excluded" and the validator's enforcement of it can never
    drift apart."""
    rel_path = rel_path.replace("\\", "/")
    for entry in excludes:
        e = entry.replace("\\", "/").rstrip("/")
        if rel_path == e or rel_path.startswith(e + "/"):
            return True
    return False


def _stdlib_names() -> set[str]:
    return set(getattr(sys, "stdlib_module_names", ()))


def _all_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imports_in(tree: ast.AST) -> Iterable[str]:
    """Yield top-level module names imported by the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                yield ""
            elif node.module:
                yield node.module


def _allowed_import(module: str, allowed_prefixes: tuple[str, ...]) -> bool:
    if not module:
        return True
    top = module.split(".")[0]
    if top in _stdlib_names() or top in {"__future__"}:
        return True
    return any(module == p or module.startswith(p + ".") for p in allowed_prefixes)


@register("validator", "syntax")
class SyntaxValidator:
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        agent_dir = out_dir / "task_agent"
        for py in _all_python_files(agent_dir):
            try:
                ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except SyntaxError as exc:
                rel = py.relative_to(agent_dir)
                errors.append(f"{rel}: SyntaxError at line {exc.lineno}: {exc.msg}")
        return errors


@register("validator", "signature")
class SignatureValidator:
    """<workflow_filename> must define run_task(task) -> AgentOutput.

    The runner accepts either an AgentOutput return or a bare value (which
    it wraps), so the return type is not AST-enforced here. The argument
    must be a single positional arg named ``task``.

    ``workflow_filename`` (default ``"workflow.py"``, every existing project
    relies on this) overrides which file gets the AST check -- set it when
    the framework-mandated ``workflow.py`` (required verbatim by
    ``platform_core.runner``'s hardcoded ``import workflow``) is kept as a
    trivial, permanently-excluded re-export, and the real
    `def run_task(task): ...` implementation lives in a separate,
    also-excluded file instead (e.g. db_mas's `workflow_adapter.py`) -- a
    bare `from workflow_adapter import run_task` in workflow.py is an
    `ast.ImportFrom` node, not a `FunctionDef`, so it would never satisfy
    this check on `workflow.py` itself; redirecting the check to the file
    that actually defines the function is the fix, not requiring
    workflow.py to also contain a wrapper function.
    """

    def __init__(self, *, workflow_filename: str = "workflow.py") -> None:
        self.workflow_filename = workflow_filename

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        path = out_dir / "task_agent" / self.workflow_filename
        if not path.exists():
            return [f"{self.workflow_filename} is missing"]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            return []  # SyntaxValidator covers it
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run_task":
                args = node.args
                if args.vararg or args.kwarg or args.kwonlyargs:
                    return ["run_task must take exactly one positional arg"]
                if len(args.args) != 1 or args.args[0].arg != "task":
                    return [
                        "run_task signature must be run_task(task) -> AgentOutput"
                    ]
                return []
        return [f"{self.workflow_filename} does not define run_task at module level"]


@register("validator", "imports")
class ImportValidator:
    """workflow.py and tool_wrapper.py may only reach the platform via the
    sanctioned entry points."""

    WORKFLOW_ALLOWED = (
        "platform_core.llm_wrapper",
        "platform_core.runner",
        "platform_core.trace",
        "tool_wrapper",
    )
    WRAPPER_ALLOWED = ("platform_core.tools", "platform_core.trace", "platform_core", "mutable_tools")

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        agent_dir = out_dir / "task_agent"
        for fname, allowed in (
            ("workflow.py", self.WORKFLOW_ALLOWED),
            ("tool_wrapper.py", self.WRAPPER_ALLOWED),
        ):
            path = agent_dir / fname
            if not path.exists():
                errors.append(f"{fname} is missing")
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for mod in _imports_in(tree):
                if mod and mod.startswith("platform_core") and not _allowed_import(mod, allowed):
                    errors.append(
                        f"{fname}: forbidden import {mod!r} "
                        f"(allowed platform paths: {', '.join(allowed)})"
                    )
        return errors


@register("validator", "schema_wrapper_consistency")
class SchemaWrapperConsistencyValidator:
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        agent_dir = out_dir / "task_agent"
        schema_path = agent_dir / "tools_schema.json"
        if not schema_path.exists():
            return ["tools_schema.json is missing"]
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [f"tools_schema.json is not valid JSON: {exc}"]
        if not isinstance(schema, list):
            return ["tools_schema.json must be a JSON array of tool definitions"]

        names: list[str] = []
        for entry in schema:
            if isinstance(entry, dict):
                if entry.get("type") == "function" and isinstance(entry.get("function"), dict):
                    n = entry["function"].get("name")
                else:
                    n = entry.get("name")
                if n:
                    names.append(n)

        try:
            from platform_core import tools as immutable_tools
            immutable_set = set(immutable_tools.all_schemas())
        except Exception as exc:
            return [f"could not load platform_core.tools: {exc}"]

        mutable_dir = agent_dir / "mutable_tools"
        mutable_set: set[str] = set()
        if mutable_dir.exists():
            for py in mutable_dir.glob("*.py"):
                if py.name == "__init__.py":
                    continue
                mutable_set.add(py.stem)

        for name in names:
            in_imm = name in immutable_set
            in_mut = name in mutable_set
            if in_imm and in_mut:
                errors.append(
                    f"name collision: {name!r} exists as both immutable and mutable"
                )
            elif not in_imm and not in_mut:
                errors.append(
                    f"tools_schema.json lists {name!r} but no immutable tool nor "
                    f"mutable_tools/{name}.py provides it"
                )
        return errors


@register("validator", "mutable_tool_imports")
class MutableToolImportValidator:
    ALLOWED_PREFIXES = ("platform_core.tools", "platform_core.trace", "mutable_tools")

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        mutable_dir = out_dir / "task_agent" / "mutable_tools"
        if not mutable_dir.exists():
            return errors
        for py in mutable_dir.glob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for mod in _imports_in(tree):
                if not mod:
                    continue
                if mod.startswith("platform_core") and not _allowed_import(
                    mod, self.ALLOWED_PREFIXES
                ):
                    errors.append(
                        f"mutable_tools/{py.name}: forbidden import {mod!r} "
                        f"(mutable tools may only import platform_core.tools, "
                        f"sibling mutable_tools.*, or stdlib)"
                    )
        return errors


@register("validator", "mutable_tool_routing")
class MutableToolRoutingValidator:
    """``tool_wrapper.py`` must dispatch mutable tools via
    ``platform_core.tools.call_mutable_tool`` — never by importing
    ``mutable_tools.<name>`` and calling its ``run()`` directly.

    Routing through ``call_mutable_tool`` is what records mutable-tool calls in
    the trace (``tool_call``/``tool_result``), so the feedback gatherer and the
    behavior summarizer can see editor-added tools. This validator hard-enforces
    editor hard-rule #5 so a wrapper rewrite can't silently lose that tracing.

    Heuristic (low false-positive): if the wrapper does any *direct* mutable
    dispatch — ``importlib.import_module("mutable_tools…")`` or
    ``import``/``from`` of a ``mutable_tools`` submodule — it must ALSO call
    ``call_mutable_tool``. A wrapper that doesn't dispatch mutable tools
    directly (the seed routes entirely through ``call_mutable_tool``) passes.
    """

    @staticmethod
    def _is_mutable_str(node: ast.AST) -> bool:
        """True if ``node`` is a string (or f-string) starting 'mutable_tools'."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.startswith("mutable_tools")
        if isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            return (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("mutable_tools")
            )
        return False

    def _dispatches_mutable_directly(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            # importlib.import_module("mutable_tools...") / import_module(f"mutable_tools.{x}")
            if isinstance(node, ast.Call):
                func = node.func
                is_import_module = (
                    isinstance(func, ast.Attribute) and func.attr == "import_module"
                ) or (isinstance(func, ast.Name) and func.id == "import_module")
                if is_import_module and node.args and self._is_mutable_str(node.args[0]):
                    return True
            # from mutable_tools[.x] import ...
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "mutable_tools" or mod.startswith("mutable_tools."):
                    return True
            # import mutable_tools.x  (a specific submodule, not the bare package)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mutable_tools."):
                        return True
        return False

    @staticmethod
    def _calls_call_mutable_tool(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "call_mutable_tool":
                    return True
                if isinstance(func, ast.Name) and func.id == "call_mutable_tool":
                    return True
        return False

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        path = out_dir / "task_agent" / "tool_wrapper.py"
        if not path.exists():
            return []  # ImportValidator reports the missing file
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            return []  # SyntaxValidator covers it
        if self._dispatches_mutable_directly(tree) and not self._calls_call_mutable_tool(tree):
            return [
                "tool_wrapper.py dispatches a mutable tool directly (importing "
                "mutable_tools.<name> and calling run()) without routing through "
                "platform_core.tools.call_mutable_tool. Route mutable tools via "
                "call_mutable_tool(tool_name, **kwargs) so their calls are recorded "
                "in the trace (editor hard-rule #5)."
            ]
        return []


_ALWAYS_IGNORE_DIRS = {"__pycache__", "results"}


@register("validator", "immutable_files")
class ImmutableFilesValidator:
    """Every file in out_dir/task_agent that is *not* mutable must be
    byte-identical to the corresponding file in base_dir/task_agent.

    Two modes, mirroring ``AgentEditor``: ``mutable_exclude=None`` (default)
    is the legacy include-list — mutable means "in MUTABLE_FILES or under
    MUTABLE_DIRS". Setting ``mutable_exclude`` (see
    ``config.FrameworkConfig.mutable_exclude``) flips to an exclude-list —
    mutable means "NOT matched by ``is_excluded``". Either way, `results/`
    (generated runtime output, e.g. db-mas's own per-task JSON writes) and
    `__pycache__` are always skipped entirely -- neither mutable nor
    immutable, just ignored.
    """

    def __init__(self, mutable_exclude: list[str] | None = None) -> None:
        self.mutable_exclude = mutable_exclude

    def _is_mutable(self, rel: Path) -> bool:
        if self.mutable_exclude is not None:
            return not is_excluded(rel.as_posix(), self.mutable_exclude)
        top = rel.parts[0]
        return top in MUTABLE_FILES or top in MUTABLE_DIRS

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        out_root = out_dir / "task_agent"
        base_root = base_dir / "task_agent"
        if not base_root.exists():
            return errors  # round 0; nothing to compare against

        for path in out_root.rglob("*"):
            if path.is_dir() or set(path.parts) & _ALWAYS_IGNORE_DIRS:
                continue
            rel = path.relative_to(out_root)
            if self._is_mutable(rel):
                continue
            base_path = base_root / rel
            if not base_path.exists():
                errors.append(f"forbidden new file: {rel} (outside MUTABLE region)")
            elif not filecmp.cmp(path, base_path, shallow=False):
                errors.append(f"forbidden modification: {rel} differs from base round")
        for path in base_root.rglob("*"):
            if path.is_dir() or set(path.parts) & _ALWAYS_IGNORE_DIRS:
                continue
            rel = path.relative_to(base_root)
            if self._is_mutable(rel):
                continue
            if not (out_root / rel).exists():
                errors.append(f"forbidden deletion: {rel} (outside MUTABLE region)")
        return errors


@register("validator", "load_test")
class LoadTestValidator:
    """Imports the agent's mutable modules in a sandboxed subprocess.

    Catches what static validators can't: ``NameError`` from a typo, an
    ``ImportError`` on a missing dependency, exceptions raised at
    module top-level, mutable_tools/*.py that fail their own imports.

    Spawns a child Python (rather than importing in-process) so a
    broken candidate can't poison the parent's module cache. Uses the
    same PYTHONPATH shape as ``SubprocessEvaluator`` so the agent's
    ``from platform_core.X import Y`` lines resolve identically.
    """

    DRIVER = textwrap.dedent("""
        import importlib, json, os, sys, traceback
        errors = []
        try:
            importlib.import_module("workflow")
        except Exception:
            errors.append("workflow.py: " + traceback.format_exc())
        if os.path.isdir("mutable_tools"):
            for fname in sorted(os.listdir("mutable_tools")):
                if fname.endswith(".py") and fname != "__init__.py":
                    try:
                        importlib.import_module("mutable_tools." + fname[:-3])
                    except Exception:
                        errors.append(
                            "mutable_tools/" + fname + ": " + traceback.format_exc()
                        )
        sys.stdout.write(json.dumps(errors))
    """).strip()

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        self.timeout_s = float(timeout_s)

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        agent_dir = out_dir / "task_agent"
        if not agent_dir.exists():
            return ["task_agent directory missing"]
        env = self._child_env()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", self.DRIVER],
                capture_output=True,
                text=True,
                cwd=str(agent_dir),
                env=env,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return [f"load_test: timed out after {self.timeout_s}s"]
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[:500]
            return [f"load_test: child exit {proc.returncode}: {tail}"]
        try:
            errs = json.loads(proc.stdout.strip() or "[]")
        except json.JSONDecodeError:
            return [f"load_test: bad child output: {proc.stdout[:200]!r}"]
        return [str(e)[:1000] for e in errs]

    def _child_env(self) -> dict[str, str]:
        # Reuse the evaluator's env shape: PYTHONPATH must include the
        # platform_core parent so ``from platform_core.X import Y`` works.
        from .evaluator import _platform_core_parent

        env = os.environ.copy()
        platform_parent = str(_platform_core_parent())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{platform_parent}{os.pathsep}{existing}" if existing else platform_parent
        )
        return env


DEFAULT_VALIDATOR_NAMES = [
    "syntax",
    "signature",
    "imports",
    "schema_wrapper_consistency",
    "mutable_tool_imports",
    "mutable_tool_routing",
    "immutable_files",
    "load_test",
]

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
    """workflow.py must define run_task(task) -> AgentOutput.

    The runner accepts either an AgentOutput return or a bare value (which
    it wraps), so the return type is not AST-enforced here. The argument
    must be a single positional arg named ``task``.
    """

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        path = out_dir / "task_agent" / "workflow.py"
        if not path.exists():
            return ["workflow.py is missing"]
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
        return ["workflow.py does not define run_task at module level"]


@register("validator", "imports")
class ImportValidator:
    """workflow.py and tool_wrapper.py may only reach the platform via the
    sanctioned entry points."""

    WORKFLOW_ALLOWED = (
        "platform_core.llm_wrapper",
        "platform_core.runner",
        "tool_wrapper",
    )
    WRAPPER_ALLOWED = ("platform_core.tools", "platform_core", "mutable_tools")

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
    ALLOWED_PREFIXES = ("platform_core.tools", "mutable_tools")

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


@register("validator", "immutable_files")
class ImmutableFilesValidator:
    """Every file in out_dir/task_agent that is *not* a mutable file or under a
    mutable directory must be byte-identical to the corresponding file in
    base_dir/task_agent."""

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errors: list[str] = []
        out_root = out_dir / "task_agent"
        base_root = base_dir / "task_agent"
        if not base_root.exists():
            return errors  # round 0; nothing to compare against

        for path in out_root.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(out_root)
            top = rel.parts[0]
            if top in MUTABLE_FILES or top in MUTABLE_DIRS:
                continue
            base_path = base_root / rel
            if not base_path.exists():
                errors.append(f"forbidden new file: {rel} (outside MUTABLE region)")
            elif not filecmp.cmp(path, base_path, shallow=False):
                errors.append(f"forbidden modification: {rel} differs from base round")
        for path in base_root.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(base_root)
            top = rel.parts[0]
            if top in MUTABLE_FILES or top in MUTABLE_DIRS:
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
    "immutable_files",
    "load_test",
]

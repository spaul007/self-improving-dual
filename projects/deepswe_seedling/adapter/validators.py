"""Edit validators for the seedling project. Registered on import (benchmark/scorer.py).

Each runs before an edited agent is evaluated (and on every agentic
``run_code_validators`` call); a non-empty return rejects the edit and is fed back to
the editor. Together they must take well under ~90 s: a rejected edit costs seconds, an
accepted broken edit costs a 1-3 h DeepSWE case per train task.

  seedling_selftest       curated safety-invariant tests (adapter/selftest/run_invariants.py)
  seedling_dry_run        the real multi-role pipeline against a fake container + fake LLM;
                          enforces the ROLE MANDATE (>=1 PATCH and >=1 VERIFY run) and A0
  seedling_settings_guard settings that would burn a whole batch (effort=high -> HTTP 400)
  seedling_host_isolation seedling's Python runs ON THE HOST inside pier: editable code may
                          not read the filesystem/network/processes outside its container
                          API (hidden tests and earlier verifier output live on the host)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from meta_agent.registry import register

HERE = Path(__file__).resolve().parent
DEFAULT_PIER_PY = "/users/n.tzou/.local/share/uv/tools/datacurve-pier/bin/python"
# Frozen seedling modules (also in the config's mutable_exclude). Host isolation scans
# everything else under seedling/.
FROZEN = {"agent.py", "execpool.py", "gitops.py", "trajectory.py", "deadline.py", "llm.py",
          "tools/__init__.py"}


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "META_AGENT_TRACE_PATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


@register("validator", "seedling_selftest")
class SeedlingSelfTestValidator:
    def __init__(self, *, pier_python: str = DEFAULT_PIER_PY, timeout_s: int = 180) -> None:
        self.pier_python, self.timeout_s = pier_python, timeout_s

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        agent = out_dir / "task_agent"
        rc, text = _run([self.pier_python, str(HERE / "selftest" / "run_invariants.py"), str(agent)],
                        agent, self.timeout_s)
        if rc == 0:
            return []
        fails = [l for l in text.splitlines() if l.startswith("FAIL") or "INVARIANT" in l]
        return ["seedling_selftest: " + (l[:400]) for l in (fails[-12:] or [text[-800:]])]


_DRY_CACHE: dict[str, tuple[Optional[dict], str]] = {}


def _tree_key(agent: Path) -> str:
    h = hashlib.sha256()
    for f in sorted((agent / "seedling").rglob("*")):
        if f.is_file() and f.suffix in (".py", ".md") and "__pycache__" not in f.parts:
            h.update(f.relative_to(agent).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def dry_run_result(agent: Path, pier_python: str, timeout_s: int) -> tuple[Optional[dict], str]:
    """Dry-run once per distinct tree content (dry_run + settings_guard share it)."""
    key = _tree_key(agent)
    if key in _DRY_CACHE:
        return _DRY_CACHE[key]
    rc, text = _run([pier_python, str(HERE / "selftest" / "dry_run.py"), str(agent)], agent, timeout_s)
    res: tuple[Optional[dict], str] = (None, text)
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                res = (json.loads(line), text)
            except ValueError:
                pass
            break
    _DRY_CACHE[key] = res
    return res


@register("validator", "seedling_dry_run")
class SeedlingDryRunValidator:
    def __init__(self, *, pier_python: str = DEFAULT_PIER_PY, timeout_s: int = 240) -> None:
        self.pier_python, self.timeout_s = pier_python, timeout_s

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        res, text = dry_run_result(out_dir / "task_agent", self.pier_python, self.timeout_s)
        if res is None:
            return ["seedling_dry_run: the pipeline did not run to completion: " + text[-900:]]
        return check_dry_run(res)


def check_dry_run(res: dict) -> list[str]:
    errs: list[str] = []
    if res.get("raised"):
        errs.append(f"seedling_dry_run: run() raised {res['raised']}")
    if res.get("outcome") != "completed":
        errs.append(f"seedling_dry_run: outcome={res.get('outcome')!r} (expected 'completed'; "
                    "'contained:X' means an exception inside the pipeline/roles)")
    rs = res.get("role_stats") or []
    roles = [r.get("role") for r in rs]
    for need in ("patch", "verify"):
        if need not in roles:
            errs.append(f"seedling_dry_run: ROLE MANDATE violated -- no {need.upper()} role ran "
                        f"(roles run: {roles}). seedling must stay multi-agent: PATCH and VERIFY.")
    for r in rs:
        if r.get("report_ok") is False:
            errs.append(f"seedling_dry_run: {r.get('role')}.{r.get('attempt')} failed to deliver its "
                        "structured `finish` report even though the scripted model always answers "
                        "the report call correctly -- the report path is broken")
        if r.get("missing_keys"):
            errs.append(f"seedling_dry_run: role_stats for {r.get('role')} lacks {r['missing_keys']} "
                        "(the scorer and meta-agent read these)")
    seen = set(res.get("sys_sha_seen") or [])
    for role, sha in (res.get("role_prompt_sha") or {}).items():
        if role in roles and sha not in seen:
            errs.append(f"seedling_dry_run: the {role.upper()} role's own system prompt never "
                        "reached the model as messages[0] (A0: every role must run under its "
                        "own prompt, including on a shared conversation)")
    for e in res.get("prompt_errors") or []:
        errs.append(f"seedling_dry_run: prompt load error {e}")
    return errs


@register("validator", "seedling_settings_guard")
class SeedlingSettingsGuardValidator:
    """Reads the candidate's settings through the dry run (same import path pier uses)."""

    def __init__(self, *, pier_python: str = DEFAULT_PIER_PY, timeout_s: int = 240) -> None:
        self.pier_python, self.timeout_s = pier_python, timeout_s

    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        errs = check_settings_source(out_dir / "task_agent" / "seedling")
        res, text = dry_run_result(out_dir / "task_agent", self.pier_python, self.timeout_s)
        if res is None:
            return errs + ["seedling_settings_guard: could not import settings: " + text[-500:]]
        return errs + check_settings(res.get("settings") or {})


def check_settings(s: dict) -> list[str]:
    errs = []
    if s.get("REASONING_EFFORT") not in ("low", "medium"):
        errs.append(f"settings: REASONING_EFFORT={s.get('REASONING_EFFORT')!r} -- must be 'low' or "
                    "'medium' (Qwen3.8 rejects 'high' with HTTP 400; 'xhigh' blows the wall budget)")
    mt = s.get("MAX_TOKENS")
    if not isinstance(mt, int) or not (16384 <= mt <= 32768):
        errs.append(f"settings: MAX_TOKENS={mt!r} -- must be in [16384, 32768] (must exceed a full "
                    "thinking block; a larger value is not accepted by the server budget)")
    fr = s.get("wall_frac") or {}
    multi = [fr.get(k) for k in ("baseline", "patch", "verify") if k in fr]
    if any(not isinstance(x, (int, float)) or x <= 0 for x in multi):
        errs.append(f"settings: every multi-mode role wall_frac must be > 0, got {fr}")
    elif sum(multi) > 1.0:
        errs.append(f"settings: baseline+patch+verify wall_frac sum {sum(multi):.2f} > 1.0")
    mpa = s.get("MAX_PATCH_ATTEMPTS")
    if not isinstance(mpa, int) or mpa < 1:
        errs.append(f"settings: MAX_PATCH_ATTEMPTS={mpa!r} must be an int >= 1")
    return errs


def check_settings_source(pkg: Path) -> list[str]:
    """No new SEEDLING_* env reads: settings must come from code, not a job's environment."""
    allowed = set()
    base = HERE.parent / "seed" / "seedling"
    for f in base.rglob("*.py"):
        allowed |= _env_names(f)
    errs = []
    for f in pkg.rglob("*.py"):
        new = _env_names(f) - allowed
        if new:
            errs.append(f"settings: {f.relative_to(pkg)} reads new environment variable(s) "
                        f"{sorted(new)} -- configuration must be in code")
    return errs


def _env_names(f: Path) -> set[str]:
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith("SEEDLING_"):
            names.add(node.value)
    return names


# ------------------------------------------------------------------ host isolation
FORBIDDEN_IMPORTS = {"subprocess", "socket", "urllib", "requests", "httpx", "http", "ftplib",
                     "shutil", "glob", "pty", "multiprocessing", "ctypes", "importlib", "pickle",
                     "marshal", "tempfile", "fcntl", "signal", "resource", "sqlite3", "zipfile",
                     "tarfile", "webbrowser", "asyncio.subprocess"}
FORBIDDEN_OS = {"system", "popen", "listdir", "scandir", "walk", "remove", "unlink", "rmdir",
                "rename", "replace", "chmod", "chown", "open", "fork", "kill", "makedirs", "mkdir",
                "symlink", "link", "startfile", "execv", "execve", "execl", "execlp", "execvp",
                "spawnl", "spawnv", "posix_spawn"}
FORBIDDEN_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "breakpoint", "input"}
FS_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes", "open", "glob", "rglob",
              "iterdir", "unlink", "rmdir", "mkdir", "touch", "rename", "replace", "symlink_to"}
FORBIDDEN_STRINGS = ("/groups", "/users/", "deep-swe", "sid_pier_jobs", "seedling_jobs",
                     "reward.json", "test.patch", "test-stdout", "ctrf.json", "/logs/verifier",
                     "verifier/")


@register("validator", "seedling_host_isolation")
class SeedlingHostIsolationValidator:
    def validate(self, out_dir: Path, base_dir: Path) -> list[str]:
        pkg = out_dir / "task_agent" / "seedling"
        errs: list[str] = []
        for f in sorted(pkg.rglob("*.py")):
            rel = f.relative_to(pkg).as_posix()
            if rel in FROZEN or "__pycache__" in f.parts:
                continue
            errs += [f"host_isolation: seedling/{rel}:{ln}: {msg}" for ln, msg in scan_source(f.read_text(encoding="utf-8"))]
        return errs[:20]


def scan_source(src: str) -> list[tuple[int, str]]:
    """(line, message) for every host-escape pattern. seedling's editable code talks to the
    world only through the container API (h.pool / tools) and the frozen LLM client."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[tuple[int, str]] = []
    # Names bound from an expression rooted at PROMPTS (e.g. ``p = PROMPTS / self.prompt_file``)
    # may be READ -- that is how a role loads its prompt file. Nothing else may touch the host.
    prompt_vars = {"PROMPTS"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and "PROMPTS" in ast.unparse(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id != "PROMPTS":
                    prompt_vars.add(t.id)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPTS"
                                                for t in node.targets):
            if ast.unparse(node.value) != "Path(__file__).parent / 'prompts'":
                out.append((node.lineno, "PROMPTS must stay Path(__file__).parent / 'prompts'"))
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORTS or a.name in FORBIDDEN_IMPORTS:
                    out.append((ln, f"import {a.name} is not allowed in editable code"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] in FORBIDDEN_IMPORTS:
                out.append((ln, f"from {mod} import ... is not allowed in editable code"))
            if mod == "os":
                for a in node.names:
                    if a.name in FORBIDDEN_OS:
                        out.append((ln, f"from os import {a.name} is not allowed"))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_BUILTINS:
                out.append((ln, f"builtin {f.id}() is not allowed in editable code"))
            elif isinstance(f, ast.Attribute):
                recv = ast.unparse(f.value)
                if recv in ("os", "os.path") and f.attr in FORBIDDEN_OS:
                    out.append((ln, f"os.{f.attr}() is not allowed in editable code"))
                elif f.attr in FS_METHODS and not (
                        f.attr in ("read_text", "read_bytes")
                        and (recv in prompt_vars or recv.startswith("PROMPTS"))):
                    out.append((ln, f"host filesystem access {recv}.{f.attr}() is not allowed "
                                    "(only the role prompt files under PROMPTS may be read)"))
                elif f.attr == "exec" and (recv.endswith("environment") or recv.endswith("env")):
                    out.append((ln, "environment.exec is only allowed in execpool.py"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value
            for s in FORBIDDEN_STRINGS:
                if s in low:
                    out.append((ln, f"string literal mentions host path {s!r}"))
                    break
            else:
                if low.startswith("../") or "/../" in low:
                    out.append((ln, "string literal with a '..' path component"))
    return out

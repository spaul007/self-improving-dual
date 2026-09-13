"""Docker-free confinement for the agentic editor's ``bash`` tool.

Each command runs in a fresh ``bash -c`` (this is what HGM's seed agent
actually does too — its ``BashSession`` is rebuilt on every call), wrapped in
bubblewrap (``bwrap``) when available:

* only the ``PathPolicy`` read roots are bind-mounted read-only, the write
  files/dirs read-write, plus the minimal system (``/usr``, ``/etc``, the
  Python prefix); everything else — including the project's ``benchmark/``
  and ``data/`` — simply does not exist inside the sandbox;
* no network (``--unshare-all``), a scrubbed environment (``--clearenv`` +
  a handful of ``--setenv``), so neither the model API nor the database is
  reachable and the task agent cannot be run on cases from inside;
* ``--remount-ro /`` last, so the scaffold directories bwrap creates for the
  bind targets are not writable either.

``mode="auto"`` probes bwrap once per process and falls back to a plain
cwd-confined subprocess (env scrubbed the same way) with a printed warning;
``mode="bwrap"`` refuses to run unconfined; ``mode="none"`` never probes.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .policy import PathPolicy

SANDBOX_MODES = ("auto", "bwrap", "none")
SYSTEM_RO_DIRS = ("/usr", "/etc")
# On merged-/usr systems these are symlinks into /usr and must be recreated
# as symlinks (bwrap cannot bind-mount over a symlink target it created).
SYSTEM_MAYBE_SYMLINK = ("/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx32")
# Env var names that must never reach the sandbox (fallback mode builds the
# env from scratch, so these only document the intent — nothing is copied).
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
PROBE_TIMEOUT_S = 10.0

_probe_lock = threading.Lock()
_probe_cache: Optional[tuple[bool, str]] = None
_warned = False


@dataclass
class BashResult:
    stdout: str
    stderr: str
    returncode: Optional[int]
    timed_out: bool
    elapsed_s: float


def probe_bwrap(timeout_s: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Can bwrap create a user-namespace sandbox here? Cached per process."""
    global _probe_cache
    with _probe_lock:
        if _probe_cache is not None:
            return _probe_cache
        exe = shutil.which("bwrap")
        if not exe:
            _probe_cache = (False, "bwrap not on PATH")
            return _probe_cache
        argv = [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--unshare-all", "true"]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _probe_cache = (False, f"probe failed: {exc}")
            return _probe_cache
        if proc.returncode != 0:
            _probe_cache = (False, f"probe exit {proc.returncode}: "
                                   f"{(proc.stderr or '').strip()[:200]}")
        else:
            _probe_cache = (True, "ok")
        return _probe_cache


def python_prefixes() -> list[Path]:
    """Directories that must be visible for the running interpreter to work
    inside the sandbox (conda env / venv / system prefix)."""
    candidates = [
        Path(sys.prefix), Path(sys.base_prefix), Path(sys.exec_prefix),
        Path(sys.executable).resolve().parent.parent,
    ]
    out: list[Path] = []
    for c in candidates:
        c = Path(os.path.realpath(c))
        if c.exists() and c not in out and str(c) not in ("/", "/usr"):
            out.append(c)
    return out


def sandbox_env(policy: PathPolicy, *, path: str) -> dict[str, str]:
    """The complete environment a sandboxed command sees. Built from
    scratch — nothing is inherited from the parent process."""
    env = {
        "PATH": path,
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(policy.repo_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "TERM": "dumb",
    }
    if policy.project_name:
        env["META_AGENT_PROJECT"] = policy.project_name
    return env


def bwrap_argv(
    policy: PathPolicy,
    *,
    command: str,
    bwrap_exe: str = "bwrap",
    prefixes: Optional[list[Path]] = None,
) -> list[str]:
    """Build the full bwrap command line for one ``bash -c command``.

    Bind order matters: later binds overlay earlier ones, so the writable
    mutable files are bound *after* the read-only run root that contains
    them, and ``--remount-ro /`` comes last so bwrap's own scaffold dirs are
    sealed too.
    """
    prefixes = python_prefixes() if prefixes is None else prefixes
    argv: list[str] = [bwrap_exe]
    for d in SYSTEM_RO_DIRS:
        if os.path.isdir(d):
            argv += ["--ro-bind", d, d]
    for p in SYSTEM_MAYBE_SYMLINK:
        if os.path.islink(p):
            argv += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            argv += ["--ro-bind", p, p]
    for prefix in prefixes:
        argv += ["--ro-bind", str(prefix), str(prefix)]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for root in policy.read_roots:
        # --ro-bind-try: a root that vanished between policy build and this
        # call (e.g. an archive dir being rotated) must not abort the command.
        argv += ["--ro-bind-try", str(root), str(root)]
    for d in policy.write_dirs:
        if d.exists():
            argv += ["--bind", str(d), str(d)]
    for f in policy.write_files:
        if f.exists():
            argv += ["--bind", str(f), str(f)]
    argv += ["--remount-ro", "/"]
    argv += ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    bin_dirs = [str(p / "bin") for p in prefixes] + ["/usr/bin", "/bin"]
    for k, v in sandbox_env(policy, path=":".join(bin_dirs)).items():
        argv += ["--setenv", k, v]
    argv += ["--chdir", str(policy.task_agent)]
    argv += ["bash", "-c", command]
    return argv


def fallback_env(policy: PathPolicy) -> dict[str, str]:
    """Unconfined mode: same keys as the sandbox env, but the parent's PATH
    so the conda tools resolve. Secrets, ``LLM_*`` and ``*_DATABASE_ROOT``
    are never copied because nothing is copied."""
    return sandbox_env(policy, path=os.environ.get("PATH", "/usr/bin:/bin"))


class Sandbox:
    def __init__(
        self,
        policy: PathPolicy,
        *,
        mode: str = "auto",
        bash_timeout_s: float = 120.0,
    ) -> None:
        if mode not in SANDBOX_MODES:
            raise ValueError(f"sandbox must be one of {SANDBOX_MODES}, got {mode!r}")
        self.policy = policy
        self.mode = mode
        self.bash_timeout_s = float(bash_timeout_s)
        self._effective: Optional[str] = "none" if mode == "none" else None
        self._bwrap_exe: Optional[str] = None

    @property
    def effective_mode(self) -> str:
        if self._effective is None:
            self._decide()
        return self._effective  # type: ignore[return-value]

    def _decide(self) -> None:
        global _warned
        ok, reason = probe_bwrap()
        if ok:
            self._bwrap_exe = shutil.which("bwrap")
            self._effective = "bwrap"
            return
        if self.mode == "bwrap":
            raise RuntimeError(
                f"editor sandbox=bwrap but bubblewrap is unusable here ({reason})"
            )
        if not _warned:
            print(
                f"[editor:agentic] bwrap unavailable ({reason}); bash runs "
                "UNCONFINED (cwd-only, env scrubbed) — the path policy is "
                "enforced by the editor tool and the validators only",
                flush=True,
            )
            _warned = True
        self._effective = "none"

    def run(self, command: str, *, timeout_s: Optional[float] = None) -> BashResult:
        timeout = float(timeout_s) if timeout_s is not None else self.bash_timeout_s
        if self.effective_mode == "bwrap":
            argv = bwrap_argv(self.policy, command=command,
                              bwrap_exe=self._bwrap_exe or "bwrap")
            cwd, env = None, None
        else:
            argv = ["bash", "-c", command]
            cwd, env = str(self.policy.task_agent), fallback_env(self.policy)
        started = time.time()
        timed_out = False
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
        return BashResult(
            stdout=out or "", stderr=err or "",
            returncode=None if timed_out else proc.returncode,
            timed_out=timed_out, elapsed_s=time.time() - started,
        )

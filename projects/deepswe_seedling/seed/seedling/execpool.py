"""safe_exec -- the ONLY way this package touches the container.

WHY THIS EXISTS (verified in pier 0.3.1, do not remove):

  pier/environments/docker/docker.py:521
      raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

`environment.exec(..., timeout_sec=N)` RAISES a bare RuntimeError on timeout, and throws
away the output it already captured. A bare RuntimeError is not AgentTimeoutError, so it
escapes run() into trial.py's outer `except Exception`, which SKIPS _run_collect_hooks().
No model.patch is generated, verification never runs, and the task scores 0 -- from one
build that ran a little long. That is the single biggest silent-zero risk in this harness.

  pier/environments/docker/docker.py:502
      stderr=asyncio.subprocess.STDOUT

`ExecResult.stderr` is therefore ALWAYS None on the Docker backend. Never branch on it.
We merge with 2>&1 and read everything from stdout.

Design consequences, all load-bearing:
  * safe_exec is a TOTAL function: never raises except CancelledError, never returns None.
  * An INNER `timeout -k 5 Ns` fires before Pier's outer timeout (+15s slack), so we get
    rc 124 WITH the partial output instead of a RuntimeError that discards the build log.
  * Every timeout is clamped to the soft deadline so no command can outlive our budget.
  * All git-touching commands serialise behind one lock (.git/index.lock contention with
    the watchdog committer), with a single stale-lock retry.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from collections import Counter
from dataclasses import dataclass, field

# Cap bytes brought back from the container. Large enough for a failing build log's tail,
# small enough that we never blow up the trajectory or stall the shared event loop.
DEFAULT_OUTPUT_CAP = 200_000

RC_DEADLINE_EXHAUSTED = 126  # our sentinel: no budget left to even start
RC_EXEC_ERROR = 125          # our sentinel: exec raised (incl. the timeout RuntimeError)
RC_TIMEOUT = 124             # GNU timeout's own code


@dataclass
class ExecOut:
    """Uniform result shape. Decoupled from pier's ExecResult on purpose.

    `stderr` is deliberately absent -- it is always None on Docker (see module docstring).
    """

    stdout: str
    return_code: int
    duration_sec: float = 0.0
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.return_code == 0

    def has(self, sentinel: str) -> bool:
        """Success is detected by sentinel in stdout, never by rc and never by stderr."""
        return sentinel in self.stdout

    def sentinel(self, name: str) -> str | None:
        """Read `NAME=value` emitted by our own shell snippets."""
        m = re.search(rf"^{re.escape(name)}=(.*)$", self.stdout, re.MULTILINE)
        return m.group(1).strip() if m else None

    def tail(self, n_chars: int = 4000) -> str:
        return self.stdout[-n_chars:] if self.stdout else ""


@dataclass
class ExecStats:
    n_calls: int = 0
    n_timeouts: int = 0
    n_errors: int = 0
    total_sec: float = 0.0
    by_label: Counter = field(default_factory=Counter)
    slowest: list = field(default_factory=list)

    def record(self, label: str, dur: float, out: ExecOut) -> None:
        self.n_calls += 1
        self.total_sec += dur
        self.by_label[label or "unlabelled"] += 1
        if out.timed_out:
            self.n_timeouts += 1
        if out.error:
            self.n_errors += 1
        self.slowest.append((round(dur, 1), label or "unlabelled"))
        self.slowest.sort(reverse=True)
        del self.slowest[8:]

    def snapshot(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_timeouts": self.n_timeouts,
            "n_errors": self.n_errors,
            "total_sec": round(self.total_sec, 1),
            "by_label": dict(self.by_label),
            "slowest": self.slowest,
        }


class ExecPool:
    """Owns every container command. Construct one per trial (never module-level state)."""

    def __init__(self, environment, deadline, logger, cwd: str = "/app") -> None:
        self._env = environment
        self._deadline = deadline
        self._log = logger
        self._cwd = cwd
        self._git_lock = asyncio.Lock()
        self._has_timeout = True  # re-probed by gitops.bootstrap()
        self.stats = ExecStats()

    def set_has_timeout(self, present: bool) -> None:
        self._has_timeout = bool(present)

    def _wrap(self, command: str, budget: float, cap: int) -> str:
        # pipefail so the cap-pipe does not mask the real exit code.
        inner = f"set -o pipefail; {{ {command} ; }} 2>&1 | tail -c {cap}"
        if not self._has_timeout:
            return inner
        # -k 5: SIGKILL 5s after SIGTERM if it ignores the term.
        return f"timeout -k 5 {int(max(1, budget))}s bash -c {shlex.quote(inner)}"

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: float = 120.0,
        env: dict | None = None,
        user=None,
        label: str = "",
        cap: int = DEFAULT_OUTPUT_CAP,
        critical: bool = False,
    ) -> ExecOut:
        """Total function. Never raises except CancelledError. Never returns None."""
        budget = self._deadline.clamp(timeout_sec, critical=critical)
        if budget < 2.0:
            return ExecOut(
                stdout="[seedling] deadline exhausted; command not started",
                return_code=RC_DEADLINE_EXHAUSTED,
                error="deadline_exhausted",
            )

        wrapped = self._wrap(command, budget, cap)
        # Outer > inner so OUR timeout fires first and we keep the partial output.
        outer = int(budget) + 15
        started = time.monotonic()
        out: ExecOut
        try:
            r = await self._env.exec(
                command=wrapped,
                cwd=cwd or self._cwd,
                env=self._env.agent_process_env(env) if env is not None else None,
                user=user,
                timeout_sec=outer,
            )
            dur = time.monotonic() - started
            rc = getattr(r, "return_code", 1)
            out = ExecOut(
                stdout=getattr(r, "stdout", None) or "",
                return_code=rc,
                duration_sec=dur,
                timed_out=(rc == RC_TIMEOUT),
            )
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # noqa: BLE001 -- containment is the whole point
            dur = time.monotonic() - started
            # This is the docker.py:521 RuntimeError path, among others.
            out = ExecOut(
                stdout=f"[seedling-exec-error] {type(e).__name__}: {e}",
                return_code=RC_EXEC_ERROR,
                duration_sec=dur,
                timed_out="timed out" in str(e).lower(),
                error=f"{type(e).__name__}: {e}",
            )
            self._log.warning("safe_exec contained %s on %r", type(e).__name__, label or command[:80])

        self.stats.record(label, out.duration_sec, out)
        return out

    async def git(self, command: str, *, timeout_sec: float = 300.0, label: str = "git",
                  critical: bool = False) -> ExecOut:
        """Serialised git command with one stale-index.lock retry.

        critical=True for commits and the apply-check: they must still run after the soft
        deadline, which is the entire point of reserving time for them.
        """
        async with self._git_lock:
            out = await self.run(command, timeout_sec=timeout_sec, label=label,
                                 critical=critical)
            if "index.lock" in out.stdout and not out.ok:
                self._log.warning("stale .git/index.lock; clearing and retrying once")
                await self.run("rm -f /app/.git/index.lock", timeout_sec=30, label="git:unlock")
                out = await self.run(command, timeout_sec=timeout_sec,
                                     label=label + ":retry", critical=critical)
            return out

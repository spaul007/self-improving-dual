"""Pier boundary. Every Pier-specific gotcha is isolated in this file.

THE EXCEPTION CONTRACT (verified in pier 0.3.1 trial/trial.py):

    normal return                      -> collect hooks RUN   -> scored
    CancelledError -> AgentTimeoutError -> collect hooks RUN   -> scored
    NonZeroAgentExitCodeError          -> collect hooks RUN   -> scored
    ANY OTHER Exception                -> hooks SKIPPED       -> GUARANTEED 0

So run() propagates ONLY CancelledError and swallows every Exception. Never raise a bare
TimeoutError either -- trial.py:981 catches AgentTimeoutError *by type*.

CONSTRUCTOR: the factory injects `extra_env` (which BaseAgent.__init__ does not declare),
and injects task_dir/trial_paths/agent_timeout_sec ONLY for the oracle agent
(trial/execution.py:163-168). So we take **kwargs, and the deadline arrives via
`--ak agent_timeout_sec=10800`. All --ak values may arrive as strings; coerce defensively.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
import traceback
import uuid
from pathlib import Path

from pier.agents.base import BaseAgent


def task_signals(role_stats: list, git: dict) -> dict:
    """Task-level aggregates of the per-role failure-class signals.

    Every one of these was diagnosed BY HAND from trajectories today (explore-without-edit,
    summariser stubs, upstream-fork fetch, truncation-as-stop, report failures). A
    self-evolution loop's diagnose step cannot read trajectories at scale; it reads this.
    """
    rs = role_stats or []
    return {
        "roles_run": len(rs),
        "zero_edit_roles": sum(1 for r in rs if r.get("zero_edit")),
        "wall_terminated_roles": sum(1 for r in rs if r.get("wall_terminated")),
        "report_failures": sum(1 for r in rs if not r.get("report_ok", True)),
        "compactions": sum(r.get("compactions", 0) for r in rs),
        "stub_rejections": sum(r.get("stub_rejections", 0) for r in rs),
        "stub_attempts": sum(r.get("stub_attempts", 0) for r in rs),
        "truncations": sum(r.get("truncations", 0) for r in rs),
        "net_fetch": sum(r.get("net_fetch", 0) for r in rs),
        "fork_fetch": sum(r.get("fork_fetch", 0) for r in rs),
        "total_steps": sum(r.get("steps", 0) for r in rs),
        "patch_bytes": (git or {}).get("patch_bytes"),
        "n_patch_files": (git or {}).get("n_patch_files"),
        # EXP-027 signals: harness-message hygiene, confinement, per-turn truncation, git
        # attribution by role, and the one acceptance event.
        "nudges": sum(r.get("nudges", 0) for r in rs),
        "pushes": sum(r.get("pushes", 0) for r in rs),
        "transients_deleted": sum(r.get("transients_deleted", 0) for r in rs),
        "source_writes_refused": sum(r.get("source_writes_refused", 0) for r in rs),
        "finish_length": sum(r.get("finish_length", 0) for r in rs),
        "cap_hits": sum(r.get("cap_hits", 0) for r in rs),
        "reads_without_limit": sum(r.get("reads_without_limit", 0) for r in rs),
        "sys_sha_mismatch": sum(1 for r in rs if r.get("sys_sha") and r.get("sys_sha_end")
                                and r["sys_sha"] != r["sys_sha_end"]),
        "verify_pass": any(r.get("role") == "verify" and r.get("verdict") == "pass" for r in rs),
        "verify_runs": sum(1 for r in rs if r.get("role") == "verify"),
        "patch_attempts": sum(1 for r in rs if r.get("role") == "patch"),
        "nonpatch_source_files": sum((r.get("git") or {}).get("source_files", 0)
                                     for r in rs if r.get("role") != "patch"),
        "stop_reasons": {k: sum(1 for r in rs if r.get("stop_reason") == k)
                         for k in ("end_turn", "wall", "soft_deadline", "llm_error", "finish_call")},
    }

from . import pipeline, settings
from .blackboard import Blackboard
from .deadline import Deadline
from .execpool import ExecPool
from .gitops import GitOps
from .llm import LLM
from .roles import Harness
from .trajectory import TrajectoryBuilder

VERSION = "0.1.0"
DEFAULT_AGENT_TIMEOUT_SEC = 10800.0   # DeepSWE [agent].timeout_sec
CHECKPOINT_INTERVAL_SEC = 300.0       # watchdog cadence; bounds cancel damage to <=5 min


def _f(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


class SeedlingAgent(BaseAgent):
    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:            # NOTE: static -- the factory calls it on the class
        return "seedling"

    def __init__(
        self,
        logs_dir,
        model_name: str | None = None,
        extra_env: dict | None = None,          # always injected; BaseAgent doesn't declare it
        agent_timeout_sec=DEFAULT_AGENT_TIMEOUT_SEC,
        checkpoint_interval_sec=CHECKPOINT_INTERVAL_SEC,
        finalize_reserve_sec=180.0,
        mode: str = "multi",                     # multi | solo | smoke
        smoke: bool = False,                     # legacy alias for mode=smoke
        **kwargs,                                # mandatory: swallow future factory args
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name,
                         **{k: v for k, v in kwargs.items()
                            if k in ("logger", "mcp_servers", "skills_dir")})
        self._extra_env = extra_env or {}
        self._timeout_given = agent_timeout_sec is not None
        self._agent_timeout_sec = _f(agent_timeout_sec, DEFAULT_AGENT_TIMEOUT_SEC)
        self._checkpoint_interval = _f(checkpoint_interval_sec, CHECKPOINT_INTERVAL_SEC)
        self._finalize_reserve = _f(finalize_reserve_sec, 180.0)
        self._mode = "smoke" if _b(smoke) else str(mode or "multi").strip().lower()
        self._llm: LLM | None = None
        self._session_id = uuid.uuid4().hex[:12]
        self._t0 = time.monotonic()
        self._traj: TrajectoryBuilder | None = None
        self._git: GitOps | None = None
        self._pool: ExecPool | None = None
        self._deadline: Deadline | None = None
        self._outcome = "unknown"
        self._bb = None
        self._last_flush = 0.0

    def version(self) -> str:
        return VERSION

    async def setup(self, environment) -> None:
        return  # host-orchestrated: nothing is installed in the container

    # ---------------------------------------------------------------- run ------------
    async def run(self, instruction: str, environment, context) -> None:
        self._t0 = time.monotonic()                       # FIRST statement, before any await
        self._deadline = Deadline(self._t0, self._agent_timeout_sec,
                                  self._finalize_reserve, safety_sec=120.0)
        self._traj = TrajectoryBuilder("seedling", VERSION, self._session_id, self.model_name)
        self._traj.add_user_step(instruction)             # steps min_length=1 satisfied now
        if not self._timeout_given:
            self._traj.notes = ("agent_timeout_sec was NOT supplied via --ak; assumed "
                                f"{DEFAULT_AGENT_TIMEOUT_SEC}s.")

        self._pool = ExecPool(environment, self._deadline, self.logger)
        self._git = GitOps(self._pool, self.logger)
        watchdog = None

        try:
            await self._git.bootstrap()
            watchdog = asyncio.create_task(self._watchdog())
            watchdog.add_done_callback(lambda t: t.cancelled() or t.exception())
            await self._orchestrate(instruction)
            self._outcome = "completed"

        except asyncio.CancelledError:
            self._outcome = "cancelled"
            with contextlib.suppress(BaseException):
                self._traj.add_system_step("cancelled: Pier agent_timeout_sec reached")
            raise                                          # -> AgentTimeoutError -> hooks run

        except BaseException as e:                         # noqa: BLE001 -- containment
            self._outcome = f"contained:{type(e).__name__}"
            self.logger.exception("contained exception in run(); collect hooks preserved")
            with contextlib.suppress(BaseException):
                self._traj.add_system_step("contained: " + traceback.format_exc()[-4000:])

        finally:
            if watchdog is not None:
                watchdog.cancel()                          # synchronous

            # (a) SYNCHRONOUS FIRST -- always survives, even after cancellation.
            with contextlib.suppress(BaseException):
                self._populate_context(context)
            with contextlib.suppress(BaseException):
                self._traj.write(Path(self.logs_dir) / "trajectory.json")
            with contextlib.suppress(BaseException):
                self._write_run_summary()

            # (b) BEST-EFFORT ASYNC -- may be cut short; must never raise.
            if self._outcome != "cancelled":
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(
                        self._git.finalize(),
                        timeout=max(5.0, self._deadline.remaining() - 30))
            else:
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(asyncio.shield(self._git.finalize()), timeout=20)

            # (c) refresh the sync artifacts with whatever (b) learned.
            with contextlib.suppress(BaseException):
                self._populate_context(context)
                self._traj.write(Path(self.logs_dir) / "trajectory.json")
                self._write_run_summary()

    # ------------------------------------------------------------- internals ---------
    async def _watchdog(self) -> None:
        """Commit on a timer no matter what the orchestrator is doing.

        This is the load-bearing protection: it bounds what a hard cancel can destroy to
        one interval, and does not depend on the orchestrator behaving.
        """
        try:
            while not self._deadline.expired():
                await asyncio.sleep(self._checkpoint_interval)
                if self._deadline.expired():
                    return
                res = await self._git.checkpoint("seedling: watchdog checkpoint")
                self.logger.info("watchdog checkpoint: %s", res)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001
            self.logger.exception("watchdog error (non-fatal)")

    async def _orchestrate(self, instruction: str) -> None:
        """Dispatch to the pipeline. Roles are CALLED; the model never self-delegates."""
        if self._mode == "smoke":
            return await self._smoke_edit()

        base = (self._extra_env.get("OPENAI_BASE_URL")
                or self._extra_env.get("OPENAI_API_BASE")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE"))
        self._llm = LLM(self.model_name, base, self.logger,
                        api_key=self._extra_env.get("OPENAI_API_KEY", "dummy"),
                        deadline=self._deadline)
        h = Harness(pool=self._pool, llm=self._llm, traj=self._traj, git=self._git,
                    deadline=self._deadline, logger=self.logger,
                    on_progress=self._flush_observability,
                    ledger=self._ledger, snapshot=self._snapshot)
        bb = Blackboard(task=instruction)
        self._bb = bb
        with contextlib.suppress(BaseException):
            self._write_manifest(base)

        await self._pool.run(f"mkdir -p {settings.SCRATCH}", timeout_sec=30,
                             label="setup:scratch")
        if self._mode == "solo":
            await pipeline.solve_single(bb, h)
        else:
            await pipeline.solve(bb, h)

    async def _smoke_edit(self) -> None:
        """P0 plumbing check: one trivial edit, no LLM. Kept as a permanent probe."""
        rec = self._traj.role("smoke", brief="P0 plumbing smoke: make one trivial edit")
        out = await self._pool.run(
            "printf '%s\\n' 'seedling P0 plumbing check' > /app/SEEDLING_SMOKE.md && "
            "echo WROTE && ls -l /app/SEEDLING_SMOKE.md",
            timeout_sec=60, label="smoke:write")
        rec.add_agent_step("write marker file", llm_call_count=0)
        rec.finish("ok" if out.ok else "fail", out.tail(400))
        self._traj.dispatch("smoke", rec, "trivial edit", out.tail(400))
        await self._git.checkpoint("seedling: P0 smoke edit")

    def _sync_tokens(self) -> None:
        """Pull token totals from the LLM client into the trajectory. Never raises."""
        with contextlib.suppress(BaseException):
            if self._llm is not None:
                snap = self._llm.snapshot()
                self._traj.set_tokens(prompt=snap.get("prompt_tokens", 0),
                                      completion=snap.get("completion_tokens", 0),
                                      cached=snap.get("cached_tokens", 0),
                                      cost=snap.get("cost_usd", 0.0))

    def _flush_observability(self, force: bool = False) -> None:
        """Synchronous, throttled (unless force), never raises. Called after each role completes."""
        now = time.monotonic()
        if not force and now - self._last_flush < 20.0:
            return
        self._last_flush = now
        self._sync_tokens()
        with contextlib.suppress(BaseException):
            self._traj.write(Path(self.logs_dir) / "trajectory.json")
        with contextlib.suppress(BaseException):
            self._write_run_summary()

    def _summary(self) -> dict:
        _rs = getattr(getattr(self, "_bb", None), "role_stats", []) or []
        _g = self._git.snapshot() if self._git else {}
        return {
            "role_stats": _rs,
            "signals": task_signals(_rs, _g),
            "outcome": self._outcome,
            "session_id": self._session_id,
            "agent_version": VERSION,
            "model_name": self.model_name,
            "agent_timeout_sec_source": "--ak" if self._timeout_given else "assumed-default",
            "deadline": self._deadline.snapshot() if self._deadline else {},
            "git": self._git.snapshot() if self._git else {},
            "exec": self._pool.stats.snapshot() if self._pool else {},
            "llm": self._llm.snapshot() if self._llm else {},
            "mode": self._mode,
            "verify_passes": getattr(getattr(self, "_bb", None), "verify_passes", 0),
            "roles": [{"role": r["role"], "attempt": r["attempt"],
                       "incomplete": bool(r["output"].get("_incomplete"))}
                      for r in getattr(getattr(self, "_bb", None), "history", []) or []],
        }

    def _populate_context(self, context) -> None:
        """Nothing is auto-derived for a plain BaseAgent (trial.py:543 early-returns)."""
        self._sync_tokens()
        t = self._traj
        all_steps = t.all_steps()
        context.n_input_tokens = t.totals["prompt"] or None
        context.n_output_tokens = t.totals["completion"] or None
        context.n_cache_tokens = t.totals["cached"] or None
        context.cost_usd = t.totals["cost"] or None
        context.n_agent_steps = sum(1 for s in all_steps if s.get("source") == "agent")
        context.metadata = self._summary()

    def _write_run_summary(self) -> None:
        d = Path(self.logs_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "run_summary.json.part"
        tmp.write_text(json.dumps(self._summary(), indent=2, default=str))
        os.replace(tmp, d / "run_summary.json")

    # ------------------------------------------------------- EXP-027 observability -----
    def _ledger(self, record: dict) -> None:
        """One JSON line per tool call, appended AS IT HAPPENS -- survives SIGKILL. Never raises
        into the role (roles.py wraps the call), but keep it total anyway."""
        try:
            d = Path(self.logs_dir)
            d.mkdir(parents=True, exist_ok=True)
            with (d / "exec_log.jsonl").open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:  # noqa: BLE001
            self.logger.exception("exec ledger append failed (non-fatal)")

    def _snapshot(self, role: str, attempt: int, messages: list, meta: dict | None = None) -> None:
        """The conversation exactly as the model saw it at role end, post-transient-sweep.
        conv/<role>.<attempt>.json. This is the artifact that would have shown, on its first
        trial, that VERIFY ran under baseline.md (A0) and that the report prompt was
        compounding (EXP-025)."""
        d = Path(self.logs_dir) / "conv"
        d.mkdir(parents=True, exist_ok=True)
        payload = {"role": role, "attempt": attempt, "n_messages": len(messages),
                   "sys_sha": (meta or {}).get("sys_sha"), "stop_reason": (meta or {}).get("stop_reason"),
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "messages": messages, "turns": (meta or {}).get("turns")}
        tmp = d / f"{role}.{attempt}.json.part"
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, d / f"{role}.{attempt}.json")

    def _write_manifest(self, base_url) -> None:
        """Which CODE produced this trial: sha of every package file and prompt, the settings
        in force, the endpoint. The plugin is imported live from the working tree, so a
        mid-run edit changes later trials silently -- this pins what each trial actually ran."""
        pkg = Path(__file__).resolve().parent
        files = sorted(list(pkg.glob("*.py")) + list(pkg.glob("tools/*.py")) + list(pkg.glob("prompts/*.md")))
        shas = {str(p.relative_to(pkg)): hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in files}
        tree_sha = hashlib.sha256("".join(f"{k}:{v}" for k, v in shas.items()).encode()).hexdigest()[:12]
        cfg = {k: getattr(settings, k) for k in dir(settings)
               if k.isupper() and not k.startswith("_")}
        payload = {"tree_sha": tree_sha, "files": shas, "settings": cfg,
                   "package_dir": str(pkg), "base_url": base_url, "model_name": self.model_name,
                   "agent_timeout_sec": self._agent_timeout_sec, "mode": self._mode,
                   "agent_version": VERSION, "session_id": self._session_id,
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        d = Path(self.logs_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))
        self.logger.warning("manifest tree_sha=%s package=%s model=%s", tree_sha, pkg, self.model_name)

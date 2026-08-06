"""SubprocessEvaluator — runs the task agent against a benchmark in isolated
child processes with wall-time, memory, and CPU caps.

Each benchmark case spawns a fresh Python child running
``python -m platform_core.runner`` with:
    cwd      = round_dir/task_agent
    PYTHONPATH ⊇ platform_core's parent
    env META_AGENT_TRACE_PATH = round_dir/logs/trace.jsonl
    stdin    = JSON Task (description, case_id, context)

The runner imports ``workflow.run_task``, calls it with the Task, and
writes a JSON envelope to stdout: ``{"ok": True, "output": {...}}`` on
success, ``{"ok": False, "error": "..."}`` otherwise. The parent parses
the envelope, builds an :class:`AgentOutput`, hands it to the benchmark's
scorer, and produces an :class:`EvaluationResult`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from .models import CaseResult, EvaluationResult
from .registry import register


class Evaluator(Protocol):
    def run(
        self,
        round_dir: Path,
        benchmark_dir: Path,
        *,
        case_ids: Optional[list[str]] = None,
    ) -> EvaluationResult: ...


def _platform_core_parent() -> Path:
    """Return the directory whose presence on PYTHONPATH makes
    `import platform_core.*` resolvable."""
    import platform_core  # type: ignore
    return Path(platform_core.__file__).resolve().parent.parent


def _load_scorer(benchmark_dir: Path):
    scorer_path = benchmark_dir / "scorer.py"
    if not scorer_path.exists():
        raise FileNotFoundError(f"scorer.py not found in {benchmark_dir}")
    spec = importlib.util.spec_from_file_location(
        f"_scorer_{benchmark_dir.name}", scorer_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "score"):
        raise AttributeError(f"{scorer_path} must define a score(case, output) function")
    return mod


def load_cases(benchmark_dir: Path) -> list[dict[str, Any]]:
    cases_path = benchmark_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.jsonl not found in {benchmark_dir}")
    cases: list[dict[str, Any]] = []
    for i, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


@register("evaluator", "subprocess")
class SubprocessEvaluator:
    def __init__(
        self,
        *,
        wall_time_s_per_case: float = 60.0,
        memory_mb: int = 512,
        cpu_seconds: int | None = None,
        parallelism: int = 1,
        max_cases: int | None = None,
        scorer: Any = None,
    ) -> None:
        self.wall_time_s = float(wall_time_s_per_case)
        self.memory_bytes = int(memory_mb) * 1024 * 1024
        self.cpu_seconds = int(cpu_seconds) if cpu_seconds is not None else None
        self.parallelism = max(1, int(parallelism))
        self.max_cases = max_cases
        # Optional scorer instance constructed by config.build_components
        # from the registered ``<project>_default`` class. When ``None``
        # the evaluator falls back to the module-level ``score()`` function
        # in ``benchmark_dir/scorer.py``.
        self.scorer = scorer

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        round_dir: Path,
        benchmark_dir: Path,
        *,
        case_ids: Optional[list[str]] = None,
    ) -> EvaluationResult:
        cases = load_cases(benchmark_dir)
        if case_ids is not None:
            by_id = {
                str(c.get("id") or c.get("case_id")): c for c in cases
            }
            missing = [cid for cid in case_ids if cid not in by_id]
            if missing:
                raise KeyError(
                    f"case_ids not found in benchmark: {missing[:5]}"
                    + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
                )
            cases = [by_id[cid] for cid in case_ids]
        if self.max_cases is not None:
            cases = cases[: self.max_cases]
        # Always import the benchmark's scorer.py so any @register("scorer", ...)
        # decorators inside it run (it's also the fallback module-level
        # score() path). When a scorer instance was injected at construction
        # time, prefer that; otherwise use the module.
        scorer_module = _load_scorer(benchmark_dir)
        scorer = self.scorer if self.scorer is not None else scorer_module

        logs_dir = round_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        trace_path = logs_dir / "trace.jsonl"
        # Truncate any prior trace.
        trace_path.write_text("", encoding="utf-8")

        cwd = round_dir / "task_agent"
        env = self._child_env(trace_path)
        # Per-evaluation isolated scratch dir, exported to every case
        # subprocess (see platform_core.trace.SCRATCH_DIR_ENV). Generic:
        # projects that keep mutable on-disk state write it under here so two
        # concurrent run() calls (e.g. dual-optimization variants over the
        # same case ids) never collide. Distinct round_dir per concurrent
        # run() makes the scratch roots inherently distinct. Set only on this
        # local env dict — never os.environ — so it stays child-only and
        # thread-isolated between concurrent run() calls.
        scratch_dir = logs_dir / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        env["META_AGENT_SCRATCH_DIR"] = str(scratch_dir)

        started = time.time()
        results: list[CaseResult] = []
        crashed = False

        if self.parallelism == 1:
            for case in cases:
                cr, did_crash = self._run_one(case, cwd, env, scorer, logs_dir)
                results.append(cr)
                crashed = crashed or did_crash
        else:
            # Each case still runs in its own subprocess; threads only oversee them.
            with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
                futures = {
                    pool.submit(self._run_one, case, cwd, env, scorer, logs_dir): case
                    for case in cases
                }
                for fut in as_completed(futures):
                    cr, did_crash = fut.result()
                    results.append(cr)
                    crashed = crashed or did_crash
            results.sort(key=lambda r: r.case_id)

        wall = time.time() - started

        # Attribute LLM round-trips to cases from the (now-complete) trace.
        # Per-case llm count = number of kind=="llm_call" events whose
        # payload.case_id matches. A case that ran but never called an LLM gets
        # 0 (distinct from None = "no data"). Re-persist each case_<id>.json so
        # the per-case file carries both wall_time_s (set in _run_one) and the
        # now-known llm_calls; observability only — never fail the run on OSError.
        llm_by_case = self._llm_calls_by_case(trace_path)
        for r in results:
            r.llm_calls = llm_by_case.get(str(r.case_id), 0)
            try:
                (logs_dir / f"case_{r.case_id}.json").write_text(
                    r.model_dump_json(indent=2), encoding="utf-8"
                )
            except OSError:
                pass

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        score = sum(r.score for r in results) / len(results) if results else 0.0
        return EvaluationResult(
            score=score,
            metrics={"mean_score": score},
            passed=passed,
            failed=failed,
            per_case=results,
            wall_time_s=wall,
            crashed=crashed,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _child_env(self, trace_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        platform_parent = str(_platform_core_parent())
        env["PYTHONPATH"] = (
            f"{platform_parent}{os.pathsep}{existing}" if existing else platform_parent
        )
        env["META_AGENT_TRACE_PATH"] = str(trace_path)
        return env

    def _preexec(self):
        # POSIX-only resource caps; safe under Linux per spec.
        try:
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes)
            )
            if self.cpu_seconds is not None:
                resource.setrlimit(
                    resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds)
                )
        except Exception:
            pass

    def _run_one(
        self,
        case: dict[str, Any],
        cwd: Path,
        env: dict[str, str],
        scorer: Any,
        logs_dir: Path,
    ) -> tuple[CaseResult, bool]:
        from platform_core.runner import AgentOutput, Task

        case_id = str(case.get("id") or case.get("case_id") or len(case))
        stderr_path = logs_dir / f"case_{case_id}.stderr"

        # Per-case env overrides from ``case["env"]`` (anything project
        # tools need to scope their behaviour to this case).
        per_case_env = case.get("env") or {}
        if per_case_env:
            env = {**env, **{str(k): str(v) for k, v in per_case_env.items()}}

        task = Task(
            description=str(case.get("input", "")),
            case_id=case_id,
            context=dict(case.get("context") or {}),
        )

        _t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "platform_core.runner"],
                input=json.dumps(task.to_dict()),
                capture_output=True,
                text=True,
                cwd=str(cwd),
                env=env,
                timeout=self.wall_time_s,
                preexec_fn=self._preexec if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - _t0
            stderr_path.write_text(
                f"TIMEOUT after {self.wall_time_s}s\n{(exc.stderr or '') if hasattr(exc, 'stderr') else ''}",
                encoding="utf-8",
            )
            return self._finish(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    error=f"timeout after {self.wall_time_s}s",
                    wall_time_s=elapsed,
                ),
                True,
                logs_dir,
            )

        # Per-case (per-plan) wall time for every non-timeout return path below.
        elapsed = time.perf_counter() - _t0

        if proc.stderr:
            stderr_path.write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            return self._finish(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    error=f"child exit code {proc.returncode}",
                    wall_time_s=elapsed,
                ),
                True,
                logs_dir,
            )

        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return self._finish(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    error=f"unparseable child output: {proc.stdout[:200]!r}",
                    wall_time_s=elapsed,
                ),
                True,
                logs_dir,
            )

        if not payload.get("ok"):
            return self._finish(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    error=str(payload.get("error", "unknown error"))[:1000],
                    wall_time_s=elapsed,
                ),
                False,
                logs_dir,
            )

        agent_output = AgentOutput.from_dict(payload.get("output") or {})
        # Preserve the raw plan + agent metadata (iterations, budget_exhausted)
        # in the persisted case file. Without this, post-hoc debugging of a
        # low-scoring case requires rerunning the whole eval — the plan and
        # iteration count are gone after the subprocess exits.
        # ``query`` is the task input (``case["input"]``) — not ground truth.
        # Carrying it on the per-case result lets the (generic) feedback
        # gatherer build "query + plan + what failed" examples without ever
        # reading the benchmark/cases file itself.
        agent_artifact = {
            "query": case.get("input"),
            "raw_result": agent_output.result,
            "agent_metadata": dict(agent_output.metadata or {}),
        }
        try:
            scored = scorer.score(case, agent_output)
        except Exception as exc:
            return self._finish(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    error=f"scorer raised: {exc!r}",
                    details=agent_artifact,
                    wall_time_s=elapsed,
                ),
                False,
                logs_dir,
            )

        details = dict(scored.get("details", {}))
        details.update(agent_artifact)
        return self._finish(
            CaseResult(
                case_id=case_id,
                passed=bool(scored.get("passed", False)),
                score=float(scored.get("score", 0.0)),
                details=details,
                wall_time_s=elapsed,
            ),
            False,
            logs_dir,
        )

    @staticmethod
    def _llm_calls_by_case(trace_path: Path) -> dict[str, int]:
        """Count llm_call events per ``str(payload['case_id'])`` from trace.jsonl.

        Robust to blank/malformed lines and to events emitted outside any case
        scope (no case_id -> skipped). The trace is fully written only once all
        case subprocesses have exited, so call this after the batch completes.
        """
        counts: dict[str, int] = {}
        try:
            text = trace_path.read_text(encoding="utf-8")
        except OSError:
            return counts
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") != "llm_call":
                continue
            cid = (event.get("payload") or {}).get("case_id")
            if cid is None:
                continue
            key = str(cid)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _finish(
        self,
        result: CaseResult,
        crashed: bool,
        logs_dir: Path,
    ) -> tuple[CaseResult, bool]:
        """Persist the per-case result so progress is visible mid-run.

        Each thread writes its own ``case_<id>.json`` (one file per
        case_id, distinct paths) so no locking is needed.
        """
        try:
            (logs_dir / f"case_{result.case_id}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError:
            # Persistence is observability — never fail the case for it.
            pass
        return result, crashed

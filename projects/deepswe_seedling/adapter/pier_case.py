"""Run ONE DeepSWE task with the candidate seedling tree, via the Pier CLI.

Called from the (frozen) ``seed/workflow.py`` inside the evaluator's case subprocess
(framework venv, py3.12, cwd = ``<round>/task_agent``). Stdlib + ``platform_core`` only.

Contract (see projects/deepswe_seedling/README.md):
  * NEVER raises and NEVER prints to stdout (the runner parses its last stdout line).
  * Pier runs in its OWN session/process group with stdin=DEVNULL and stdout/stderr to
    files OUTSIDE the round dir, under an inner watchdog that fires before the
    evaluator's per-case timeout (which can only SIGKILL this process, orphaning pier).
  * The Pier job dir lives OUTSIDE ``round/logs`` so hidden grader output
    (verifier/test-stdout.txt, ctrf.json, reports/) can never reach the meta-agent;
    only a WHITELIST (rendered transcripts, run_summary.json, reward.json,
    model.patch, exec_log.txt, report.md) is copied into
    ``$META_AGENT_SCRATCH_DIR/<case>/<run-id>/``.
  * The scorer re-reads the trial's result.json itself (``metadata.trial_dir``).

Configuration comes from env (the YAML ``env:`` block), all with safe defaults:
  SID_PIER_BIN, SID_PIER_JOBS_ROOT, SID_EXPERIMENT, SID_MODEL, SID_BASE_URL,
  SID_AGENT_TIMEOUT_S, SID_PIER_WALL_S, SID_LEDGER, SID_RESULT_CACHE_DIR,
  SID_RESULT_CACHE_READ.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

from platform_core import trace
from platform_core.runner import AgentOutput

from .render import render_exec_log, render_trial, run_report
from .trial import dispatch_outputs, find_trial_dir, load_outcome

DEFAULTS = {
    "SID_PIER_BIN": "/users/n.tzou/.local/bin/pier",
    "SID_PIER_JOBS_ROOT": "/groups/AIC-MV/n.tzou/sid_pier_jobs",
    "SID_EXPERIMENT": "default",
    "SID_MODEL": "openai/Qwen/Qwen3.8-27B",
    "SID_BASE_URL": "http://gpu-aic-mv-02-st-p5-node-6:8001/v1",
    "SID_AGENT_TIMEOUT_S": "10800",
    # agent 10800 + verifier 1800 + env/image setup 900. The evaluator's
    # wall_time_s_per_case must be larger (config: this + 900).
    "SID_PIER_WALL_S": "13500",
    "SID_LEDGER": "",
    "SID_RESULT_CACHE_DIR": "",
    "SID_RESULT_CACHE_READ": "0",
}
GRACE_S = 120
POLL_S = 10
# Only these env vars reach pier -- never the framework's PYTHONPATH or trace path.
PASS_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP",
            "DOCKER_HOST", "DOCKER_CONFIG", "UV_CACHE_DIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
WHITELIST = ("run_summary.json",)          # copied verbatim from <trial>/agent/


def _cfg(key: str) -> str:
    return os.environ.get(key) or DEFAULTS[key]


# --------------------------------------------------------------------------- helpers
def tree_hash(agent_dir: Path) -> str:
    """sha256 over the seedling package's .py/.md files (the mutable + frozen code that
    pier imports). Identical trees -> identical hash -> cache hit."""
    h = hashlib.sha256()
    pkg = Path(agent_dir) / "seedling"
    for f in sorted(pkg.rglob("*")):
        if f.is_file() and f.suffix in (".py", ".md") and "__pycache__" not in f.parts:
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def server_inflight(base_url: str) -> Optional[dict]:
    """vLLM /metrics running+waiting totals (other users' load is part of the measurement)."""
    try:
        url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception:  # noqa: BLE001
        return None
    run = wait = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running{"):
            run += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_waiting{"):
            wait += float(line.rsplit(" ", 1)[1])
    return {"running": run, "waiting": wait}


def _pier_env(agent_dir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env["PYTHONPATH"] = str(agent_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _docker(*args: str, timeout: int = 60) -> str:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return ""


def reap_trial(trial_dir: Optional[Path]) -> list[str]:
    """Remove the trial's own containers/networks (compose project = trial dir name,
    lowercased). Scoped by name -- never a global prune on a shared node."""
    if trial_dir is None:
        return []
    proj = trial_dir.name.lower()
    removed = []
    ids = _docker("ps", "-aq", "--filter", f"name={proj}").split()
    if ids:
        _docker("rm", "-f", *ids, timeout=120)
        removed += [f"container:{i}" for i in ids]
    nets = _docker("network", "ls", "-q", "--filter", f"name={proj}").split()
    for n in nets:
        _docker("network", "rm", n)
        removed.append(f"network:{n}")
    return removed


def _ledger(event: dict) -> None:
    path = _cfg("SID_LEDGER")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), "pid": os.getpid(), **event}) + "\n")
    except OSError:
        pass


def _emit_trace(o: dict, trial_dir: Optional[Path]) -> None:
    """Synthetic trace events so the gatherer's llm_calls / tool_usage reflect the real
    run (seedling uses its own litellm client, so nothing else would emit them)."""
    try:
        for r in o.get("role_stats") or []:
            for _ in range(int(r.get("llm_calls") or 0)):
                trace.emit("llm_call", {"role": r.get("role"), "attempt": r.get("attempt"),
                                        "synthetic": True})
        src = Path(trial_dir) / "agent" / "exec_log.jsonl" if trial_dir else None
        if src and src.is_file():
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                trace.emit("tool_call", {"name": row.get("tool"), "role": row.get("role"),
                                         "rc": row.get("rc"), "synthetic": True})
    except Exception:  # noqa: BLE001
        pass


def export_artifacts(trial_dir: Optional[Path], dest: Path, o: dict, report: str) -> None:
    """Whitelisted, rendered artifacts only. Never copies anything under verifier/ except
    the numeric reward fields."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.md").write_text(report + "\n", encoding="utf-8")
    if trial_dir is None:
        return
    for name in WHITELIST:
        src = trial_dir / "agent" / name
        if src.is_file():
            shutil.copyfile(src, dest / name)
    patch = trial_dir / "artifacts" / "model.patch"
    if patch.is_file():
        shutil.copyfile(patch, dest / "model.patch")
    (dest / "reward.json").write_text(json.dumps(
        {k: o.get(k) for k in ("reward", "f2p", "p2p", "partial", "f2p_total", "f2p_passed",
                               "p2p_total", "p2p_passed", "infra_class")}, indent=2),
        encoding="utf-8")
    render_trial(trial_dir, dest / "transcripts")
    render_exec_log(trial_dir, dest)


def _index(scratch_root: Path, case_id: str, run_id: str, line: str) -> None:
    try:
        with open(scratch_root / "INDEX.md", "a", encoding="utf-8") as fh:
            fh.write(f"- {case_id}/{run_id}/ : {line}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- main
def run_case(task: Any, agent_dir: Path) -> AgentOutput:
    """Never raises."""
    t0 = time.time()
    meta: dict[str, Any] = {"status": "adapter_error"}
    try:
        return _run_case(task, Path(agent_dir).resolve(), meta, t0)
    except Exception as exc:  # noqa: BLE001 -- the frozen entry point must never raise
        meta.update({"status": "adapter_error", "error": repr(exc)[:500],
                     "wall_s": round(time.time() - t0, 1)})
        return AgentOutput(result=f"INFRA-EXCLUDED (adapter_error: {exc!r})"[:1000], metadata=meta)


def _run_case(task: Any, agent_dir: Path, meta: dict, t0: float) -> AgentOutput:
    case_id = str(task.case_id)
    task_dir = Path((task.context or {}).get("task_dir") or "")
    if not (task_dir / "task.toml").is_file():
        meta.update({"status": "bad_task_dir", "task_dir": str(task_dir)})
        return AgentOutput(result=f"INFRA-EXCLUDED (bad task_dir {task_dir})", metadata=meta)

    base_url, model = _cfg("SID_BASE_URL"), _cfg("SID_MODEL")
    agent_timeout = int(float(_cfg("SID_AGENT_TIMEOUT_S")))
    wall = float(_cfg("SID_PIER_WALL_S"))
    run_id = uuid.uuid4().hex[:10]
    round_name = agent_dir.parent.name            # round_NNN (or _smoke_test)
    job_parent = Path(_cfg("SID_PIER_JOBS_ROOT")) / _cfg("SID_EXPERIMENT") / round_name / case_id
    job_name = f"sid-{run_id}"
    job_dir = job_parent / job_name
    scratch_root = Path(os.environ.get(trace.SCRATCH_DIR_ENV) or (agent_dir.parent / "logs" / "scratch"))
    dest = scratch_root / case_id / run_id
    thash = tree_hash(agent_dir)
    meta.update({"status": "started", "case_id": case_id, "run_id": run_id, "job_dir": str(job_dir),
                 "scratch_dir": str(dest), "tree_hash": thash[:16], "model": model,
                 "base_url": base_url, "agent_timeout_s": agent_timeout,
                 "inflight_start": server_inflight(base_url)})

    # ---- opt-in result cache (explicit crash-restart replays only)
    cache_dir = _cfg("SID_RESULT_CACHE_DIR")
    key = hashlib.sha256(json.dumps([thash, case_id, model, base_url, agent_timeout]).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.json" if cache_dir else None
    trial_dir: Optional[Path] = None
    if cache_file and _cfg("SID_RESULT_CACHE_READ") == "1" and cache_file.is_file():
        cached = json.loads(cache_file.read_text())
        cand = Path(cached.get("trial_dir") or "")
        if (cand / "result.json").is_file():
            trial_dir, meta["status"], meta["cache_hit"] = cand, "cached", True

    if trial_dir is None:
        job_parent.mkdir(parents=True, exist_ok=True)
        cmd = [_cfg("SID_PIER_BIN"), "run", "-p", str(task_dir),
               "--agent-import-path", "seedling.agent:SeedlingAgent", "-m", model,
               "--ak", f"agent_timeout_sec={agent_timeout}", "--ak", "mode=multi",
               "--ae", f"OPENAI_BASE_URL={base_url}", "--ae", f"OPENAI_API_BASE={base_url}",
               "--ae", "OPENAI_API_KEY=dummy",
               "-o", str(job_parent), "--job-name", job_name, "-n", "1", "--env", "docker"]
        (job_parent / f"{job_name}.cmd").write_text(" ".join(cmd) + "\n")
        _ledger({"event": "launch", "case": case_id, "round": round_name, "job_dir": str(job_dir)})
        with open(job_parent / f"{job_name}.out", "wb") as out, \
                open(job_parent / f"{job_name}.err", "wb") as err:
            proc = subprocess.Popen(cmd, cwd=str(agent_dir), env=_pier_env(agent_dir),
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    start_new_session=True)
            meta["pier_pid"] = proc.pid
            killed = None
            while proc.poll() is None:
                if time.time() - t0 > wall:
                    killed = "watchdog"
                    _signal_group(proc, signal.SIGINT)
                    try:
                        proc.wait(timeout=GRACE_S)
                    except subprocess.TimeoutExpired:
                        _signal_group(proc, signal.SIGKILL)
                        killed = "watchdog_sigkill"
                        proc.wait(timeout=60)
                    break
                time.sleep(POLL_S)
        meta["pier_rc"] = proc.returncode
        trial_dir = find_trial_dir(job_dir)
        if killed:
            meta["status"] = killed
            meta["reaped"] = reap_trial(trial_dir)
        else:
            meta["status"] = "ok" if proc.returncode == 0 else f"pier_rc_{proc.returncode}"
        _ledger({"event": "exit", "case": case_id, "job_dir": str(job_dir), "status": meta["status"]})
        if cache_file and trial_dir is not None and not killed:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"trial_dir": str(trial_dir), "case": case_id,
                                              "tree_hash": thash}))

    o = load_outcome(trial_dir)
    if meta["status"] in ("watchdog", "watchdog_sigkill") and o.get("reward") is None:
        o["infra_class"] = "watchdog_kill"
    report = run_report(o, dispatch_outputs(trial_dir) if trial_dir else [])
    export_artifacts(trial_dir, dest, o, report)
    _emit_trace(o, trial_dir)
    _index(scratch_root, case_id, run_id, report.splitlines()[0][:300])
    meta.update({"trial_dir": str(trial_dir) if trial_dir else None,
                 "wall_s": round(time.time() - t0, 1),
                 "inflight_end": server_inflight(base_url)})
    return AgentOutput(result=report, metadata=meta)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass

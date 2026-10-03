"""Fault injection: run() must never let a non-CancelledError escape, and must always
leave trajectory.json + run_summary.json behind. An escape means Pier skips the collect
hooks and the task scores 0 regardless of how good the patch was."""
import asyncio, json, logging, tempfile
from pathlib import Path
from pier.models.agent.context import AgentContext
from pier.models.trajectories.trajectory import Trajectory
from seedling.agent import SeedlingAgent


class FakeExec:
    def __init__(self, mode="ok"): self.mode, self.calls = mode, []
    def agent_process_env(self, env): return env
    async def exec(self, command, cwd=None, env=None, user=None, timeout_sec=None):
        self.calls.append(command)
        if self.mode == "slow":
            await asyncio.sleep(30)   # park on an await so cancellation can land mid-run
        if self.mode == "timeout_runtimeerror":      # the docker.py:521 path
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        if self.mode == "boom":
            raise ValueError("catastrophic backend failure")
        class R: stdout, stderr, return_code = "SEED_BOOTSTRAP_OK\nSEED_BASE=abc123\nSEED_HAS_TIMEOUT=1\nSEED_CHECKPOINT_OK\nSEED_PATCH_BYTES=42\nWROTE", None, 0
        return R()


def make(tmp, **kw):
    # mirrors AgentFactory: logs_dir/model_name/extra_env/**kwargs/logger
    return SeedlingAgent(logs_dir=Path(tmp), model_name="openai/qwen38-27b",
                         extra_env={}, logger=logging.getLogger("t"),
                         agent_timeout_sec="600", smoke="true", **kw)


def run_case(mode, cancel_after=None):
    tmp = tempfile.mkdtemp()
    agent, ctx, env = make(tmp), AgentContext(), FakeExec(mode)
    async def go():
        t = asyncio.ensure_future(agent.run("fix it", env, ctx))
        if cancel_after is not None:
            await asyncio.sleep(cancel_after); t.cancel()
        return await t
    raised = None
    try: asyncio.run(go())
    except BaseException as e: raised = type(e).__name__
    d = Path(tmp)
    traj = d / "trajectory.json"
    return {
        "raised": raised,
        "trajectory": traj.exists(),
        "traj_valid": bool(Trajectory.model_validate_json(traj.read_text())) if traj.exists() else False,
        "summary": (d / "run_summary.json").exists(),
        "outcome": json.loads((d / "run_summary.json").read_text())["outcome"]
                   if (d / "run_summary.json").exists() else None,
    }


CASES = [
    ("healthy run",                    "ok",                   None, None),
    ("exec timeout -> RuntimeError",   "timeout_runtimeerror", None, None),
    ("arbitrary backend exception",    "boom",                 None, None),
    ("hard cancellation",              "slow",                 0.05, "CancelledError"),
]
print(f"{'case':34s} {'raised':16s} {'traj':5s} {'valid':6s} {'summary':8s} outcome")
fails = 0
for label, mode, cancel, want in CASES:
    r = run_case(mode, cancel)
    ok = (r["raised"] == want) and r["trajectory"] and r["traj_valid"] and r["summary"]
    fails += not ok
    print(f"{label:34s} {str(r['raised']):16s} {str(r['trajectory']):5s} "
          f"{str(r['traj_valid']):6s} {str(r['summary']):8s} {r['outcome']}")
print("\nRESULT:", "ALL PASS" if not fails else f"{fails} FAILURE(S)")
raise SystemExit(1 if fails else 0)

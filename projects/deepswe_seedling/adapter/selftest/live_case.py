"""Phase-3 live check: run real DeepSWE case(s) through the REAL evaluator + scorer path.

    python live_case.py --config <yaml> <case_id>[,<case_id>...] [--pier-wall S]
                        [--eval-wall S] [--parallelism N]

Assembles the framework from the config exactly as main_loop does (env exports included),
copies the unedited seed into a fresh round dir, calls ``evaluator.run`` on the cases, and
prints binary checks per case: envelope parsed / reward equals the trial's result.json /
whitelisted artifacts present / no hidden grader output under the round dir / infra cases
flagged. ``--pier-wall`` shrinks the inner watchdog (forced-kill test); ``--eval-wall``
shrinks the evaluator's per-case timeout (orphan test -> the sbatch trap must reap).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402

HIDDEN_MARKERS = ("test-stdout.txt", "ctrf.json", "base.xml", "new.xml", "gate.xml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("cases")
    ap.add_argument("--pier-wall", type=float)
    ap.add_argument("--eval-wall", type=float)
    ap.add_argument("--parallelism", type=int)
    a = ap.parse_args()

    cfg = cfg_mod.load(a.config)
    runtime_env.apply_all(cfg)
    if a.pier_wall:
        os.environ["SID_PIER_WALL_S"] = str(a.pier_wall)
    fw = cfg_mod.build_components(cfg)
    ev = fw.evaluator
    if a.eval_wall:
        ev.wall_time_s = float(a.eval_wall)
    if a.parallelism:
        ev.parallelism = a.parallelism

    stamp = time.strftime("%Y%m%d_%H%M%S")
    round_dir = Path(fw.runs_root) / f"live_{stamp}_{os.environ.get('SLURM_JOB_ID', 'local')}" / "round_000"
    shutil.copytree(fw.seed_dir, round_dir / "task_agent", ignore=shutil.ignore_patterns("__pycache__"))
    ids = [c for c in a.cases.split(",") if c]
    print(f"live: round_dir={round_dir} cases={ids} pier_wall={os.environ.get('SID_PIER_WALL_S')} "
          f"eval_wall={ev.wall_time_s} parallelism={ev.parallelism}", flush=True)
    t0 = time.time()
    result = ev.run(round_dir, fw.benchmark_dir, case_ids=ids)
    print(f"live: evaluator.run returned in {time.time() - t0:.0f}s", flush=True)

    fails = 0
    hidden = [str(p) for p in round_dir.rglob("*") if p.name in HIDDEN_MARKERS or "/verifier/" in str(p)]
    for c in result.per_case:
        d = c.details or {}
        meta = d.get("agent_metadata") or {}
        trial = meta.get("trial_dir")
        truth = None
        if trial and (Path(trial) / "result.json").is_file():
            truth = ((json.loads((Path(trial) / "result.json").read_text()).get("verifier_result") or {})
                     .get("rewards") or {}).get("reward")
        scratch = Path(meta.get("scratch_dir") or "/nonexistent")
        present = sorted(p.name for p in scratch.iterdir()) if scratch.is_dir() else []
        checks = {
            "envelope_parsed": c.error is None or "timeout" in (c.error or ""),
            "reward_matches_result_json": d.get("excluded") or (d.get("reward") == truth),
            "artifacts_whitelisted": set(present) <= {"report.md", "run_summary.json", "model.patch",
                                                      "reward.json", "exec_log.txt", "transcripts"},
            "report_present": "report.md" in present or bool(c.error),
        }
        ok = all(checks.values())
        fails += not ok
        print(json.dumps({"case": c.case_id, "ok": ok, "score": c.score, "error": c.error,
                          "excluded": d.get("excluded"), "infra_class": d.get("infra_class"),
                          "reward": d.get("reward"), "truth": truth, "status": meta.get("status"),
                          "wall_s": meta.get("wall_s"), "classes": d.get("failure_classes"),
                          "roles": [f"{r.get('role')}.{r.get('attempt')}:{r.get('stop_reason')}"
                                    for r in d.get("roles") or []],
                          "artifacts": present, "checks": checks}), flush=True)
    print(f"live: hidden grader files under round dir: {hidden or 'none'}")
    fails += bool(hidden)
    print("LIVE CHECK PASS" if not fails else f"LIVE CHECK: {fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

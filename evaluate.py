"""Standalone evaluation: run a specific task_agent against the FULL
benchmark (no train/eval split applied).

Use this to compare the unedited seed agent against any round produced
by the optimization loop. Results land in
``runs/eval_<stamp>_<agent_basename>/round_eval/``.

    python3 evaluate.py --config configs/travel.yaml \\
                        --agent task_agent_seeds/travel_baseline

    python3 evaluate.py --config configs/travel.yaml \\
                        --agent runs/20260506_140000_travel_default/round_003/task_agent
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.models import EvaluationResult


def run(config_path: Path, agent_path: Path) -> EvaluationResult:
    if not agent_path.exists() or not agent_path.is_dir():
        raise FileNotFoundError(f"agent directory not found: {agent_path}")

    cfg = cfg_mod.load(config_path)
    runtime_env.apply_all(cfg)

    fw = cfg_mod.build_components(cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = fw.runs_root / f"eval_{stamp}_{agent_path.name}"
    round_dir = out_root / "round_eval"
    (round_dir / "logs").mkdir(parents=True)
    shutil.copytree(agent_path, round_dir / "task_agent")
    out_root.joinpath("config.snapshot.yaml").write_text(
        Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
    )

    # No case_ids → run every case in the benchmark.
    result = fw.evaluator.run(round_dir, fw.benchmark_dir)

    (round_dir / "eval_result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )

    total = result.passed + result.failed
    print(
        f"composite_score: {result.score:.4f}  "
        f"passed: {result.passed}/{total}  "
        f"wall: {result.wall_time_s:.1f}s\n"
        f"artifacts: {round_dir}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a specific task_agent against the full benchmark."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to a task_agent directory (e.g. task_agent_seeds/travel_baseline "
        "or runs/<exp>/round_NNN/task_agent)",
    )
    args = parser.parse_args()
    run(args.config, args.agent)

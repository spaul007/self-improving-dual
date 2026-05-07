"""Entry point. Loads a YAML config, assembles components, and hands control
to the manager. There is intentionally no round loop here — the manager owns
the optimization regime so it can be swapped via config alone.

Every pluggable component (manager, evaluator, gatherer, validators, editor)
is built by :func:`meta_agent.config.build_components` from the YAML;
the only thing this entry point does is push environment variables that
subprocesses inherit (model, reasoning effort, tool-package allow-list,
travel database root) before instantiating things.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.models import EvolutionOutcome


def run(config_path: Path) -> EvolutionOutcome:
    cfg = cfg_mod.load(config_path)

    runtime_env.apply_all(cfg)

    fw = cfg_mod.build_components(cfg)

    experiment_dir = cfg_mod.init_experiment_dir(cfg, config_path, fw.runs_root)

    outcome = fw.manager.evolve(
        editor=fw.editor,
        evaluator=fw.evaluator,
        gatherer=fw.gatherer,
        seed_dir=fw.seed_dir,
        benchmark_dir=fw.benchmark_dir,
        experiment_dir=experiment_dir,
        max_rounds=cfg.loop.max_rounds,
        score_target=cfg.loop.score_target,
        train_case_ids=fw.train_case_ids,
        eval_case_ids=fw.eval_case_ids,
    )

    print(
        f"Experiment dir: {experiment_dir}\n"
        f"Best round: {outcome.best_round}  "
        f"Final score: {outcome.final_score:.3f}"
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the meta-agent self-evolution loop.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)

"""One-off debug runner: start an HGM run with round_000 SEEDED from a donor
run's seed evaluation instead of re-evaluating the base agent.

Only for component smoke/debugging — the seed baseline is borrowed, so scores
are comparable only if the donor used the same seed agent, task model, and
train ids. Delete this file when done; regular runs keep using main_loop.py.

    PYTHONPATH=. python3 run_smoke_seeded.py \
        --config configs/hgm_travel_smoke_beliefs2stage.yaml \
        --donor  /path/to/donor_run/round_000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import main_loop
from meta_agent.managers import hgm as hgm_mod
from meta_agent.managers.hgm_tree import HGMNode
from meta_agent.models import CaseResult, EvolutionStrategy


def make_seeded_run_seed(donor_round: Path):
    donor = json.loads((donor_round / "eval_result.json").read_text(
        encoding="utf-8"))
    by_id = {c["case_id"]: c for c in donor["per_case"]}
    print(f"[seeded] donor {donor_round} -> {len(by_id)} case results "
          f"(donor score {donor.get('score', 0.0):.4f})", flush=True)

    def _run_seed(self, seed_dir, evaluator, gatherer):
        # Mirrors HGMManager._run_seed except the evaluator.run call, which is
        # replaced by the donor's per-case results.
        out_dir = self._experiment_dir / "round_000"
        agent_dst = out_dir / "task_agent"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
        agent_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, agent_dst)
        (out_dir / "logs").mkdir(exist_ok=True)

        node = HGMNode(node_id=0, parent_id=None, round_dir=out_dir)
        self._tree.add(node)
        self._next_id = 1

        missing = [cid for cid in self._train_case_ids if cid not in by_id]
        if missing:
            raise SystemExit(f"[seeded] donor round_000 lacks train cases: "
                             f"{missing}")
        for cid in self._train_case_ids:
            node.record(CaseResult(**by_id[cid]))

        zero_strategy = EvolutionStrategy(
            target_files=[],
            optimization_goal="Seed agent (HGM tree root).",
            proposed_changes="(none — seed pre-eval reused from donor run)",
            rationale="HGM tree root.",
        )
        self._feedback[0] = gatherer.compile(
            0, 0, zero_strategy, self._build_eval_result(node), out_dir
        )
        self._write_node_sidecar(node)
        print(
            f"node 0: SEED reused from donor -> mean={node.mean_utility:.3f} "
            f"n={node.n_evals} (no evaluation run)",
            flush=True,
        )

    return _run_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--donor", required=True,
                    help="donor run's round_000 directory")
    args = ap.parse_args()
    donor_round = Path(args.donor)
    if not (donor_round / "eval_result.json").exists():
        raise SystemExit(f"[seeded] no eval_result.json under {donor_round}")
    hgm_mod.HGMManager._run_seed = make_seeded_run_seed(donor_round)
    main_loop.run(args.config)


if __name__ == "__main__":
    main()

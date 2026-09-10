"""Start an HGM run with round_000 REUSED from a donor run's seed evaluation.

The seed pre-evaluation (the full train set on the task model) is the most
expensive free step of a run and is identical for every run that shares the
same seed agent, task model, task-agent effort and train ids. This runner
copies the donor's ``round_000`` evidence into the new run — ``logs/`` (the
trace and per-case results the feedback gatherer reads) and ``verbose/`` —
replays the donor's per-case results into node 0, recompiles node 0's
feedback from the copied trace, and then continues with round 1 exactly as
``main_loop.py`` would (init expansions, evaluations, finalize).

Use it ONLY with a donor produced by the same config: the donor's
``config.snapshot.yaml`` must agree on ``task_agent`` and ``split`` (checked;
``--force`` overrides), otherwise the borrowed baseline is not comparable.

    PYTHONPATH=. python3 run_seeded.py \\
        --config configs/hgm_travel_1000_qwen122b_dsv4pro_beliefs2stage.yaml \\
        --donor  runs/<donor_run>/round_000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

import main_loop
from meta_agent.managers import hgm as hgm_mod
from meta_agent.managers.hgm_tree import HGMNode
from meta_agent.models import CaseResult, EvolutionStrategy

DONOR_MARKER = "seed_donor.json"


def _check_donor_matches(config_path: Path, donor_round: Path, *,
                         force: bool) -> None:
    snap = donor_round.parent / "config.snapshot.yaml"
    if not snap.exists():
        if force:
            return
        raise SystemExit(f"[seeded] donor has no config.snapshot.yaml next to "
                         f"{donor_round}; pass --force to skip the check")
    mine = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    theirs = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
    mismatched = [k for k in ("task_agent", "split", "project")
                  if mine.get(k) != theirs.get(k)]
    if mismatched and not force:
        raise SystemExit(f"[seeded] donor config differs on {mismatched} — the "
                         "seed baseline would not be comparable (--force to "
                         "override)")
    if mismatched:
        print(f"[seeded] WARNING: donor config differs on {mismatched}; "
              "continuing under --force", flush=True)


def make_seeded_run_seed(donor_round: Path):
    donor = json.loads((donor_round / "eval_result.json").read_text(
        encoding="utf-8"))
    by_id = {c["case_id"]: c for c in donor["per_case"]}
    print(f"[seeded] donor {donor_round} -> {len(by_id)} case results "
          f"(donor score {donor.get('score', 0.0):.4f})", flush=True)

    def _run_seed(self, seed_dir, evaluator, gatherer):
        # Mirrors HGMManager._run_seed except the evaluator.run call, which is
        # replaced by the donor's per-case results, and the donor's logs,
        # which are copied so the recompiled feedback (tool usage, LLM
        # calls, failure report) is derived from the same trace.
        out_dir = self._experiment_dir / "round_000"
        agent_dst = out_dir / "task_agent"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
        agent_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, agent_dst)
        for sub in ("logs", "verbose"):
            src = donor_round / sub
            if src.is_dir():
                dst = out_dir / sub
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        (out_dir / "logs").mkdir(exist_ok=True)
        (out_dir / DONOR_MARKER).write_text(json.dumps({
            "donor_round": str(donor_round.resolve()),
            "donor_score": donor.get("score"),
            "n_cases": len(by_id),
        }, indent=2) + "\n", encoding="utf-8")

        node = HGMNode(node_id=0, parent_id=None, round_dir=out_dir)
        self._tree.add(node)
        self._next_id = 1

        missing = [cid for cid in self._train_case_ids if cid not in by_id]
        if missing:
            raise SystemExit(f"[seeded] donor round_000 lacks train cases: "
                             f"{missing}")
        # Replay in the DONOR's per-case order (its evaluator's completion
        # order): the failure report samples representative failures in
        # encounter order, so this keeps node 0's feedback byte-identical
        # to the donor's rather than an equally valid but different sample.
        wanted = set(self._train_case_ids)
        for c in donor["per_case"]:
            if c["case_id"] in wanted:
                node.record(CaseResult(**c))

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
            f"n={node.n_evals} (no evaluation run; logs copied)",
            flush=True,
        )

    return _run_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--donor", required=True,
                    help="donor run's round_000 directory")
    ap.add_argument("--force", action="store_true",
                    help="skip the donor-config compatibility check")
    args = ap.parse_args()
    donor_round = Path(args.donor)
    if not (donor_round / "eval_result.json").exists():
        raise SystemExit(f"[seeded] no eval_result.json under {donor_round}")
    _check_donor_matches(Path(args.config), donor_round, force=args.force)
    hgm_mod.HGMManager._run_seed = make_seeded_run_seed(donor_round)
    main_loop.run(args.config)


if __name__ == "__main__":
    main()

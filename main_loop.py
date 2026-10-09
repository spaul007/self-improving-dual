"""Entry point. Loads a YAML config, assembles components, and hands control
to the manager. There is intentionally no round loop here — the manager owns
the optimization regime so it can be swapped via config alone.

Every pluggable component (manager, evaluator, gatherer, validators, editor)
is built by :func:`meta_agent.config.build_components` from the YAML;
the only thing this entry point does is push environment variables that
subprocesses inherit (model, reasoning effort, tool-package allow-list,
project database root) before instantiating things.

After the manager returns, a Markdown ``run_summary.md`` is written into the
experiment directory naming the best round and detailing the top-3 agents
(Stage A vs Stage B winner, optimization goal, train mean, full-benchmark
score when ``full_eval_top_k`` is set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from meta_agent import config as cfg_mod
from meta_agent import runtime_env
from meta_agent.models import EvolutionOutcome


# Config keys a resume may change (everything else must match the run's snapshot, or
# the continued search would silently mix two experiments).
RESUME_MUTABLE_KEYS = {
    ("evaluator", "config", "parallelism"),
    ("loop", "max_rounds"),
    ("manager", "config", "eval_budget"),
}


def _flatten(d: Any, prefix: tuple = ()) -> dict[tuple, Any]:
    if isinstance(d, dict):
        out: dict[tuple, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, prefix + (k,)))
        return out
    return {prefix: d}


def resume_config_diff(snapshot_text: str, new_text: str) -> tuple[list, list]:
    """(allowed_changes, forbidden_changes) between a run's config snapshot and a
    config offered on resume, as lists of (key-path, old, new)."""
    import yaml
    a = _flatten(yaml.safe_load(snapshot_text) or {})
    b = _flatten(yaml.safe_load(new_text) or {})
    allowed, forbidden = [], []
    for k in sorted(set(a) | set(b), key=str):
        if a.get(k) != b.get(k):
            (allowed if k in RESUME_MUTABLE_KEYS else forbidden).append((".".join(map(str, k)), a.get(k), b.get(k)))
    return allowed, forbidden


def run(config_path: Optional[Path], resume_dir: Optional[Path] = None) -> EvolutionOutcome:
    if resume_dir is not None:
        resume_dir = Path(resume_dir).resolve()
        snap = resume_dir / "config.snapshot.yaml"
        if not snap.is_file():
            raise SystemExit(f"--resume: {snap} not found")
        if config_path is not None:
            allowed, forbidden = resume_config_diff(snap.read_text(), Path(config_path).read_text())
            if forbidden:
                raise SystemExit("--resume: config differs from the run's snapshot in keys a resume may "
                                 f"not change: {forbidden}")
            for k, old, new in allowed:
                print(f"resume: config change {k}: {old} -> {new}", flush=True)
        else:
            config_path = snap
    cfg = cfg_mod.load(config_path)

    runtime_env.apply_all(cfg)

    fw = cfg_mod.build_components(cfg)

    if resume_dir is not None:
        experiment_dir = resume_dir
        if Path(config_path).resolve() != snap.resolve():
            n = len(list(resume_dir.glob("config.resume-*.yaml"))) + 1
            (resume_dir / f"config.resume-{n}.yaml").write_text(Path(config_path).read_text())
    else:
        experiment_dir = cfg_mod.init_experiment_dir(cfg, config_path, fw.runs_root)
    print(f"Experiment dir: {experiment_dir}", flush=True)

    from meta_agent.managers.hgm import RunPaused
    try:
        outcome = _evolve(fw, cfg, experiment_dir, resume=resume_dir is not None)
    except RunPaused:
        print(f"PAUSED -- resume with: main_loop.py --resume {experiment_dir}", flush=True)
        sys.exit(3)
    return _after(experiment_dir, outcome)


def _evolve(fw: Any, cfg: Any, experiment_dir: Path, resume: bool) -> EvolutionOutcome:
    return fw.manager.evolve(
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
        summarizer=fw.summarizer,
        failure_summarizer=fw.failure_summarizer,
        block_suggester=fw.block_suggester,
        unit_selector=fw.unit_selector,
        **({"resume": True} if resume else {}),
        reflector=fw.reflector,
    )


def _after(experiment_dir: Path, outcome: EvolutionOutcome) -> EvolutionOutcome:
    summary_path: Optional[Path] = None
    try:
        summary_path = _write_run_summary(experiment_dir, outcome)
    except Exception as exc:  # noqa: BLE001
        print(f"[run_summary] skipped — {exc!r}", flush=True)

    print(
        f"Experiment dir: {experiment_dir}\n"
        f"Best round: {outcome.best_round}  "
        f"Final score: {outcome.final_score:.3f}"
    )
    if summary_path is not None:
        print(f"Run summary: {summary_path}")
    return outcome


# ---------------------------------------------------------------------- #
# Post-run summary
# ---------------------------------------------------------------------- #


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_round(round_dir: Path) -> Optional[dict[str, Any]]:
    node = _read_json(round_dir / "hgm_node.json")
    if node is None:
        return None
    strategy = _read_json(round_dir / "strategy.json") or {}
    selection = _read_json(round_dir / "variants" / "selection.json")
    eval_score = _read_json(round_dir / "eval_score.json")
    full_eval = _read_json(round_dir / "full_eval_score.json")
    return {
        "round_dir": round_dir,
        "node_id": node.get("node_id"),
        "parent_id": node.get("parent_id"),
        "edit_failed": bool(node.get("edit_failed", False)),
        "mean_utility": float(node.get("mean_utility", 0.0)),
        "n_evals": int(node.get("n_evals", 0)),
        "cmp": node.get("cmp"),
        "strategy": strategy,
        "selection": selection,
        "eval_score": eval_score,
        "full_eval": full_eval,
    }


def _depth(node_id: int, by_id: dict[int, dict[str, Any]]) -> int:
    d = 0
    cur = node_id
    seen: set[int] = set()
    while True:
        if cur in seen:
            break
        seen.add(cur)
        info = by_id.get(cur)
        if info is None:
            break
        parent = info.get("parent_id")
        if parent is None or parent == cur:
            break
        d += 1
        cur = parent
    return d


def _truncate(text: str, limit: int = 600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _render_agent_block(
    rank: int, info: dict[str, Any], by_id: dict[int, dict[str, Any]]
) -> str:
    nid = info["node_id"]
    pid = info["parent_id"]
    depth = _depth(nid, by_id)
    strategy = info["strategy"] or {}
    selection = info["selection"]
    full_eval = info["full_eval"]
    eval_score = info["eval_score"]

    stage_label = "Stage A (intermediate)"
    category_block = ""
    pool_block = ""
    if selection:
        winner = selection.get("winner") or {}
        if winner.get("index", -1) != -1:
            cat_name = winner.get("category_name") or "?"
            cat_id = winner.get("category_id") or "?"
            stage_label = f"Stage B specialist — {cat_name} ({cat_id})"
        pool = selection.get("pool") or []
        if pool:
            rows = []
            for entry in pool:
                label = (
                    "Stage A intermediate"
                    if entry.get("index", -1) == -1
                    else f"Stage B var_{entry.get('index')} — "
                    f"{entry.get('category_name') or entry.get('category_id') or '?'}"
                )
                score = entry.get("mean_score")
                score_s = "n/a" if score is None else f"{score:.3f}"
                rows.append(f"    - {label}: mean={score_s}, n={entry.get('n_cases', 0)}")
            pool_block = "\n  - Variant pool:\n" + "\n".join(rows)

    goal = _truncate(strategy.get("optimization_goal") or "(none)")
    changes = _truncate(strategy.get("proposed_changes") or "(none)", limit=400)

    lines = [
        f"### #{rank}. Round {nid:03d} (node {nid})",
        f"  - Parent: node {pid}; depth: {depth}",
        f"  - Train mean: **{info['mean_utility']:.3f}** over {info['n_evals']} case(s)",
        f"  - Winner: {stage_label}",
    ]
    if full_eval is not None:
        passed = full_eval.get("passed")
        n_cases = full_eval.get("n_cases")
        lines.append(
            f"  - Full-benchmark score: **{full_eval.get('composite_score', 0.0):.3f}**"
            f" (passed {passed}/{n_cases})"
        )
    if eval_score is not None:
        lines.append(
            f"  - Held-out eval score: **{eval_score.get('composite_score', 0.0):.3f}**"
            f" (passed {eval_score.get('passed')}/"
            f"{(eval_score.get('passed', 0) + eval_score.get('failed', 0))})"
        )
    if category_block:
        lines.append(category_block)
    if pool_block:
        lines.append(pool_block)
    lines.append(f"  - Optimization goal:\n\n    > {goal.replace(chr(10), chr(10) + '    > ')}")
    lines.append(f"  - Proposed changes:\n\n    > {changes.replace(chr(10), chr(10) + '    > ')}")
    return "\n".join(lines)


def _write_run_summary(experiment_dir: Path, outcome: EvolutionOutcome) -> Path:
    round_dirs = sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and d.name.startswith("round_")
    )
    rounds: list[dict[str, Any]] = []
    for rd in round_dirs:
        info = _parse_round(rd)
        if info is not None:
            rounds.append(info)
    by_id = {r["node_id"]: r for r in rounds if r["node_id"] is not None}

    successes = [r for r in rounds if not r["edit_failed"]]
    successes.sort(key=lambda r: r["mean_utility"], reverse=True)
    top3 = successes[:3]

    best_info = by_id.get(outcome.best_round)
    best_lines: list[str] = []
    if best_info is not None:
        best_lines.append(
            f"- Round: **{outcome.best_round:03d}** (node {outcome.best_round})"
        )
        best_lines.append(
            f"- Train mean: **{outcome.final_score:.3f}** over "
            f"{best_info['n_evals']} case(s)"
        )
        full_eval = best_info["full_eval"]
        if full_eval is not None:
            best_lines.append(
                f"- Full-benchmark score: **{full_eval.get('composite_score', 0.0):.3f}**"
                f" (passed {full_eval.get('passed')}/{full_eval.get('n_cases')})"
            )
        eval_score = best_info["eval_score"]
        if eval_score is not None:
            best_lines.append(
                f"- Held-out eval score: **{eval_score.get('composite_score', 0.0):.3f}**"
            )
        goal = _truncate(
            (best_info["strategy"] or {}).get("optimization_goal") or "(none)",
            limit=400,
        )
        best_lines.append(f"- Optimization goal: {goal}")
    else:
        best_lines.append(f"- Round: **{outcome.best_round:03d}**")
        best_lines.append(f"- Final score: **{outcome.final_score:.3f}**")

    total_nodes = len(rounds)
    failed = sum(1 for r in rounds if r["edit_failed"])
    out_lines = [
        f"# Run summary — {experiment_dir.name}",
        "",
        "## Best round (LCB-selected)",
        *best_lines,
        "",
        "## Top 3 agents (by train mean)",
        "",
    ]
    if not top3:
        out_lines.append("(no successful expansions)")
    else:
        for rank, info in enumerate(top3, start=1):
            out_lines.append(_render_agent_block(rank, info, by_id))
            out_lines.append("")

    out_lines += [
        "## Run stats",
        f"- Total nodes generated: **{total_nodes}** (incl. seed root)",
        f"- Edit-failed nodes: {failed}",
        f"- Successful expansions: {len(successes) - (1 if successes else 0)}",
        "",
    ]

    summary_path = experiment_dir / "run_summary.md"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the meta-agent self-evolution loop.")
    parser.add_argument("--config", type=Path, help="Path to YAML config")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="Continue an existing (paused or killed) run dir. --config is optional "
                             "and may only change parallelism, max_rounds or eval_budget.")
    args = parser.parse_args()
    if args.config is None and args.resume is None:
        parser.error("--config is required (unless --resume)")
    run(args.config, resume_dir=args.resume)

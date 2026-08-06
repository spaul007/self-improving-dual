"""Plot budget-vs-performance curves, averaged over repeated runs of a method.

Reads the per-budget ``snapshots/eval_at_budget_<B>.json`` files written by
``snapshot_eval.py`` and produces two figures, each overlaying every method:

    1. composite score vs budget
    2. pass rate vs budget   (pass rate = passed_mean / n_cases)

A "method" is a set of runs that should be averaged together (e.g. the two
``hgm`` runs, or the two ``hgm_dual`` runs). At each budget the mean/std POOL
every individual eval run across all experiments of that method (e.g. exp1
run1/run2 + exp2 run1/run2 -> 4 values), so the shaded band reflects both
within- and between-experiment variance.

This script is domain-agnostic — point it at any benchmark's runs by editing
the CONFIG block below:

  * ``RUNS_BASE``  — directory that holds the run folders (or set to None and
    give absolute paths in the method lists).
  * ``METHODS``    — one entry per curve: label -> list of run dirs. Each entry
    may be a bare folder name (resolved under RUNS_BASE) or an absolute path.
  * ``BUDGET_ZERO``— optional seed/baseline anchor at budget 0, per method.
  * ``CASE_NAME``  — used in titles and output filenames.

Then run:

    python3 plot_budget_curves.py
    python3 plot_budget_curves.py --out-dir /tmp/plots --no-band
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CONFIG — edit this block for a new benchmark/domain.                         #
# --------------------------------------------------------------------------- #

# Base directory holding the run folders. Set to None to require absolute
# paths in the method lists below.
RUNS_BASE = Path(
    "/home/ubuntu/sudipta/agentic_reasoning/"
    "meta-agent-dev-v2-improved-feedback-shopping/runs"
)


def R(name: str) -> Path:
    """Resolve a run entry: absolute path as-is, else joined under RUNS_BASE."""
    p = Path(name)
    if p.is_absolute() or RUNS_BASE is None:
        return p
    return RUNS_BASE / name


HGM_RUNS = [
    R("20260619_071723_shopping_hgm_4000"),
    R("20260621_004834_shopping_hgm_4000"),
]

HGM_DUAL_RUNS = [
    R("20260619_071807_shopping_hgm_dual_4000"),
    R("20260621_004847_shopping_hgm_dual_4000"),
]

METHODS: dict[str, list[Path]] = {
    "HGM (Feedback optimized)": HGM_RUNS,
    "HGM-dual": HGM_DUAL_RUNS,
}

# Optional budget=0 anchor point per method (e.g. the seed/baseline agent's
# performance before any optimization). Set the composite score and/or pass
# rate you want each curve to start from at budget 0. Use None (or leave a
# method out) to skip the anchor for that method. pass_rate is a fraction 0-1.
BUDGET_ZERO: dict[str, dict] = {
    "HGM (Feedback optimized)": {"composite": .82, "pass_rate": .50},
    "HGM-dual":                 {"composite": .82, "pass_rate": .50},
}

# Title prefix for the figures / output filenames.
CASE_NAME = "shopping"
# --------------------------------------------------------------------------- #


_BUDGET_RE = re.compile(r"eval_at_budget_(\d+)\.json$")


def load_run_points(run_dir: Path) -> dict[int, dict]:
    """Return {budget: {...}} for one run directory.

    Budget key is the JSON's ``requested_budget`` (falls back to the filename).
    Pass rate is ``passed_mean / n_cases`` (None if n_cases missing/zero).
    Also returns the raw individual eval runs (repeats) so a method's std can
    pool every run across experiments.
    """
    snap_dir = run_dir / "snapshots"
    points: dict[int, dict] = {}
    if not snap_dir.is_dir():
        print(f"  [warn] no snapshots/ in {run_dir}")
        return points
    for jf in sorted(snap_dir.glob("eval_at_budget_*.json")):
        m = _BUDGET_RE.search(jf.name)
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] could not read {jf}: {exc}")
            continue
        budget = int(data.get("requested_budget") or (m.group(1) if m else 0))
        n_cases = data.get("n_cases")
        composite = data.get("composite_score")
        passed = data.get("passed_mean")
        pass_rate = (
            float(passed) / float(n_cases)
            if passed is not None and n_cases
            else None
        )
        # Raw individual eval runs (repeats) underneath this experiment, so a
        # method's std can pool every run across experiments rather than just
        # the experiment-level means. composite -> composite_score_runs;
        # pass rate -> per_run[*].passed / per_run[*].n_cases.
        composite_runs = [
            float(x) for x in (data.get("composite_score_runs") or []) if x is not None
        ]
        pass_runs: list[float] = []
        for r in data.get("per_run") or []:
            rc = r.get("n_cases") or n_cases
            rp = r.get("passed")
            if rp is not None and rc:
                pass_runs.append(float(rp) / float(rc))
        points[budget] = {
            "composite": None if composite is None else float(composite),
            "pass_rate": pass_rate,
            # Fall back to the experiment-level value if raw runs are absent.
            "composite_runs": composite_runs or (
                [] if composite is None else [float(composite)]
            ),
            "pass_runs": pass_runs or ([] if pass_rate is None else [pass_rate]),
        }
    return points


def aggregate_method(run_dirs: list[Path]) -> dict[int, dict]:
    """Average each metric across runs, per budget.

    Mean and std POOL every individual eval run across all experiments of the
    method (e.g. exp1 run1/run2 + exp2 run1/run2 -> 4 values), so the std band
    reflects both within- and between-experiment variance.

    Returns {budget: {composite_mean, composite_std, pass_mean, pass_std, n}}
    where n is how many individual runs contributed at that budget.
    """
    by_budget_comp: dict[int, list[float]] = defaultdict(list)
    by_budget_pass: dict[int, list[float]] = defaultdict(list)
    for rd in run_dirs:
        pts = load_run_points(Path(rd))
        for budget, vals in pts.items():
            by_budget_comp[budget].extend(vals["composite_runs"])
            by_budget_pass[budget].extend(vals["pass_runs"])

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        n = len(xs)
        m = sum(xs) / n
        s = (sum((x - m) ** 2 for x in xs) / n) ** 0.5 if n > 1 else 0.0
        return m, s

    out: dict[int, dict] = {}
    for budget in sorted(set(by_budget_comp) | set(by_budget_pass)):
        comp = by_budget_comp.get(budget, [])
        pas = by_budget_pass.get(budget, [])
        cm, cs = _mean_std(comp) if comp else (None, None)
        pm, ps = _mean_std(pas) if pas else (None, None)
        out[budget] = {
            "composite_mean": cm,
            "composite_std": cs,
            "pass_mean": pm,
            "pass_std": ps,
            "n": max(len(comp), len(pas)),
        }
    return out


def _plot_metric(
    agg: dict[str, dict],
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    show_band: bool,
) -> None:
    plt.figure(figsize=(8, 5.5))
    for label, per_budget in agg.items():
        budgets = sorted(b for b, v in per_budget.items() if v[mean_key] is not None)
        if not budgets:
            continue
        ys = [per_budget[b][mean_key] for b in budgets]
        es = [per_budget[b][std_key] or 0.0 for b in budgets]
        line = plt.plot(budgets, ys, marker="o", linewidth=2, label=label)[0]
        if show_band:
            lo = [y - e for y, e in zip(ys, es)]
            hi = [y + e for y, e in zip(ys, es)]
            plt.fill_between(budgets, lo, hi, alpha=0.18, color=line.get_color())
    plt.xlabel("Budget (evaluations spent)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("./plots"),
                    help="Directory to write the PNGs (default: ./plots).")
    ap.add_argument("--no-band", action="store_true",
                    help="Hide the ±1 std shaded band.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agg: dict[str, dict] = {}
    for label, run_dirs in METHODS.items():
        print(f"# {label}: {len(run_dirs)} run(s)")
        agg[label] = aggregate_method(run_dirs)
        # Inject the optional budget=0 anchor (seed/baseline) if set.
        zero = BUDGET_ZERO.get(label) or {}
        zc, zp = zero.get("composite"), zero.get("pass_rate")
        if zc is not None or zp is not None:
            agg[label][0] = {
                "composite_mean": None if zc is None else float(zc),
                "composite_std": 0.0,
                "pass_mean": None if zp is None else float(zp),
                "pass_std": 0.0,
                "n": 0,
            }
        for b in sorted(agg[label]):
            v = agg[label][b]
            cm = "n/a" if v["composite_mean"] is None else f"{v['composite_mean']:.4f}"
            pm = "n/a" if v["pass_mean"] is None else f"{v['pass_mean']:.4f}"
            print(f"    budget={b:>6}  composite={cm}  pass_rate={pm}  (n={v['n']})")

    show_band = not args.no_band
    _plot_metric(
        agg, "composite_mean", "composite_std",
        ylabel="Composite score",
        title=f"{CASE_NAME}: composite score vs budget",
        out_path=args.out_dir / f"{CASE_NAME}_composite_vs_budget.png",
        show_band=show_band,
    )
    _plot_metric(
        agg, "pass_mean", "pass_std",
        ylabel="Case Accuracy",
        title=f"{CASE_NAME}: Case Accuracy vs budget",
        out_path=args.out_dir / f"{CASE_NAME}_passrate_vs_budget.png",
        show_band=show_band,
    )


if __name__ == "__main__":
    main()

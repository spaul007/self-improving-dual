"""C Tier 1 — what can the current measurement actually detect?

Everything here reads scores already on disk. No GPU, no API.

Five diagnostics:

  1. noise floor      — variance decomposition from repeat evaluations of the SAME
                        agent code on the SAME cases
  2. power            — cases needed to resolve a given effect, paired and unpaired
  3. discriminability — between-node vs within-node variance for a search run
  4. winner's curse   — simulate a tree of nodes with IDENTICAL true quality under
                        the run's own eval counts, and see how big an apparent
                        "best minus seed" the selection rule manufactures anyway
  5. arm re-analysis  — are the published arm differences distinguishable from noise?

Usage:
    PYTHONPATH=. python3 study/noise.py --repeats <dir> [<dir> ...] \\
        [--run runs/<search run>] [--arms name=path[,name=path...]] [--out report.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Any, Optional

Z80 = 0.8416  # one-sided z for 80% power
Z95 = 1.9600  # two-sided z at alpha=0.05


def _load_eval(d: Path) -> Optional[dict[str, Any]]:
    """Per-case scores from an eval or round directory."""
    hits = sorted(glob.glob(str(d / "round_*" / "eval_result.json"))) or \
        sorted(glob.glob(str(d / "eval_result.json")))
    if not hits:
        return None
    j = json.loads(Path(hits[0]).read_text())
    per = {c["case_id"]: c["score"] for c in (j.get("per_case") or [])
           if c.get("score") is not None}
    return {"dir": str(d), "score": j.get("score"), "per_case": per}


def decompose(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Split per-observation variance into case difficulty and run-to-run noise.

    Repeats of the same code on the same cases, so any spread WITHIN a case is
    the agent being stochastic, and the spread of case means (corrected for it)
    is real difficulty.
    """
    sets = [set(r["per_case"]) for r in runs]
    cases = sorted(set.intersection(*sets))
    k = len(runs)
    if k < 2 or not cases:
        raise SystemExit("need >=2 repeats sharing >=1 case")
    means = [st.mean(r["per_case"][c] for c in cases) for r in runs]
    per_case = [[r["per_case"][c] for r in runs] for c in cases]
    case_means = [st.mean(v) for v in per_case]
    within = st.mean(st.pvariance(v) for v in per_case)          # run-to-run
    between_raw = st.pvariance(case_means)
    between = max(between_raw - within / k, 0.0)                 # de-biased
    return {
        "n_repeats": k, "n_cases": len(cases),
        "run_means": means,
        "grand_mean": st.mean(means),
        "sd_across_runs": st.stdev(means) if k > 1 else 0.0,
        "var_between_case": between, "sd_between_case": between ** 0.5,
        "var_within_case": within, "sd_within_case": within ** 0.5,
        "icc": between / (between + within) if (between + within) else 0.0,
        "n_deterministic_cases": sum(1 for v in per_case if st.pvariance(v) == 0),
    }


def se_table(dec: dict[str, Any], ns=(16, 32, 60, 120)) -> list[dict[str, Any]]:
    sb, sw = dec["var_between_case"], dec["var_within_case"]
    out = []
    for n in ns:
        out.append({
            "n": n,
            "se_mean_fresh": math.sqrt((sb + sw) / n),   # fresh cases each time
            "se_paired": math.sqrt(2 * sw / n),          # same cases, two runs
            "se_unpaired": math.sqrt(2 * (sb + sw) / n),
        })
    return out


def power_table(dec: dict[str, Any],
                effects=(0.02, 0.03, 0.05, 0.10)) -> list[dict[str, Any]]:
    """Cases needed for 80% power at alpha=0.05 (two-sided)."""
    sw, sb = dec["var_within_case"], dec["var_between_case"]
    sd_paired = math.sqrt(2 * sw)              # per-case paired difference
    sd_unpaired = math.sqrt(2 * (sb + sw))
    out = []
    for e in effects:
        out.append({
            "effect": e,
            "n_paired": math.ceil(((Z95 + Z80) * sd_paired / e) ** 2),
            "n_unpaired": math.ceil(((Z95 + Z80) * sd_unpaired / e) ** 2),
        })
    return out


def _nodes_of(run: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(run.glob("round_*/hgm_node.json")):
        d = json.loads(p.read_text())
        n = int(d.get("n_evals") or 0)
        if n <= 0:
            continue
        out.append({"node": d.get("node_id"), "parent": d.get("parent_id"),
                    "n": n, "mean": float(d.get("mean_utility") or 0.0)})
    return out


def discriminability(run: Path, dec: dict[str, Any]) -> dict[str, Any]:
    """How much of the spread between node means is real?

    Each node mean carries sampling error sqrt((sb+sw)/n_i). Subtract the mean of
    those from the observed spread; what is left is the between-node signal.
    """
    nodes = _nodes_of(run)
    if len(nodes) < 2:
        return {}
    sb, sw = dec["var_between_case"], dec["var_within_case"]
    means = [x["mean"] for x in nodes]
    obs = st.pvariance(means)
    err = st.mean((sb + sw) / x["n"] for x in nodes)
    sig = max(obs - err, 0.0)
    seed = next((x for x in nodes if x["parent"] is None), None)
    best = max(nodes, key=lambda x: x["mean"])
    return {
        "n_nodes": len(nodes),
        "observed_var": obs, "observed_sd": obs ** 0.5,
        "sampling_var": err, "sampling_sd": err ** 0.5,
        "signal_var": sig, "signal_sd": sig ** 0.5,
        "icc_nodes": sig / obs if obs else 0.0,
        "seed": seed, "best": best,
        "best_minus_seed": (best["mean"] - seed["mean"]) if seed else None,
        "nodes": nodes,
    }


def winners_curse(run: Path, dec: dict[str, Any], *, trials: int = 20000,
                  seed_rng: int = 0) -> dict[str, Any]:
    """Null model: every node is EXACTLY as good as the seed.

    Draw each node's mean with its own real n_evals and the measured noise, take
    the max, and record max - seed. If the observed gap sits inside this
    distribution, the search's headline improvement is consistent with pure
    selection noise.
    """
    nodes = _nodes_of(run)
    seed_node = next((x for x in nodes if x["parent"] is None), None)
    if not seed_node or len(nodes) < 2:
        return {}
    sd_obs = math.sqrt(dec["var_between_case"] + dec["var_within_case"])
    rng = random.Random(seed_rng)
    others = [x for x in nodes if x["parent"] is not None]
    gaps = []
    for _ in range(trials):
        s = rng.gauss(0.0, sd_obs / math.sqrt(seed_node["n"]))
        m = max(rng.gauss(0.0, sd_obs / math.sqrt(x["n"])) for x in others)
        gaps.append(m - s)
    gaps.sort()
    obs = max(x["mean"] for x in nodes) - seed_node["mean"]

    def q(p: float) -> float:
        return gaps[min(len(gaps) - 1, int(p * len(gaps)))]

    return {
        "trials": trials, "observed_gap": obs,
        "null_median": q(0.50), "null_p90": q(0.90), "null_p95": q(0.95),
        "null_p99": q(0.99),
        "p_value": sum(1 for g in gaps if g >= obs) / len(gaps),
    }


def arm_compare(arms: list[tuple[str, Path]], dec: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-node score per arm, with the sampling error that best node carries."""
    sb, sw = dec["var_between_case"], dec["var_within_case"]
    out = []
    for name, path in arms:
        nodes = _nodes_of(path)
        if not nodes:
            continue
        best = max(nodes, key=lambda x: x["mean"])
        seed = next((x for x in nodes if x["parent"] is None), None)
        out.append({
            "arm": name, "run": path.name, "n_nodes": len(nodes),
            "best_node": best["node"], "best_mean": best["mean"], "best_n": best["n"],
            "se_best": math.sqrt((sb + sw) / best["n"]),
            "seed_mean": seed["mean"] if seed else None,
        })
    return out


def report(dec: dict[str, Any], *, disc=None, wc=None, arms=None,
           repeat_dirs=None) -> str:
    L: list[str] = []
    A = L.append
    A("# C Tier 1 — what the current measurement can detect")
    A("")
    A("## 1. Noise floor")
    A("")
    A(f"{dec['n_repeats']} repeat evaluations of the **same agent code** on the "
      f"**same {dec['n_cases']} cases**.")
    A("")
    if repeat_dirs:
        A("| repeat | " + " | ".join(Path(d).name for d in repeat_dirs) + " |")
        A("|---" * (len(repeat_dirs) + 1) + "|")
        A("| score | " + " | ".join(f"{m:.4f}" for m in dec["run_means"]) + " |")
        A("")
    A(f"- run-to-run sd at n={dec['n_cases']}: **{dec['sd_across_runs']:.4f}**")
    A(f"- between-case variance (true difficulty): {dec['var_between_case']:.4f} "
      f"(sd {dec['sd_between_case']:.4f})")
    A(f"- within-case variance (same code, same case, different run): "
      f"{dec['var_within_case']:.4f} (sd {dec['sd_within_case']:.4f})")
    A(f"- **ICC = {dec['icc']:.3f}** — only {100 * dec['icc']:.0f}% of per-observation "
      f"variance is case difficulty; the rest is the agent being stochastic")
    A(f"- cases fully deterministic across all repeats: "
      f"{dec['n_deterministic_cases']}/{dec['n_cases']}")
    A("")
    A("### Standard error by design")
    A("")
    A("| n cases | SE of one mean (fresh cases) | SE of paired Δ (same cases) | SE of unpaired Δ |")
    A("|---|---|---|---|")
    for r in se_table(dec):
        A(f"| {r['n']} | ±{r['se_mean_fresh']:.4f} | ±{r['se_paired']:.4f} "
          f"| ±{r['se_unpaired']:.4f} |")
    A("")
    A("## 2. Power — cases needed to resolve an effect (80% power, α=0.05)")
    A("")
    A("| effect Δ | paired cases | unpaired cases |")
    A("|---|---|---|")
    for r in power_table(dec):
        A(f"| {r['effect']:.2f} | {r['n_paired']:,} | {r['n_unpaired']:,} |")
    A("")
    A("The verdict band in `edit_outcome.NEUTRAL_BAND` is ±0.02.")
    A("")

    if disc:
        A("## 3. Discriminability of the search's own node scores")
        A("")
        A(f"- {disc['n_nodes']} evaluated nodes")
        A(f"- observed spread of node means: sd **{disc['observed_sd']:.4f}**")
        A(f"- spread expected from sampling error alone: sd "
          f"**{disc['sampling_sd']:.4f}**")
        A(f"- residual real between-node signal: sd **{disc['signal_sd']:.4f}** "
          f"→ **{100 * disc['icc_nodes']:.0f}%** of the observed spread")
        if disc.get("seed"):
            A(f"- seed node {disc['seed']['node']}: {disc['seed']['mean']:.4f} "
              f"over {disc['seed']['n']} evals")
        A(f"- best node {disc['best']['node']}: {disc['best']['mean']:.4f} over "
          f"{disc['best']['n']} evals")
        if disc.get("best_minus_seed") is not None:
            A(f"- **best − seed = {disc['best_minus_seed']:+.4f}**")
        A("")

    if wc:
        A("## 4. Winner's curse — the null model")
        A("")
        A("Every node given **identical true quality**, drawn with its own real "
          "`n_evals` and the measured noise; take the max over the tree.")
        A("")
        A(f"- observed best − seed: **{wc['observed_gap']:+.4f}**")
        A(f"- null distribution of that same statistic: median "
          f"{wc['null_median']:+.4f}, p90 {wc['null_p90']:+.4f}, "
          f"p95 {wc['null_p95']:+.4f}, p99 {wc['null_p99']:+.4f}")
        A(f"- **P(null gap ≥ observed) = {wc['p_value']:.3f}** "
          f"({wc['trials']:,} trials)")
        A("")
        if wc["p_value"] > 0.05:
            A("> The headline improvement is **not distinguishable** from what the "
              "selection rule manufactures out of noise alone.")
        else:
            A("> The headline improvement exceeds what selection noise alone "
              "produces.")
        A("")

    if arms:
        A("## 5. Published arm comparison, with the noise attached")
        A("")
        A("| arm | run | nodes | best node | best mean | n evals | SE |")
        A("|---|---|---|---|---|---|---|")
        for a in arms:
            A(f"| {a['arm']} | `{a['run']}` | {a['n_nodes']} | {a['best_node']} "
              f"| {a['best_mean']:.4f} | {a['best_n']} | ±{a['se_best']:.4f} |")
        A("")
        if len(arms) >= 2:
            best = max(arms, key=lambda a: a["best_mean"])
            worst = min(arms, key=lambda a: a["best_mean"])
            gap = best["best_mean"] - worst["best_mean"]
            se = math.hypot(best["se_best"], worst["se_best"])
            A(f"- widest gap: **{best['arm']} − {worst['arm']} = {gap:+.4f}** "
              f"± {se:.4f} → **{gap / se:.1f}σ** before any winner's-curse "
              f"correction, and each arm's *best* node is itself a max over "
              f"~30 nodes.")
        A("")
        A("> These are best-node scores, each already selected as a maximum. The "
          "winner's-curse section above applies to every one of them, so the "
          "true arm gap is smaller than the nominal one.")
        A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="+", required=True, type=Path,
                    help="eval dirs of the SAME agent on the SAME cases")
    ap.add_argument("--run", type=Path, default=None, help="a search run to analyse")
    ap.add_argument("--arms", default=None, help="name=path,name=path")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    runs = [r for r in (_load_eval(d) for d in args.repeats) if r]
    if len(runs) < 2:
        raise SystemExit("need >=2 readable repeat eval dirs")
    dec = decompose(runs)

    disc = wc = None
    if args.run:
        disc = discriminability(args.run, dec)
        wc = winners_curse(args.run, dec)

    arms = None
    if args.arms:
        pairs = []
        for item in args.arms.split(","):
            name, _, path = item.partition("=")
            pairs.append((name, Path(path)))
        arms = arm_compare(pairs, dec)

    text = report(dec, disc=disc, wc=wc, arms=arms,
                  repeat_dirs=[r["dir"] for r in runs])
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(
            {"decompose": dec, "se": se_table(dec), "power": power_table(dec),
             "discriminability": disc, "winners_curse": wc, "arms": arms},
            indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render EDIT_MEMORY.md — a worked example of the edit memory's artefacts
assembled from ONE finished run. Read-only: nothing under the run directory
is written.

Usage:
    PYTHONPATH=. python3 study/render_edit_memory_example.py <run_dir> [--node N] [--out EDIT_MEMORY.md]

Without --node the run's own best round (from run_summary.md) is used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meta_agent.edit_memory_render import _load_records, build_ledger, judge_ledger_lines  # noqa: E402


def _read(p: Path, default: str = "") -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return default


def _json(p: Path) -> str:
    try:
        return json.dumps(json.loads(_read(p, "{}")), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return "(unreadable)"


def best_round(run: Path) -> int | None:
    m = re.search(r"- Round: \*\*(\d+)\*\*", _read(run / "run_summary.md"))
    return int(m.group(1)) if m else None


def section(title: str, body: str, fence: str | None = None) -> str:
    body = body.rstrip("\n")
    if fence is not None:
        body = f"```{fence}\n{body}\n```"
    return f"## {title}\n\n{body}\n"


def belief_excerpt(doc: str, n_sections: int = 3) -> tuple[str, list[tuple[str, str]]]:
    """The summary + the first n belief sections, and every slug's track line."""
    parts = re.split(r"(?m)^(?=### belief:)", doc)
    head = parts[0].rstrip()
    sections = parts[1:]
    tracks = []
    for s in sections:
        slug = re.match(r"### belief:([a-z0-9][a-z0-9_-]*)", s)
        tr = re.search(r"(?m)^- track: (.*)$", s)
        tracks.append((slug.group(1) if slug else "?", tr.group(1) if tr else "(none)"))
    shown = "\n".join(s.rstrip() for s in sections[:n_sections])
    more = len(sections) - n_sections
    tail = f"\n\n(+{more} more belief section(s) not shown)" if more > 0 else ""
    return head + "\n\n" + shown + tail, tracks


def report_excerpt(run: Path, max_lines: int = 80) -> str:
    prompts = sorted((run / "edit_memory_beliefs_prompts").glob("update_*.txt"))
    if not prompts:
        return "(no belief update prompt found)"
    text = _read(prompts[-1])
    m = re.search(r"(?ms)^## Calibration report\n(.*?)(?=^## |\Z)", text)
    body = (m.group(0) if m else "(no calibration report in the last prompt)").rstrip()
    lines = body.split("\n")
    if len(lines) > max_lines:
        body = "\n".join(lines[:max_lines]) + f"\n… (+{len(lines) - max_lines} lines)"
    return f"from `{prompts[-1].relative_to(run)}`:\n\n{body}"


def steering_excerpt(rd: Path) -> str:
    text = _read(rd / "verbose" / "editor_attempt_1_user.txt")
    if not text:
        return "(no verbose editor prompt for this round — run with `verbose: true`)"
    out = []
    steer, _, rest = text.partition("## Belief document")
    steer = steer.split("## Steering context", 1)[-1] if "## Steering context" in steer else steer
    out.append(steer.rstrip() + "\n\n## Belief document\n(the document itself — see §4 above)")
    prop = re.search(r"(?ms)^## Planning-pass proposal.*?(?=^## |\Z)", rest)
    if prop:
        out.append(prop.group(0).rstrip())
    nodes = re.findall(r"(?m)^### Retrieved node (\d+) \(([^)]*)\)", rest)
    if nodes:
        out.append("## Retrieved records and implementations (headers only)\n"
                   + "\n".join(f"- node {n} ({why})" for n, why in nodes))
    return "\n\n".join(out)


def artefact_map(run: Path, rd: Path) -> str:
    root = sorted(p.name + ("/" if p.is_dir() else "") for p in run.iterdir()
                  if not p.name.startswith("round_") and p.name != "snapshots")
    rnd = sorted(p.name + ("/" if p.is_dir() else "") for p in rd.iterdir())
    return ("run root: " + ", ".join(root) + "\n\n"
            f"{rd.name}/: " + ", ".join(rnd))


def render(run: Path, node: int | None) -> str:
    snap = yaml.safe_load(_read(run / "config.snapshot.yaml", "{}")) or {}
    em = ((snap.get("edit_memory") or {}).get("config") or {})
    beliefs = em.get("beliefs") or {}
    label = em.get("strategy_label")
    records = _load_records(run)
    n_judged = sum(1 for r in records.values() if r.get("effect_by_edit"))
    node = node if node is not None else (best_round(run) or 0)
    rd = run / f"round_{node:03d}"

    mode = (f"`strategy_label: {label}`" if label else
            "delta-labelled, pre-v7 code (no `strategy_label` key; analysis v6 — "
            "records carry implementation verdicts but no `effect` lines)")
    parts = [
        "# EDIT_MEMORY.md — a worked example from one run\n",
        f"Rendered by `study/render_edit_memory_example.py` from `{run.name}` "
        f"(node {node}). Mode: {mode}. Records with judge effect lines: "
        f"{n_judged} of {len(records)}. Judge lines (`effect`, `regressions`, "
        "`targets`) appear only in judge-mode runs — re-run this script on the "
        "first judge-mode run to refresh the example. The layouts themselves are "
        "specified in `EDIT_MEMORY_SPEC.md`.\n",
    ]
    summ = _read(run / "run_summary.md")
    best = re.search(r"(?ms)^## Best round.*?(?=^## Top|\Z)", summ)
    header = (best.group(0).rstrip() if best else "(no run_summary.md)")
    models = {k: (snap.get(k) or {}).get("model") or ((snap.get(k) or {}).get("config") or {}).get("model")
              for k in ("task_agent", "editor", "edit_memory")}
    keys = {k: em.get(k) for k in ("steering_mode", "strategy_label", "analysis_min_own_evals",
                                   "judge_min_evidence", "min_shared", "verdict_threshold",
                                   "max_strategies", "max_subedits")}
    header += ("\n\nmodels: " + ", ".join(f"{k}={v}" for k, v in models.items())
               + "\nedit_memory keys: " + ", ".join(f"{k}={v}" for k, v in keys.items())
               + "\nbeliefs keys: " + ", ".join(f"{k}={v}" for k, v in beliefs.items()))
    parts.append(section("§0 Run", header))
    parts.append(section(f"§1 The node record — `{rd.name}/edit_memory.md`",
                         _read(rd / "edit_memory.md", "(missing)"), fence="markdown"))
    parts.append(section(f"§2 Its pre-registered prediction — `{rd.name}/belief_prediction.json`",
                         _json(rd / "belief_prediction.json"), fence="json"))
    parts.append(section(f"§3 The planning pass's prediction — `{rd.name}/edit_prediction.json`",
                         _json(rd / "edit_prediction.json"), fence="json"))
    doc = _read(run / "edit_memory_beliefs.md")
    excerpt, tracks = belief_excerpt(doc) if doc else ("(no belief document)", [])
    track_tbl = "\n".join(f"| `{s}` | {t} |" for s, t in tracks)
    parts.append(section("§4 The belief document at the end of the run (summary + first sections)",
                         excerpt, fence="markdown"))
    parts.append(section("§4b Every belief's `- track:` line",
                         "| belief | track |\n|---|---|\n" + (track_tbl or "| (none) | |")))
    parts.append(section("§5 The calibration report the maintainer last saw", report_excerpt(run),
                         fence="markdown"))
    parts.append(section(f"§6 What the editor saw for this expand — `{rd.name}/verbose/editor_attempt_1_user.txt`",
                         steering_excerpt(rd), fence="markdown"))
    parts.append(section(f"§7 What retrieval showed it — `{rd.name}/retrieval_manifest.json`",
                         _json(rd / "retrieval_manifest.json"), fence="json"))
    try:
        registry = json.loads(_read(run / "edit_memory_registry.json", "{}"))
        ledger = build_ledger(registry, records,
                              threshold=float(em.get("verdict_threshold", 0.02)),
                              min_shared=int(em.get("min_shared", 8)))
        rows = judge_ledger_lines(ledger)
    except Exception as exc:  # noqa: BLE001
        rows = [f"(ledger unavailable: {exc!r})"]
    parts.append(section("§8 Per-strategy outcomes (rendered live from the records)",
                         "\n".join(rows)))
    parts.append(section("§9 Artefact map", artefact_map(run, rd)))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--node", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write here (default: stdout)")
    args = ap.parse_args()
    text = render(args.run_dir, args.node)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(text):,} chars)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retrofit edit-memory v2 records for ANY finished run — project-agnostic.

For every node of a run that has ``round_NNN/edit_memory.md``, generates under
``<run>/edit_memory_v2/`` (originals are never touched):

  node_NNN.md          original record + "## Usage (retrofit)" +
                       "## Analysis (LLM, retrofit)"
  node_NNN_usage.json  deterministic sidecar (surface, tool-call counts, raw
                       mutable_log events from the surviving trace)
  prompts/node_NNN.txt exact LLM prompt (provenance)
  summary.md           one row per node

Nothing here is task-specific:
  - the mutable surface comes from the framework contract
    (``meta_agent.edit_diff`` / ``editor_validators``);
  - trace parsing uses the platform event kinds (``tool_call``/``mutable_log``);
  - per-case failed checks come from the run's own ``per_check_recipe``
    (``edit_memory_registry.json``, discovered by the live setup pass) via
    ``meta_agent.edit_outcome.extract_checks``;
  - model / base_url / effort / project name default to the run's
    ``config.snapshot.yaml``.

Usage:
  python3 retrofit_edit_memory.py --run runs/<run_dir>            # full
  python3 retrofit_edit_memory.py --run runs/<run_dir> --nodes 2,8 --out /tmp/x
  python3 retrofit_edit_memory.py --run runs/<run_dir> --no-llm   # counts only

CAVEAT (retrofit-only): ``trace.jsonl`` is truncated at the start of every eval
batch, so counts cover each node's LAST surviving batch — lower bounds. During
a live HGM run the same capture must instead happen at every
``refresh_outcomes`` call and accumulate across batches (see the edit-memory
v2 plan); this script exists for post-hoc analysis of already-finished runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from meta_agent.edit_diff import changed_mutable_files, diff_mutable_files  # noqa: E402
from meta_agent.edit_outcome import extract_checks  # noqa: E402
from platform_core.llm_wrapper import call_llm  # noqa: E402

MAX_EVENT_LINES = 120       # mutable_log lines shown to the LLM
MAX_CASES = 12              # ground-truth case excerpts shown to the LLM
DIFF_CAP = 6000

LABEL_RE = re.compile(r"^(?:label\s*=\s*)?f?['\"]([^'\"]+)['\"]")
LABEL_KW_RE = re.compile(r"label\s*=\s*f?['\"]([^'\"]+)['\"]")
NAME_KW_RE = re.compile(r"(?<![\w_])name\s*=\s*f?['\"]([^'\"]+)['\"]")
ALIAS_IMPORT_RE = re.compile(
    r"from\s+platform_core\.trace\s+import\s+log(?:\s+as\s+(\w+))?")


# ------------------------------------------------------------- run config --
def load_run_config(run: Path) -> dict:
    """Model/effort/base_url for the analysis call and the project name,
    from the run's own config snapshot (edit_memory block first, then
    task_agent). All overridable from the CLI."""
    cfg: dict = {"project": "unknown-project", "model": None,
                 "base_url": None, "reasoning_effort": "medium"}
    snap = run / "config.snapshot.yaml"
    if snap.exists():
        try:
            import yaml
            y = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
            cfg["project"] = y.get("project") or cfg["project"]
            em = ((y.get("edit_memory") or {}).get("config") or {})
            ta = y.get("task_agent") or {}
            cfg["model"] = em.get("model") or ta.get("model")
            cfg["base_url"] = em.get("base_url") or ta.get("base_url")
            cfg["reasoning_effort"] = (em.get("reasoning_effort")
                                       or cfg["reasoning_effort"])
        except Exception as exc:  # noqa: BLE001
            print(f"[config] could not parse {snap}: {exc!r}", flush=True)
    return cfg


def load_recipe(run: Path) -> tuple[str | None, str | None]:
    """The run's per-case failed-check recipe, as discovered by the live
    edit-memory setup pass. (None, None) => no per-check data available."""
    reg = run / "edit_memory_registry.json"
    try:
        r = (json.loads(reg.read_text(encoding="utf-8"))
             .get("per_check_recipe") or {})
        return r.get("mode"), r.get("path")
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------- surface --
def _tool_stems(root: Path) -> set[str]:
    d = root / "task_agent" / "mutable_tools"
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.py") if p.name != "__init__.py"}


def _schema_names(root: Path) -> set[str]:
    p = root / "task_agent" / "tools_schema.json"
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    names: set[str] = set()
    for t in js if isinstance(js, list) else []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        n = fn.get("name")
        if n:
            names.add(n)
    return names


def _log_call_re(file_text: str) -> re.Pattern:
    """Regex matching this file's trace-log call sites, whatever the editor
    named the import (``log``, ``trace_log``, ``trace.log`` ...)."""
    names = {"log", "trace.log"}
    for m in ALIAS_IMPORT_RE.finditer(file_text):
        names.add(m.group(1) or "log")
    alt = "|".join(re.escape(n) for n in sorted(names))
    return re.compile(r"(?<![\w.])(?:%s)\(\s*(.{0,300})" % alt, re.S)


def _log_pairs(text: str, alias_text: str | None = None) -> set[tuple[str, str | None]]:
    """(label, name) pairs at trace-log call sites in ``text``. ``alias_text``
    (default: ``text``) is where import aliases are looked up — pass the full
    file when scanning only its added lines."""
    pairs: set[tuple[str, str | None]] = set()
    for m in _log_call_re(alias_text or text).finditer(text):
        window = m.group(1)
        lm = LABEL_RE.match(window.strip()) or LABEL_KW_RE.search(window)
        if not lm:
            continue
        nm = NAME_KW_RE.search(window)
        pairs.add((lm.group(1), nm.group(1) if nm else None))
    return pairs


def added_surface(parent_dir: Path, child_dir: Path) -> dict:
    tools_added = sorted((_tool_stems(child_dir) - _tool_stems(parent_dir))
                         | (_schema_names(child_dir) - _schema_names(parent_dir)))
    tools_removed = sorted((_tool_stems(parent_dir) - _tool_stems(child_dir))
                           | (_schema_names(parent_dir) - _schema_names(child_dir)))
    parent_pairs: set[tuple[str, str | None]] = set()
    added_pairs: set[tuple[str, str | None]] = set()
    for rel in changed_mutable_files(parent_dir, child_dir):
        p_path = parent_dir / "task_agent" / rel
        c_path = child_dir / "task_agent" / rel
        p_text = p_path.read_text(encoding="utf-8", errors="replace") if p_path.exists() else ""
        c_text = c_path.read_text(encoding="utf-8", errors="replace") if c_path.exists() else ""
        p_lines = set(p_text.splitlines())
        added_text = "\n".join(l for l in c_text.splitlines() if l not in p_lines)
        parent_pairs |= _log_pairs(p_text)
        added_pairs |= _log_pairs(added_text, alias_text=c_text)
    new_pairs = sorted(p for p in added_pairs if p not in parent_pairs)
    inherited = sorted(added_pairs & parent_pairs)
    return {
        "tools": tools_added,
        "removed_tools": tools_removed,
        "labels": [{"label": l, "name": n} for l, n in new_pairs],
        "inherited_labels": [{"label": l, "name": n} for l, n in inherited],
    }


# ------------------------------------------------------------------ trace --
def parse_trace(trace_path: Path):
    tools: dict[str, dict] = defaultdict(lambda: {"calls": 0, "case_ids": set()})
    events: list[dict] = []
    case_ids: set[str] = set()
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        return {}, [], set(), None
    raw = trace_path.read_bytes()
    sig = hashlib.blake2b(raw, digest_size=8).hexdigest()
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, payload = ev.get("kind"), ev.get("payload", {})
        cid = payload.get("case_id")
        if cid is not None:
            case_ids.add(str(cid))
        if kind == "tool_call":
            name = payload.get("name")
            if name:
                tools[name]["calls"] += 1
                if cid is not None:
                    tools[name]["case_ids"].add(str(cid))
        elif kind == "mutable_log":
            events.append(payload)
    return dict(tools), events, case_ids, sig


def _dyn_eq(pattern: str | None, value) -> bool:
    """Exact match, except f-string placeholders in extracted patterns
    (``tool_{tc.name}``) match any runtime substring."""
    if pattern is None:
        return True
    if "{" in pattern:
        parts = re.split(r"\{[^{}]*\}", pattern)
        rx = ".*".join(re.escape(p) for p in parts)
        return re.fullmatch(rx, str(value or "")) is not None
    return pattern == value


def _match_events(events: list[dict], label: str, name: str | None) -> list[dict]:
    return [e for e in events
            if _dyn_eq(label, e.get("label")) and _dyn_eq(name, e.get("name"))]


def usage_lines(surface: dict, tools: dict, events: list[dict], case_ids: set,
                sig: str | None) -> list[str]:
    if sig is None:
        return ["- **usage**: no trace data survived for this node "
                "(crashed batch or missing trace.jsonl)"]
    lines = []
    basis = f"last surviving eval batch only, {len(case_ids)} cases"
    tool_bits = []
    for t in surface["tools"]:
        info = tools.get(t)
        if info:
            tool_bits.append(f"`{t}` {info['calls']} calls / {len(info['case_ids'])} cases")
        else:
            tool_bits.append(f"`{t}` **0 calls**")
    if tool_bits:
        lines.append(f"- **new tools** ({basis}): " + " · ".join(tool_bits))
    for lab in surface["labels"]:
        evs = _match_events(events, lab["label"], lab["name"])
        key = f"{lab['label']}/{lab['name']}" if lab["name"] else lab["label"]
        if not evs:
            lines.append(f"- **new log point `{key}`**: **never fired** ({basis})")
            continue
        verdicts = defaultdict(int)
        for e in evs:
            verdicts[str(e.get("verdict", "-"))] += 1
        vs = " / ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
        lines.append(f"- **new log point `{key}`**: fired {len(evs)}x ({vs}) ({basis})")
    if not surface["tools"] and not surface["labels"]:
        lines.append(f"- **usage**: edit added no new tools or instrumented log points "
                     f"(prompt/logic-only change); {basis}")
    return lines


# ------------------------------------------------------------------- LLM ---
ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_edit_analysis",
        "description": "Submit the grounded analysis of this node's edit.",
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "component": {"type": "string",
                                          "description": "tool name or label/name of the verifier/checker"},
                            "activated": {"type": "string",
                                          "description": "how often it actually ran, per the evidence"},
                            "verdict_behavior": {"type": "string",
                                                 "description": "pass/fail pattern when it ran"},
                            "agreement": {"type": "string",
                                          "description": "when it passed, did the agent's output actually satisfy that component per the scorer? cite case ids"},
                            "assessment": {"type": "string",
                                           "enum": ["effective", "partially-effective",
                                                    "ineffective", "harmful", "unmeasured"]},
                        },
                        "required": ["component", "activated", "verdict_behavior",
                                     "agreement", "assessment"],
                    },
                },
                "constraint_effect": {
                    "type": "string",
                    "description": "which specific benchmark checks/constraints this edit helped or hurt, grounded in the checks-moved data and case evidence",
                },
                "overall": {"type": "string",
                            "enum": ["helped", "hurt", "neutral", "mixed", "unmeasured"]},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["components", "constraint_effect", "overall", "confidence"],
        },
    },
}

SYSTEM_TMPL = """You analyze one edit made by a self-improving agent for the "{project}" task.
The edit added/changed code (tools, verifiers, constraint checkers, prompts) in the agent.
You receive: the edit's description, the code diff, deterministic usage counts,
raw runtime verifier logs (mutable_log events), per-case ground truth from the
benchmark scorer, and the measured score delta vs the parent agent.

Answer two questions, grounded ONLY in the supplied evidence:
1. For each verifier/tool/constraint-checker component the edit added: when it
   activated and reported "pass", did the agent actually produce a correct
   output for that component according to the scorer? (A verifier that says
   pass while the scorer fails the targeted check is a missed detection; one
   that says fail on outputs the scorer accepts is over-strict.)
2. Did the edit help the specific constraints/checks it targeted, and overall?

Rules: cite case ids for every agreement claim; never invent numbers; if the
evidence is too thin for a claim, say so and use assessment "unmeasured".
The runtime trace covers only the final evaluation batch — treat counts as
lower bounds and say "in the observed batch" rather than overgeneralizing."""


def _failed_messages(obj, depth: int = 0, cap: int = 5) -> list[str]:
    """Schema-agnostic: collect message strings from dicts shaped like
    ``{"passed": False, "message": ...}`` anywhere in the case details."""
    out: list[str] = []
    if depth > 4 or len(out) >= cap:
        return out
    if isinstance(obj, dict):
        if obj.get("passed") is False and isinstance(obj.get("message"), str):
            out.append(obj["message"][:150])
        for v in obj.values():
            if len(out) >= cap:
                break
            out.extend(_failed_messages(v, depth + 1, cap - len(out)))
    elif isinstance(obj, list):
        for v in obj[:20]:
            if len(out) >= cap:
                break
            out.extend(_failed_messages(v, depth + 1, cap - len(out)))
    return out


def _case_excerpt(round_dir: Path, cid: str, recipe: tuple) -> dict | None:
    p = round_dir / "logs" / f"case_{cid}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    checks = extract_checks(d, recipe[0], recipe[1])
    out = {"case_id": cid, "passed": d.get("passed"), "score": d.get("score")}
    if d.get("error"):
        out["error"] = str(d["error"])[:200]
    if checks is not None:
        out["failed_checks"] = sorted(checks)
    msgs = _failed_messages(d.get("details"))
    if msgs:
        out["failed_check_messages"] = msgs
    return out


def build_prompt(node: int, record_text: str, surface: dict, tools: dict,
                 events: list[dict], u_lines: list[str], diff: str,
                 round_dir: Path, parent_dir: Path, recipe: tuple) -> str:
    # stratified event sample: up to 3 per (label, name, verdict)
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        by_group[(e.get("label"), e.get("name"), str(e.get("verdict", "-")))].append(e)
    sampled: list[dict] = []
    for _, evs in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        sampled.extend(evs[:3])
    sampled = sampled[:MAX_EVENT_LINES]
    ev_lines = [json.dumps(e, ensure_ascii=False)[:300] for e in sampled]

    cids: list[str] = []
    for e in sampled:
        c = e.get("case_id")
        if c is not None and str(c) not in cids:
            cids.append(str(c))
    cases = []
    for cid in cids[:MAX_CASES]:
        child_c = _case_excerpt(round_dir, cid, recipe)
        if child_c is None:
            continue
        parent_c = _case_excerpt(parent_dir, cid, recipe)
        entry: dict = {"child": child_c}
        if parent_c is not None:
            entry["parent_same_case"] = {"passed": parent_c.get("passed"),
                                         "failed_checks": parent_c.get("failed_checks")}
        cases.append(entry)

    parts = [
        f"# Node {node} — edit record (description + measured outcome)",
        record_text.strip(),
        "\n# Deterministic usage (retrofit)",
        "\n".join(u_lines),
        "\n# Surface detected (what this edit added vs parent)",
        json.dumps(surface, indent=1),
        f"\n# Runtime mutable_log events (sampled {len(sampled)} of {len(events)}; final batch only)",
        "\n".join(ev_lines) if ev_lines else "(none)",
        f"\n# Ground truth for cases the components touched ({len(cases)} cases; "
        "parent_same_case = same case on the PARENT agent, for before/after comparison)",
        json.dumps(cases, indent=1) if cases else "(no overlapping case files)",
        "\n# Code diff vs parent (mutable surface, truncated)",
        diff or "(no diff)",
    ]
    return "\n".join(parts)


def render_analysis(a: dict | None, ran_llm: bool) -> str:
    if not ran_llm:
        return "## Analysis (LLM, retrofit)\n\n*(skipped — --no-llm)*\n"
    if not a:
        return "## Analysis (LLM, retrofit)\n\n*(LLM analysis unavailable — call failed)*\n"
    lines = ["## Analysis (LLM, retrofit)", ""]
    for c in a.get("components", []):
        lines.append(f"- **`{c.get('component')}`** — activated: {c.get('activated')}")
        lines.append(f"  - verdicts: {c.get('verdict_behavior')}")
        lines.append(f"  - pass-vs-scorer agreement: {c.get('agreement')}")
        lines.append(f"  - assessment: **{c.get('assessment')}**")
    lines.append(f"- **constraint effect**: {a.get('constraint_effect')}")
    lines.append(f"- **overall**: **{a.get('overall')}** · confidence: {a.get('confidence')}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main ---
class Retrofit:
    def __init__(self, run: Path, out: Path, cfg: dict, recipe: tuple,
                 use_llm: bool):
        self.run, self.out, self.cfg, self.recipe = run, out, cfg, recipe
        self.use_llm = use_llm

    def analyze(self, node: int, prompt: str) -> dict | None:
        try:
            resp = call_llm(
                messages=[{"role": "system",
                           "content": SYSTEM_TMPL.format(project=self.cfg["project"])},
                          {"role": "user", "content": prompt}],
                tools=[ANALYSIS_TOOL], model=self.cfg["model"],
                base_url=self.cfg["base_url"],
                reasoning_effort=self.cfg["reasoning_effort"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[node {node}] llm call failed: {exc!r}", flush=True)
            return None
        for tc in (getattr(resp, "tool_calls", None) or []):
            if getattr(tc, "name", None) == "submit_edit_analysis":
                args = tc.arguments
                return args if isinstance(args, dict) else None
        m = re.search(r"```json\s*(\{.*?\})\s*```",
                      getattr(resp, "content", "") or "", re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        print(f"[node {node}] no structured output", flush=True)
        return None

    def process_node(self, round_dir: Path) -> dict | None:
        rec_path = round_dir / "edit_memory.md"
        if not rec_path.exists():
            return None
        text = rec_path.read_text(encoding="utf-8")
        fm_node = re.search(r"^node:\s*(\d+)", text, re.M)
        fm_parent = re.search(r"^parent:\s*(\d+)", text, re.M)
        if not fm_node or not fm_parent:
            return None
        node, parent = int(fm_node.group(1)), int(fm_parent.group(1))
        parent_dir = self.run / f"round_{parent:03d}"

        surface = added_surface(parent_dir, round_dir)
        tools, events, case_ids, sig = parse_trace(round_dir / "logs" / "trace.jsonl")
        u_lines = usage_lines(surface, tools, events, case_ids, sig)
        diff = diff_mutable_files(parent_dir, round_dir, char_cap=DIFF_CAP) or ""

        sidecar = {
            "version": 1, "node": node, "parent": parent,
            "note": "retrofit: counts cover only the last surviving eval batch",
            "surface": surface,
            "batches": [sig] if sig else [],
            "batch_case_ids": sorted(case_ids, key=str),
            "tools": {k: {"calls": v["calls"], "case_ids": sorted(v["case_ids"])}
                      for k, v in sorted(tools.items())},
            "events": events,
        }
        (self.out / f"node_{node:03d}_usage.json").write_text(
            json.dumps(sidecar, indent=1, ensure_ascii=False), encoding="utf-8")

        prompt = build_prompt(node, text, surface, tools, events, u_lines, diff,
                              round_dir, parent_dir, self.recipe)
        (self.out / "prompts" / f"node_{node:03d}.txt").write_text(
            f"### SYSTEM\n{SYSTEM_TMPL.format(project=self.cfg['project'])}"
            f"\n\n### USER\n{prompt}", encoding="utf-8")

        analysis = self.analyze(node, prompt) if self.use_llm else None

        out_md = (text.rstrip() + "\n\n## Usage (retrofit)\n\n"
                  + "\n".join(u_lines) + "\n\n"
                  + render_analysis(analysis, self.use_llm))
        (self.out / f"node_{node:03d}.md").write_text(out_md, encoding="utf-8")

        delta_m = re.search(r"delta shared\*\*:\s*([+\-0-9.]+)", text)
        print(f"[node {node:03d}] done  surface_tools={len(surface['tools'])} "
              f"labels={len(surface['labels'])} events={len(events)} "
              f"analysis={'ok' if analysis else ('skipped' if not self.use_llm else 'FAILED')}",
              flush=True)
        return {
            "node": node, "parent": parent,
            "delta": delta_m.group(1) if delta_m else "n/a",
            "tools": surface["tools"],
            "labels": [f"{l['label']}/{l['name']}" if l["name"] else l["label"]
                       for l in surface["labels"]],
            "n_events": len(events),
            "overall": (analysis or {}).get("overall",
                                            "-" if not self.use_llm else "call-failed"),
            "confidence": (analysis or {}).get("confidence", "-"),
            "assessments": [f"{c.get('component')}: {c.get('assessment')}"
                            for c in (analysis or {}).get("components", [])],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path,
                    help="run directory (contains round_*/edit_memory.md)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <run>/edit_memory_v2)")
    ap.add_argument("--nodes", default=None,
                    help="comma-separated node ids to process (default: all)")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic usage layer only, skip the analysis call")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=None, help="override config.snapshot.yaml")
    ap.add_argument("--base-url", default=None, help="override config.snapshot.yaml")
    ap.add_argument("--effort", default=None, help="override reasoning effort")
    args = ap.parse_args()

    run = args.run.resolve()
    out = (args.out or run / "edit_memory_v2").resolve()
    cfg = load_run_config(run)
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.effort:
        cfg["reasoning_effort"] = args.effort
    recipe = load_recipe(run)
    print(f"run={run.name} project={cfg['project']} model={cfg['model']} "
          f"recipe={recipe} llm={not args.no_llm}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    wanted = ({int(n) for n in args.nodes.split(",")} if args.nodes else None)
    round_dirs = []
    for d in sorted(run.glob("round_*")):
        if not (d / "edit_memory.md").exists():
            continue
        m = re.search(r"round_(\d+)$", d.name)
        if wanted is not None and (not m or int(m.group(1)) not in wanted):
            continue
        round_dirs.append(d)
    print(f"{len(round_dirs)} nodes to process", flush=True)

    rf = Retrofit(run, out, cfg, recipe, use_llm=not args.no_llm)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for r in pool.map(rf.process_node, round_dirs) if r]
    rows.sort(key=lambda r: r["node"])

    lines = [f"# Edit memory v2 retrofit — {run.name}", "",
             "| node | Δ shared | new tools | new log points | events (last batch) | overall (LLM) | conf | component assessments |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {node} | {delta} | {tools} | {labels} | {n_events} | {overall} | {confidence} | {asmt} |".format(
            node=r["node"], delta=r["delta"],
            tools=", ".join(r["tools"]) or "—",
            labels=", ".join(r["labels"]) or "—",
            n_events=r["n_events"], overall=r["overall"],
            confidence=r["confidence"],
            asmt="; ".join(r["assessments"]) or "—"))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} records + summary.md to {out}", flush=True)


if __name__ == "__main__":
    main()

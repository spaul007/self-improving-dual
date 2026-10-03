"""Per-case DOSSIERS: a small, deterministic evidence file per task run, and a ranked node index.

Why: the meta-agent diagnosed at the category level (EXP-033: every failure was
``near_miss + verify_false_pass``) and never opened the MB-sized transcripts it could read.
A dossier puts the decisive facts of one run in ~3-4 KB:

  header      reward, f2p/p2p counts, failure classes, each role's wall / stop reason
  coverage    the task's requirement checklist (adapter/requirements/<task>.json, extracted
              once from instruction.md -- the statement the agent itself reads) against what
              PATCH listed (``requirements``), what VERIFY enumerated (``behaviours``) and
              tested (``behaviours_tested``), and the patch's added lines
  excerpts    VERIFY's test commands (exec log), PATCH's last summary, VERIFY's last report,
              files touched
  contrast    other nodes' runs of the SAME task with a DIFFERENT reward, and what differed

Hidden-test isolation: built only from the agent's own artifacts + the numeric grade. Never
reads ``verifier/``. Matching is heuristic (normalised token overlap, literals weighted) and
is marked as such; it is a pointer to evidence, not a verdict.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

def _clip(text: str, n: int) -> str:
    """ONE-line clip (render._clip inserts a multi-line elision marker that breaks tables)."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."
from .trial import failure_classes

HERE = Path(__file__).resolve().parent
REQ_DIR = HERE / "requirements"
MAX_ROWS = 18            # coverage rows shown (the rest summarised)
MAX_CONTRASTS = 3
_STOP = set("""a an the and or of to in on for with by is are be when that this it its as at from
into must should shall will can may not no each any all only than then so if else also both
either via per using use used new existing given return returns returned value values""".split())
_TOK = re.compile(r"[a-z0-9_][a-z0-9_.\-]*")
_TEST_CMD = re.compile(r"\b(test|pytest|jest|vitest|mocha|go test|cargo test|deno test|stestr|tox|"
                       r"npm|pnpm|yarn|bun|node|python3?|tsx|ts-node|go run|make)\b", re.I)


# ----------------------------------------------------------------------------- matching
def _tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall((text or "").lower()) if len(t) > 2 and t not in _STOP}


def _lits(req: dict) -> list[str]:
    return [l.lower() for l in req.get("literals") or [] if len(l.strip()) >= 3]


def match(req: dict, candidates: list[str]) -> str:
    """'Y' (matched), '?' (partial), '-' (absent) for one requirement vs a list of texts."""
    if not candidates:
        return "-"
    key = _tokens(req.get("text", ""))
    lits = _lits(req)
    # Stricter when the requirement carries exact literals (names, strings, formats): a
    # compound VERIFY behaviour mentioning the topic is not coverage of the exact literal.
    best = "-"
    for c in candidates:
        cl = (c or "").lower()
        tok = len(key & _tokens(cl)) / len(key) if key else 0.0
        if lits:
            lit_hit = sum(1 for l in lits if l in cl) / len(lits)
            mark = "Y" if (lit_hit >= 0.5 and tok >= 0.4) else ("?" if (lit_hit > 0 or tok >= 0.5) else "-")
        else:
            mark = "Y" if tok >= 0.6 else ("?" if tok >= 0.4 else "-")
        if mark == "Y":
            return "Y"
        if mark == "?":
            best = "?"
    return best


def match_diff(req: dict, added: str) -> str:
    lits = _lits(req)
    if not added:
        return "-"
    if lits:
        hit = sum(1 for l in lits if l in added)
        return "Y" if hit == len(lits) else ("?" if hit else "-")
    key = _tokens(req.get("text", ""))
    frac = len(key & _tokens(added)) / len(key) if key else 0.0
    return "?" if frac >= 0.5 else "-"


# ----------------------------------------------------------------------------- inputs
def load_requirements(task: str) -> list[dict]:
    try:
        return json.loads((REQ_DIR / f"{task}.json").read_text())["requirements"]
    except (OSError, ValueError, KeyError):
        return []


def _last(dispatches: list[dict], role: str) -> dict:
    return next((d["output"] for d in reversed(dispatches) if d.get("role") == role), {}) or {}


def _latest_nonempty(dispatches: list[dict], role: str, field: str) -> list[str]:
    """The most recent NON-EMPTY list a role reported for ``field``. A run that ends on the
    deadline leaves a synthesized, empty last report; using it would claim 'VERIFY enumerated
    0 behaviours' for a run whose earlier VERIFY rounds enumerated dozens."""
    for d in reversed(dispatches):
        if d.get("role") == role:
            vals = [str(x) for x in (d.get("output") or {}).get(field) or [] if str(x).strip()]
            if vals:
                return vals
    return []


def _ids(ids: list[str], cap: int = 24) -> str:
    return (", ".join(ids[:cap]) + (f" (+{len(ids) - cap} more)" if len(ids) > cap else "")) or "none"


def _added_lines(patch_text: str) -> tuple[str, list[str]]:
    files, added = [], []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                files.append(parts[1])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added).lower(), files


def _verify_test_cmds(trial_dir: Path, limit: int = 5) -> list[str]:
    src = trial_dir / "agent" / "exec_log.jsonl"
    if not src.is_file():
        return []
    rows = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        cmd = r.get("command") or ""
        if r.get("role") == "verify" and r.get("tool") == "Bash" and _TEST_CMD.search(cmd):
            rows.append(f"verify.{r.get('attempt')} rc={r.get('rc')} {r.get('dur_s')}s: "
                        f"{_clip(cmd.replace(chr(10), ' '), 220)}")
    return rows[-limit:]


# ----------------------------------------------------------------------------- build
def build(case_id: str, run_id: str, round_name: str, trial_dir: Optional[Path], o: dict,
          dispatches: list[dict]) -> dict:
    """Machine-readable dossier facts (also the input to contrasts)."""
    reqs = load_requirements(case_id)
    patch = trial_dir / "artifacts" / "model.patch" if trial_dir else None
    added, files = _added_lines(patch.read_text(encoding="utf-8", errors="replace")
                                if patch and patch.is_file() else "")
    lp, lv = _last(dispatches, "patch"), _last(dispatches, "verify")
    p_req = _latest_nonempty(dispatches, "patch", "requirements")
    v_beh = _latest_nonempty(dispatches, "verify", "behaviours")
    v_tst = _latest_nonempty(dispatches, "verify", "behaviours_tested")
    cov = {}
    for r in reqs:
        cov[r["id"]] = {"p": match(r, p_req), "v": match(r, v_beh), "t": match(r, v_tst),
                        "d": match_diff(r, added)}
    return {
        "case": case_id, "run": run_id, "round": round_name,
        "reward": o.get("reward"), "f2p": o.get("f2p"), "p2p": o.get("p2p"),
        "f2p_passed": o.get("f2p_passed"), "f2p_total": o.get("f2p_total"),
        "infra": o.get("infra_class"), "classes": failure_classes(o),
        "roles": [f"{r.get('role')}.{r.get('attempt')} {r.get('wall_sec')}s/{r.get('planned_wall_sec')}s "
                  f"stop={r.get('stop_reason')} edits={r.get('edits')} verdict={r.get('verdict')}"
                  for r in o.get("role_stats") or []],
        "files": files, "coverage": cov,
        "counts": {"reqs": len(reqs), "p_listed": len(p_req), "v_beh": len(v_beh), "v_tested": len(v_tst)},
        "req_text": {r["id"]: r["text"] for r in reqs},
        "patch_summary": _clip(str(lp.get("summary") or ""), 500),
        "verify_report": {"verdict": lv.get("verdict"),
                          "issues": [_clip(str(i), 200) for i in (lv.get("issues") or [])][:6],
                          "test_command": _clip(str(lv.get("test_command") or ""), 200)},
        "verify_cmds": _verify_test_cmds(trial_dir) if trial_dir else [],
    }


def find_contrasts(exp_dir: Path, case_id: str, run_id: str, reward: Any) -> list[dict]:
    """Other rounds' dossiers for the same task whose reward differs."""
    out = []
    for f in sorted(exp_dir.glob(f"round_*/logs/scratch/{case_id}/*/dossier.json")):
        if f.parent.name == run_id:
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("reward") is not None and reward is not None and d["reward"] != reward:
            d["_path"] = f.parent.relative_to(exp_dir).as_posix()
            out.append(d)
    return out[:MAX_CONTRASTS]


def render(d: dict, contrasts: list[dict]) -> str:
    cov, rt = d["coverage"], d["req_text"]
    c = d["counts"]
    untested = [rid for rid, m in cov.items() if m["t"] == "-"]
    not_enum = [rid for rid, m in cov.items() if m["v"] == "-"]
    head = (f"# {d['case']}  ({d['round']}, run {d['run']})\n"
            f"reward={d['reward']} f2p={d.get('f2p_passed')}/{d.get('f2p_total')} p2p={d.get('p2p')} "
            f"classes={d['classes'] or '-'}" + (f" INFRA={d['infra']}" if d.get("infra") else "") + "\n"
            + "\n".join(f"- {r}" for r in (d["roles"] if len(d["roles"]) <= 9
                                              else d["roles"][:1] + ["..."] + d["roles"][-7:])) + "\n")
    lines = [head,
             "## Requirement coverage (heuristic match: Y / ? partial / - absent)",
             f"{c['reqs']} requirements from instruction.md | PATCH listed {c['p_listed']} edge cases | "
             f"VERIFY enumerated {c['v_beh']} behaviours, tested {c['v_tested']}",
             f"NOT enumerated by VERIFY: {_ids(not_enum)}",
             f"NOT in VERIFY's tested list: {_ids(untested)}",
             f"only PARTIALLY matched by VERIFY's tests (?): {_ids([r for r, m in cov.items() if m['t'] == '?'])}",
             "cols: PATCH.req VERIFY.beh VERIFY.tested in-diff"]
    shown = sorted(cov, key=lambda rid: (cov[rid]["t"] == "Y", cov[rid]["v"] == "Y",
                                         int(rid[1:]) if rid[1:].isdigit() else 0))[:MAX_ROWS]
    for rid in shown:
        m = cov[rid]
        lines.append(f"  {rid:<4} {m['p']} {m['v']} {m['t']} {m['d']}  {_clip(rt.get(rid, ''), 110)}")
    if len(cov) > len(shown):
        lines.append(f"  (+{len(cov) - len(shown)} more rows; untested ones are listed first)")
    lines += ["", "## Key excerpts"]
    lines += [f"- {x}" for x in d["verify_cmds"]] or ["- (no VERIFY test commands found)"]
    vr = d["verify_report"]
    lines.append(f"- VERIFY final report: verdict={vr['verdict']} test_command={vr['test_command']!r}")
    for i in vr["issues"]:
        lines.append(f"  - issue: {i}")
    lines.append(f"- PATCH last summary: {d['patch_summary'] or '(none)'}")
    lines.append(f"- files in patch: {', '.join(d['files'][:12]) or '(none)'}")
    if contrasts:
        lines += ["", "## Contrast: same task, other agents, different result"]
        for o in contrasts:
            ocov = o.get("coverage", {})
            gained = [rid for rid in cov if ocov.get(rid, {}).get("t") == "Y" and cov[rid]["t"] != "Y"]
            lost = [rid for rid in cov if cov[rid]["t"] == "Y" and ocov.get(rid, {}).get("t") != "Y"]
            of = set(o.get("files") or [])
            lines.append(f"- {o.get('round')} run {o.get('run')}: reward={o.get('reward')} "
                         f"f2p={o.get('f2p_passed')}/{o.get('f2p_total')}  (another agent's run: {o['_path']})")
            lines.append(f"  tested there but not here: {', '.join(gained) or 'none'}; "
                         f"tested here but not there: {', '.join(lost) or 'none'}")
            lines.append(f"  files only there: {', '.join(sorted(of - set(d['files'])))[:300] or 'none'}; "
                         f"only here: {', '.join(sorted(set(d['files']) - of))[:300] or 'none'}")
    lines.append("\nFull evidence: transcripts/<role>.<attempt>.txt, exec_log.txt, model.patch in this folder.")
    return "\n".join(_clip_line(l) for l in "\n".join(lines).splitlines()) + "\n"


def _clip_line(line: str, cap: int = 400) -> str:
    return line if len(line) <= cap else line[: cap - 3] + "..."


def write(dest: Path, case_id: str, run_id: str, round_name: str, trial_dir: Optional[Path],
          o: dict, dispatches: list[dict]) -> dict:
    """Build + write dossier.json / dossier.md into ``dest``. Never raises into the caller."""
    d = build(case_id, run_id, round_name, trial_dir, o, dispatches)
    exp_dir = dest.parents[4] if len(dest.parents) > 4 else dest.parent
    contrasts = find_contrasts(exp_dir, case_id, run_id, d.get("reward"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "dossier.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    (dest / "dossier.md").write_text(render(d, contrasts), encoding="utf-8")
    return d


# ----------------------------------------------------------------------------- node index
def write_index(round_dir: Path) -> Optional[Path]:
    """``<round>/logs/DOSSIERS.md``: this node's runs, most informative failures first."""
    scratch = round_dir / "logs" / "scratch"
    rows = []
    for f in sorted(scratch.glob("*/*/dossier.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        root = sorted((round_dir.parent / "round_000" / "logs" / "scratch" / d["case"]).glob("*/dossier.json"))
        root_reward = None
        if root and round_dir.name != "round_000":
            try:
                root_reward = json.loads(root[-1].read_text()).get("reward")
            except (OSError, ValueError):
                pass
        cls = d.get("classes") or []
        if d.get("infra"):
            rank, why = 5, f"infra ({d['infra']})"
        elif root_reward is not None and d.get("reward") != root_reward:
            rank, why = 0, f"FLIP vs root ({root_reward} -> {d.get('reward')})"
        elif d.get("reward") == 1:
            rank, why = 6, "solved"
        elif "near_miss" in cls and (d.get("f2p") or 0) >= 0.8:
            rank, why = 1, f"near miss f2p={d.get('f2p_passed')}/{d.get('f2p_total')}"
        elif "role_wall" in cls:
            rank, why = 2, "a role ran out of wall time"
        else:
            rank, why = 3, ", ".join(cls) or "failed"
        untested = [rid for rid, m in (d.get("coverage") or {}).items() if m.get("t") == "-"]
        rel = f.parent.relative_to(round_dir).as_posix()
        rows.append((rank, d["case"], f"- **{d['case']}** reward={d.get('reward')} -- {why}; "
                     f"{len(untested)} stated requirements untested by VERIFY -> {rel}/dossier.md"))
    if not rows:
        return None
    rows.sort()
    out = round_dir / "logs" / "DOSSIERS.md"
    out.write_text("# Dossiers for this agent's evaluated tasks (most informative first)\n"
                   "Open a dossier, then the transcript/patch lines it points to.\n\n"
                   + "\n".join(r[2] for r in rows) + "\n", encoding="utf-8")
    return out

#!/usr/bin/env python3
"""Standalone experiment: let a meta-agent REDESIGN the travel_mas_refactored harness around a weak runtime model.

Thesis under test: the runtime model (Qwen3.5-35B-A3B, ~3B active parameters) is unreliable -- it skips required tool calls, fabricates names and
prices, and cannot compose a long multi-day plan in one generation -- so the harness should let CODE own everything deterministic (all tool calls,
scheduling, dedup, exact names/prices, budget arithmetic, rendering, self-check and repair) and use the LLM only for small, single-purpose,
verifiable subtasks (structured extraction of the request, choosing one item from a code-supplied candidate list).

The meta-agent (local Qwen3.5-122B on node-5) gets the seed's own source as an editable workspace, ONE stored baseline pass of the TRAIN split
(query, plan, failed-check labels and grader messages), a train-only digest, the train-only trace of that pass and the project's check descriptions
-- the same information categories HGM's editor sees. It never sees the gold scorer's source, benchmark data, converted plans or any eval-split
log. Tools: list_cases / show_case / read_file / grep over that corpus and the live workspace, write_file / str_replace_file on the workspace
(guarded edit policy), run_python (offline sandbox on the stored train plans; tool lookups work, LLM calls and file/env access do not) and
evaluate_variant (real end-to-end run on a few TRAIN cases, labeled results like HGM's failure report; a disclosed deviation from HGM's timing).

The final variant is measured on all held-out EVAL cases with fresh runs (default 3 repeats) against the already-run baseline passes on the same
cases: overall composite, paired per-case bootstrap CI, no-plan rate, per-dimension pass rates, per-check deltas, LLM calls per case and local-vLLM
failure rates. Task agent: local Qwen3.5-35B (node-6); meta-agent: local Qwen3.5-122B (node-5); nothing routes to OpenRouter / Gemma.

Standalone: depends only on framework modules (meta_agent.*, platform_core.*) and the stored baseline directory.
Usage: python3 experiment_harness_redesign.py --out-dir redesign_run --seed 42
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402
from meta_agent.llm_failure_health import analyze_trace_file, incidence_rate_pct  # noqa: E402
from platform_core.llm_wrapper import call_llm  # noqa: E402

CONFIG = Path(os.environ["REDESIGN_CONFIG_PATH"]) if os.environ.get("REDESIGN_CONFIG_PATH") else \
    REPO_ROOT / "configs/eval_baseline_travel_mas_refactored_local_vllm.yaml"  # override for a different task-agent config
PROJECT_DIR = REPO_ROOT / "projects/travel_mas_refactored"
DEFAULT_BASELINE = REPO_ROOT / "seed_baseline_travel_mas_refactored_local_5x"
SEMANTICS_FILES = ["error_semantics.json", "harness_error_semantics.json"]
# Meta-agent endpoint; override with REDESIGN_META_MODEL / REDESIGN_META_BASE_URL (must be a LOCAL Qwen endpoint; asserted at start-up).
META_MODEL = os.environ.get("REDESIGN_META_MODEL", "Qwen/Qwen3.5-122B-A10B")
META_BASE_URL = os.environ.get("REDESIGN_META_BASE_URL", "http://gpu-aic-mv-02-st-p5-node-5:8000/v1")
TASK_MODEL = "Qwen/Qwen3.5-35B-A3B"
TOOL_OUTPUT_CAP = 30000
# Some models occasionally leak a pseudo-tool-call as literal text (e.g. DeepSeek via OpenRouter emitting
# "<\uff5cDSML\uff5ctool_calls><\uff5cDSML\uff5cinvoke name=\"...\">...") instead of a real structured tool call.
# Detected so it gets a specific corrective retry instead of burning the generic "no tool calls" nudge budget.
_MALFORMED_TOOLCALL_RE = re.compile(r"invoke\s+name\s*=|tool_calls>", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def assert_local_only(baseline_dir: Path) -> str:
    base = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "")
    assert "node-6" in base and "openrouter" not in base.lower(), f"task agent must use local node-6, got {base!r}"
    assert "qwen" in model.lower() and "gemma" not in model.lower(), f"task agent must be local Qwen, got {model!r}"
    assert "qwen" in os.environ.get("TRAVEL_CONVERT_MODEL", "").lower(), "scorer conversion model must be local Qwen"
    assert not os.environ.get("LLM_PROVIDER_PREFERENCE"), "LLM_PROVIDER_PREFERENCE is an OpenRouter field; must be unset"
    # META-AGENT identity is an explicit experimental variable across this project's harness-redesign runs (local Qwen-122B,
    # local Qwen-27B, Claude via a controlled subagent+CLI, or an external API such as OpenRouter/DeepSeek) -- only Gemma is
    # excluded (this project's known-unreliable TASK backbone, never a legitimate meta-agent choice). The TASK agent lock above
    # (node-6, local Qwen) is the actual experimental control and is never relaxed by this.
    assert "gemma" not in META_MODEL.lower(), f"meta-agent must not be Gemma, got {META_MODEL!r}"
    if "openrouter" in META_BASE_URL.lower():
        log(f"NOTE: meta-agent is EXTERNAL (OpenRouter, {META_MODEL}) -- task agent remains local Qwen ({model}). Deliberate, explicit comparison.")
    for url, mid in ((base, model), (META_BASE_URL, META_MODEL)):
        ids = [m["id"] for m in _http_json(url.rstrip("/") + "/models")["data"]]
        assert mid in ids, f"{mid} not served at {url}"
    live = _http_json(base.rsplit("/v1", 1)[0] + "/version")
    want = json.loads((baseline_dir / "server_info.json").read_text()).get("vllm_version")
    assert live == want, f"node-6 vLLM version {live} != baseline's {want}: scores depend on it"
    log(f"local-only check OK; vLLM {live} matches the baseline")
    return json.dumps(live)


def gold_label_sets(details: dict[str, Any]) -> tuple[set[str], set[str], bool]:
    """(applicable labels, failed labels, no_plan) from one scored case's details (grader OUTPUT fields only)."""
    dd = details.get("dimension_details")
    if not dd and not details.get("failed_checks") and not details.get("hard_constraints"):
        return set(), set(), True
    applicable: set[str] = set()
    for dim, body in (dd or {}).items():
        for chk in (body or {}).get("checks", []):
            applicable.add(f"commonsense:{dim}:{chk.get('name')}")
    for name in (details.get("hard_constraints") or {}):
        applicable.add(f"hard:{name}")
    failed = set(details.get("failed_checks") or [])
    return applicable | failed, failed, False


def failed_messages(d: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for dim, body in (d.get("dimension_details") or {}).items():
        for chk in (body or {}).get("checks", []):
            if not chk.get("passed"):
                out[f"commonsense:{dim}:{chk.get('name')}"] = str(chk.get("message") or "")
    for name, hc in (d.get("hard_constraints") or {}).items():
        if isinstance(hc, dict) and not hc.get("passed"):
            out[f"hard:{name}"] = str(hc.get("message") or "")
    for lab in d.get("failed_checks") or []:
        out.setdefault(lab, "")
    return out


def compact(details: Optional[dict], score: Any) -> dict:
    app, failed, no_plan = gold_label_sets(details or {})
    return {"applicable": sorted(app), "failed": sorted(failed), "no_plan": bool(no_plan), "score": float(score or 0.0)}


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(statistics.fmean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def _trim(text: str, cap: int = TOOL_OUTPUT_CAP) -> str:
    return text if len(text) <= cap else text[:cap] + f"\n...[output truncated at {cap} chars]"


def read_baseline(base: Path) -> tuple[int, dict[tuple[str, int], dict]]:
    n_pass = len(list(base.glob("pass_*.json")))
    assert n_pass >= 1, f"no baseline passes under {base}"
    rec: dict[tuple[str, int], dict] = {}
    for k in range(1, n_pass + 1):
        for c in json.loads((base / f"pass_{k}.json").read_text())["per_case"]:
            rec[(str(c["case_id"]), k)] = c
    return n_pass, rec


# --------------------------------------------------------------------------- #
# Stage 1: corpus for the meta-agent (TRAIN split, one baseline pass) + eval targets
# --------------------------------------------------------------------------- #

def prepare(args: argparse.Namespace, fw: Any, out: Path) -> dict:
    base = args.baseline_dir
    train = [str(x) for x in fw.train_case_ids]
    evals = [str(x) for x in fw.eval_case_ids]
    train_set = set(train)
    n_pass, rec = read_baseline(base)
    ev_pass = 1  # the single stored pass the meta-agent sees (a parent's one full train evaluation)

    corpus = out / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    (corpus / "cases").mkdir(parents=True)
    (corpus / "semantics").mkdir()
    index, n_noplan, comp, dim_scores, fail_count = [], 0, [], {}, {}
    for cid in train:
        c = rec.get((cid, ev_pass))
        if c is None:
            continue
        d = c.get("details") or {}
        msgs = failed_messages(d)
        _, _, np_ = gold_label_sets(d)
        n_noplan += np_
        comp.append(float(c.get("score") or 0.0))
        for dim, v in (d.get("dimension_scores") or {}).items():
            dim_scores.setdefault(dim, []).append(float(v))
        for lab in msgs:
            fail_count[lab] = fail_count.get(lab, 0) + 1
        row = {"case_id": cid, "composite": c.get("score"), "query": d.get("query"),
               "raw_plan_text": d.get("raw_plan_text") or d.get("raw_result") or "",
               "no_plan": bool(np_), "error": str(d.get("error") or c.get("error") or "")[:600],
               "commonsense_score": d.get("commonsense_score"), "hard_score": d.get("hard_score"),
               "dimension_scores": d.get("dimension_scores"), "dimension_details": d.get("dimension_details"),
               "hard_constraints": d.get("hard_constraints"), "failed_checks": msgs}
        (corpus / "cases" / f"{cid}.json").write_text(json.dumps(row, indent=1))
        index.append({"case_id": cid, "composite": c.get("score"), "n_failed_checks": len(msgs)})
    (corpus / "index.json").write_text(json.dumps(index, indent=1))
    digest = [f"# Train-split digest of one baseline evaluation (pass {ev_pass}, {len(index)} train cases)", "",
              f"- mean composite: {statistics.fmean(comp):.4f}", f"- no_plan_rate: {n_noplan / max(len(index), 1):.3f}",
              "- dimension means: " + ", ".join(f"{k}={statistics.fmean(v):.3f}" for k, v in sorted(dim_scores.items())), "",
              "## Failed checks (number of train cases failing the check)", ""]
    digest += [f"- {lab}: {n}" for lab, n in sorted(fail_count.items(), key=lambda kv: -kv[1])]
    (corpus / "digest.md").write_text("\n".join(digest))
    tp = base / f"pass_{ev_pass}" / "logs" / "trace.jsonl"
    events = []
    for line in tp.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    ids = {e["payload"]["id"] for e in events
           if e.get("kind") in ("llm_call", "tool_call") and str(e["payload"].get("case_id")) in train_set}
    n_trace = 0
    with (corpus / "trace.jsonl").open("w") as fo:
        for e in events:
            p = e.get("payload") or {}
            if (e.get("kind") in ("llm_call", "tool_call") and str(p.get("case_id")) in train_set) or \
               (e.get("kind") in ("llm_response", "tool_result") and p.get("id") in ids):
                fo.write(json.dumps(e) + "\n")
                n_trace += 1
    for name in SEMANTICS_FILES:
        src = PROJECT_DIR / "adapter" / name
        if src.exists():
            shutil.copy(src, corpus / "semantics" / name)
    log(f"corpus: {len(index)} train cases (pass {ev_pass} only), {n_trace} train trace events; eval-split logs excluded")

    rng = random.Random(args.seed)
    mean_train = {cid: statistics.fmean(float(rec[(cid, k)].get("score") or 0.0) for k in range(1, n_pass + 1) if (cid, k) in rec) for cid in train}
    by_low = sorted(train, key=lambda c: (mean_train[c], rng.random()))
    n_low = max(1, round(args.inloop_cases * 2 / 3))
    rest = by_low[n_low:]
    rng.shuffle(rest)
    inloop = by_low[:n_low] + rest[: max(0, args.inloop_cases - n_low)]
    info = {"inloop_train_cases": inloop,
            "inloop_pass1": {c: float(rec[(c, ev_pass)].get("score") or 0.0) for c in train if (c, ev_pass) in rec},
            "final_eval_cases": evals,
            "stored_control": {c: [compact(rec[(c, k)].get("details"), rec[(c, k)].get("score")) for k in range(1, n_pass + 1)] for c in evals},
            "n_baseline_passes": n_pass, "corpus_mean_composite": statistics.fmean(comp)}
    (out / "targets.json").write_text(json.dumps(info))
    log(f"in-loop train sample: {len(inloop)} cases; final measurement: all {len(evals)} eval cases vs {n_pass} stored baseline passes")
    return info


# --------------------------------------------------------------------------- #
# Workspace edit policy
# --------------------------------------------------------------------------- #

FROZEN_FILES = {"workflow.py", "tool_wrapper.py", "tools_schema.json"}
DENY_PATTERNS = [
    r"scorer_impl", r"constraints_commonsense", r"constraints_hard", r"eval_converted", r"\bscorer\b", r"\bbenchmark\b",
    r"\badapter\b", r"\bmeta_agent\b", r"\bprojects\.", r"_eval\b", r"cases\.jsonl", r"meta_info", r"\bgold\b",
    r"\bsubprocess\b", r"os\.system", r"__import__", r"\bimportlib\b", r"\bexec\s*\(", r"\beval\s*\(", r"\bopen\s*\(",
    r"\bsocket\b", r"\brequests\b", r"\burllib\b", r"\bpathlib\b", r"\bshutil\b", r"\bglob\b", r"os\.listdir", r"os\.walk",
    r"\benviron\b", r"\bgetenv\b", r"\bbase_url\b", r"LLM_BASE_URL", r"LLM_MODEL", r"\bmodel\s*=", r"MAS_LLM_BACKBONE_CFG",
    r"openrouter", r"gemma", r"node-5",
    r"^\s*(import|from)\s+(os|io|ctypes|pickle|marshal)\b", r"\bctypes\b", r"\bpickle\b", r"\bmarshal\b", r"\bsys\.modules\b",
]
YAML_FREE_KEYS = {"temperature", "max_output_tokens", "reasoning_effort", "enable_thinking"}


def _yaml_problems(new_src: str) -> list[str]:
    try:
        new = yaml.safe_load(new_src) or {}
    except yaml.YAMLError as e:
        return [f"invalid YAML: {e}"]
    sections = {"default": dict(new.get("default") or {})}
    for a, v in (new.get("agents") or {}).items():
        sections[f"agents.{a}"] = dict(v or {})
    probs = []
    for name, body in sections.items():
        for k, v in body.items():
            if k in ("model", "base_url"):
                if v is not None:
                    probs.append(f"{name}.{k} must stay null (the task agent is fixed to local Qwen3.5-35B on node-6)")
            elif k not in YAML_FREE_KEYS:
                probs.append(f"{name}.{k}: only temperature / max_output_tokens / reasoning_effort / enable_thinking may be set")
    return probs


def edit_problems(rel: str, new_src: str, orig_root: Path) -> list[str]:
    """Problems with proposing `new_src` for `rel`, judged against the ORIGINAL seed file (never an earlier edit)."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        return ["path must be relative inside the workspace"]
    if rel in FROZEN_FILES or rel.startswith("agents/immutable/"):
        return [f"{rel} is frozen (framework entry point / tool layer / message contract)"]
    if p.suffix == ".yaml":
        return _yaml_problems(new_src) if rel == "mas_llm_backbone.yaml" else ["only mas_llm_backbone.yaml may be edited among yaml files"]
    if p.suffix != ".py":
        return ["only .py files (and mas_llm_backbone.yaml) are editable"]
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    orig_file = orig_root / rel
    old_src = orig_file.read_text() if orig_file.exists() else ""
    added = "\n".join(l[1:] for l in difflib.unified_diff(old_src.splitlines(), new_src.splitlines(), lineterm="", n=0)
                      if l.startswith("+") and not l.startswith("+++"))
    code_added = "\n".join(l for l in added.splitlines() if not l.strip().startswith("#"))
    probs = []
    for pat in DENY_PATTERNS:
        m = re.search(pat, code_added, re.IGNORECASE if pat in ("openrouter", "gemma") else re.MULTILINE)
        if m:
            probs.append(f"forbidden token {m.group(0)!r} in added code (no grader/benchmark/adapter access, no file/network/process/env "
                         "access; the model and endpoint are fixed)")
    return probs


def workspace_problems(ws: Path, orig: Path) -> list[str]:
    probs = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix not in (".py", ".yaml"):
            continue
        rel = p.relative_to(ws).as_posix()
        o = orig / rel
        if o.exists() and o.read_text() == p.read_text():
            continue
        probs += [f"{rel}: {x}" for x in edit_problems(rel, p.read_text(), orig)]
    return probs


def variant_profile(orig: Path, ws: Path) -> dict:
    call_re = re.compile(r"call_llm\(|run_tool_stage\(|run_notool_stage\(")
    prof: dict[str, Any] = {"files_changed": [], "files_added": [], "llm_call_sites": 0, "llm_call_sites_seed": 0,
                            "code_statements_delta": 0}

    def stats(src: str) -> tuple[int, int]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return (0, 0)
        return (len(call_re.findall(src)), sum(isinstance(n, ast.stmt) for n in ast.walk(tree)))

    seed_calls = 0
    for p in sorted(orig.rglob("*.py")):
        if "__pycache__" not in p.parts:
            seed_calls += stats(p.read_text())[0]
    total_calls = 0
    for p in sorted(ws.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix not in (".py", ".yaml"):
            continue
        rel = p.relative_to(ws).as_posix()
        o = orig / rel
        new_src = p.read_text()
        if p.suffix == ".py":
            total_calls += stats(new_src)[0]
        if not o.exists():
            prof["files_added"].append(rel)
            old_src = ""
        else:
            old_src = o.read_text()
            if old_src == new_src:
                continue
            prof["files_changed"].append(rel)
        if p.suffix == ".py":
            prof["code_statements_delta"] += stats(new_src)[1] - stats(old_src)[1]
    prof["llm_call_sites"], prof["llm_call_sites_seed"] = total_calls, seed_calls
    return prof


def route_problems(round_dir: Path) -> list[str]:
    """Every task-agent LLM call must have gone to node-6 / Qwen3.5-35B."""
    tp = round_dir / "logs" / "trace.jsonl"
    if not tp.exists():
        return []
    bad: set[tuple[str, str]] = set()
    for line in tp.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "llm_call":
            p = e.get("payload") or {}
            if p.get("model") != TASK_MODEL or "node-6" not in str(p.get("base_url")):
                bad.add((str(p.get("model")), str(p.get("base_url"))))
    return [f"LLM call to {m} @ {b}" for m, b in sorted(bad)]


def run_smoke(ws: Path) -> Optional[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    try:
        r = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0, '.'); import workflow, mas_workflow"],
                           cwd=ws, env=env, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return "import of workflow/mas_workflow timed out"
    return None if r.returncode == 0 else (r.stderr or r.stdout)[-1500:]


# --------------------------------------------------------------------------- #
# run_python: the agent runs its own code offline on the stored TRAIN plans
# --------------------------------------------------------------------------- #

RUN_ALLOWED_IMPORTS = {
    "re", "json", "math", "datetime", "itertools", "collections", "statistics", "typing", "dataclasses", "functools", "textwrap",
    "string", "copy", "heapq", "bisect", "random", "decimal", "fractions", "operator", "enum", "abc", "__future__", "time", "unicodedata",
}
RUN_BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                     "input", "breakpoint", "help", "exit", "quit", "memoryview"}
RUN_BLOCKED_ATTRS = {"os", "sys", "subprocess", "builtins", "importlib", "shutil", "pathlib", "io", "socket", "system", "popen", "environ",
                     "getenv", "putenv", "listdir", "walk", "scandir", "remove", "unlink", "rmdir", "rename", "execv", "fork", "spawn",
                     "kill", "open", "read_text", "write_text", "read_bytes", "write_bytes", "modules", "load_module", "import_module"}
RUN_LEAK_PATTERNS = [r"scorer_impl", r"constraints_commonsense", r"constraints_hard", r"eval_converted", r"\bscorer\b", r"\bbenchmark\b",
                     r"\badapter\b", r"\bmeta_agent\b", r"_eval\b", r"cases\.jsonl", r"meta_info", r"\bgold\b", r"openrouter", r"gemma"]

RUNNER_PRELUDE = r'''
import sys, json, types, os as _os
_ws, _data, _codefile = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [_ws]
_d = json.load(open(_data))
_cases = _d["cases"]
_m = types.ModuleType("train_data")
_m.CASES = _cases
def _by_id(cid):
    for c in _cases:
        if str(c["case_id"]) == str(cid):
            return c
    return None
_m.by_id = _by_id
_m.failing = lambda label: [c for c in _cases if any(label in k for k in c["failed_checks"])]
def _use(cid):
    _os.environ["TRAVEL_SAMPLE_ID"] = str(cid)
_m.use_case = _use
sys.modules["train_data"] = _m
_code = open(_codefile).read()
exec(compile(_code, "<run_python>", "exec"), {"__name__": "__main__"})
'''


def python_problems(code: str, ws: Path) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    ws_mods = {p.stem for p in ws.glob("*.py")} | {p.name for p in ws.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    allowed = RUN_ALLOWED_IMPORTS | ws_mods | {"train_data"}
    probs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    probs.append(f"import {a.name!r} not allowed (allowed: stdlib data modules, train_data, your workspace modules)")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in allowed:
                probs.append(f"from {node.module!r} import ... not allowed")
        elif isinstance(node, ast.Name) and node.id in RUN_BLOCKED_NAMES:
            probs.append(f"name {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in RUN_BLOCKED_ATTRS):
            probs.append(f"attribute {node.attr!r} not allowed")
    for pat in RUN_LEAK_PATTERNS:
        m = re.search(pat, code, re.IGNORECASE)
        if m:
            probs.append(f"forbidden token {m.group(0)!r}")
    return sorted(set(probs))[:6]


# --------------------------------------------------------------------------- #
# The meta-agent's virtual filesystem (allow-list only) and tools
# --------------------------------------------------------------------------- #

class Vfs:
    """corpus files (read-only) + the live workspace (`workspace/<rel>`). Nothing else exists."""

    def __init__(self, corpus: Path, ws: Path):
        self.corpus, self.ws = corpus, ws
        self.cases = {p.stem: json.loads(p.read_text()) for p in sorted((corpus / "cases").glob("*.json"))}
        self.files: dict[str, Path] = {}
        self.refresh()

    def refresh(self) -> None:
        f: dict[str, Path] = {"corpus/index.json": self.corpus / "index.json", "corpus/digest.md": self.corpus / "digest.md",
                              "corpus/trace.jsonl": self.corpus / "trace.jsonl"}
        for p in sorted((self.corpus / "cases").glob("*.json")):
            f[f"corpus/cases/{p.name}"] = p
        for name in SEMANTICS_FILES:
            if (self.corpus / "semantics" / name).exists():
                f[f"semantics/{name}"] = self.corpus / "semantics" / name
        for p in sorted(self.ws.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".yaml", ".json", ".md", ".txt"):
                f["workspace/" + p.relative_to(self.ws).as_posix()] = p
        self.files = f

    def resolve(self, path: str) -> Path:
        key = path.strip().lstrip("./").lstrip("/")
        if key in self.files:
            return self.files[key]
        if f"{key}.json" in self.files:
            return self.files[f"{key}.json"]
        raise FileNotFoundError(f"no such file {path!r}. Readable: corpus/index.json, corpus/digest.md, corpus/cases/<case_id>.json, "
                                "corpus/trace.jsonl, semantics/*.json, workspace/<file>")


def tool_list_cases(v: Vfs, a: dict) -> str:
    flt = str(a.get("failed_check") or "")
    table: dict[str, int] = {}
    cases = []
    for cid, row in v.cases.items():
        for lab in row["failed_checks"]:
            table[lab] = table.get(lab, 0) + 1
        if flt and not any(flt in lab for lab in row["failed_checks"]):
            continue
        cases.append({"case_id": cid, "composite": row["composite"], "no_plan": row["no_plan"], "failed_checks": sorted(row["failed_checks"])})
    return json.dumps({"n_train_cases": len(v.cases),
                       "failure_table": [{"label": k, "cases_failing": n} for k, n in sorted(table.items(), key=lambda kv: -kv[1])],
                       "cases": cases[: int(a.get("limit") or 200)]}, indent=1)


def tool_show_case(v: Vfs, a: dict) -> str:
    cid = str(a.get("case_id", "")).strip()
    row = v.cases.get(cid)
    if row is None:
        return f"no train case {cid!r}; use list_cases"
    out = [f"CASE {cid}  composite={row['composite']}  commonsense={row['commonsense_score']}  hard={row['hard_score']}", "REQUEST:", row["query"] or "", "",
           "RAW PLAN:", row["raw_plan_text"] or "(no plan)"]
    if row["error"]:
        out += ["", f"ERROR / NO-PLAN NOTE: {row['error']}"]
    out += ["", "FULL GRADER BREAKDOWN (every check this case was scored on -- ALL of them, not just the failing ones; "
                "a dimension is 1.0 only if every one of its checks below is passed=true):"]
    for dim, body in sorted((row.get("dimension_details") or {}).items()):
        out.append(f"  [{dim}]  score={body.get('score')}  ({body.get('passed')}/{body.get('total')} checks passed)")
        for chk in body.get("checks", []):
            mark = "PASS" if chk.get("passed") else "FAIL"
            out.append(f"    - {mark}  {chk.get('name')}" + (f": {chk['message']}" if chk.get("message") else ""))
    hc = row.get("hard_constraints") or {}
    if hc:
        out.append("  [hard constraints -- which apply depends on what THIS request asked for]")
        for name, v in sorted(hc.items()):
            mark = "PASS" if (isinstance(v, dict) and v.get("passed")) else "FAIL"
            msg = v.get("message") if isinstance(v, dict) else None
            out.append(f"    - {mark}  {name}" + (f": {msg}" if msg else ""))
    out += ["", "FAILED CHECKS ONLY (quick summary, same messages as above):"]
    out += [f"  {lab}: {msg or '(no message)'}" for lab, msg in sorted(row["failed_checks"].items())] or ["  (none)"]
    return "\n".join(out)


def tool_read_file(v: Vfs, a: dict) -> str:
    p = v.resolve(str(a.get("path", "")))
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    offset = max(0, int(a.get("offset") or 0))
    limit = min(int(a.get("limit") or 12000), TOOL_OUTPUT_CAP)
    chunk = text[offset:offset + limit]
    more = f" -- more available, call again with offset={offset + limit}" if offset + limit < len(text) else ""
    return f"[{a.get('path')}] chars {offset}-{offset + len(chunk)} of {len(text)}{more}\n{chunk}"


def tool_grep(v: Vfs, a: dict) -> str:
    try:
        rx = re.compile(str(a.get("pattern", "")), re.IGNORECASE)
    except re.error as e:
        return f"bad regex: {e}"
    cap = int(a.get("max_matches") or 40)
    pat = str(a.get("path") or "corpus/cases/*").strip().lstrip("./").lstrip("/")
    keys = [k for k in v.files if fnmatch.fnmatch(k, pat)]
    if not keys:
        return "no files match that path/glob"
    hits: list[str] = []
    for key in keys:
        p = v.files[key]
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{key}:{i}: {line.strip()[:400]}")
                if len(hits) >= cap:
                    return "\n".join(hits) + f"\n...[stopped at max_matches={cap}]"
    return "\n".join(hits) or "no matches"


TOOLS = [
    {"name": "list_cases", "description": "Overview of the baseline TRAIN evaluation: failure table (train cases failing each labeled check) and per-case composite + failed checks. failed_check filters by label substring.",
     "input_schema": {"type": "object", "properties": {"failed_check": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "show_case", "description": "One baseline train case: the request, the FULL raw plan the seed produced, its dimension scores and every failed check with the grader's message.",
     "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]}},
    {"name": "read_file", "description": "Read a file, paged by character offset. Paths: workspace/<file> (the agent's own source), corpus/digest.md, corpus/trace.jsonl (baseline train LLM/tool events), semantics/error_semantics.json, semantics/harness_error_semantics.json.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "grep", "description": "Case-insensitive regex search over readable files (glob ok, e.g. workspace/agents/*.py, corpus/trace.jsonl).",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "max_matches": {"type": "integer"}}, "required": ["pattern"]}},
    {"name": "write_file", "description": "Overwrite (or create) a workspace file with COMPLETE content.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "str_replace_file", "description": "Targeted edit of a workspace file: replace old_str (must occur exactly once) with new_str.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}},
    {"name": "check_workspace", "description": (
        "Sanity check of your CURRENT workspace, in two parts. FAST part (a second or two, no LLM calls): full-tree policy/syntax scan "
        "of every changed file, a real import of workflow.py and mas_workflow.py together, plus the FULL suite of HGM's own real editor "
        "validators (syntax, undefined names, function signatures, import allow-list, schema/wrapper consistency, mutable-tool import and "
        "routing rules, immutable-files, mas_llm_backbone.yaml structure, a subprocess import of every mutable module) -- catches typos, "
        "undefined names, forbidden edits and cross-file mistakes a single write_file/str_replace_file call can't see. SLOWER part (makes "
        "ONE real local LLM run on a fixed train case, does not consume an evaluate_variant call): confirms a plan is actually PRODUCED for "
        "that case, not merely that nothing crashed -- a run that completes with no exception but emits no itinerary (e.g. ran out of output "
        "budget) is the single most common regression in this project and none of the fast checks can see it. Use this after a batch of "
        "edits, BEFORE spending a limited evaluate_variant call on something that would just crash, get rejected, or silently produce no plan."),
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_python", "description": (
        "Run YOUR OWN Python code offline (no LLM, no network) on the stored TRAIN plans and see its stdout/stderr. `import train_data` gives CASES "
        "(list of dicts: case_id, query, plan = the seed's raw plan text, composite, no_plan, failed_checks = {label: grader message}), by_id(cid), "
        "failing(label_substring) and use_case(cid), which must be called before any tool lookup for that case. You may import your workspace modules "
        "(e.g. `from tool_wrapper import ToolWrapper` to call the search tools, your own modules) and stdlib data modules (re, json, math, datetime, "
        "itertools, collections, statistics, ...). No file/env/network access. Use it to inspect REAL plans and tool outputs, and to test your pipeline "
        "on the stored train requests BEFORE spending an evaluate_variant call. Print what you want to see."),
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["code"]}},
    {"name": "evaluate_variant", "description": (
        "Run your CURRENT workspace through the real evaluator and get the labeled failure report for that child. Limited calls. "
        "Without case_ids: evaluates ALL 60 TRAIN cases (can take a while -- this is the real signal used for accept/rollback, so it is "
        "thorough by default). Pass case_ids (a list of TRAIN case ids, e.g. the ones a failure table showed you) to evaluate just that "
        "subset instead -- much faster, useful while iterating on one specific failure mode before spending a full call on all 60. "
        "case_ids outside the TRAIN split are rejected without consuming a call."),
     "input_schema": {"type": "object", "properties": {"case_ids": {"type": "array", "items": {"type": "string"},
                      "description": "optional: specific TRAIN case ids to evaluate instead of all 60"}}}},
    {"name": "submit_variant", "description": "Finish: the current workspace is your final variant.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

SYSTEM_PROMPT = """You are a meta-agent redesigning the harness of a multi-agent travel-planning system (the "task agent") around a WEAK runtime model.

The task agent turns a traveler's natural-language request into a full day-by-day itinerary using LLM stages (Flight -> Train -> Sightseeing ->
Accounting) that call search tools. A hidden grader scores every itinerary against many named checks. You do not have the grader's source or data;
you learn from real runs, like the system's own self-improvement loop: one baseline evaluation of the TRAIN cases (scored, with failed-check labels
and grader messages), a digest, the trace of LLM/tool calls, the project's human-written check descriptions and the agent's source in workspace/.

GRADING STRUCTURE (visible in the logs' dimension_scores / hard_score): composite = (commonsense + hard) / 2. Each commonsense dimension counts
(1/8) only if EVERY check in it passes; the hard score counts only if EVERY hard constraint of the case passes. A case with no plan scores 0.
Fixing one check rarely moves the score; a plan must satisfy whole dimensions consistently.

YOUR TARGET: maximize the composite on held-out cases. The runtime model is Qwen3.5-35B-A3B (about 3B active parameters) and is UNRELIABLE:
it skips required tool calls even when told to, fabricates venue names and prices, cannot reliably compose a long multi-day itinerary in one
generation (about 11% of runs produce no plan at all), and piling more prompt rules onto it makes things worse.

DESIGN RULES -- restructure mas_workflow.py and the stage modules broadly (new modules allowed) so that CODE owns everything deterministic and the
LLM is used ONLY for small, single-purpose, verifiable subtasks:
 1. All tool calls are made by code (through the existing ToolWrapper): venues, coordinates, route/commute times (query_road_route_info),
    opening hours, visit durations, transport options. The LLM must never be responsible for remembering to call a tool.
 2. Use the LLM only for narrow tasks with constrained output, for example (a) extract the traveler's request into a fixed JSON schema (cities,
    dates, people/rooms, budget, must-visit / must-eat items, hotel and transport preferences), (b) choose ONE item from a short candidate list
    that code supplies (so it cannot invent names). Code parses and validates every answer; on failure retry once, then fall back to a
    deterministic default. Do NOT ask the model to write the whole itinerary. Keep each call small (reasoning_effort low for calls without tools).
 3. Code builds the itinerary: schedule the days respecting real commute times between consecutive stops, opening/service hours, closure days,
    realistic visit and meal durations, lunch/dinner slots, the return to the hotel each night, the intercity legs; never repeat an attraction or
    restaurant; use exact names and prices from tool results; compute the budget arithmetic; render the exact required output line format.
 4. Code re-checks its own plan against the constraints you can infer from the grader's messages in the stored logs and repairs it
    deterministically before returning. A valid deterministic fallback plan is better than no plan.
 5. Keep the workflow.py -> mas_workflow.run_task(task) -> AgentOutput contract; the final result is the complete itinerary text.

RULES
- No hard-coding of case ids, cities or answers; general behavior only. No file/network/process/env access. The model and endpoint are fixed
  (local Qwen3.5-35B): do not try to change them. workflow.py, tool_wrapper.py, tools_schema.json and agents/immutable/ are frozen.
- If you don't want the LLM to call a tool for a given case (e.g. because code already decided the answer deterministically), remove that
  tool from the schema you pass to run_tool_stage for that case -- do NOT just tell it not to call the tool in the prompt text. Confirmed
  on this exact codebase: told not to call a specific tool while that tool stayed in its schema, the weak model called it anyway in every
  sampled case, and at least once used the tool's own result over the value it was given, breaking a case that passed before. A schema the
  model literally cannot call is enforced; a request in the prompt is not, no matter how explicit.
- run_python is free and makes no LLM calls; evaluate_variant runs on ALL 60 train cases by default (takes a while) unless you pass case_ids
  for a faster subset check; either way it uses the local model and you have {eval_rounds} calls total.
  Turn budget: {max_turns}.
- Test on the ACTUAL cases where a check currently fails, ALL of them if you can, not just one or two -- and be aware that a small, fixed
  sample looking clean is not the same as being fixed. It is easy to overfit to whatever small set you keep re-testing against without ever
  re-checking the wider set the fix is meant to generalize over. run_python against train_data.failing(...) is free -- use it to check every
  failing case offline before spending a limited evaluate_variant call, and spend at least one evaluate_variant call with NO case_ids (the
  full 60) before you submit, however good a small sample looks.
- In addition to the real stored cases, feel free to write your OWN small synthetic test inputs for any parsing/extraction logic you write --
  short constructed strings covering variations you'd expect in real requests (different phrasings, punctuation, quoting styles, name formats
  with apostrophes or multi-word names) -- and check your function against them directly with run_python, the same way you'd test any parsing
  code. The real stored cases tell you whether it works on what's already in front of you; synthetic edge cases tell you whether the
  underlying logic actually generalizes, rather than happening to match only the specific inputs you've seen so far.

WORKFLOW
1. Read corpus/digest.md and several train cases (show_case) -- real requests, plans and grader messages; read the agent files
   (workspace/mas_workflow.py, workspace/agents/*.py, tool_wrapper.py, tools_schema.json) and grep corpus/trace.jsonl to see real tool outputs.
2. Use run_python heavily: print REAL stored plans and REAL tool outputs first and parse those (do NOT assume formats from the prompt spec -- the
   square brackets in the format spec are placeholders, not literal). Build the pipeline incrementally and test it offline on the stored train
   requests (tool lookups work offline through ToolWrapper after train_data.use_case(cid); LLM calls do not).
3. Write the code into the workspace, then evaluate_variant, read the per-case results (failed checks, no-plan reasons), refine.
4. Before calling submit_variant, re-read your OWN changed code and check every claim you are about to write in the summary against what the
   code actually does line by line -- not what you intended it to do. A summary that overstates what a change does (e.g. "the LLM no longer
   decides X" when the LLM's tool schema still lets it decide X) is worse than an accurate but modest one: it hides exactly the failure mode
   most worth catching before this gets built on further.
5. submit_variant with a short summary of the design and what you found."""


# --------------------------------------------------------------------------- #
# One redesign session (develop) and its measurement
# --------------------------------------------------------------------------- #

class Session:
    def __init__(self, out: Path, fw: Any, args: argparse.Namespace, info: dict):
        self.out, self.fw, self.args, self.info = out, fw, args, info
        self.dir = out / "work"
        self.orig, self.ws, self.corpus = self.dir / "orig", self.dir / "workspace", out / "corpus"
        self.rounds_used = 0
        self.eval_log: list[dict] = []
        self.route_violations: list[str] = []
        self.py_calls = 0

    def setup(self, start_workspace: Optional[Path] = None) -> None:
        """orig = the pristine seed (edit-policy reference); workspace = the seed, or an existing workspace to continue from."""
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        ig = shutil.ignore_patterns("__pycache__")
        shutil.copytree(self.fw.seed_dir, self.orig, ignore=ig)
        shutil.copytree(start_workspace or self.fw.seed_dir, self.ws, ignore=ig)
        # HGM's real editor_validators expect <out_dir>/task_agent; symlink (not copy) so these always see live content.
        self._val_out_dir, self._val_base_dir = self.dir / "_valcheck", self.dir / "_valcheck_base"
        self._val_out_dir.mkdir()
        self._val_base_dir.mkdir()
        (self._val_out_dir / "task_agent").symlink_to(self.ws.resolve(), target_is_directory=True)
        (self._val_base_dir / "task_agent").symlink_to(self.orig.resolve(), target_is_directory=True)
        self._validator_cache: Optional[list[tuple[str, Any]]] = None

    def run_cases(self, name: str, case_ids: list[str]) -> Any:
        rd = self.dir / name
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        shutil.copytree(self.ws, rd / "task_agent", ignore=shutil.ignore_patterns("__pycache__"))
        res = self.fw.evaluator.run(rd, self.fw.benchmark_dir, case_ids=case_ids)
        self.route_violations += route_problems(rd)
        return res

    def _validators(self) -> list[tuple[str, Any]]:
        """The FULL set of HGM's real editor_validators (meta_agent/editor_validators.py), not just the (much sparser) subset this
        experiment's own YAML config happens to declare -- built the same way build_components() would (registry lookup + inject
        only the kwargs each validator's __init__ actually declares), so this is genuine reuse of HGM's own production checks, not a
        reimplementation of them."""
        if self._validator_cache is None:
            import inspect
            from meta_agent import registry as _reg
            from meta_agent.editor_validators import DEFAULT_VALIDATOR_NAMES
            names = list(DEFAULT_VALIDATOR_NAMES) + ["llm_backbone_config"]  # smoke_test is folded into the real-case check below
            injections = {"mutable_exclude": self.fw.config.mutable_exclude, "evaluator": self.fw.evaluator, "benchmark_dir": self.fw.benchmark_dir}
            cache = []
            for name in names:
                cls = _reg.get("validator", name)
                sig = inspect.signature(cls).parameters
                kwargs = {k: v for k, v in injections.items() if k in sig}
                cache.append((name, cls(**kwargs)))
            self._validator_cache = cache
        return self._validator_cache

    def check_workspace(self) -> str:
        probs = workspace_problems(self.ws, self.orig)
        if probs:
            return "check_workspace: POLICY/SYNTAX PROBLEMS -- " + "; ".join(probs[:10])
        smoke = run_smoke(self.ws)
        if smoke:
            return "check_workspace: IMPORT FAILED --\n" + smoke
        lines, any_fail = [], False
        for name, validator in self._validators():
            try:
                errs = validator.validate(self._val_out_dir, self._val_base_dir)
            except Exception as exc:  # noqa: BLE001 -- a validator crashing is a report line, not a tool crash
                errs = [f"validator itself raised {exc!r}"]
            if errs:
                any_fail = True
                lines.append(f"  [{name}] FAILED: " + "; ".join(str(e)[:300] for e in errs[:5]))
            else:
                lines.append(f"  [{name}] OK")
        # Real end-to-end check on ONE fixed TRAIN case: not just "did it crash" (HGM's smoke_test validator) but also "did it
        # actually produce a plan" -- a case that runs to completion with no exception yet emits nothing is the single most common
        # regression in this project's history (see corpus/digest.md's no_plan_rate) and neither ast-based checks nor a bare
        # import/exception check would ever catch it.
        try:
            case_id = sorted(self.fw.train_case_ids, key=int)[0]
            res = self.run_cases("check_workspace_smoke", [case_id])
            c = res.per_case[0]
            if c.error:
                any_fail = True
                lines.append(f"  [smoke_test+plan_check] CRASHED on train case {c.case_id}: {str(c.error)[:400]}")
            else:
                _, _, no_plan = gold_label_sets(c.details or {})
                if no_plan:
                    any_fail = True
                    lines.append(f"  [smoke_test+plan_check] NO PLAN produced for train case {c.case_id} (ran without a code exception, "
                                 "but the grader sees no usable itinerary -- e.g. it ran out of output budget or the wrap-up produced no "
                                 "<itinerary> tag). This is a regression even though nothing crashed.")
                else:
                    lines.append(f"  [smoke_test+plan_check] OK -- train case {c.case_id} produced a plan (its composite score is informational "
                                 "only; this check is pass/no-plan, not pass/fail on quality)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  [smoke_test+plan_check] could not run: {exc!r}")
        header = "check_workspace: " + ("SOME CHECKS FAILED (see below)" if any_fail else "ALL CHECKS OK") + \
            " -- policy/syntax clean, workflow.py + mas_workflow.py import fine, plus the full HGM validator suite:"
        return "\n".join([header] + lines)

    def run_python(self, code: str, timeout_s: int = 120) -> str:
        probs = python_problems(code, self.ws)
        if probs:
            return "run_python REJECTED (nothing executed): " + "; ".join(probs)
        scratch = self.dir / "scratch"
        scratch.mkdir(exist_ok=True)
        data = scratch / "train_cases.json"
        if not data.exists():
            rows = []
            for f in sorted((self.corpus / "cases").glob("*.json")):
                r = json.loads(f.read_text())
                rows.append({"case_id": r["case_id"], "query": r["query"], "plan": r["raw_plan_text"], "composite": r["composite"],
                             "no_plan": r["no_plan"], "failed_checks": r["failed_checks"]})
            data.write_text(json.dumps({"cases": rows}))
        self.py_calls += 1
        codefile = scratch / f"user_{self.py_calls}.py"
        codefile.write_text(code)
        (scratch / "prelude.py").write_text(RUNNER_PRELUDE)
        env = self.fw.evaluator._child_env(scratch / "trace_offline.jsonl")
        env.pop("META_AGENT_TRACE_PATH", None)
        env["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"   # offline: any LLM call fails fast
        env["OPENAI_API_KEY"] = "EMPTY"
        try:
            r = subprocess.run([sys.executable, "-B", str(scratch / "prelude.py"), str(self.ws), str(data), str(codefile)],
                               cwd=scratch, env=env, capture_output=True, text=True, timeout=max(5, min(int(timeout_s or 120), 300)),
                               preexec_fn=self.fw.evaluator._preexec)   # the bound method itself: it sets the child's rlimits
        except subprocess.TimeoutExpired:
            return f"run_python TIMED OUT after {timeout_s}s"
        out = r.stdout[-9000:] if len(r.stdout) > 9000 else r.stdout
        err = r.stderr[-2500:]
        return f"exit code {r.returncode}\n--- stdout ---\n{out}" + (f"\n--- stderr ---\n{err}" if err.strip() else "")

    def evaluate_variant(self, case_ids: Optional[list] = None) -> str:
        if self.rounds_used >= self.args.eval_rounds:
            return f"no evaluation calls left (limit {self.args.eval_rounds}); refine if you must, then submit_variant"
        probs = workspace_problems(self.ws, self.orig)
        if probs:
            return "workspace violates the edit policy (evaluation NOT run, call NOT consumed):\n" + "\n".join(probs[:8])
        smoke = run_smoke(self.ws)
        if smoke:
            return "workspace does not import (evaluation NOT run, call NOT consumed):\n" + smoke
        train_set = {str(x) for x in self.fw.train_case_ids}
        if case_ids:
            req = [str(c).strip() for c in case_ids if str(c).strip()]
            bad = [c for c in req if c not in train_set]
            if bad:
                return (f"REJECTED (evaluation NOT run, call NOT consumed): case_ids not in the TRAIN split (only train cases are visible to "
                        f"you): {bad[:10]}")
            ids = sorted(set(req), key=int)
        else:
            ids = sorted(train_set, key=int)  # default: the full 60 train cases, not a small sample
        self.rounds_used += 1
        try:
            res = self.run_cases(f"inloop_{self.rounds_used}", ids)
        except Exception as exc:  # noqa: BLE001 -- an infrastructure failure must not cost the agent an evaluation call
            self.rounds_used -= 1
            return f"evaluation FAILED for an infrastructure reason ({exc!r}); your evaluation call was NOT consumed. Try again shortly."
        by = {str(c.case_id): c for c in res.per_case}
        lines, comp, base = [], [], []
        n_noplan = 0
        for cid in ids:
            cr = by.get(cid)
            d = (cr.details if cr else None) or {}
            rec = compact(d, cr.score if cr else 0.0)
            comp.append(rec["score"])
            base.append(self.info["inloop_pass1"].get(cid, 0.0))
            if rec["no_plan"]:
                n_noplan += 1
                why = str(d.get("error") or (cr.error if cr else "") or "no plan")
                why = why[-600:] if "Traceback" in why else why[:300]
                lines.append(f"  case {cid}: NO PLAN/CRASH ({why}) | composite 0.000 (baseline pass 1 {base[-1]:.3f})")
            else:
                msgs = failed_messages(d)
                bad_dims = sorted({l.split(":")[1] for l in rec["failed"] if l.startswith("commonsense:")})
                hard_bad = [l for l in rec["failed"] if l.startswith("hard:")]
                lines.append(f"  case {cid}: composite {rec['score']:.3f} (baseline pass 1 {base[-1]:.3f}) | failing dimensions: {bad_dims or 'none'} | "
                             f"failing hard: {len(hard_bad)}")
                for lab, m in list(msgs.items())[:6]:
                    lines.append(f"      - {lab.split(':', 1)[-1]}: {m[:170]}")
        head = (f"evaluation call {self.rounds_used}/{self.args.eval_rounds} on {len(ids)} train cases: mean composite {statistics.fmean(comp):.3f} "
                f"(baseline pass 1 on the same cases {statistics.fmean(base):.3f}); no-plan cases {n_noplan}/{len(ids)}")
        self.eval_log.append({"call": self.rounds_used, "composite": statistics.fmean(comp), "baseline_pass1": statistics.fmean(base), "no_plan": n_noplan})
        return head + "\n" + "\n".join(lines)

    def develop(self, system_prompt: Optional[str] = None, user_message: Optional[str] = None) -> dict:
        args = self.args
        v = Vfs(self.corpus, self.ws)
        transcript = (self.dir / "transcript.jsonl").open("a", encoding="utf-8")
        system = (system_prompt or SYSTEM_PROMPT).format(eval_rounds=args.eval_rounds, max_turns=args.max_turns)
        user = user_message or ("Redesign the harness per the design rules. Start by reading corpus/digest.md, several train cases (show_case) and the agent source in "
                                "workspace/, then use run_python to look at real plans and tool outputs before you write any code.")
        history: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        impl = {"list_cases": tool_list_cases, "show_case": tool_show_case, "read_file": tool_read_file, "grep": tool_grep}
        used: dict[str, int] = {}
        submitted, nudges, malformed_nudges, summary, turn, failed_call = False, 0, 0, "", -1, False

        def do_write(a: dict, replace: bool) -> str:
            rel = str(a.get("path", "")).strip().lstrip("./").lstrip("/")
            rel = rel[len("workspace/"):] if rel.startswith("workspace/") else rel
            target = self.ws / rel
            old = target.read_text() if target.exists() else None
            if replace:
                if old is None:
                    return f"no such file {rel}"
                o, n = str(a.get("old_str", "")), str(a.get("new_str", ""))
                if old.count(o) != 1:
                    return f"old_str must occur exactly once in {rel}, found {old.count(o)}; nothing changed"
                new = old.replace(o, n)
            else:
                new = str(a.get("content", ""))
            probs = edit_problems(rel, new, self.orig)
            if probs:
                return "EDIT REJECTED (file unchanged): " + "; ".join(probs)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new)
            return f"wrote {rel} ({len(new)} chars, policy OK)"

        for turn in range(args.max_turns):
            if self.rounds_used == 0 and turn in (args.max_turns // 3, (2 * args.max_turns) // 3):
                history.append({"role": "user", "content": f"Turn {turn}/{args.max_turns}: you have not called evaluate_variant yet. Test what you have "
                                "offline with run_python if you haven't, then call evaluate_variant -- untested changes usually crash or regress."})
            if turn == args.max_turns - 6:
                history.append({"role": "user", "content": f"Only 6 turns left ({max(0, args.eval_rounds - self.rounds_used)} evaluation calls left). "
                                "Prefer small str_replace_file edits; make sure your last evaluated state is what you submit, then call submit_variant."})
            try:
                resp = call_llm(messages=history, tools=TOOLS, model=META_MODEL, base_url=META_BASE_URL,
                                reasoning_effort=os.environ.get("REDESIGN_META_REASONING", "medium"),
                                max_output_tokens=int(os.environ.get("REDESIGN_META_MAX_TOKENS", "32768")))
            except Exception as exc:  # noqa: BLE001
                log(f"meta-agent call failed on turn {turn}: {exc!r}")
                failed_call = True
                break
            calls = getattr(resp, "tool_calls", None) or []
            usage = getattr(getattr(resp, "raw", None), "usage", None)
            reasoning_tok = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None) if usage else None
            transcript.write(json.dumps({"turn": turn, "tool": "_meta_call_info", "args": "",
                                         "output": json.dumps({"stop_reason": getattr(resp, "stop_reason", None),
                                                               "has_content": bool(getattr(resp, "content", None)),
                                                               "n_tool_calls": len(calls), "reasoning_tokens": reasoning_tok,
                                                               "output_tokens": getattr(usage, "output_tokens", None) if usage else None})}) + "\n")
            if getattr(resp, "content", None):
                transcript.write(json.dumps({"turn": turn, "tool": "_assistant_text", "args": "", "output": resp.content[:2500]}) + "\n")
            transcript.flush()
            if not calls:
                content = getattr(resp, "content", None) or ""
                if _MALFORMED_TOOLCALL_RE.search(content):
                    malformed_nudges += 1
                    transcript.write(json.dumps({"turn": turn, "tool": "_malformed_tool_call_detected", "args": "", "output": content[:1500]}) + "\n")
                    transcript.flush()
                    if malformed_nudges > 4:
                        log(f"too many malformed pseudo-tool-call attempts ({malformed_nudges}); ending session")
                        break
                    history.append({"role": "user", "content": "Your previous response looked like an attempted tool call written as plain text "
                                    "(not a real tool call), so nothing was executed. Please make that SAME call again using the actual tool-calling "
                                    "mechanism, not as text in your message."})
                    continue
                nudges += 1
                if nudges > 2:
                    break
                history.append({"role": "user", "content": "Continue with the tools, or call submit_variant if done."})
                continue
            for idx, call in enumerate(calls):
                cid = getattr(call, "id", None) or getattr(call, "call_id", None) or f"call_{turn}_{idx}"
                a = call.arguments if isinstance(call.arguments, dict) else {}
                malformed = (not isinstance(call.arguments, dict)) or "_raw_arguments" in a
                history.append({"type": "function_call", "call_id": cid, "name": call.name, "arguments": json.dumps({} if malformed else a)})
                used[call.name] = used.get(call.name, 0) + 1
                v.refresh()
                try:
                    if malformed:
                        out_txt = "your tool-call arguments were not valid JSON; retry (prefer small str_replace_file edits)"
                    elif call.name in impl:
                        out_txt = impl[call.name](v, a)
                    elif call.name == "write_file":
                        out_txt = do_write(a, False)
                    elif call.name == "str_replace_file":
                        out_txt = do_write(a, True)
                    elif call.name == "check_workspace":
                        out_txt = self.check_workspace()
                    elif call.name == "run_python":
                        out_txt = self.run_python(str(a.get("code", "")), int(a.get("timeout_s") or 120))
                    elif call.name == "evaluate_variant":
                        out_txt = self.evaluate_variant(a.get("case_ids"))
                    elif call.name == "submit_variant":
                        if self.rounds_used == 0 and turn < args.max_turns - 1:
                            out_txt = "NOT submitted: you have not called evaluate_variant yet. Call it first, fix what it shows, then submit."
                        else:
                            submitted, summary = True, str(a.get("summary", ""))[:3000]
                            out_txt = "submitted"
                    else:
                        out_txt = f"unknown tool {call.name!r}"
                except Exception as exc:  # noqa: BLE001
                    out_txt = f"tool error: {exc!r}"
                out_txt = _trim(out_txt)
                history.append({"type": "function_call_output", "call_id": cid, "output": out_txt})
                transcript.write(json.dumps({"turn": turn, "tool": call.name, "args": str(a)[:500], "output": out_txt[:2500]}) + "\n")
                transcript.flush()
            if submitted:
                break
            _history_trim(history)
        transcript.close()
        info = {"submitted": submitted, "summary": summary, "turns_used": turn + 1, "tool_usage": used, "eval_calls_used": self.rounds_used,
                "evaluated_in_loop": self.rounds_used > 0, "meta_call_failed": failed_call, "inloop_rounds": self.eval_log,
                "route_violations": sorted(set(self.route_violations))[:5]}
        (self.dir / "develop.json").write_text(json.dumps(info, indent=1))
        return info

    def measure(self) -> dict:
        ids = self.info["final_eval_cases"]
        control = self.info["stored_control"]
        prof = variant_profile(self.orig, self.ws)
        m: dict[str, Any] = {"profile": prof, "evaluated_in_loop": self.rounds_used > 0, "inloop_rounds": self.eval_log}
        probs = workspace_problems(self.ws, self.orig)
        if not (prof["files_changed"] or prof["files_added"]):
            m["status"] = "no_variant"
        elif probs:
            m["status"], m["policy_problems"] = "invalid", probs[:10]
        else:
            smoke = run_smoke(self.ws)
            if smoke:
                m["status"], m["smoke_error"] = "invalid", smoke
        if "status" in m:
            (self.dir / "measurement.json").write_text(json.dumps(m, indent=1))
            return m
        recs: dict[str, list[dict]] = {c: [] for c in ids}
        run_dirs = []
        for r in range(self.args.final_repeats):
            log(f"measuring repeat {r + 1}/{self.args.final_repeats} on {len(ids)} eval cases ...")
            res = self.run_cases(f"final_{r + 1}", ids)
            run_dirs.append(self.dir / f"final_{r + 1}")
            by = {str(c.case_id): c for c in res.per_case}
            for cid in ids:
                cr = by.get(cid)
                recs[cid].append(compact(cr.details if cr else None, cr.score if cr else 0.0))
            log(f"   repeat {r + 1} composite {res.score:.4f}")
        m["status"] = "invalid" if self.route_violations else "ok"
        if self.route_violations:
            m["route_violations"] = sorted(set(self.route_violations))[:5]
        m["records"] = recs
        m.update(summarize(ids, recs, control))
        m["llm_calls_per_case"] = llm_calls_per_case(run_dirs)
        (self.dir / "measurement.json").write_text(json.dumps(m, indent=1))
        return m


def _history_trim(history: list[dict[str, Any]], keep_last: int = 8, max_chars: int = 700_000) -> None:
    if sum(len(str(h.get("output", "")) + str(h.get("content", ""))) for h in history) <= max_chars:
        return
    outs = [i for i, h in enumerate(history) if h.get("type") == "function_call_output"]
    for i in outs[:-keep_last]:
        history[i]["output"] = "[older tool output trimmed to save context]"


def llm_calls_per_case(run_dirs: list[Path]) -> Optional[float]:
    n_calls = n_cases = 0
    for rd in run_dirs:
        tp = rd / "logs" / "trace.jsonl"
        if not tp.exists():
            continue
        seen = set()
        for line in tp.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "llm_call":
                n_calls += 1
                seen.add((e.get("payload") or {}).get("case_id"))
        n_cases += len(seen)
    return n_calls / n_cases if n_cases else None


# --------------------------------------------------------------------------- #
# Summary statistics and report
# --------------------------------------------------------------------------- #

def dimension_rates(recs: dict[str, list[dict]], ids: list[str]) -> dict[str, float]:
    """Share of (case, repeat) plans in which a dimension fully passes (no plan counts as failing); hard = every hard constraint passes."""
    out: dict[str, list[int]] = {}
    for c in ids:
        for r in recs[c]:
            dims = {l.split(":")[1] for l in r["applicable"] if l.startswith("commonsense:")}
            for d in dims or set():
                out.setdefault(d, []).append(0 if r["no_plan"] or any(l.startswith(f"commonsense:{d}:") for l in r["failed"]) else 1)
            if any(l.startswith("hard:") for l in r["applicable"]):
                out.setdefault("HARD (all constraints)", []).append(0 if r["no_plan"] or any(l.startswith("hard:") for l in r["failed"]) else 1)
    return {k: statistics.fmean(v) for k, v in out.items()}


def check_rate(recs: dict[str, list[dict]], ids: list[str], label: str) -> Optional[float]:
    vals = []
    for c in ids:
        for r in recs[c]:
            if r["no_plan"]:
                if label in {l for x in recs[c] for l in x["applicable"]}:
                    vals.append(0)
            elif label in r["applicable"]:
                vals.append(0 if label in r["failed"] else 1)
    return statistics.fmean(vals) if vals else None


def summarize(ids: list[str], recs: dict[str, list[dict]], control: dict[str, list[dict]]) -> dict:
    per_v = {c: statistics.fmean(r["score"] for r in recs[c]) for c in ids}
    per_c = {c: statistics.fmean(r["score"] for r in control[c]) for c in ids}
    d = [per_v[c] - per_c[c] for c in ids]
    lo, hi = boot_ci(d, seed=11)
    n_runs = len(next(iter(recs.values())))
    out: dict[str, Any] = {
        "n_cases": len(ids), "repeats": n_runs, "variant_composite": statistics.fmean(per_v.values()),
        "control_composite": statistics.fmean(per_c.values()), "paired_diff": statistics.fmean(d), "ci95": [lo, hi],
        "per_run_composite": [statistics.fmean(recs[c][r]["score"] for c in ids) for r in range(n_runs)],
        "variant_no_plan_rate": statistics.fmean(x["no_plan"] for c in ids for x in recs[c]),
        "control_no_plan_rate": statistics.fmean(x["no_plan"] for c in ids for x in control[c]),
        "cases_better": sum(x > 0.02 for x in d), "cases_worse": sum(x < -0.02 for x in d),
        "dimension_pass_variant": dimension_rates(recs, ids), "dimension_pass_control": dimension_rates(control, ids),
    }
    labels = sorted({l for c in ids for x in control[c] for l in x["applicable"]})
    pc = {}
    for lab in labels:
        a, b = check_rate(recs, ids, lab), check_rate(control, ids, lab)
        if a is not None and b is not None:
            pc[lab] = {"variant": a, "control": b}
    out["per_check"] = pc
    return out


def build_report(out: Path, m: dict, dev: dict, args: argparse.Namespace, live: str, rel: dict) -> str:
    t = ["# Harness redesign experiment", "",
         f"Meta-agent {META_MODEL} redesigned the harness (thesis: code owns everything deterministic, the weak LLM only small verifiable subtasks). "
         f"Task agent {TASK_MODEL}; vLLM {live}. Development: {dev.get('turns_used')} turns, {dev.get('eval_calls_used')} in-loop evaluations, "
         f"submitted={dev.get('submitted')}. In-loop evaluations (TRAIN sample): "
         + "; ".join(f"call {r['call']}: composite {r['composite']:.3f} vs baseline pass 1 {r['baseline_pass1']:.3f}, no-plan {r['no_plan']}" for r in dev.get("inloop_rounds", [])), ""]
    if m.get("status") != "ok":
        t += [f"**Measurement status: {m.get('status')}** {m.get('policy_problems') or m.get('smoke_error') or m.get('route_violations') or ''}"]
        return "\n".join(t)
    t += ["## Overall (all held-out EVAL cases)", "",
          "| | Composite | No-plan rate |", "|---|---|---|",
          f"| Stored baseline ({args.baseline_dir.name}, {m['n_cases']} cases) | {m['control_composite']:.4f} | {m['control_no_plan_rate']:.3f} |",
          f"| Redesigned harness ({m['repeats']} fresh repeats) | **{m['variant_composite']:.4f}** | {m['variant_no_plan_rate']:.3f} |", "",
          f"Paired per-case difference: **{m['paired_diff']:+.4f}**, 95% bootstrap CI [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]. "
          f"Per-run composites: {[round(x, 4) for x in m['per_run_composite']]}. Cases better / worse by more than 0.02: {m['cases_better']} / {m['cases_worse']}.", "",
          "## Dimension pass rates (share of plans where the dimension fully passes; no plan = fail)", "",
          "| Dimension | Baseline | Redesign | Diff |", "|---|---|---|---|"]
    for d, v in sorted(m["dimension_pass_variant"].items()):
        c = m["dimension_pass_control"].get(d, float("nan"))
        t.append(f"| {d} | {c:.2f} | {v:.2f} | {v - c:+.2f} |")
    t += ["", "## Per-check success (largest changes)", "", "| Check | Baseline | Redesign | Diff |", "|---|---|---|---|"]
    rows = sorted(m["per_check"].items(), key=lambda kv: kv[1]["variant"] - kv[1]["control"])
    shown = rows if len(rows) <= 16 else rows[:8] + [(None, None)] + rows[-8:]
    for lab, v in shown:
        if lab is None:
            t.append("| ... | | | |")
            continue
        t.append(f"| {lab} | {v['control']:.2f} | {v['variant']:.2f} | {v['variant'] - v['control']:+.2f} |")
    pr = m["profile"]
    t += ["", "## What the variant changed", "",
          f"- files changed: {pr['files_changed']}; files added: {pr['files_added']}",
          f"- LLM call sites in the code: {pr['llm_call_sites']} (seed: {pr['llm_call_sites_seed']}); code statements delta: {pr['code_statements_delta']:+d}",
          f"- LLM calls per case (measured): {m.get('llm_calls_per_case') and round(m['llm_calls_per_case'], 1)}",
          f"- meta-agent summary: {dev.get('summary', '')[:1500]}", "",
          f"## Local-vLLM reliability (all task-agent traces of the measurement runs)\n\n{rel}",
          "", "Feedback mode: in-session evaluate_variant on TRAIN cases (a disclosed deviation from HGM's later-round feedback); no eval-split information reaches the meta-agent."]
    return "\n".join(t)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--eval-rounds", type=int, default=5, help="evaluate_variant calls")
    ap.add_argument("--inloop-cases", type=int, default=8, help="TRAIN cases per evaluate_variant call")
    ap.add_argument("--final-repeats", type=int, default=3)
    ap.add_argument("--parallelism", type=int, default=15, help="evaluator case parallelism")
    ap.add_argument("--case-timeout", type=float, default=3000.0, help="per-case wall-time cap in seconds")
    ap.add_argument("--redo", action="store_true", help="discard an existing development / measurement in --out-dir")
    args = ap.parse_args()
    assert args.max_turns >= 8, "--max-turns must be >= 8"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out}")

    cfg = cfg_mod.load(str(CONFIG))
    runtime_env.apply_all(cfg)
    live = assert_local_only(args.baseline_dir)
    fw = cfg_mod.build_components(cfg)
    fw.evaluator.parallelism = args.parallelism
    fw.evaluator.wall_time_s = args.case_timeout

    tj = out / "targets.json"
    info = json.loads(tj.read_text()) if tj.exists() and not args.redo else prepare(args, fw, out)
    sess = Session(out, fw, args, info)

    dj = sess.dir / "develop.json"
    if dj.exists() and sess.ws.exists() and not args.redo:
        dev = json.loads(dj.read_text())
        sess.eval_log, sess.rounds_used = dev.get("inloop_rounds", []), dev.get("eval_calls_used", 0)
        log("development already done; measuring")
    else:
        sess.setup()
        log(f"developing (meta-agent {META_MODEL}) ...")
        t0 = time.time()
        dev = sess.develop()
        log(f"develop done in {time.time() - t0:.0f}s: submitted={dev['submitted']} turns={dev['turns_used']} eval_calls={dev['eval_calls_used']} "
            f"meta_call_failed={dev['meta_call_failed']}")
    sess.route_violations = list(dev.get("route_violations", []))

    mp = sess.dir / "measurement.json"
    m = json.loads(mp.read_text()) if mp.exists() and not args.redo else sess.measure()
    tot = {"n_llm_responses": 0, "n_status_failed_retries": 0, "n_exception_retries": 0, "n_terminal_failed_responses": 0}
    for tr in (sess.dir).glob("final_*/logs/trace.jsonl"):
        fh = analyze_trace_file(tr)
        for k in tot:
            tot[k] += fh.get(k, 0)
    tot["incidence_rate_pct"] = incidence_rate_pct(tot) if tot["n_llm_responses"] else 0.0
    rep = build_report(out, m, dev, args, live, tot)
    (out / "REPORT.md").write_text(rep)
    (out / "summary.json").write_text(json.dumps({k: v for k, v in m.items() if k != "records"}, indent=1))
    print("\n" + rep)
    print(f"\noutputs: {out}")


if __name__ == "__main__":
    main()

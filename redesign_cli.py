#!/usr/bin/env python3
"""CLI environment that exposes the SAME tools the Qwen meta-agent gets in experiment_harness_redesign.py to an external meta-agent
(e.g. a Claude subagent working through a shell), with the same allow-list virtual filesystem, edit policy, offline run_python sandbox and
evaluate_variant. The environment enforces the information rule: only the TRAIN corpus (one baseline pass), the live workspace and the semantics files
are readable; the gold scorer's source, benchmark data and every eval-split log are unreachable.

Agent-facing commands (all operate on --out-dir):
  list_cases [--failed-check S] [--limit N]      show_case CASE_ID       read PATH [--offset N] [--limit N]      grep PATTERN [--path GLOB] [--max N]
  write PATH            (new full file content on stdin)              replace PATH   (stdin: OLD text, a line '=====>>>=====', NEW text)
  python [--timeout S]  (code on stdin)                               eval [--wait S]    submit --summary TEXT      status
Operator-only commands: init, measure.   Internal: _eval_worker.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import experiment_harness_redesign as h  # noqa: E402
from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402

SPLIT = "=====>>>====="
MAX_EVALS = 5


def state_path(out: Path) -> Path:
    return out / "cli_state.json"


def load_state(out: Path) -> dict:
    p = state_path(out)
    return json.loads(p.read_text()) if p.exists() else {"rounds_used": 0, "py_calls": 0, "eval_log": [], "pending": None, "submitted": False,
                                                          "summary": "", "n_cmds": 0, "max_evals": MAX_EVALS, "budget": 150}


def save_state(out: Path, st: dict) -> None:
    state_path(out).write_text(json.dumps(st, indent=1))


def make_args(ns: argparse.Namespace, max_evals: int = MAX_EVALS) -> argparse.Namespace:
    return argparse.Namespace(baseline_dir=h.DEFAULT_BASELINE, seed=ns.seed, max_turns=150, eval_rounds=max_evals, inloop_cases=ns.inloop_cases,
                              final_repeats=3, parallelism=15, case_timeout=3000.0, redo=False)


def build_session(out: Path, ns: argparse.Namespace, need_fw: bool = True) -> h.Session:
    st0 = load_state(out)
    args = make_args(ns, st0.get("max_evals", MAX_EVALS))
    fw = None
    if need_fw:
        cfg = cfg_mod.load(str(h.CONFIG))
        runtime_env.apply_all(cfg)
        fw = cfg_mod.build_components(cfg)
        fw.evaluator.parallelism = args.parallelism
        fw.evaluator.wall_time_s = args.case_timeout
    info = json.loads((out / "targets.json").read_text())
    sess = h.Session(out, fw, args, info)
    st = load_state(out)
    sess.rounds_used, sess.eval_log, sess.py_calls = st["rounds_used"], st["eval_log"], st["py_calls"]
    return sess


def vfs(out: Path) -> h.Vfs:
    return h.Vfs(out / "corpus", out / "work" / "workspace")


def do_write(sess_ws: Path, orig: Path, rel: str, content: str) -> str:
    rel = rel.strip().lstrip("./").lstrip("/")
    rel = rel[len("workspace/"):] if rel.startswith("workspace/") else rel
    probs = h.edit_problems(rel, content, orig)
    if probs:
        return "EDIT REJECTED (file unchanged): " + "; ".join(probs)
    target = sess_ws / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {rel} ({len(content)} chars, policy OK)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inloop-cases", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("init_from"); p.add_argument("--start-workspace", required=True); p.add_argument("--max-evals", type=int, default=5); p.add_argument("--budget", type=int, default=150)
    sub.add_parser("status")
    p = sub.add_parser("list_cases"); p.add_argument("--failed-check", default=""); p.add_argument("--limit", type=int, default=200)
    p = sub.add_parser("show_case"); p.add_argument("case_id")
    p = sub.add_parser("read"); p.add_argument("path"); p.add_argument("--offset", type=int, default=0); p.add_argument("--limit", type=int, default=12000)
    p = sub.add_parser("grep"); p.add_argument("pattern"); p.add_argument("--path", default="corpus/cases/*"); p.add_argument("--max", type=int, default=40)
    p = sub.add_parser("write"); p.add_argument("path")
    p = sub.add_parser("replace"); p.add_argument("path")
    p = sub.add_parser("python"); p.add_argument("--timeout", type=int, default=120)
    p = sub.add_parser("eval"); p.add_argument("--wait", type=int, default=540)
    p.add_argument("--case-ids", nargs="*", default=None,
                    help="specific TRAIN case ids to evaluate instead of all 60 (faster, for iterating on one failure mode)")
    p = sub.add_parser("submit"); p.add_argument("--summary", default="")
    p = sub.add_parser("_eval_worker"); p.add_argument("--case-ids", nargs="*", default=None)
    sub.add_parser("measure")
    ns = ap.parse_args()
    out = ns.out_dir.resolve()

    if ns.cmd == "init":
        out.mkdir(parents=True, exist_ok=True)
        cfg = cfg_mod.load(str(h.CONFIG)); runtime_env.apply_all(cfg)
        h.assert_local_only(h.DEFAULT_BASELINE)
        fw = cfg_mod.build_components(cfg)
        args = make_args(ns)
        info = h.prepare(args, fw, out)
        sess = h.Session(out, fw, args, info)
        sess.setup()
        save_state(out, load_state(out))
        print(f"initialised {out}: workspace at {sess.ws}, {len(info['inloop_train_cases'])} in-loop train cases, {len(info['final_eval_cases'])} held-out eval cases (hidden)")
        return

    if ns.cmd == "init_from":
        import experiment_harness_staged as sg
        out.mkdir(parents=True, exist_ok=True)
        cfg = cfg_mod.load(str(h.CONFIG)); runtime_env.apply_all(cfg)
        h.assert_local_only(h.DEFAULT_BASELINE)
        fw = cfg_mod.build_components(cfg)
        args = make_args(ns, ns.max_evals)
        start = Path(ns.start_workspace).resolve()
        probs = h.workspace_problems(start, Path(fw.seed_dir))
        assert not probs, f"start workspace violates the edit policy: {probs[:3]}"
        assert h.run_smoke(start) is None, "start workspace does not import"
        train_ids = [str(x) for x in fw.train_case_ids]
        h.log("evaluating the starting workspace on all 60 train cases (its own graded plans become the corpus) ...")
        res = sg.eval_train(fw, start, out / "eval_start", train_ids)
        info = sg.build_stage_corpus(out, res, h.DEFAULT_BASELINE, out / "eval_start", ns.seed, ns.inloop_cases)
        sess = h.Session(out, fw, args, info)
        sess.setup(start_workspace=start)
        stt = load_state(out); stt["max_evals"], stt["budget"] = ns.max_evals, ns.budget; save_state(out, stt)
        cs = sg.summarize_eval(res)
        print(f"initialised {out} from {start}: start composite on train {cs['composite']:.4f}, no-plan {cs['no_plan_rate']:.3f}; "
              f"{len(info['inloop_train_cases'])} in-loop train cases; budgets: {ns.budget} commands, {ns.max_evals} evaluations")
        return

    st = load_state(out)
    if ns.cmd not in ("_eval_worker", "measure"):
        st["n_cmds"] += 1
        save_state(out, st)

    if ns.cmd == "status":
        me = st.get("max_evals", MAX_EVALS)
        left = me - st["rounds_used"]
        print(f"commands used: {st['n_cmds']} (budget {st.get('budget', 150)}) | evaluate_variant calls used: {st['rounds_used']}/{me} ({left} left) | "
              f"run_python calls: {st['py_calls']} | pending evaluation: {bool(st['pending'])} | submitted: {st['submitted']}")
        for r in st["eval_log"]:
            print(f"  eval {r['call']}: composite {r['composite']:.3f} (baseline pass 1 {r['baseline_pass1']:.3f}), no-plan {r['no_plan']}")
        return
    if ns.cmd == "list_cases":
        print(h._trim(h.tool_list_cases(vfs(out), {"failed_check": ns.failed_check, "limit": ns.limit}))); return
    if ns.cmd == "show_case":
        print(h._trim(h.tool_show_case(vfs(out), {"case_id": ns.case_id}))); return
    if ns.cmd == "read":
        try:
            print(h._trim(h.tool_read_file(vfs(out), {"path": ns.path, "offset": ns.offset, "limit": ns.limit})))
        except FileNotFoundError as e:
            print(f"tool error: {e}")
        return
    if ns.cmd == "grep":
        print(h._trim(h.tool_grep(vfs(out), {"pattern": ns.pattern, "path": ns.path, "max_matches": ns.max}))); return
    ws, orig = out / "work" / "workspace", out / "work" / "orig"
    if ns.cmd == "write":
        print(do_write(ws, orig, ns.path, sys.stdin.read())); return
    if ns.cmd == "replace":
        rel = ns.path.strip().lstrip("./").lstrip("/")
        rel = rel[len("workspace/"):] if rel.startswith("workspace/") else rel
        target = ws / rel
        if not target.exists():
            print(f"no such file {rel}"); return
        raw = sys.stdin.read()
        if SPLIT not in raw:
            print(f"stdin must be: OLD text, a line '{SPLIT}', NEW text"); return
        old_s, new_s = raw.split(SPLIT, 1)
        old_s, new_s = old_s.strip("\n"), new_s.strip("\n")
        cur = target.read_text()
        if cur.count(old_s) != 1:
            print(f"old text must occur exactly once in {rel}, found {cur.count(old_s)}; nothing changed"); return
        print(do_write(ws, orig, rel, cur.replace(old_s, new_s))); return
    if ns.cmd == "python":
        sess = build_session(out, ns)
        res = h._trim(sess.run_python(sys.stdin.read(), ns.timeout))
        st = load_state(out); st["py_calls"] = sess.py_calls; save_state(out, st)
        print(res); return
    if ns.cmd == "submit":
        if st["rounds_used"] == 0:
            print("NOT submitted: you have not called `eval` yet. Call it first, fix what it shows, then submit."); return
        if st["pending"]:
            print("NOT submitted: an evaluation is still running; wait for it (`eval`) first."); return
        st["submitted"], st["summary"] = True, ns.summary[:3000]; save_state(out, st)
        print("submitted"); return
    if ns.cmd == "_eval_worker":
        sess = build_session(out, ns)
        text = sess.evaluate_variant(ns.case_ids)
        st = load_state(out)
        st["rounds_used"], st["eval_log"], st["pending"] = sess.rounds_used, sess.eval_log, None
        save_state(out, st)
        (out / f"eval_result_{st['rounds_used'] if st['rounds_used'] else 0}_{int(time.time())}.txt").write_text(text)
        (out / "eval_last.txt").write_text(text)
        return
    if ns.cmd == "eval":
        if not st["pending"]:
            if st["rounds_used"] >= st.get("max_evals", MAX_EVALS):
                print(f"no evaluation calls left (limit {st.get('max_evals', MAX_EVALS)}); refine if you must, then submit"); return
            (out / "eval_last.txt").unlink(missing_ok=True)
            st["pending"] = {"started": time.time()}; save_state(out, st)
            worker_cmd = [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out), "--seed", str(ns.seed),
                          "--inloop-cases", str(ns.inloop_cases), "_eval_worker"]
            if ns.case_ids:
                worker_cmd += ["--case-ids", *ns.case_ids]
            subprocess.Popen(worker_cmd, start_new_session=True, stdout=open(out / "eval_worker.log", "a"), stderr=subprocess.STDOUT)
            print(f"evaluation started (real end-to-end run on {len(ns.case_ids) if ns.case_ids else 'ALL 60'} TRAIN case(s); takes a while).")
        deadline = time.time() + ns.wait
        while time.time() < deadline:
            p = out / "eval_last.txt"
            if p.exists():
                print(h._trim(p.read_text())); return
            time.sleep(5)
        print("still running -- call `eval` again to keep waiting (this does NOT consume another evaluation call).")
        return
    if ns.cmd == "measure":   # operator only
        import experiment_harness_redesign as hh
        tj = out / "targets.json"
        info0 = json.loads(tj.read_text())
        if not info0.get("final_eval_cases"):   # started via init_from: add the held-out eval targets (operator side only)
            cfg = cfg_mod.load(str(h.CONFIG)); runtime_env.apply_all(cfg)
            fwx = cfg_mod.build_components(cfg)
            tmp = out / "prep_tmp"; tmp.mkdir(exist_ok=True)
            pinfo = h.prepare(make_args(ns), fwx, tmp)
            info0["final_eval_cases"], info0["stored_control"] = pinfo["final_eval_cases"], pinfo["stored_control"]
            tj.write_text(json.dumps(info0))
        sess = build_session(out, ns)
        dev = {"submitted": st["submitted"], "summary": st["summary"], "turns_used": st["n_cmds"], "eval_calls_used": st["rounds_used"],
               "inloop_rounds": st["eval_log"], "route_violations": []}
        m = sess.measure()
        tot = {"n_llm_responses": 0, "n_status_failed_retries": 0, "n_exception_retries": 0, "n_terminal_failed_responses": 0}
        for tr in sess.dir.glob("final_*/logs/trace.jsonl"):
            fh = hh.analyze_trace_file(tr)
            for k in tot:
                tot[k] += fh.get(k, 0)
        tot["incidence_rate_pct"] = hh.incidence_rate_pct(tot) if tot["n_llm_responses"] else 0.0
        live = json.dumps(hh._http_json(__import__("os").environ["LLM_BASE_URL"].rsplit("/v1", 1)[0] + "/version"))
        rep = hh.build_report(out, m, dev, make_args(ns), live, tot)
        rep = rep.replace("Meta-agent " + hh.META_MODEL, "External meta-agent (Claude subagent working through redesign_cli.py; same tools, same information)")
        (out / "REPORT.md").write_text(rep)
        (out / "summary.json").write_text(json.dumps({k: v for k, v in m.items() if k != "records"}, indent=1))
        print(rep); return


if __name__ == "__main__":
    main()

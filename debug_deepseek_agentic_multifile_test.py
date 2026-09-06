"""Follow-up to debug_deepseek_agentic_prototype.py: does the winning
agentic read_file/write_file/run_code_validators architecture (0/8 malformed
writes on a single-file fix) hold when a trial genuinely requires writing
DIFFERENT content to MULTIPLE files, not just repeated attempts at one file?

Reuses node 31's real historical fix scope from the earlier DeepSeek sanity
run (mas_workflow.py + agents/flight.py + agents/train.py +
agents/sightseeing.py, 0.754 score) as the task: add a one-step recovery
loop so a failed Sightseeing stage retries with diagnostic context instead
of terminating empty, and propagate upstream Flight/Train failures for
detection -- a genuine coordinated 4-file change, each file needing
DIFFERENT edits (not the same comment-add to all 4, which the earlier
non-agentic bundled-call A/B test already showed was trivially easy at 8/8
valid JSON).

Same manual-history-accumulation mechanism as the single-file prototype
(never previous_response_id, never raw response.output replay -- both
confirmed broken against OpenRouter's Responses-API layer).

Usage: python3 debug_deepseek_agentic_multifile_test.py <n_trials>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
from openai import OpenAI
from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path("/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
SCRATCH_ROOT = Path(
    "/tmp/claude-10210/-groups-AIC-MV-v-kulkarni1/0612ae5b-7995-4127-b44c-d9799589290f"
    "/scratchpad/agentic_multifile_trials"
)

cfg = cfg_mod.load(str(REPO_ROOT / "configs/hgm_travel_deepseek_editor_sanity_1case.yaml"))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

MODEL = "deepseek/deepseek-v4-pro-0813"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_TURNS = 28
TARGET_FILES = [
    "mas_workflow.py",
    "agents/flight.py",
    "agents/train.py",
    "agents/sightseeing.py",
]

SYSTEM = (
    "You are the self-improvement module of a self-evolving agent. You "
    "have tools: read_file(path) to view a file's current content, "
    "write_file(path, content) to submit ONE file's full new content "
    "(call this once per file you change -- never bundle multiple files "
    "into one call), run_code_validators() to check your current changes "
    "for syntax/import/signature errors before finishing, and "
    "submit_done(optimization_goal, proposed_changes, rationale) to finish "
    "once validators pass.\n\n"
    "Diagnosis: when the Sightseeing stage fails outright, the workflow "
    "just terminates with an empty result instead of retrying with "
    "diagnostic context about what went wrong. Separately, when the Flight "
    "or Train stage fails, that failure is swallowed silently instead of "
    "being signalled upstream, so later stages and the orchestrator can't "
    "detect or react to it.\n\n"
    "Fix this with a genuinely coordinated change across all 4 of these "
    "files (each needs DIFFERENT edits -- this is not the same change "
    "copy-pasted 4 times):\n"
    "  1. agents/flight.py -- when the flight stage fails, make the failure "
    "explicit/detectable in its return value/state instead of failing "
    "silently.\n"
    "  2. agents/train.py -- same as flight.py: make train-stage failure "
    "explicit/detectable instead of silent.\n"
    "  3. agents/sightseeing.py -- add a one-step retry when the "
    "sightseeing stage fails, including diagnostic context about the "
    "failure (including any upstream flight/train failure signal) in the "
    "retry attempt, instead of terminating with an empty result.\n"
    "  4. mas_workflow.py -- wire it together: pass upstream flight/train "
    "failure signals into the sightseeing stage call, and make sure the "
    "orchestrator itself can see/react to a stage failure.\n\n"
    "You have a LIMITED number of turns. Do NOT explore the codebase "
    "broadly -- you only need exactly these 4 files, nothing else. For "
    "each of the 4 files: call read_file ONCE, then write_file ONCE with "
    "your corrected version. After all 4 are written, call "
    "run_code_validators ONCE; if it flags anything, fix it with one more "
    "write_file call to the relevant file(s), then call submit_done. Do "
    "not read or write any file outside this list of 4, and do not call "
    "run_code_validators before you have written all 4 files."
)

TOOLS = [
    {
        "type": "function", "name": "read_file",
        "description": "Read a file's current content from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "type": "function", "name": "write_file",
        "description": "Submit ONE file's full new content. Call once per changed file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "type": "function", "name": "run_code_validators",
        "description": "Run the real validator suite (syntax, imports, signatures, etc.) against your current changes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function", "name": "submit_done",
        "description": "Finish the self-improvement after all files are written and validators pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "optimization_goal": {"type": "string"},
                "proposed_changes": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["optimization_goal", "proposed_changes", "rationale"],
        },
    },
]


def one_trial(trial_idx: int) -> dict:
    scratch = SCRATCH_ROOT / f"trial_{trial_idx}"
    if scratch.exists():
        shutil.rmtree(scratch)
    base_dir = scratch / "base"
    out_dir = scratch / "out"
    base_dir.mkdir(parents=True)
    shutil.copytree(fw.seed_dir, base_dir / "task_agent")
    shutil.copytree(fw.seed_dir, out_dir / "task_agent")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL, max_retries=0)
    history: list = [{"role": "user", "content": SYSTEM + "\n\nBegin."}]
    write_file_results = []
    turn_log = []
    done = False

    for turn in range(MAX_TURNS):
        print(f"    [trial {trial_idx}] turn {turn}: calling API...", flush=True)
        t0 = time.time()
        try:
            response = client.responses.create(
                model=MODEL, input=history, tools=TOOLS,
                reasoning={"effort": "medium"}, max_output_tokens=16384,
            )
        except Exception as exc:
            turn_log.append(f"turn {turn}: api_error {exc!r}")
            break
        print(f"    [trial {trial_idx}] turn {turn}: got response in {time.time()-t0:.1f}s", flush=True)

        function_calls = [item for item in (response.output or []) if getattr(item, "type", None) == "function_call"]
        if not function_calls:
            turn_log.append(f"turn {turn}: no tool call")
            break

        for fc in function_calls:
            history.append({
                "type": "function_call",
                "call_id": fc.call_id,
                "name": fc.name,
                "arguments": fc.arguments,
            })
        for fc in function_calls:
            name = fc.name
            raw_args = fc.arguments
            try:
                args = json.loads(raw_args)
                parse_ok = True
            except json.JSONDecodeError as e:
                parse_ok = False
                args = {}
                turn_log.append(f"turn {turn}: {name} MALFORMED_JSON pos={e.pos}/{len(raw_args)} msg={e.msg}")

            if name == "read_file" and parse_ok:
                path = (args.get("path") or "").lstrip("/")
                fpath = out_dir / "task_agent" / path
                content = fpath.read_text() if fpath.exists() else f"(file not found: {path})"
                output = content
            elif name == "write_file":
                if parse_ok:
                    path = (args.get("path") or "").lstrip("/")
                    content = args.get("content") or ""
                    fpath = out_dir / "task_agent" / path
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(content)
                    write_file_results.append((path, "valid_json", len(content)))
                    output = f"written {path} ({len(content)} chars)"
                else:
                    write_file_results.append((raw_args[:60], "malformed_json", len(raw_args)))
                    output = "ERROR: your arguments were not valid JSON. Try again with careful escaping."
            elif name == "run_code_validators":
                errors = fw.editor._run_validators(out_dir, base_dir)
                output = "All validators passed." if not errors else "Validator errors:\n" + "\n".join(f"- {e}" for e in errors)
            elif name == "submit_done":
                done = True
                output = "acknowledged" if parse_ok else "ERROR: malformed JSON in submit_done"
                if not parse_ok:
                    turn_log.append(f"turn {turn}: submit_done MALFORMED_JSON")
            else:
                output = "unknown tool"

            history.append({"type": "function_call_output", "call_id": fc.call_id, "output": output})
            print(f"    [trial {trial_idx}] turn {turn}: {name}(parse_ok={parse_ok}) -> {output[:100]!r}", flush=True)

        if done:
            break

    files_written = sorted({p for p, outcome, _ in write_file_results if outcome == "valid_json"})
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "trial": trial_idx,
        "write_file_calls": write_file_results,
        "distinct_files_written": files_written,
        "all_4_written": set(files_written) == set(TARGET_FILES),
        "turn_log": turn_log,
        "done": done,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    all_write_outcomes = []
    all_4_count = 0
    for i in range(n):
        r = one_trial(i)
        print(f"=== trial {i+1}/{n} (done={r['done']}, all_4_written={r['all_4_written']}) ===", flush=True)
        print(f"  distinct files written: {r['distinct_files_written']}", flush=True)
        print(f"  write_file calls: {r['write_file_calls']}", flush=True)
        if r["turn_log"]:
            print(f"  turn log: {r['turn_log']}", flush=True)
        all_write_outcomes.extend(o for _, o, _ in r["write_file_calls"])
        if r["all_4_written"]:
            all_4_count += 1

    print()
    print("=== SUMMARY: write_file call outcomes across all trials ===")
    print(dict(Counter(all_write_outcomes)))
    print(f"Trials that wrote all 4 target files: {all_4_count}/{n}")

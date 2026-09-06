"""Test whether agentic read_file/write_file/run_code_validators tool
access (one file per call, in a multi-turn loop, with real validator
feedback) reduces DeepSeek's malformed-JSON rate, vs. bundling all changed
files' full content into one `files: [...]` array in a single tool call
(the current agent_editor.py shape, confirmed to fail ~65% of the time on
the real prompt, and to still fail 5/5 even with strict:true).

Key structural difference under test: each write_file call carries exactly
ONE path + ONE content string, nothing nested alongside it. Also gives the
model a real read_file (actual seed source) and run_code_validators (the
REAL configured validator stack, via fw.editor._run_validators against a
real on-disk workspace) so it can self-correct before finishing, mirroring
how a real coding agent iterates.

Usage: python3 deepseek_agentic_test.py <n_trials>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
from openai import OpenAI
from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path("/groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual")
SCRATCH_ROOT = Path("/tmp/claude-10210/-groups-AIC-MV-v-kulkarni1/0612ae5b-7995-4127-b44c-d9799589290f/scratchpad/agentic_trials")

cfg = cfg_mod.load(str(REPO_ROOT / "configs/hgm_travel_deepseek_editor_sanity_1case.yaml"))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

MODEL = "deepseek/deepseek-v4-pro-0813"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_TURNS = 16

SYSTEM = (
    "You are the self-improvement module of a self-evolving agent. You "
    "have tools: read_file(path) to view a file's current content, "
    "write_file(path, content) to submit ONE file's full new content "
    "(call this once per file you change -- never bundle multiple files "
    "into one call), run_code_validators() to check your current changes "
    "for syntax/import/signature errors before finishing, and "
    "submit_done(optimization_goal, proposed_changes, rationale) to finish "
    "once validators pass.\n\n"
    "Diagnosis: the sightseeing stage's wrap-up retry sometimes produces "
    "an <itinerary> tag with placeholder attraction/restaurant names "
    "instead of ones verified via tool results, because the retry drops "
    "the accumulated tool-call context. Fix agents/sightseeing.py so the "
    "retry keeps a summary of key extracted entity names/prices from the "
    "tool results and includes them in its wrap-up prompt.\n\n"
    "You have a LIMITED number of turns. Do NOT explore the codebase "
    "broadly -- you only need agents/sightseeing.py, which is the only "
    "file you need to change. Call read_file('agents/sightseeing.py') "
    "ONCE, then immediately call write_file with your corrected version, "
    "then run_code_validators ONCE, fix anything it flags with one more "
    "write_file call if needed, then call submit_done. Do not read any "
    "other file and do not call run_code_validators before you have "
    "written at least one file."
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
    # OpenRouter's Responses-API layer supports neither replaying raw
    # response.output items (rejects "reasoning" item shapes) nor
    # previous_response_id (rejects any non-null value) -- so history must
    # be manually accumulated using only the well-defined item types:
    # {"role": "user"/"assistant", "content": str} and manually
    # reconstructed {"type": "function_call", ...} / {"type":
    # "function_call_output", ...} pairs. Reasoning items are simply
    # omitted from the replay -- not required for the model to continue.
    history: list = [{"role": "user", "content": SYSTEM + "\n\nBegin."}]
    write_file_results = []
    turn_log = []
    done = False

    for turn in range(MAX_TURNS):
        print(f"    [trial {trial_idx}] turn {turn}: calling API...", flush=True)
        t0 = __import__("time").time()
        try:
            response = client.responses.create(
                model=MODEL, input=history, tools=TOOLS,
                reasoning={"effort": "medium"}, max_output_tokens=16384,
            )
        except Exception as exc:
            turn_log.append(f"turn {turn}: api_error {exc!r}")
            break
        print(f"    [trial {trial_idx}] turn {turn}: got response in {__import__('time').time()-t0:.1f}s", flush=True)

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

    shutil.rmtree(scratch, ignore_errors=True)
    return {"trial": trial_idx, "write_file_calls": write_file_results, "turn_log": turn_log, "done": done}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    all_write_outcomes = []
    for i in range(n):
        r = one_trial(i)
        print(f"=== trial {i+1}/{n} (done={r['done']}) ===", flush=True)
        print(f"  write_file calls: {r['write_file_calls']}", flush=True)
        if r["turn_log"]:
            print(f"  turn log: {r['turn_log']}", flush=True)
        all_write_outcomes.extend(o for _, o, _ in r["write_file_calls"])

    print()
    print("=== SUMMARY: write_file call outcomes across all trials ===")
    print(dict(Counter(all_write_outcomes)))

#!/usr/bin/env python3
"""Communication-trace export for `projects/travel_mas` specifically.

`travel_mas`'s 4 pipeline stages (Flight/Train/Sightseeing/Accounting) are
plain module-level functions in `seed/workflow.py`, not methods on a class
with a `.name` attribute -- so the generic `platform_core.runner
--patch-agent-run "module:Class.method"` CLI (which assumes a `self`-like
first argument) doesn't fit here, and the plumbing needed
(`patch_agent_function`'s `agent_id`/`prompt_from`/`result_parser`
callbacks) can't be expressed as CLI strings anyway. See
`platform_core/communication_instrumentation.py::patch_agent_function`'s
own docstring for the general mechanism this project-specific script wires
up.

Two modes, mirroring `export_communication_batch.py`'s own shape:

  --case-id ID          WORKER mode: run exactly one case, write one trace
                         JSON. Invoked as a subprocess by driver mode below
                         (each case gets its own clean process -- avoids any
                         ContextVar-across-threads concern entirely, the
                         same reasoning integrating.md flags for db_mas's
                         ThreadPoolExecutor-based specialists).

  (no --case-id)         DRIVER mode: fan out over every case in the
                         benchmark (or --limit/--case-ids-file), each as a
                         WORKER subprocess, bounded by --concurrency.

Usage:
    PYTHONPATH=. python3 export_travel_mas_communication.py \\
        --agent-dir projects/travel_mas/seed \\
        --benchmark projects/travel_mas/benchmark \\
        --out-dir communication_traces/travel_mas \\
        --concurrency 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()


# --------------------------------------------------------------------------- #
# travel_mas-specific extraction callbacks -- see
# platform_core/communication_instrumentation.py::patch_agent_function's
# docstring for what each of these is for.
# --------------------------------------------------------------------------- #


def _stage_prompts(wf_module: Any) -> dict[str, str]:
    return {
        "flight": wf_module.FLIGHT_SYSTEM_PROMPT,
        "train": wf_module.TRAIN_SYSTEM_PROMPT,
        "sightseeing": wf_module.SIGHTSEEING_SYSTEM_PROMPT,
        "accounting": wf_module.ACCOUNTING_SYSTEM_PROMPT,
    }


def _make_agent_id_fn(wf_module: Any):
    by_prompt = _stage_prompts(wf_module)

    def agent_id_from_system_prompt(bound: dict[str, Any]) -> str | None:
        messages = bound.get("messages") or []
        if not messages:
            return None
        first = messages[0]
        sysmsg = first.get("content") if isinstance(first, dict) else None
        for stage, prompt in by_prompt.items():
            if sysmsg == prompt:
                return stage
        return "unknown"

    return agent_id_from_system_prompt


def _prompt_from(bound: dict[str, Any]) -> str | None:
    """Render the full `messages` list as a role-labeled transcript, not
    just the last user-role message. `_run_tool_stage`'s `messages` list
    only ever grows (`.append`/`.extend`, never truncated), so each call's
    list is a superset of every earlier call's within the same stage --
    combined with `patch_agent_function`'s last-call-wins overwrite, the
    recorded prompt ends up being the complete conversation for the whole
    stage, not a single-call snapshot. This also fixes hand-off detection:
    Sightseeing's system/user turns embed Flight's/Train's notes verbatim
    somewhere in the transcript, so `find_producers` (substring match
    against previously registered agent outputs) now always has that text
    to search, regardless of which specific call happened to fire."""
    messages = bound.get("messages") or []
    if not messages:
        return None
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "unknown"
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False) if content is not None else ""
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts) if parts else None


def _result_parser(result: Any) -> dict[str, Any]:
    tool_calls = getattr(result, "tool_calls", None) or []
    # `.strip()`, to match what workflow.py actually splices into the next
    # stage's prompt (e.g. `flight_note = text.strip() or "..."` at
    # workflow.py:278/317/490) -- registering the raw, unstripped content
    # here would make find_producers' exact-substring match silently miss
    # every hand-off, since the embedded copy never has the surrounding
    # whitespace the raw response did.
    content = getattr(result, "content", None)
    if isinstance(content, str):
        content = content.strip()
    return {
        "text": content,
        "tool_calls": [
            {"name": getattr(tc, "name", None), "arguments": getattr(tc, "arguments", None)}
            for tc in tool_calls
        ],
    }


# --------------------------------------------------------------------------- #
# Worker mode
# --------------------------------------------------------------------------- #


def _run_worker(agent_dir: Path, benchmark_dir: Path, case_id: str, out_path: Path) -> None:
    from platform_core.communication_instrumentation import (
        CommunicationRecorder,
        patch_agent_function,
        recording_scope,
    )
    from platform_core.runner import _load_case, _task_from_case  # noqa: SLF001 -- reuse, don't reimplement

    agent_dir = agent_dir.resolve()
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    case = _load_case(benchmark_dir.resolve(), case_id)
    for k, v in (case.get("env") or {}).items():
        os.environ[str(k)] = str(v)
    task = _task_from_case(case)

    import workflow as wf  # the travel_mas seed's own workflow.py, now on sys.path

    recorder = CommunicationRecorder(
        task_id=task.case_id,
        task_prompt=task.description,
        ground_truth=case.get("meta_info"),
    )
    undo = patch_agent_function(
        "workflow:call_llm",
        _make_agent_id_fn(wf),
        prompt_from=_prompt_from,
        result_parser=_result_parser,
    )
    # Sightseeing's hand-off to Accounting is NOT a verbatim splice like
    # Flight/Train's notes into Sightseeing's prompt -- `_extract_itinerary`
    # strips the `<itinerary>`/`</itinerary>` tags out first (workflow.py's
    # own `sightseeing_body = _extract_itinerary(text)`), so the raw
    # response text registered above is no longer a literal substring of
    # what actually lands in Accounting's prompt, and find_producers'
    # exact-substring match silently misses the hand-off.
    #
    # Tried routing this through patch_transform_function (find whichever
    # agent's registered output exactly equals _extract_itinerary's input,
    # then register its output too) -- ran into a real wrapper-ordering
    # bug (a stripping step has to run *before* patch_transform_function's
    # own argument-binding sees the value, not inside the wrapped callable)
    # and even fixed, it's solving a harder problem than this case needs:
    # `_extract_itinerary` is only ever called from within
    # `_run_sightseeing_stage`, so there's no ambiguity about which agent
    # produced its input -- just register the result directly as
    # "sightseeing" output, unconditionally, no producer lookup needed.
    from platform_core.communication_instrumentation import _active as _recorder_var

    _orig_extract_itinerary = wf._extract_itinerary

    def _extract_itinerary_and_register(text):
        result = _orig_extract_itinerary(text)
        if result:
            rec = _recorder_var.get()
            if rec is not None:
                rec.register_output("sightseeing", result)
        return result

    wf._extract_itinerary = _extract_itinerary_and_register
    try:
        with recording_scope(recorder):
            wf.run_task(task)
    finally:
        wf._extract_itinerary = _orig_extract_itinerary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(recorder.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Driver mode
# --------------------------------------------------------------------------- #


def _sanitize(case_id: str) -> str:
    return case_id.replace("/", "__").replace("\\", "__")


def _load_all_case_ids(benchmark_dir: Path) -> list[str]:
    ids: list[str] = []
    with (benchmark_dir / "cases.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            ids.append(str(case.get("id") or case.get("case_id")))
    return ids


def _run_one_subprocess(
    agent_dir: Path, benchmark_dir: Path, case_id: str, out_dir: Path, timeout_s: int
) -> tuple[str, bool, float, str]:
    out_path = out_dir / f"{_sanitize(case_id)}.json"
    cmd = [
        sys.executable, str(THIS_FILE),
        "--agent-dir", str(agent_dir),
        "--benchmark", str(benchmark_dir),
        "--case-id", case_id,
        "--out-path", str(out_path),
    ]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return case_id, False, time.time() - started, f"timed out after {timeout_s}s"
    elapsed = time.time() - started
    ok = proc.returncode == 0 and out_path.exists()
    err = "" if ok else (proc.stderr or "")[-800:]
    return case_id, ok, elapsed, err


def _driver_main(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.case_ids_file:
        case_ids = [
            line.strip() for line in args.case_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        case_ids = _load_all_case_ids(args.benchmark)
    if args.limit:
        case_ids = case_ids[: args.limit]

    print(
        f"Running {len(case_ids)} cases against {args.agent_dir} "
        f"(concurrency={args.concurrency}, out_dir={args.out_dir})",
        file=sys.stderr,
    )

    ok_count = 0
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(_run_one_subprocess, args.agent_dir, args.benchmark, cid, args.out_dir, args.timeout): cid
            for cid in case_ids
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            case_id, ok, elapsed, err = fut.result()
            if ok:
                ok_count += 1
            else:
                failures.append({"case_id": case_id, "elapsed_s": round(elapsed, 1), "error": err})
                print(f"[{i}/{len(case_ids)}] FAILED {case_id} ({elapsed:.1f}s): {err[:200]}", file=sys.stderr)
            if i % 10 == 0 or i == len(case_ids):
                rate = i / (time.time() - t0)
                eta_s = (len(case_ids) - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{len(case_ids)}] ok={ok_count} fail={len(failures)} eta={eta_s/60:.1f}min",
                    file=sys.stderr,
                )

    total_s = time.time() - t0
    print(f"Done in {total_s/60:.1f}min. ok={ok_count} fail={len(failures)} out_dir={args.out_dir}", file=sys.stderr)
    if failures:
        failures_path = args.out_dir / "_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote failure details to {failures_path}", file=sys.stderr)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-dir", required=True, type=Path)
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--case-id", default=None, help="WORKER mode: run exactly this one case")
    p.add_argument("--out-path", type=Path, default=None, help="WORKER mode: where to write this case's trace JSON")
    p.add_argument("--out-dir", type=Path, default=None, help="DRIVER mode: where to write all trace JSONs")
    p.add_argument("--case-ids-file", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=3600, help="per-case subprocess timeout, seconds")
    args = p.parse_args(argv)

    if args.case_id:
        if not args.out_path:
            print("--case-id requires --out-path", file=sys.stderr)
            return 2
        try:
            _run_worker(args.agent_dir, args.benchmark, args.case_id, args.out_path)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            return 1
        return 0

    if not args.out_dir:
        print("driver mode requires --out-dir", file=sys.stderr)
        return 2
    return _driver_main(args)


if __name__ == "__main__":
    sys.exit(main())

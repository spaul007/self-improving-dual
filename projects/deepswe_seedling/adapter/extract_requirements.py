"""One-off: split each DeepSWE task's instruction.md into an atomic requirement checklist.

    python extract_requirements.py [--only task1,task2] [--workers 8] [--force]

Writes ``adapter/requirements/<task>.json``:
    {"task": ..., "model": ..., "requirements": [{"id": "r1", "text": ..., "literals": [...]}, ...]}

The checklist feeds the per-case dossier's coverage matrix (adapter/dossier.py). It is built
from ``instruction.md`` only -- the problem statement the agent itself reads -- never from the
hidden tests, so it leaks nothing. Extracted ONCE (cached, committed), so every node's dossier
uses the identical list and no LLM runs at evaluation time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "requirements"
CASES = HERE.parent / "benchmark" / "cases.jsonl"
BASE = "http://gpu-aic-mv-02-st-p5-node-6:8001/v1"
MODEL = "Qwen/Qwen3.8-27B"

SYSTEM = (
    "You turn a software task statement into a checklist of atomic, independently testable "
    "requirements. You never invent requirements: every item must be stated or directly implied "
    "by the text. You output JSON only."
)
INSTRUCTION = """Split the task statement above into atomic requirements.

Rules:
- One observable behaviour per item (an API, an output format, an error, an edge case, a default,
  a precedence/conflict rule). Split "X and Y" into two items when they are separately testable.
- Keep exact literals verbatim in "literals": error message strings, names of functions/classes/
  flags/options/files, exact output formats, numeric values. Empty list if none.
- "text" is one short sentence (<= 30 words), specific enough to write a test from.
- Cover EVERY requirement the statement contains, including ones in "Constraints"/"Notes"
  sections. Do not add generic items like "code compiles" or "existing tests pass".
- Order as they appear. Usually 5-40 items.

Output exactly one JSON object, no prose, no code fences:
{"requirements": [{"id": "r1", "text": "...", "literals": ["..."]}, ...]}"""


def call(instruction: str, effort: str = "low") -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "<task_statement>\n" + instruction + "\n</task_statement>\n\n" + INSTRUCTION},
        ],
        "max_tokens": 16384,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return r["choices"][0]["message"].get("content") or ""


def parse(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0) if m else text)
    reqs = obj["requirements"]
    out = []
    for i, r in enumerate(reqs, 1):
        t = str(r.get("text") or "").strip()
        if not t:
            continue
        lits = [str(x) for x in (r.get("literals") or []) if str(x).strip()]
        out.append({"id": f"r{len(out) + 1}", "text": t, "literals": lits})
    if not out:
        raise ValueError("empty requirement list")
    return out


def one(case: dict, force: bool, lock: threading.Lock) -> str:
    task = case["id"]
    dest = OUT / f"{task}.json"
    if dest.exists() and not force:
        return f"skip {task}"
    last = None
    for attempt in range(3):
        try:
            reqs = parse(call(case["input"], effort="low" if attempt < 2 else "medium"))
            dest.write_text(json.dumps({"task": task, "model": MODEL, "requirements": reqs},
                                       indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            return f"ok {task}: {len(reqs)} requirements"
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    return f"FAIL {task}: {last!r}"[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cases = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    lock = threading.Lock()
    fails = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda c: one(c, a.force, lock), cases):
            print(msg, flush=True)
            fails += msg.startswith("FAIL")
    n = len(list(OUT.glob("*.json")))
    print(f"done: {n} requirement files, {fails} failures", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

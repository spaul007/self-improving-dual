"""One-off generator: builds cases.jsonl from the ORIGINAL standalone
wikihop_mas project's own downloaded+converted 2WikiMultihopQA dev split --
NOT a vendored copy. dev.jsonl alone is ~53MB/12,576 rows, train.jsonl
~676MB, train.parquet ~343MB; vendoring any of data/ into the seed dir
would make it get shutil.copytree'd on every HGM tree node
(managers/hgm.py:850) and every editor attempt (agent_editor.py:236's
_copy_workspace) -- both currently unfiltered, no size-based exclusion.
Reads the dataset from its original location on disk once, at generation
time; never touched again once cases.jsonl is checked in.

Deliberately its own env var name (WIKIHOP_DEV_JSONL), NOT the vendored
project's own MAS_WIKIHOP_DEV_PATH (see config.py) -- that one governs what
the standalone repo's own run_inference.py CLI reads at its own runtime, a
different concern from this one-off generation step.

Ground truth (`answer`/`type`/`supporting_facts`/`evidences`) is written
only to each case's `meta_info` -- never into `context` -- so it never
reaches the agent (see workflow.py's `_to_wikihop_item` docstring).
`context` carries the raw `[title, [sentence, ...]]` paragraph list --
required agent input, not a label.

WIKIHOP_GEN_LIMIT caps how many of dev.jsonl's 12,576 rows are turned into
cases (default 200 -- far more than any HGM sanity/optimization run's
train_size needs, small enough to keep cases.jsonl a reasonable checked-in
size). Set via env var to generate more for a larger future run.

Rerun whenever the source dev.jsonl changes; the generated cases.jsonl is
checked in like any other project's benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_DEV_JSONL = "/groups/AIC-MV/v.kulkarni1/wikihop_mas/data/2wikimultihopqa/dev.jsonl"
_DEV_JSONL = Path(os.environ.get("WIKIHOP_DEV_JSONL", _DEFAULT_DEV_JSONL))
_OUT = Path(__file__).resolve().parent / "cases.jsonl"
_LIMIT = int(os.environ["WIKIHOP_GEN_LIMIT"]) if os.environ.get("WIKIHOP_GEN_LIMIT") else 200


def main() -> None:
    if not _DEV_JSONL.exists():
        raise FileNotFoundError(
            f"{_DEV_JSONL} not found -- set WIKIHOP_DEV_JSONL to the "
            "2WikiMultihopQA dev.jsonl path (see wikihop_mas/README.md's "
            "download+convert steps)."
        )

    lines: list[str] = []
    with _DEV_JSONL.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if _LIMIT is not None and idx >= _LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("_id", idx))
            case = {
                "id": case_id,
                "input": row["question"],
                "context": {"paragraphs": row.get("context", [])},
                "meta_info": {
                    "answer": row.get("answer", ""),
                    "type": row.get("type", ""),
                    "supporting_facts": row.get("supporting_facts", []),
                    "evidences": row.get("evidences", []),
                },
            }
            lines.append(json.dumps(case, ensure_ascii=False))

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} cases -> {_OUT}  (source: {_DEV_JSONL})")


if __name__ == "__main__":
    main()

"""One-off generator: builds cases.jsonl from MARBLE's database_main.jsonl,
the same source file db-mas's own benchmark.py reads. Rerun whenever that
source file changes; the generated cases.jsonl is checked in like any other
project's benchmark.

Ground truth (`root_causes`) is written only to each case's `meta_info` --
never into `context` -- so it never reaches the agent (see workflow.py's
`_to_dbmas_task` docstring). `env.DBMAS_PORT` gives each case a distinct,
pre-assigned Postgres host port (no worker-slot-index concept exists in the
framework's evaluator, unlike db-mas's own `run_many`'s dynamic slot queue),
using the same base as db-mas's own `config.PARALLEL_DB_PORT_BASE`.
"""
from __future__ import annotations

import json
from pathlib import Path

_MARBLE_JSONL = Path(
    "/groups/AIC-MV/v.kulkarni1/MARBLE/multiagentbench/database/database_main.jsonl"
)
_OUT = Path(__file__).resolve().parent / "cases.jsonl"
# Deliberately NOT db-mas/config.py's PARALLEL_DB_PORT_BASE (15432): this host
# is shared with other db-mas checkouts (e.g. another user's copy under
# /groups/AIC-MV/sudipta.paul/...) that use that same default range for their
# own concurrent runs. A disjoint base avoids colliding with them.
_PORT_BASE = 25432


def main() -> None:
    with _MARBLE_JSONL.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    lines = []
    for idx, rec in enumerate(records):
        task_block = dict(rec["task"])
        root_causes = task_block.pop("root_causes")
        case = {
            "id": str(rec["task_id"]),
            "input": task_block.get("content", ""),
            "env": {"DBMAS_PORT": str(_PORT_BASE + idx)},
            "context": {
                "environment": rec["environment"],
                "task": task_block,
            },
            "meta_info": {"root_causes": root_causes},
        }
        lines.append(json.dumps(case, ensure_ascii=False))

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} cases -> {_OUT}")


if __name__ == "__main__":
    main()

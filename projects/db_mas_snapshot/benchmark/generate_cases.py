"""One-off generator: builds cases.jsonl from the vendored copy's own
data/marble-db/database_tasks.jsonl (100 records) -- NOT read at inference
time by mas_workflow.run_task itself (confirmed: it only ever reads the
per-task db_cache/<unique_id>.json snapshot, never the flattened jsonl), so
this only runs once, offline, to produce the checked-in cases.jsonl.

Ground truth (`root_causes`) is written only to each case's `meta_info` --
never into `context` -- so it never reaches the agent (see workflow.py's
`_to_db_item` docstring). `context` carries `labels`/`number_of_labels_pred`
-- legitimate non-gold task input (the 5-label set is byte-identical across
every task and already embedded verbatim in the `problem` prose itself;
`number_of_labels_pred` only reveals a *count*, never *which* labels).

The `id` field is the single most fragile piece of this whole integration:
it MUST exactly match a `db_cache/<id>.json` snapshot filename (workflow.py
feeds it straight into `mas_workflow.run_task`'s `_snapshot_path(unique_id)`
resolution) -- this script asserts every emitted id has a matching snapshot
file on disk and fails loudly otherwise, rather than silently producing
cases that would all resolve to `snapshot_found=False` at eval time.

Drops `answer` (confirmed unused placeholder) and `marble_number_of_labels_pred`
(confirmed provenance-only, unused) -- neither is read anywhere on the
run_task/scoring path.
"""
from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_JSONL = _PROJECT_ROOT / "db_mas" / "data" / "marble-db" / "database_tasks.jsonl"
_DB_CACHE_DIR = _PROJECT_ROOT / "db_mas" / "data" / "marble-db" / "db_cache"
_OUT = Path(__file__).resolve().parent / "cases.jsonl"


def main() -> None:
    if not _SOURCE_JSONL.exists():
        raise FileNotFoundError(f"{_SOURCE_JSONL} not found -- expected the vendored db_mas/data/.")

    lines: list[str] = []
    missing_snapshots: list[str] = []
    with _SOURCE_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row["unique_id"])

            snapshot_path = _DB_CACHE_DIR / f"{case_id}.json"
            if not snapshot_path.exists():
                missing_snapshots.append(case_id)
                continue

            case = {
                "id": case_id,
                "input": row["problem"],
                "context": {
                    "labels": row.get("labels", []),
                    "number_of_labels_pred": row.get("number_of_labels_pred"),
                },
                "meta_info": {
                    "root_causes": row.get("root_causes", []),
                },
            }
            lines.append(json.dumps(case, ensure_ascii=False))

    if missing_snapshots:
        raise RuntimeError(
            f"{len(missing_snapshots)} task(s) have no matching db_cache/<id>.json "
            f"snapshot file (a silent id-format mismatch here would degrade those "
            f"cases to snapshot_found=False at eval time, not raise an error): "
            f"{missing_snapshots[:10]}{' ...' if len(missing_snapshots) > 10 else ''}"
        )

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} cases -> {_OUT}  (source: {_SOURCE_JSONL})")


if __name__ == "__main__":
    main()

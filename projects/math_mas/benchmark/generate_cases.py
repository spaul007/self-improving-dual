"""One-off generator: builds cases.jsonl from the vendored math_mas's own
Math500 data file. Rerun whenever that source file changes; the generated
cases.jsonl is checked in like any other project's benchmark.

Ground truth (`answer`) is written only to each case's `meta_info` -- never
into `context` -- so it never reaches the agent (see workflow.py's
`_to_math_item` docstring). `solution` (the full worked reference) is
dropped entirely -- scoring is exact-match against `answer` only, the MAS
never needs the reference solution.

Unlike db_mas, no `env` block is needed per case -- math_mas has no
Docker/port/per-case infra to configure.
"""
from __future__ import annotations

import json
from pathlib import Path

_MATH500_JSONL = (
    Path(__file__).resolve().parents[1] / "math_mas" / "data" / "math-500" / "math_500.jsonl"
)
_OUT = Path(__file__).resolve().parent / "cases.jsonl"


def main() -> None:
    with _MATH500_JSONL.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    lines = []
    for idx, rec in enumerate(records):
        case_id = str(rec.get("unique_id") or idx)
        case = {
            "id": case_id,
            "input": rec["problem"],
            "context": {},
            "meta_info": {
                "answer": rec["answer"],
                "subject": rec.get("subject"),
                "level": rec.get("level"),
            },
        }
        lines.append(json.dumps(case, ensure_ascii=False))

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} cases -> {_OUT}")


if __name__ == "__main__":
    main()

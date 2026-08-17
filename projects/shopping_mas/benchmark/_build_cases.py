"""One-shot builder for cases.jsonl.

Reads the three per-level query files from the reference repo at
``/users/n.tzou/cl/shopping_agent/data/level_{1,2,3}_query_meta.json``
(or wherever ``--source`` points) and emits a single 120-line jsonl
under ``projects/shopping_mas/benchmark/cases.jsonl``.

Not re-run by default -- ``cases.jsonl`` was seeded as a byte-identical copy
of the sibling ``projects/shopping`` project's own ``benchmark/cases.jsonl``
(same 120 cases, same ids/env/context/meta_info -- the case dict carries no
project-specific content, only the query itself), so the two projects stay
directly comparable case-for-case. Kept here for provenance/reproducibility,
matching ``projects/shopping/benchmark/_build_cases.py``'s own convention
(and ``math_mas``'s/``math_mas_pathology``'s analogous ``generate_cases.py``).

Run from the repo root:

    PYTHONPATH=. python3 projects/shopping_mas/benchmark/_build_cases.py

Existing cases.jsonl is overwritten. Output shape per case:

    {
      "id": "L1-1",                                  # `L{level}-{sample_id}`
      "input": "<natural-language query>",            # the user prompt
      "env": {"SHOPPING_SAMPLE_ID": "1", "SHOPPING_LEVEL": "1"},
      "context": {"level": 1},                       # unused by shopping_mas's own
                                                       # workflow.py (which reads level
                                                       # from the env vars above, the
                                                       # same channel the scorer uses),
                                                       # kept for parity with the
                                                       # sibling project's case shape
      "meta_info": {"level": 1}                       # scorer reads case["meta_info"]
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path("/users/n.tzou/cl/shopping_agent/data")
DEFAULT_TARGET = Path(__file__).resolve().parent / "cases.jsonl"


def build_one(level: int, source: Path) -> list[dict]:
    src = source / f"level_{level}_query_meta.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    out: list[dict] = []
    for entry in raw:
        sid = str(entry["id"])
        out.append(
            {
                "id": f"L{level}-{sid}",
                "input": entry["query"],
                "env": {
                    "SHOPPING_SAMPLE_ID": sid,
                    "SHOPPING_LEVEL": str(level),
                },
                "context": {"level": level},
                "meta_info": {"level": level},
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory holding level_{1,2,3}_query_meta.json.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Where to write cases.jsonl.",
    )
    args = parser.parse_args()

    cases: list[dict] = []
    for level in (1, 2, 3):
        cases.extend(build_one(level, args.source))

    args.target.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"wrote {len(cases)} cases to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

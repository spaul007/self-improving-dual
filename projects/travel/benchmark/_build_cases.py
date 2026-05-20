"""Regenerate `cases.jsonl` from the reference TravelPlanning query file.

The original `_build_cases.py` was deleted in the 2026-05-06 cleanup
after producing the first version of `cases.jsonl`. That first version
shipped terse, bullet-style user queries — apparently programmatically
compressed — which stripped natural-language details like explicit
room counts and budget caps. On 2026-05-20 we discovered all 120 case
input strings differed from the reference's source `query` field
(`/users/n.tzou/cl/work/travel_agent/data/travelplanning_query_en.json`)
and that "room" appeared in 98/120 reference queries but only 15/120 of
ours, which explained the entire Cost Calculation Accuracy gap.

This script rebuilds `cases.jsonl` using the reference's full `query`
text as the `input` field, while preserving:
  - the existing `id` (string, "0" through "119")
  - `env: {"TRAVEL_SAMPLE_ID": <id>}`  (per-case env merged into the
    child subprocess by the evaluator)
  - `meta_info` (byte-identical to the reference's, already)

Run from the repo root:
    python3 projects/travel/benchmark/_build_cases.py

The script is idempotent: re-running with the same inputs produces a
byte-identical `cases.jsonl`. It refuses to overwrite if the reference
file is missing or differs in case count from the existing
`cases.jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path

REFERENCE_QUERY_FILE = Path(
    "/users/n.tzou/cl/work/travel_agent/data/travelplanning_query_en.json"
)
HERE = Path(__file__).resolve().parent
OUTPUT_FILE = HERE / "cases.jsonl"


def main() -> None:
    if not REFERENCE_QUERY_FILE.exists():
        raise SystemExit(f"reference query file missing: {REFERENCE_QUERY_FILE}")

    with open(REFERENCE_QUERY_FILE, "r", encoding="utf-8") as fh:
        ref_entries = json.load(fh)

    if len(ref_entries) != 120:
        raise SystemExit(
            f"expected 120 reference entries, got {len(ref_entries)} — "
            f"refusing to rebuild cases.jsonl from an unexpected source"
        )

    cases = []
    for entry in ref_entries:
        sample_id = str(entry["id"])
        query = entry.get("query", "")
        meta_info = entry.get("meta_info", {})
        if not query:
            raise SystemExit(f"reference entry id={sample_id} has empty query")
        cases.append(
            {
                "id": sample_id,
                "input": query,
                "env": {"TRAVEL_SAMPLE_ID": sample_id},
                "meta_info": meta_info,
            }
        )

    cases.sort(key=lambda c: int(c["id"]))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"wrote {len(cases)} cases to {OUTPUT_FILE}")
    avg_len = sum(len(c["input"]) for c in cases) / len(cases)
    print(f"avg input length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()

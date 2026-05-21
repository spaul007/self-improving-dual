"""Verify `cases.jsonl` is field-equivalent to the reference query file.

Companion to `_build_cases.py`. Loads both files and confirms that every
corresponding field matches:

    cases.jsonl                       reference travelplanning_query_en.json
    ---------------------------       ---------------------------------------
    id                          ==    id
    input                       ==    query
    meta_info  (deep equality)  ==    meta_info  (deep equality)
    env.TRAVEL_SAMPLE_ID        ==    id                  (derived from id)

The reference's `query_with_constraints` field has no counterpart in
`cases.jsonl` and is intentionally not compared.

Comparison is keyed by `id`, not file order. Run from the repo root:

    python3 projects/travel/benchmark/_verify_cases.py

Exit code 0 if every case matches; 1 if any field mismatches, with a
per-case diff printed to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REFERENCE_QUERY_FILE = Path(
    "/users/n.tzou/cl/work/travel_agent/data/travelplanning_query_en.json"
)
HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.jsonl"

FIELD_MAP = [
    # (cases_field_path, ref_field_path, label)
    (("id",), ("id",), "id"),
    (("input",), ("query",), "input/query"),
    (("meta_info",), ("meta_info",), "meta_info"),
]


def get_path(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def diff_value(label: str, a: Any, b: Any, path: str = "") -> list[str]:
    """Return a list of human-readable diff lines. Empty list = equal."""
    if a is _MISSING or b is _MISSING:
        if a is b:
            return []
        return [f"  [{label}{path}] presence mismatch: cases={a!r} ref={b!r}"]
    if type(a) is not type(b):
        return [f"  [{label}{path}] type mismatch: cases={type(a).__name__} ref={type(b).__name__}"]
    if isinstance(a, dict):
        diffs: list[str] = []
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            sub = diff_value(label, a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}")
            diffs.extend(sub)
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"  [{label}{path}] list length mismatch: cases={len(a)} ref={len(b)}"]
        diffs = []
        for i, (av, bv) in enumerate(zip(a, b)):
            diffs.extend(diff_value(label, av, bv, f"{path}[{i}]"))
        return diffs
    if a != b:
        a_repr = repr(a)
        b_repr = repr(b)
        if len(a_repr) > 120 or len(b_repr) > 120:
            return [
                f"  [{label}{path}] value mismatch:",
                f"      cases: {a_repr[:200]}{'…' if len(a_repr) > 200 else ''}",
                f"      ref:   {b_repr[:200]}{'…' if len(b_repr) > 200 else ''}",
            ]
        return [f"  [{label}{path}] value mismatch: cases={a_repr} ref={b_repr}"]
    return []


def load_cases() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with open(CASES_FILE, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("id")
            if sid is None:
                raise SystemExit(f"{CASES_FILE}:{line_no} missing 'id'")
            if sid in by_id:
                raise SystemExit(f"{CASES_FILE}:{line_no} duplicate id: {sid!r}")
            by_id[sid] = obj
    return by_id


def load_reference() -> dict[str, dict]:
    if not REFERENCE_QUERY_FILE.exists():
        raise SystemExit(f"reference query file missing: {REFERENCE_QUERY_FILE}")
    with open(REFERENCE_QUERY_FILE, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    by_id: dict[str, dict] = {}
    for entry in entries:
        sid = str(entry["id"])
        if sid in by_id:
            raise SystemExit(f"reference has duplicate id: {sid!r}")
        by_id[sid] = entry
    return by_id


def main() -> int:
    cases = load_cases()
    reference = load_reference()

    cases_ids = set(cases.keys())
    ref_ids = set(reference.keys())

    failures: list[str] = []

    # Set parity
    only_cases = cases_ids - ref_ids
    only_ref = ref_ids - cases_ids
    if only_cases:
        failures.append(f"id(s) in cases.jsonl but not in reference: {sorted(only_cases, key=int)}")
    if only_ref:
        failures.append(f"id(s) in reference but not in cases.jsonl: {sorted(only_ref, key=int)}")

    print(f"cases.jsonl: {len(cases)} entries  ·  reference: {len(reference)} entries")
    if len(cases) != len(reference):
        failures.append(f"count mismatch: cases={len(cases)} reference={len(reference)}")

    common_ids = sorted(cases_ids & ref_ids, key=int)
    mismatch_count = 0
    for sid in common_ids:
        c = cases[sid]
        r = reference[sid]
        case_diffs: list[str] = []

        for cases_path, ref_path, label in FIELD_MAP:
            cv = get_path(c, cases_path)
            rv = get_path(r, ref_path)
            case_diffs.extend(diff_value(label, cv, rv))

        # env.TRAVEL_SAMPLE_ID should equal id
        env = c.get("env") or {}
        tsid = env.get("TRAVEL_SAMPLE_ID")
        if tsid != sid:
            case_diffs.append(
                f"  [env.TRAVEL_SAMPLE_ID] mismatch: env={tsid!r} id={sid!r}"
            )

        if case_diffs:
            mismatch_count += 1
            failures.append(f"id={sid}:\n" + "\n".join(case_diffs))

    if failures:
        print()
        print(f"FAILED — {mismatch_count} case(s) with field mismatches:")
        print()
        for msg in failures:
            print(msg)
            print()
        return 1

    print(f"OK — all {len(common_ids)} cases match across "
          f"{', '.join(label for _, _, label in FIELD_MAP)}, env.TRAVEL_SAMPLE_ID.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

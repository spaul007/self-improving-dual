"""IMMUTABLE — deterministic date/number comparator, the Concluder's only tool.

For comparison-type questions the Concluder must not do date/number arithmetic
itself (models are unreliable at this); it calls this tool instead. Parse
failures return a diagnosable JSON string, never an exception, so the LLM can
retry with cleaner values -- still bounded by config.CONCLUDER_MAX_ROUNDS.
"""

import json
import re
from datetime import datetime

from dateutil import parser as dtparser

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _parse_number(s: str) -> float:
    m = _NUMBER.search(s)
    if not m:
        raise ValueError(f"no number found in {s!r}")
    return float(m.group(0).replace(",", ""))


def _parse_date(s: str) -> datetime:
    # dateutil fills unspecified fields from `default`; a fixed sentinel default
    # makes comparisons between two differently-specified dates well-defined.
    return dtparser.parse(s, default=datetime(1, 1, 1), fuzzy=True)


def compare_values(a: str, b: str, kind: str) -> str:
    """Returns a JSON string: {"a_parsed", "b_parsed", "result": "a<b"|"a>b"|"a==b"}
    or {"error": ...} on a parse failure (never raises)."""
    try:
        if kind == "number":
            av, bv = _parse_number(a), _parse_number(b)
        elif kind == "date":
            av, bv = _parse_date(a), _parse_date(b)
        else:
            return json.dumps({"error": f"unknown kind {kind!r}, expected 'date' or 'number'"})
    except Exception as exc:
        return json.dumps({"error": f"could not parse a value: {exc}"})

    result = "a==b" if av == bv else ("a>b" if av > bv else "a<b")
    return json.dumps({"a_parsed": str(av), "b_parsed": str(bv), "result": result})


COMPARE_VALUES_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_values",
        "description": (
            "Deterministically compare two values of the same kind and report which is "
            "greater/earlier or if they're equal. Always use this instead of doing date "
            "or number arithmetic yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First value, e.g. '12 March 1990' or '42'."},
                "b": {"type": "string", "description": "Second value, same kind as a."},
                "kind": {"type": "string", "enum": ["date", "number"]},
            },
            "required": ["a", "b", "kind"],
            "additionalProperties": False,
        },
    },
}


def make_compare_values_handler():
    def _handler(args: dict) -> str:
        return compare_values(str(args.get("a", "")), str(args.get("b", "")), str(args.get("kind", "")))
    return _handler

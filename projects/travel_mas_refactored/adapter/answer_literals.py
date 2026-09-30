"""Answer literals of a travel benchmark case, for the ``hardcoded_answers``
validator (``meta_agent/editor_validators.py``).

A case's ``meta_info.hard_constraints`` holds, per constraint, the entity the
scorer expects (e.g. ``inbound_train_no: "G729"`` for "the latest direct
train"). The scorer's failure messages repeat these (``Required return train
not found: G729``), so an editor reading case logs sees them. The ones the
case's query does not state are answers the agent must find with its tools;
those are what an edit may not hard-code.

Only entity identifiers count: train and flight numbers and hotel,
restaurant and attraction names. Tags, services, types, prices, ratings and
seat counts are excluded -- they are generic words or numbers that
legitimate code contains.
"""
from __future__ import annotations

from typing import Any, Iterable

ENTITY_FIELDS = (
    "outbound_train_no", "inbound_train_no",
    "outbound_flight_no", "inbound_flight_no",
    "hotel_name", "restaurant_name",
    "attraction_name", "attraction_names",
)


def answer_literals(case: dict[str, Any]) -> list[str]:
    """String values of ``ENTITY_FIELDS`` in the case's hard constraints
    that do not appear in its query text (``case["input"]``)."""
    query = str(case.get("input") or "")
    constraints = (case.get("meta_info") or {}).get("hard_constraints") or {}
    if not isinstance(constraints, dict):
        return []
    out: list[str] = []
    for constraint in constraints.values():
        if not isinstance(constraint, dict):
            continue
        for field in ENTITY_FIELDS:
            for value in _strings(constraint.get(field)):
                if value and value not in query and value not in out:
                    out.append(value)
    return out


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value.strip()
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                yield v.strip()

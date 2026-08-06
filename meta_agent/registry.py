from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = {
    "manager": {},
    "evaluator": {},
    "gatherer": {},
    "validator": {},
    "scorer": {},
    "editor": {},
    "summarizer": {},
}


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    if kind not in _REGISTRIES:
        raise ValueError(f"Unknown registry kind {kind!r}. Known: {list(_REGISTRIES)}")

    def deco(cls: type[T]) -> type[T]:
        _REGISTRIES[kind][name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    if kind not in _REGISTRIES:
        raise KeyError(f"Unknown registry kind {kind!r}. Known: {list(_REGISTRIES)}")
    bucket = _REGISTRIES[kind]
    if name not in bucket:
        raise KeyError(
            f"Unknown {kind}: {name!r}. Available: {sorted(bucket)}"
        )
    return bucket[name]


def available(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}))

"""Tool registry and per-role permissions.

Every tool is a host-side function that marshals its arguments into one or more
safe_exec calls. Nothing is installed in the container.

PERMISSIONS. A role declares an allowlist plus an optional `readonly` flag:
  * write_file / edit_file are HARD-DENIED under readonly, and their target path is
    checked against DENY_WRITE_GLOBS first. Fully enforced, host-side, before any exec.
  * `bash` under readonly is ADVISORY -- a shell can do anything and we do not pretend
    otherwise. We DETECT instead: gitops snapshots `git status` around a readonly role
    and records a violation. Stated plainly because a permission that looks enforced but
    isn't is worse than one documented as advisory.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Awaitable, Callable

# Mirrors the anti-cheat tripwire paths named in DeepSWE's tests/test.sh. Editing these
# is both a build hazard and a cheating signal.
DENY_WRITE_GLOBS = [
    "*/conftest.py", "conftest.py", "*/sitecustomize.py", "*pytest.ini", "*tox.ini",
    "*.lock", "*package-lock.json", "*go.sum", "*Cargo.lock", "*/test.sh", "test.sh",
]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Awaitable[str]]
    mutates: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def denied_path(path: str) -> str | None:
    p = (path or "").strip()
    for g in DENY_WRITE_GLOBS:
        if fnmatch.fnmatch(p, g) or fnmatch.fnmatch(p.lstrip("/"), g):
            return f"refused: {p} matches protected pattern {g!r}"
    return None


def registry() -> dict[str, Tool]:
    """Built fresh per call; never a mutable module-level global (concurrent trials)."""
    from .shell import ALL
    return {t.name: t for t in ALL}


def for_role(names: list[str], readonly: bool = False) -> dict[str, Tool]:
    reg = registry()
    out = {}
    for n in names:
        t = reg.get(n)
        if t is None:
            continue
        if readonly and t.mutates:
            continue
        out[n] = t
    return out

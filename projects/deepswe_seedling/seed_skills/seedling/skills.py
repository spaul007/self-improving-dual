"""Skill library for the roles: short procedure documents the model loads on demand.

Layout (inside prompts/ so every read goes through PROMPTS -- host-isolation rule):

    prompts/skills/INDEX.md        one line per skill:
                                   - <name> | roles: <role>[, <role>...] | <when to use it>
    prompts/skills/<name>.md       the procedure

A role's system prompt ends with the FULL text of the skills tagged for it (Role.system_prompt,
so the text is deterministic and the A0 sys_sha check still holds). The host-side `Skill`
tool (tools/shell.py) can still read one by name, but no role is given it by default. Only
names listed in INDEX.md can be read (no path is ever built from model input beyond a
validated name). Role names: patch, verify, baseline.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def index_entries() -> list[dict]:
    try:
        text = (PROMPTS / "skills" / "INDEX.md").read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or line.count("|") < 2:
            continue
        name, roles, when = (p.strip() for p in line[2:].split("|", 2))
        roles = roles.split(":", 1)[1] if roles.lower().startswith("roles:") else roles
        if not _NAME.match(name):
            continue
        out.append({"name": name,
                    "roles": [r.strip().lower() for r in roles.split(",") if r.strip()],
                    "when": when})
    return out


def for_role(role: str) -> list[dict]:
    return [e for e in index_entries() if role.lower() in e["roles"]]


def index_text(role: str) -> str:
    """The block appended to `role`'s system prompt: the FULL text of every skill tagged
    for it ("" when it has none).

    Inlined, not loaded on demand: a live check (2026-09-24) showed VERIFY never calling the
    on-demand `Skill` tool, and travel stages never loaded theirs -- a skill the model does
    not open is inert. Deterministic, so the A0 sys_sha check still holds.
    """
    parts = [t.strip() for t in (read_skill_text(e["name"]) for e in for_role(role)) if t]
    if not parts:
        return ""
    return ("\n\n## Skills (follow these procedures)\n\n"
            "Each skill below is a short, tested procedure. When your situation matches its "
            "**When**, follow its steps and its check.\n\n" + "\n\n".join(parts) + "\n")


def read_skill_text(name: str) -> str | None:
    name = (name or "").strip()
    if name not in {e["name"] for e in index_entries()}:
        return None
    try:
        return (PROMPTS / "skills" / (name + ".md")).read_text()
    except OSError:
        return None

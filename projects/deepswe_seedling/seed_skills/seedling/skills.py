"""Skill library for the roles: short procedure documents the model loads on demand.

Layout (inside prompts/ so every read goes through PROMPTS -- host-isolation rule):

    prompts/skills/INDEX.md        one line per skill:
                                   - <name> | roles: <role>[, <role>...] | <when to use it>
    prompts/skills/<name>.md       the procedure

A role's system prompt ends with the index lines tagged for it (Role.system_prompt, so the
text is deterministic and the A0 sys_sha check still holds); the role calls the `Skill`
tool to read one. Only names listed in INDEX.md can be read (no path is ever built from
model input beyond a validated name). Role names: patch, verify, baseline.
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
    """The block appended to `role`'s system prompt ("" when it has no skills)."""
    entries = for_role(role)
    if not entries:
        return ""
    lines = "\n".join(f"- {e['name']}: {e['when']}" for e in entries)
    return ("\n\n## Skills available to you\n" + lines + "\n\n"
            "Each skill is a short, tested procedure. When your situation matches a skill's "
            "description, call the `Skill` tool with its name and follow it BEFORE acting on "
            "that part of the task.\n")


def read_skill_text(name: str) -> str | None:
    name = (name or "").strip()
    if name not in {e["name"] for e in index_entries()}:
        return None
    try:
        return (PROMPTS / "skills" / (name + ".md")).read_text()
    except OSError:
        return None

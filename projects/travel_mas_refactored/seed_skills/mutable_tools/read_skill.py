"""Skill library access for the travel stages (mutable tool + shared index parser).

Skills are short procedure documents in ``skills/<name>.md``; ``skills/INDEX.md`` lists
them, one per line:

    - <name> | stages: <stage>[, <stage>...] | <when to use it>

A stage sees the index lines tagged for it (agents/common.py appends them to its system
prompt) and calls ``read_skill(name)`` for the full text. Only names in the index can be
read. Stdlib only (mutable-tool import rule).
"""
from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def index_entries() -> list[dict]:
    """Parsed INDEX.md: [{"name", "stages", "when"}]; [] when there is no library."""
    idx = SKILLS_DIR / "INDEX.md"
    if not idx.is_file():
        return []
    out = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- ") or line.count("|") < 2:
            continue
        name, stages, when = (p.strip() for p in line[2:].split("|", 2))
        stages = stages.split(":", 1)[1] if stages.lower().startswith("stages:") else stages
        out.append({
            "name": name,
            "stages": [s.strip().lower() for s in stages.split(",") if s.strip()],
            "when": when,
        })
    return out


def read_skill_text(name: str) -> str | None:
    names = {e["name"] for e in index_entries()}
    if name not in names:
        return None
    f = SKILLS_DIR / f"{name}.md"
    return f.read_text(encoding="utf-8") if f.is_file() else None


def run(name: str = "") -> str:
    name = (name or "").strip()
    text = read_skill_text(name)
    if text is None:
        known = ", ".join(sorted(e["name"] for e in index_entries())) or "(none)"
        return f"Error: unknown skill {name!r}. Available skills: {known}"
    return text

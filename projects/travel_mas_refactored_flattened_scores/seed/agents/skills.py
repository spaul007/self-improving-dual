"""Evolvable skill library for the travel stages (inert when the library is empty).

Skills are short procedure documents in ``skills/<name>.md``; ``skills/INDEX.md`` lists them,
one per line:

    - <name> | stages: <stage>[, <stage>...] | <when to use it>

``with_inline_skills(prompt, stage)`` appends the full text of the skills tagged for a stage.
With no library (or an empty INDEX) it returns ``prompt`` unchanged, so the agent behaves
exactly as without skills. Only names listed in INDEX.md are read. Stdlib only.
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
        if stages.lower().startswith("stages:"):
            stages = stages.split(":", 1)[1]
        out.append({
            "name": name,
            "stages": [s.strip().lower() for s in stages.split(",") if s.strip()],
            "when": when,
        })
    return out


def read_skill_text(name: str) -> str | None:
    if name not in {e["name"] for e in index_entries()}:
        return None
    f = SKILLS_DIR / f"{name}.md"
    return f.read_text(encoding="utf-8") if f.is_file() else None


def with_inline_skills(system_prompt: str, stage: str) -> str:
    """``system_prompt`` + the full text of the skills tagged for ``stage`` (unchanged if none)."""
    parts = [t.strip() for t in (read_skill_text(e["name"]) for e in index_entries()
                                 if stage.lower() in e["stages"]) if t]
    if not parts:
        return system_prompt
    return (system_prompt + "\n\n## Skills (follow these procedures)\n\n"
            "Each skill below is a short, tested procedure. When your situation matches its "
            "**When**, follow its steps and its check.\n\n" + "\n\n".join(parts))

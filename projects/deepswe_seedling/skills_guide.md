---
dir: seedling/prompts/skills/
tag_key: roles
tag_values: patch, verify, baseline
---
How seedling uses skills: each role's system prompt ends with the FULL text of the skills tagged for it
(`seedling/roles.py::Role.system_prompt` -> `seedling/skills.py::index_text`); `role_stats.skills_inlined` lists them per
role run. Role names in the index must be exactly one of: patch, verify, baseline -- a skill tagged for the wrong role is
invisible to the role that needed it. Skill names are lowercase letters, digits and dashes (others are ignored). The
library lives under `seedling/prompts/` because the harness may only read host files through `PROMPTS` (host-isolation
validator). Skills cost context and therefore wall time on every step of a wall-budgeted role: keep each skill short and
procedural, and prefer sharpening an existing skill that was in the failing role's prompt over adding a near-duplicate.
Ground new skills in the dossiers (requirement coverage, VERIFY's test commands) of the failing tasks.

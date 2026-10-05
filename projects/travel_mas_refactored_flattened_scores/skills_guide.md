---
dir: skills/
tag_key: stages
tag_values: flight, train, sightseeing, accounting
---
How the travel MAS uses skills: every stage gets the FULL text of the skills tagged for it appended to its system
prompt on every run (`agents/common.py` -> `run_tool_stage` / `run_notool_stage` -> `agents/skills.py::with_inline_skills`).
Stage names in the index must be exactly one of: flight, train, sightseeing, accounting. A skill is only seen by the
stages it is tagged for, and only if INDEX.md lists it. Ground a new skill in the failure messages of the cases you read
(what the checker reported -- e.g. transfer gaps vs queried route time, business hours, duplicate venues) and give it a
concrete, checkable final step. If a stage had a relevant skill in its prompt and still failed, fix that skill's steps
rather than adding a near-duplicate.

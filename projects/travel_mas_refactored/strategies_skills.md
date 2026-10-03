# Strategies -- travel MAS with a skill library (seed_skills)

Reference hints for the block suggester when the seed has a skill library. Hints, not rules.

## General

- The agent has a SKILL LIBRARY: `skills/INDEX.md` lists skills as
  `- <name> | stages: <stage,...> | <when to use>`, with the procedure in `skills/<name>.md`. Every stage gets the FULL
  text of the skills tagged for it appended to its system prompt (`agents/common.py`: `run_tool_stage` /
  `run_notool_stage` -> `with_inline_skills`; parsing in `mutable_tools/read_skill.py`). Skills are always in context,
  so a skill's wording directly changes what the stage reads on every run.

## Block: skills

- Only files under `skills/` can change in this block (enforced). Keep `skills/INDEX.md` and the `.md` files in sync:
  every index line needs a file; a file without an index line is never seen.
- Stage names in the index must be exactly: flight, train, sightseeing, accounting.
- Before adding a skill, check whether an existing one already covers the failure; if the stage had it in context and
  failed anyway, fix its steps (make them concrete and checkable) rather than adding a near-duplicate.
- Every skill is in its stages' prompts on every run -- keep skills short; retire ones that do not help.
- Ground a new skill in the failure messages of the cases you read (what the checker reported), and give it a
  concrete, checkable final step.

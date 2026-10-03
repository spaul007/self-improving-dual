# Strategies -- travel MAS with a skill library (seed_skills)

Reference hints for the block suggester when the seed has a skill library. Hints, not rules.

## General

- The agent has a SKILL LIBRARY: `skills/INDEX.md` lists skills as
  `- <name> | stages: <stage,...> | <when to use>`, with the procedure in `skills/<name>.md`. Each tool-using stage
  (flight, train, sightseeing) sees its tagged index lines in its system prompt and loads a skill with the `read_skill`
  tool; accounting (no tools) gets its tagged skills inlined in full. Loading is wired in `agents/common.py`
  (`run_tool_stage` / `run_notool_stage`) and `mutable_tools/read_skill.py`. Skill loads appear in `logs/trace.jsonl`
  as `tool_call` events named `read_skill`.

## Block: skills

- Only files under `skills/` can change in this block (enforced). Keep `skills/INDEX.md` and the `.md` files in sync:
  every index line needs a file; a file without an index line is never seen.
- Stage names in the index must be exactly: flight, train, sightseeing, accounting.
- Before adding a skill, check whether an existing one covers the failure and whether it was loaded in the failing
  cases (grep `read_skill` in `logs/trace.jsonl`). Not loaded -> sharpen its index line (the "when"); loaded but
  failed anyway -> fix its steps.
- Accounting's skills are inlined into its prompt on every run -- keep them especially short.
- Ground a new skill in the failure messages of the cases you read (what the checker reported), and give it a
  concrete, checkable final step.

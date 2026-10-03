# Strategies — seedling (multi-agent coding harness on DeepSWE)

Reference hints for `meta_agent/block_suggester.py` when the task agent is **seedling**: a
host-orchestrated multi-agent coding harness solving one repository task inside a docker
container, graded by hidden tests (fail-to-pass `f2p` for the requested behaviour,
pass-to-pass `p2p` for everything that already worked; reward = 1 only if both are
complete). Read fresh on every `suggest()` call. Hints, not rules.

The harness (all under `harness/seedling/`):
- `pipeline.py` — orchestration: BASELINE runs once on the unmodified repo, then PATCH →
  VERIFY repeats until VERIFY's verdict is `pass`, the deadline, or `MAX_PATCH_ATTEMPTS`.
  VERIFY's verdict is final (schema-only acceptance: no hard-coded gate).
- `roles.py` — the role definitions (`PATCH`, `VERIFY`, `BASELINE`: prompt file, tools, and
  the `output` schema that becomes each role's `finish` report) and `Role.run`, the shared
  ReAct loop (end-turn detection, the report call, nudges, compaction trigger, per-role
  statistics).
- `blackboard.py` — what each role is told about the others (`context_for(role)`): the
  hand-off surface between roles.
- `prompts/{patch,verify,baseline}.md` — each role's system prompt.
- `compact.py` — conversation summarisation when a role's context grows large.
- `settings.py` — wall-clock budget fractions per role, attempts, token limits.
- `tools/shell.py` — the Bash/Read/Edit/Write tools the roles call inside the container.
- Frozen (read-only reference): `agent.py` (Pier entry, deadline, artifacts), `llm.py`,
  `execpool.py`, `gitops.py`, `trajectory.py`, `deadline.py`, `tools/__init__.py`.

The failure data you see is per-task outcome classes (e.g. `verify_false_pass`,
`near_miss`, `build_break`, `role_wall`) plus each role's own statistics and rendered
transcripts under `logs/scratch/<task>/<run>/`. You never see the hidden tests.

## General

- Diagnose from the transcripts and role statistics, not from the task text alone: a
  near-miss is usually a behaviour the task statement named that no role checked.
- The harness must remain multi-agent: at least a PATCH role and a VERIFY role, each with
  its own system prompt and structured report. Improve how they work and hand off; do not
  merge them.
- Every role's budget is a fraction of a fixed wall clock. Anything that makes one role
  spend longer is paid for by another role or by fewer PATCH → VERIFY cycles.
- A change that only rewords an instruction the model already ignores rarely helps;
  changing what information a role receives, or what it must report, usually does.
- Keep observability: every role's `role_stats` fields are what the next diagnosis reads.
- Prefer the smallest change that addresses the diagnosed problem.

## Block: individual_subagent

- One role's own prompt and behaviour: its instructions, its tool set, its `output` report
  schema, and role-specific logic in `Role.run`.
- If a role stops early or wanders, look at how its prompt defines "done" and what its
  report schema forces it to state.
- If a role repeatedly runs out of wall clock, check whether it re-discovers things another
  role already knows.

## Block: collaboration_workflow

- The hand-off surface: `pipeline.py` (order, retries, what ends the loop) and
  `blackboard.py::context_for` (what each role is told about the others' work and reports).
- Fix information at its source: if PATCH repeats a mistake VERIFY already reported, check
  whether VERIFY's findings actually reach PATCH, and in what form.
- Be explicit about which role sends and which role receives whatever you change.

## Block: verifiers

- "Verifiers" here means VERIFY's checking behaviour (and any check another role runs on
  its own output) — not a new pipeline stage.
- VERIFY passes patches the grader fails. A check is only as good as its coverage of what
  the task statement requires; a verdict should be tied to explicit evidence of each
  required behaviour.
- A check is only useful if something acts on the result.

## Block: foundation_capability

- Mechanisms shared by every role: the `Role.run` loop mechanics, compaction
  (`compact.py`), tool plumbing (`tools/shell.py`), budgets and limits (`settings.py`).
- A capacity limit hit mid-task (wall budget, token cap, output truncation) silently
  discards progress; check the limits before adding logic.
- Be cautious: shared changes affect every role at once.

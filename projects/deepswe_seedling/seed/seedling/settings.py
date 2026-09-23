"""Every tunable in one place.

This is the numeric mutation surface: a self-evolving algorithm (or a bandit/BO sweep)
can change any of these without touching control flow. Keep them plain module constants
so they are trivially readable and settable.
"""

from __future__ import annotations

# -- pipeline shape ------------------------------------------------------------------
# Verify -> Patch retries. Raised 3 -> 10: at 3, the CAP (not convergence) ended 74% of
# trials, and trials used only 31-59m of their 180m grant. With 10 the WALL becomes the
# binding constraint instead of a number I invented -- the loop already exits early on
# `deadline.expired()`, and on a clean gate + Verify agreement.
import os as _os
MAX_PATCH_ATTEMPTS = int(_os.environ.get("SEEDLING_MAX_ATTEMPTS", "8"))


# -- per-role budgets ----------------------------------------------------------------
# v8: max_wall_sec is the ONLY bound. There is no turn cap -- see BUDGETS below.
# wall_frac is a FRACTION of the working window, not an absolute. Absolutes summed to
# 8700s and silently over-ran a 3600s grant, so every role ran to the GLOBAL deadline
# instead of its own budget and nothing was left for finalize. Fractions scale with
# whatever --ak agent_timeout_sec is actually given.
# THESE CAPS ARE MINE, NOT THE BENCHMARK'S -- and they are the binding constraint.
# Measured: DeepSWE grants 10800s/task, but trials finish in 31-59m of that 180m and NO role
# was ever cut by a wall budget. Yet the patch role hit its STEP cap 62% of the time. So the
# agent is leaving ~2h of its grant unused while being cut off mid-work. Env-overridable so
# they can be swept -- exactly the kind of numeric knob a self-evolving search should own,
# rather than a number I picked by intuition and never revisited.
def _steps(name: str, default: int) -> int:
    import os
    try:
        return int(os.environ.get(f"SEEDLING_{name}_STEPS", default))
    except (TypeError, ValueError):
        return default


BUDGETS = {
    # v8: NO STEP CAP. A role runs until the model stops calling tools (end_turn), or until
    # its wall budget expires. `max_steps` is gone -- it was never a quality lever: measured,
    # the role EXPANDS to fill whatever cap it is given (60->62, 120->122, 80->77), and
    # Claude Code SOLVES tasks in fewer steps than it fails them (122 vs 164), so steps are
    # not the scarce resource. What bounded seedling was the cap; what bounds it now is time.
    #
    # wall_frac is a share of the WINDOW, per role instance, so
    #     attempts that fit = 1 / (patch_frac + verify_frac)
    # and the window cancels out. 0.22 + 0.15 = 0.37 -> ~2.7 cycles.
    "patch":   {"wall_frac": 0.22},
    "verify":  {"wall_frac": 0.15},
    # BASELINE runs once per task before the loop. 0.06 ~= 630s: discovery is ~10-30 steps,
    # plus one full suite run (dateutil 35s; Go/Rust builds and big JS suites can be minutes).
    "baseline": {"wall_frac": 0.06},
    "solo":    {"wall_frac": 0.92},
}




# -- sampling ------------------------------------------------------------------------
# Qwen3.8's THINKING-ON row (temp 1.0 / top_p 0.95 / top_k 20). Thinking is ON whenever
# reasoning_effort is set, so the 0.7/0.80 non-thinking row would be the wrong one here.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

# reasoning_effort must travel via extra_body.chat_template_kwargs, NOT the top-level
# field: vLLM validates the top-level one against none|low|medium|high and it never
# reaches Qwen's chat template, whose own enum is xhigh|medium|low.
REASONING_EFFORT = "medium"   # EXP-013: medium == xhigh in accuracy at 1/3.7 the cost

# Qwen3.8 thinks with `reasoning_effort` defaulting to 'xhigh' (its own chat template),
# and a single observed thinking block reached 33,410 chars ~= 8,350 tokens -- MORE than
# the old 8192 cap, so the model was truncated mid-thought and emitted no tool call at
# all. The cap must comfortably exceed a full deliberation plus the tool call that
# follows it. Context is 262144, so this is cheap.
MAX_TOKENS = 32768
LLM_TIMEOUT_SEC = 900
LLM_MAX_RETRIES = 3

# -- tool output ---------------------------------------------------------------------
MAX_TOOL_OUTPUT_CHARS = 12_000   # what the MODEL sees; execpool caps the wire separately
# ^ KEPT as an entry guard. Compaction handles ACCUMULATION, but nothing else guards a single
#   runaway command dumping 400K chars in one turn.

# -- context compaction (replaces truncation) ----------------------------------------
# Modelled on Claude Code's observed behaviour, read from its raw session file:
#   compactMetadata = {trigger: "auto", preTokens: 167387, durationMs: 478376,
#                      preservedMessages: {uuids: [...7 records...]}}
# i.e. it summarises at ~167K and preserves the last ~7 records VERBATIM, then continues in
# the same thread. 150K here leaves headroom under the 262,144 window.
# ⚠ COST: that compaction took 8.0 MINUTES of LLM time. Budget 1-2 per PATCH role.
SUMMARIZE_AT_TOKENS = int(_os.environ.get("SEEDLING_SUMMARIZE_AT", "150000"))
PRESERVE_TAIL_MSGS  = int(_os.environ.get("SEEDLING_PRESERVE_TAIL", "8"))
CHARS_PER_TOKEN     = 4          # crude estimator; only needs to be right to ~20%

# -- Read window ----------------------------------------------------------------------
# seedling's read_file defaulted to 400 LINES and its p90 output landed on 12,026 chars --
# exactly the cap above, i.e. >=10% of reads returned a truncated wall. Claude Code's Read
# passes an explicit window in 21 of 30 calls, at 30-130 lines. 120 matches its ceiling.
READ_DEFAULT_LIMIT = int(_os.environ.get("SEEDLING_READ_LIMIT", "120"))

# v7. Fold a bounded TAIL of the model's thinking into the assistant turn that is appended to
# the conversation. Why: vLLM splits the response into `reasoning` (the <think> block) and
# `content`, and only `content` is ever appended to `messages`. On a tool-calling turn `content`
# is frequently a stub -- measured on yaegi-go-embed-directives, the visible text was
# "calling list_dir" while the whole intent ("check _test for embed-related tests") sat in the
# thinking block. So a retry could see WHAT was done but never WHY, even though patch.md claims
# it can see what the role "concluded".
# TAIL, not head: a thinking block ends with the decision and opens with orientation.
# v8 DEFAULT 0. Measured on identical tasks at identical concurrency (v6 vs v7-n4):
# v7 steps ran 10.5s median vs v6's 7.8s -- ~35% slower -- while v7-n4 carried
# SEEDLING_REASONING_NOTE=6000 with KEEP=12, i.e. up to 72,000 chars (~18K tokens)
# appended to EVERY request against a ~40K baseline. The note taxes every call; the new
# compaction carries intent forward only when it fires, and does it better (the summary
# has a "Problem Solving"/"Errors and fixes" section). Set >0 only to re-test the note.
# REASONING_NOTE_CHARS REMOVED. Reasoning is carried natively on the `reasoning` key of
# every assistant message (see roles._assistant) and bounded by compaction. The text-note
# hack re-injected thinking as prose; and its "0 = off" default was `s[-0:]`, the whole string.
# Assistant messages are NEVER trimmed by trim_history, so these notes would otherwise grow
# without bound (80 steps x 400 chars ~ 32KB per role). Keep them on recent turns only.
REASONING_NOTE_KEEP = int(_os.environ.get("SEEDLING_REASONING_KEEP", "12"))
EXEC_TIMEOUT_SEC = 600

# -- verify --------------------------------------------------------------------------
VERIFY_BUILD_TIMEOUT_SEC = 600
# Measured in smoke #4: dateutil's regression suite (2,035 tests) hit a 1500s cap AND the
# baseline re-run hit another 1500s -- 50 MINUTES of a 3h budget on ONE gate, both timing
# out, yielding nothing. The run took 135m (vs 32m in smoke #3) and scored WORSE
# (f2p 52/67 vs 57/67). Verification time comes OUT OF patch time, it is not free.
VERIFY_TEST_TIMEOUT_SEC = 600
# Never let verification eat more than this share of the REMAINING budget.
VERIFY_MAX_BUDGET_FRAC = 0.12

# -- paths ---------------------------------------------------------------------------
REPO = "/app"
SCRATCH = "/tmp/seedling/scratch"   # NEVER write scratch tests under /app -- a leftover
                                    # test file survives into the verifier and can fail
                                    # the whole p2p bucket, scoring 0.

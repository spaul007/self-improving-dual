"""One Role dataclass, one generic ReAct loop, three instances.

Roles are DATA; the pipeline is CODE. Adding a role means adding a Role(...) instance and
a prompt file -- no new machinery.

TERMINATION AND STRUCTURED OUTPUT ARE THE SAME MECHANISM. Each role gets a synthesized
`finish` tool whose JSON schema is generated from its `output` spec. Calling it both ends
the role and carries its result. That unifies three things:
  * termination is a tool call, so it flows through the existing loop
  * structured output is just the call's arguments
  * a bad output is SELF-REPAIRING -- a missing key returns a normal tool-error
    observation and the loop continues, costing one step, with no special error path
We do not use guided decoding: tool-calling is the settled protocol for this stack, and
vLLM's guided decoding interacts badly with Qwen thinking mode.

A role that exhausts its budget without calling finish returns {"_incomplete": true, ...}.
THE PIPELINE NEVER ABORTS ON THIS -- partial work can still score, and a Patch role that
ran out of steps must still reach the commit.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import compact, settings
from .gitops import TEST_SHAPED as _TEST_SHAPED
import re as _re
def _assistant(content, reasoning, tool_calls=None) -> dict:
    """Build an assistant message that CARRIES THE MODEL'S THINKING FORWARD.

    vLLM forwards prior reasoning under the key `reasoning` (its own response field name --
    NOT `reasoning_content`, which it silently drops; measured: 76 vs 877 prompt tokens). The
    Qwen3.8 template renders it inside <think> tags, so the model reads it as its own thinking.
    This is the proper channel; v7's text note re-injected thinking as PROSE, which the model
    then treated as something it had said. One continuous conversation per role now means
    content + tool calls + tool results + REASONING, bounded by compaction.
    """
    m = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning"] = reasoning
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


import re as _re
# Commands that pull code or metadata from the internet (permitted by DeepSWE -- every task
# declares [agent] network_mode="public" -- but a signal the evolution loop must see), and the
# narrower case of fetching from the benchmark's own source fork, which carries the solution.
_NET_FETCH = _re.compile(r"(curl|wget)[^\n|;&]*?(api\.github\.com|raw\.githubusercontent\.com|"
                         r"github\.com/[^ ]+/(pull|commit|compare)|pypi\.org/simple)|"
                         r"git (clone|fetch|remote add)[^\n;|&]*https?://", _re.I)
_FORK_FETCH = _re.compile(r"blitzy-research", _re.I)
# A test runner actually being executed. Counted per role run so the pipeline can refuse a
# VERIFY `pass` that ran nothing AFTER the patch: in EXP-024 run 1, geo-shapeindex's VERIFY
# ran 2 steps in 33s, wrote no test, and cited the BASELINE run on the unmodified repo as its
# evidence -- and the evidence gate accepted it because the runner name matched.
_TEST_RUN = _re.compile(r"\b(pytest|go test|cargo test|npm (?:run )?test|yarn test|pnpm (?:exec )?(?:test|jest|vitest|mocha)|"
                        r"make test|tox\b|jest|vitest|mocha|phpunit|mvn test|gradle test|dotnet test|python -m unittest|"
                        r"node --test)\b", _re.I)


def arg_path(args: dict) -> str:
    """The filesystem path a tool call targets, whatever the parameter is called.

    v8 renamed every file tool's parameter to `file_path` to match Claude Code. Two call
    sites still looked up the OLD key, so both silently returned nothing: the `source_writes`
    attribution counter and the `touched` file list would have reported 0 for every trial of
    the whole run -- and a zero there reads as "clean", not as "broken". Same falsy-zero
    failure as the `p2p or 1.0` monitor that reported clean for 8 hours while two trials sat
    at 0.0.

    Checked against the live schemas by tests/test_wiring.py, so a future rename fails a test
    instead of silently zeroing the metrics.
    """
    for k in ("file_path", "path"):
        v = args.get(k)
        if v:
            return str(v)
    return ""


TEST_SHAPED_RE = _re.compile(_TEST_SHAPED, _re.I)
from .llm import LLMError
from .tools import for_role

PROMPTS = Path(__file__).parent / "prompts"

# Steps held back so a role can always deliver a structured answer before being cut off.
FINISH_RESERVE = 3

_JSON = {"string": {"type": "string"},
         "string_list": {"type": "array", "items": {"type": "string"}}}


@dataclass
class Harness:
    """What a role is allowed to touch. Assembled once per trial by agent.py."""
    pool: object
    llm: object
    traj: object
    git: object
    deadline: object
    logger: object
    # Called after every role finishes. agent.py wires this to a SYNCHRONOUS trajectory +
    # run_summary write, so a hard kill (not a clean cancel) still leaves observability
    # behind for every role that completed.
    on_progress: object = None
    # ledger(record: dict) -- one line per tool call, appended to exec_log.jsonl as it
    # happens (survives SIGKILL). snapshot(role, attempt, messages) -- the exact conversation
    # as the model saw it, written at role end. Both are the artifacts that would have shown
    # the A0 system-prompt bug and the report-residue bug on their first trial.
    ledger: object = None
    snapshot: object = None


class _RunState:
    """Per-run counters for ONE role invocation.

    WHY THIS IS NOT ON THE ROLE. `PATCH`/`VERIFY`/`SOLO` are module-level singletons, and pier
    runs concurrent trials as asyncio tasks in a SINGLE process (verified: one `pier run -n 3`).
    So anything assigned to `self` inside `run()` is shared by every trial at once.

    Measured harm, v8 smoke: katex made ZERO Edit/Write calls in 51 steps, but its role-end line
    reported `edits=41` -- exactly dateutil's 38 Edit + 3 Write. Worse than wrong numbers, it
    disabled a guard: the half-wall nudge fires on `edits == 0`, so katex -- the one task that
    needed it, and the task the nudge was WRITTEN for -- never got nudged, because a different
    trial had edited. The zero-edit pushback and the shared `truncations < 3` retry budget failed
    the same way. At n=4/n=8 (every production run) these guards were largely inert.
    """

    __slots__ = ("edits", "source_writes", "compactions", "reasoning_chars", "truncations",
                 "last_prompt_tokens", "net_fetch", "fork_fetch", "compact_stats", "tests_run",
                 "source_writes_refused", "nudges", "pushes", "transients_deleted", "pending",
                 "stop_reason", "turns", "tool_calls", "cap_hits", "reads_without_limit",
                 "compact_skip_until", "compact_backoffs")

    def __init__(self) -> None:
        self.edits = 0
        self.source_writes = 0
        self.compactions = 0
        self.reasoning_chars = 0
        self.truncations = 0
        self.last_prompt_tokens = 0
        self.net_fetch = 0        # Bash commands that pull code/metadata from the internet
        self.fork_fetch = 0       # ...specifically from the benchmark's source fork
        self.compact_stats = {"compacted": 0, "rejected": 0, "stub_attempts": 0}
        self.tests_run = 0        # Bash commands that ran a test runner IN THIS ROLE RUN
        self.source_writes_refused = 0   # Edit/Write outside scratch refused for a confined role
        self.nudges = 0           # half-wall "make the edit" messages issued
        self.pushes = 0           # end_turn-with-0-edits pushes issued
        self.transients_deleted = 0      # harness messages removed after the model replied
        self.pending: list = []   # harness messages awaiting deletion (by identity)
        self.stop_reason = "running"
        self.turns: list = []     # per-turn LLM record: tokens, finish_reason, latencies
        self.tool_calls = 0
        self.cap_hits = 0         # tool results clipped at MAX_TOOL_OUTPUT_CHARS
        self.reads_without_limit = 0
        self.compact_skip_until = -1     # v8.3: after a REJECTED compaction, do not retry until this step
        self.compact_backoffs = 0


def _transient(messages: list, st: _RunState, content: str, role: str = "user") -> dict:
    """Append a HARNESS-authored message that the model must see ONCE.

    Every harness message except the per-attempt context is transient: it is deleted from the
    conversation after the model's next reply, by `_sweep_transients`. Why: the report prompt
    left in a shared conversation compounded into "report again" loops (EXP-025); the half-wall
    nudge, the zero-edit push and the truncation retry persisted the same way, so attempt N+1
    read attempt N's "you are halfway through your time" as a statement about now. The model's
    own reply to a transient stays -- those are its words, not ours.
    """
    m = {"role": role, "content": content}
    messages.append(m)
    st.pending.append(m)
    return m


def _sweep_transients(messages: list, st: _RunState) -> int:
    """Delete pending harness messages by identity. Idempotent; safe after compaction (a
    summarised transient is simply no longer in the list)."""
    if not st.pending:
        return 0
    ids = {id(m) for m in st.pending}
    before = len(messages)
    messages[:] = [m for m in messages if id(m) not in ids]
    n = before - len(messages)
    st.pending = []
    st.transients_deleted += n
    return n


@dataclass
class Role:
    name: str
    prompt_file: str
    tools: list = field(default_factory=list)
    readonly: bool = False
    # v7: confine this role's MUTATING tools to settings.SCRATCH. Measured in EXP-020: VERIFY
    # was given write_file/edit_file so it could write a test, and on
    # optique-conditional-option-dependencies it used them to edit /app SOURCE instead (4 calls)
    # -- the reviewer doing the implementer's job, which confounds the very comparison the
    # experiment exists to make. The prompt said "write your test in /tmp/seedling/scratch/"
    # and never said "do not edit source", because it did not occur to me that granting a
    # capability for one purpose grants it for all. Enforce host-side rather than ask: a
    # permission that looks enforced but isn't is worse than one documented as advisory.
    flag_source_writes: bool = False
    output: dict = field(default_factory=dict)
    # Which per-task conversation this role appends to. Defaults to its own name. BASELINE
    # sets "verify" so that everything it learns establishing the test baseline -- the
    # command, the suite's quirks, the pre-existing failures -- is REMEMBERED by VERIFY on
    # every later attempt. v8's continuous per-role conversation makes a baseline file
    # unnecessary: the knowledge lives where it is used.
    conversation_key: str = ""

    @property
    def mutates_expected(self) -> bool:
        """True for roles whose job is to change files -- only those get the edit nudge and
        the zero-edit push on end_turn.

        Derived from the registry's `mutates` flag, NOT from a hardcoded name list. The
        previous version tested `{"write_file","edit_file"} & set(self.tools)`; renaming the
        tools to Claude Code's `Edit`/`Write` silently made it False for every role, which
        disabled both nudges without a single error. Ask the tools what they do."""
        if self.readonly:
            return False
        from .tools import registry
        reg = registry()
        return any(reg[t].mutates for t in self.tools if t in reg)

    # -- prompt ------------------------------------------------------------------------
    def report_prompt(self) -> str:
        """Asked ONCE after the role stops, to collect the structured output the pipeline
        needs. Deliberately framed as reporting on finished work rather than as a decision
        to stop -- the stopping already happened."""
        fields = "\n".join(f"  {k} -- {v.get('description','')}"
                            for k, v in self.output.items())
        return ("You have finished working. Report what you did by calling `finish` exactly "
                "once, with these fields:\n" + fields +
                "\n\nBe concrete: real commands you ran, real output you saw, real file "
                "paths you changed. Do not start new work.")

    def system_prompt(self) -> str:
        p = PROMPTS / self.prompt_file
        try:
            return p.read_text()
        except OSError:
            return f"You are the {self.name} role of a software engineering agent."

    def budget(self, deadline=None) -> dict:
        """Wall budget scales with the ACTUAL time granted, never a hardcoded absolute."""
        b = dict(settings.BUDGETS.get(self.name, {"wall_frac": 0.3}))
        window = (deadline.soft - deadline.t0) if deadline is not None else 7200.0
        b["max_wall_sec"] = max(120.0, window * b.pop("wall_frac", 0.3))
        return b

    # -- the synthesized finish tool ---------------------------------------------------
    def finish_schema(self) -> dict:
        props, required = {}, []
        for key, spec in self.output.items():
            t = spec.get("type", "string")
            if t == "enum":
                props[key] = {"type": "string", "enum": list(spec.get("values", []))}
            else:
                props[key] = dict(_JSON.get(t, _JSON["string"]))
            props[key]["description"] = spec.get("description", "")
            if spec.get("required", True):
                required.append(key)
        return {"type": "function", "function": {
            "name": "finish",
            "description": "Call this when done. Ends your turn and reports your result.",
            "parameters": {"type": "object", "properties": props, "required": required}}}

    def validate(self, args: dict) -> str | None:
        """Return an error string to feed back to the model, or None if the output is ok."""
        for key, spec in self.output.items():
            if spec.get("required", True) and key not in args:
                return f"finish rejected: missing required field {key!r}"
            if key in args and spec.get("type") == "enum":
                vals = list(spec.get("values", []))
                if args[key] not in vals:
                    return f"finish rejected: {key!r} must be one of {'|'.join(vals)}"
            if key in args and spec.get("type") == "string_list" \
                    and not isinstance(args[key], list):
                return f"finish rejected: {key!r} must be a list of strings"
        return None

    # -- the ReAct loop ----------------------------------------------------------------
    async def run(self, bb, h: Harness) -> dict:
        budget = self.budget(h.deadline)
        tools = for_role(self.tools, readonly=self.readonly)
        # NO `finish` tool during the loop. Termination is end_turn; the structured output
        # is collected afterwards by the report call. Offering `finish` here is what made
        # stopping an extra action competing with working (fired 1 time in 6).
        schemas = [t.schema() for t in tools.values()]
        # Record the ACTUAL context handed to this role, plus which handoff channels were
        # populated. Previously the seed step held only `bb.task[:400]`, so the handoff --
        # the thing this architecture exists to do -- could not be audited from the
        # trajectory at all. Verified the hard way: I twice mis-read an absent label in the
        # seed step as "the patch role never received the compiler errors".
        ctx = bb.context_for(self.name)
        manifest = {
            "channels_present": [lbl for lbl in
                                 ("Task", "Reviewer notes", "What the patch role changed",
                                  "The change you have made SO FAR", "Your BASELINE",
                                  "Tests that ALREADY FAIL")
                                 if lbl in ctx],
            "context_chars": len(ctx),
            "attempt": bb.attempt,
        }
        rec = h.traj.role(self.name, brief=ctx, extra={"handoff": manifest})
        started = time.monotonic()
        _started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        st = _RunState()
        touched: list = []
        assistant_texts: list = []

        # RESUME, do not restart. On attempt 1 this seeds the conversation; on later
        # attempts it appends to the SAME list, so the role still remembers what it read,
        # tried, and ruled out. One conversation per role for the whole trial.
        messages = bb.conversation(self.conversation_key or self.name)
        # THE SYSTEM PROMPT IS THE CURRENT ROLE'S, EVERY TIME. Until EXP-026 it was seeded
        # once, by whichever role OPENED the conversation. BASELINE opens VERIFY's with
        # baseline.md -- "Do NOT judge the task, do NOT write tests" -- so every VERIFY attempt
        # in four runs ran under BASELINE's prompt and verify.md never entered its context.
        # kea's VERIFY said so when nudged: "my role is VERIFY -- I'm supposed to establish a
        # baseline ... not make any changes". compact.maybe_compact() preserves messages[0],
        # so this is the single point of truth; sys_sha on the END line proves it from artifacts.
        _sys_text = self.system_prompt()
        _sys = {"role": "system", "content": _sys_text}
        if messages and messages[0].get("role") == "system":
            messages[0] = _sys
        else:
            messages.insert(0, _sys)
        sys_sha = hashlib.sha256(_sys_text.encode()).hexdigest()[:8]
        rec.add_system_step(f"system prompt: {self.prompt_file} sha={sys_sha} "
                            f"chars={len(_sys_text)} conversation={self.conversation_key or self.name} "
                            f"resumed={len(messages) > 1}")
        h.logger.warning("role %s START attempt=%d planned_wall=%.0fs sys_sha=%s prompt=%s "
                         "conv_msgs=%d ~tok=%d", self.name, bb.attempt, budget["max_wall_sec"],
                         sys_sha, self.prompt_file, len(messages), compact.est_tokens(messages))
        # v8: no trim on resume. Growth is handled by compact.maybe_compact() inside the
        # loop, which summarises the old transcript instead of clipping it.
        # A resumed conversation needs an unmistakable turn boundary: on katex, VERIFY resumed
        # the conversation its BASELINE turn had used and believed it was still on that turn.
        if len(messages) > 1:
            ctx = (f"=== NEW TURN: you are now the {self.name.upper()} role, attempt {bb.attempt}. "
                   f"Everything above is your OWN earlier work in this task (possibly under a "
                   f"different role name); the repository may have changed since. Read the "
                   f"context below and act on it. ===\n\n" + ctx)
        messages.append({"role": "user", "content": ctx})
        result: dict | None = None
        last_text = ""

        nudged = False
        nudged_empty = False
        end_turn = False
        step = -1
        # v8: NO STEP CAP. The loop ends when the model stops calling tools (end_turn), or
        # when the wall/deadline expires. `max_steps` is gone: measured, the role expands to
        # fill whatever cap it is given (60->62, 120->122, 80->77), and Claude Code SOLVES
        # tasks in fewer steps than it fails them (122 vs 164). Steps were never the scarce
        # resource; the cap was just the thing that happened to stop the role.
        while True:
            step += 1

            # In-flight heartbeat: a long role otherwise logs NOTHING until it ends, so
            # "exploring" and "stuck" look identical while it runs.
            if step and step % 10 == 0:
                h.logger.warning("role %s step %d edits=%d touched=%d ~tok=%d compactions=%d",
                                 self.name, step, st.edits, len(touched),
                                 compact.est_tokens(messages), st.compactions)

            elapsed = time.monotonic() - started
            wall = budget["max_wall_sec"]

            # Harness messages the model has now replied to are removed here, at the top of
            # the next turn -- see _transient(). Everything below that appends a nudge goes
            # through _transient(), never messages.append().
            _sweep_transients(messages, st)

            # MID-BUDGET NUDGE, now anchored to WALL not steps (there is no step count to
            # halve any more). Measured in smoke #4: katex's patch.2 spent 54 bash calls on
            # git/grep/sed and made ZERO edits while holding verbatim compiler errors.
            if (not nudged and self.mutates_expected
                    and elapsed > 0.5 * wall and st.edits == 0):
                nudged = True
                st.nudges += 1
                _transient(messages, st,
                    "You are halfway through your time and have made ZERO edits. "
                    "Investigation is not the task. Make the edit now with Edit or Write. "
                    "An unedited repository scores zero.")
                h.logger.warning("role %s: half-wall nudge issued (0 edits) at step %d",
                                 self.name, step)

            if h.deadline.expired():
                st.stop_reason = "soft_deadline"
                h.logger.info("role %s stopping: soft deadline", self.name)
                break
            if elapsed > wall:
                st.stop_reason = "wall"
                h.logger.info("role %s stopping: wall budget", self.name)
                break

            # COMPACT before the call, so the request itself is bounded. Replaces the old
            # trim_history() clip of old tool output -- full fidelity recently, a structured
            # summary of everything older.
            try:
                # Pass the REAL size of the last request. chars/4 runs ~25% low on reasoning
                # and ~15% high on code, so the estimate fires compaction at the wrong moment
                # in both directions and is the only path to a 262,144-token overflow.
                # v8.3 BACK-OFF: a rejected compaction used to be retried on EVERY following
                # turn while the conversation stayed above threshold. EXP-027 yaegi PATCH.3
                # paid two summariser tries (~200s each) on each of five consecutive steps --
                # ~2,000 of its 2,361s -- and made 0 edits. After a rejection, skip 5 steps.
                if step >= st.compact_skip_until:
                    _rej_before = st.compact_stats.get("rejected", 0)
                    if await compact.maybe_compact(messages, self.name, h,
                                                   real_tokens=st.last_prompt_tokens,
                                                   stats=st.compact_stats):
                        st.compactions += 1
                    elif st.compact_stats.get("rejected", 0) > _rej_before:
                        st.compact_skip_until = step + 5
                        st.compact_backoffs += 1
                        h.logger.warning("role %s COMPACT BACK-OFF: rejected at step %d, next try at "
                                         "step %d", self.name, step, st.compact_skip_until)
            except Exception as e:  # noqa: BLE001 -- compaction must never kill a role
                h.logger.warning("role %s compaction raised %s -- continuing",
                                 self.name, type(e).__name__)

            _t_llm = time.monotonic()
            try:
                resp = await h.llm.chat(messages, tools=schemas)
            except LLMError as e:
                st.stop_reason = "llm_error"
                rec.add_system_step(f"llm error: {e}")
                h.logger.warning("role %s llm error: %s", self.name, e)
                break
            _llm_s = time.monotonic() - _t_llm
            _u = resp.get("usage") or {}
            # PER-TURN LLM RECORD. finish_reason was recorded only on the report call before;
            # a truncation rate could not be measured retroactively for any earlier run.
            _turn = {"step": step, "prompt_tokens": _u.get("prompt_tokens", 0),
                     "completion_tokens": _u.get("completion_tokens", 0),
                     "cached_tokens": _u.get("cached_tokens", 0),
                     "reasoning_chars": len(resp.get("reasoning") or ""),
                     "finish_reason": resp.get("finish_reason"), "llm_s": round(_llm_s, 2),
                     "tool_s": 0.0, "n_calls": len(resp.get("tool_calls") or [])}
            st.turns.append(_turn)

            st.last_prompt_tokens = int((resp.get("usage") or {}).get("prompt_tokens") or 0)
            last_text = resp["content"] or last_text
            if resp["content"]:
                assistant_texts.append(resp["content"])
            rec.n_llm_calls += 1
            calls = resp["tool_calls"]

            if not calls:
                # END_TURN -- this is now the NORMAL way a role finishes.
                #
                # v7 and earlier did the opposite: they replied "Use one of the provided
                # tools, or call finish" and continued, i.e. they OVERRODE the model's own
                # stop signal and demanded a structured `finish` call instead. Measured
                # consequence: PATCH called finish in 1 of 6 role instances, and two of those
                # had already been told to and ignored it. Claude Code has no finish tool at
                # all -- its loop ends on stop_reason=end_turn, 364 tool_use turns vs 2
                # end_turn in one trial. Stopping is the model's DEFAULT behaviour; asking
                # for an extra structured action makes termination compete with working.
                rec.add_agent_step(last_text or "(no content)", llm_call_count=1,
                                   model_name=h.llm.model,
                                   reasoning_content=resp.get("reasoning"),
                                   metrics={"prompt_tokens": _turn["prompt_tokens"],
                                            "completion_tokens": _turn["completion_tokens"],
                                            "cached_tokens": _turn["cached_tokens"] or 0},
                                   extra={"turn": dict(_turn)})

                # TRUNCATION IS NOT TERMINATION. A reasoning model can spend the whole
                # max_tokens budget inside its thinking block and return content="" with no
                # tool call and finish_reason="length". Read as `end_turn` that looks exactly
                # like "the model is done" -- but its own reasoning says otherwise: measured on
                # the first v8 smoke, a 33,410-char thinking block (~8.3K tokens vs the then
                # 8192 cap) ended mid-plan with "I need to do the following: 1. Add a function
                # ...", and PATCH finished with 0 edits on 3 of 3 tasks.
                #
                # This is also what v7's much-maligned "use one of the provided tools" prompt
                # was really compensating for. The fix is not to override a genuine stop, it is
                # to STOP MISREADING a truncated one.
                if resp.get("finish_reason") == "length" and st.truncations < 3:
                    st.truncations += 1
                    h.logger.warning("role %s: TRUNCATED generation (finish_reason=length, "
                                     "reasoning=%d chars) -- retrying, not treating as "
                                     "end_turn [%d/3]", self.name,
                                     len(resp.get("reasoning") or ""), st.truncations)
                    _r = resp.get("reasoning") or ""
                    st.reasoning_chars += len(_r)
                    # Both halves of the retry exchange are transient: the synthetic
                    # assistant turn is not something the model said, and "think BRIEFLY"
                    # must not govern the rest of the role.
                    _m = _assistant(last_text or "(thinking was cut off)", _r)
                    messages.append(_m)
                    st.pending.append(_m)
                    _transient(messages, st,
                        "Your previous reply was cut off before you produced anything. Think "
                        "BRIEFLY this time, then immediately make one tool call. Do not plan "
                        "further -- act on the plan you already have.")
                    continue

                # This turn's content, not `last_text` -- that carried a PREVIOUS turn's
                # words forward and appended them as if said now.
                _r = resp.get("reasoning") or ""
                st.reasoning_chars += len(_r)
                messages.append(_assistant(resp["content"] or "", _r))
                # ONE exception: a role that is supposed to edit and has not edited anything
                # is almost certainly not done. Give it exactly one push, then respect the
                # next end_turn regardless.
                if self.mutates_expected and st.edits == 0 and not nudged_empty:
                    nudged_empty = True
                    st.pushes += 1
                    _transient(messages, st,
                        "You stopped without making any edit. If the change is genuinely "
                        "complete, say so and stop again. Otherwise make the edit now.")
                    h.logger.warning("role %s: end_turn with 0 edits -- pushed once",
                                     self.name)
                    continue
                end_turn = True
                st.stop_reason = "end_turn"
                h.logger.info("role %s: END_TURN at step %d (edits=%d)",
                              self.name, step, st.edits)
                break

            from .trajectory import Call
            recorded, assistant_calls, tool_msgs = [], [], []
            done = False
            for c in calls:
                name, args = c["name"], (c["arguments"] or {})
                if name == "finish":
                    # Unreachable via the schema list (v8 does not offer `finish` in-loop),
                    # but a model can still emit the name from habit. Honour it rather than
                    # returning "unknown tool" -- it means the same thing as end_turn.
                    h.logger.info("role %s: finish emitted at step %d (not offered)",
                                  self.name, step)
                    err = self.validate(args)
                    out_text = err or "ok"
                    if not err:
                        result, done = dict(args), True
                else:
                    tool = tools.get(name)
                    if tool is None:
                        out_text = (f"unknown tool {name!r}; available: "
                                    f"{', '.join(sorted(tools))}, finish")
                    elif "_raw" in args:
                        # The model's tool-call arguments were not valid JSON, so llm.py
                        # parked the blob under `_raw`. Dispatching it would raise
                        # "unexpected keyword argument '_raw'" -- a message about a parameter
                        # the model never wrote, which sends it looking for a bug in its
                        # tool USE rather than in its JSON. Say what actually happened.
                        h.logger.warning("role %s: malformed tool-call JSON for %s",
                                         self.name, name)
                        out_text = (f"your arguments for {name} were not valid JSON, so the "
                                    f"call was not run. Re-send it with correctly escaped "
                                    f"JSON arguments. Received: {str(args.get('_raw'))[:300]}")
                    else:
                        try:
                            # CONFINEMENT IS ENFORCED for a role flagged `flag_source_writes`
                            # (VERIFY). Edit/Write outside SCRATCH or a test-shaped path is
                            # REFUSED host-side, not just counted. Why enforce now: EXP-026 kea's
                            # VERIFY wrote /app/multicolumn-evidence.js, that file entered
                            # `repo_diff`, and PATCH.2 was then told it was "the change you have
                            # made so far". The role seam is the object of study; a reviewer
                            # editing source blurs the one boundary the experiment measures.
                            # `bash` redirects still bypass this (bash is not `mutates`) -- the
                            # per-role git attribution in pipeline.py catches those after the fact.
                            _p = arg_path(args)
                            _confined = (self.flag_source_writes and tool.mutates and _p
                                         and not _p.startswith(settings.SCRATCH)
                                         and not TEST_SHAPED_RE.search(_p))
                            # FAILURE-CLASS SIGNALS. The diagnose step of a self-evolution loop
                            # reads flags, not trajectories; these were all found by hand today.
                            if name == "Bash":
                                _cmd = str(args.get("command") or "")
                                # v8.3: a single command may not consume the role's wall. The
                                # tool's own cap (EXEC_TIMEOUT_SEC=600) was ~the whole BASELINE
                                # wall (630s): kea's `pnpm test` timed out at 602s and the role
                                # had 28s left. Cap at half the role wall, floor 60s.
                                try:
                                    _want = int(args.get("timeout") or settings.EXEC_TIMEOUT_SEC)
                                except (TypeError, ValueError):
                                    _want = settings.EXEC_TIMEOUT_SEC
                                args["timeout"] = max(60, min(_want, settings.EXEC_TIMEOUT_SEC,
                                                              int(0.5 * budget["max_wall_sec"])))
                                if _NET_FETCH.search(_cmd):
                                    st.net_fetch += 1
                                if _TEST_RUN.search(_cmd):
                                    st.tests_run += 1
                                if _FORK_FETCH.search(_cmd):
                                    st.fork_fetch += 1
                                    h.logger.warning("role %s FORK FETCH: %s", self.name, _cmd[:120])
                            if name == "Read" and not args.get("limit"):
                                st.reads_without_limit += 1
                            _t_tool = time.monotonic()
                            if _confined:
                                st.source_writes += 1
                                st.source_writes_refused += 1
                                out_text = (f"refused: the {self.name.upper()} role may only write "
                                            f"under {settings.SCRATCH}/ (or a test-shaped path). "
                                            f"The source tree belongs to the PATCH role -- report "
                                            f"what is wrong in `issues` instead of editing it.")
                                h.logger.warning("role %s SOURCE-WRITE REFUSED %s", self.name, _p[:120])
                            else:
                                out_text = await tool.fn(h.pool, **args)
                            _tool_s = time.monotonic() - _t_tool
                            _turn["tool_s"] = round(_turn["tool_s"] + _tool_s, 2)
                            st.tool_calls += 1
                            if "chars elided]" in out_text:
                                st.cap_hits += 1
                            # EXEC LEDGER: one record per tool call, appended as it happens.
                            # Post-hoc analysis (duplicate commands, test runs, same-runner,
                            # fetches) reads THIS, so the harness never needs a gate regex.
                            if h.ledger is not None:
                                _rc = None
                                if out_text.startswith("[rc="):
                                    try:
                                        _rc = int(out_text[4:out_text.index("]")].split()[0])
                                    except (ValueError, IndexError):
                                        _rc = None
                                try:
                                    h.ledger({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                              "role": self.name, "attempt": bb.attempt, "step": step,
                                              "seq": st.tool_calls, "tool": name,
                                              "path": _p[:300] if _p else None,
                                              "command": (str(args.get("command") or "")[:2000]
                                                          if name == "Bash" else None),
                                              "description": (str(args.get("description") or "")[:200]
                                                              if name == "Bash" else None),
                                              "rc": _rc, "refused": out_text.startswith("refused"),
                                              "dur_s": round(_tool_s, 2), "out_chars": len(out_text),
                                              "out_sha": hashlib.sha1(out_text.encode("utf-8", "replace")).hexdigest()[:10]})
                                except Exception:  # noqa: BLE001 -- observability never breaks a role
                                    h.logger.exception("ledger write failed (non-fatal)")
                            if tool.mutates and not out_text.startswith(("refused", "[rc=")):
                                st.edits += 1
                            # `touched` feeds _synthesize's `changed_files`. Gate on mutates:
                            # ungated, kea's failed report handed VERIFY 12 'changed' files
                            # having made 0 edits -- they were files it had only Read.
                            pth = arg_path(args)
                            if (tool.mutates and pth and pth not in touched
                                    and not out_text.startswith(("refused", "[rc="))):
                                touched.append(pth)
                        except TypeError as e:
                            out_text = f"bad arguments for {name}: {e}"
                recorded.append(Call(name, args, result=out_text))
                assistant_calls.append({"id": c["id"] or f"c{len(assistant_calls)}",
                                        "type": "function",
                                        "function": {"name": name,
                                                     "arguments": json.dumps(args)}})
                tool_msgs.append({"role": "tool",
                                  "tool_call_id": c["id"] or f"c{len(tool_msgs)}",
                                  "content": out_text})

            rec.add_agent_step(last_text or f"calling {calls[0]['name']}",
                               calls=recorded, llm_call_count=1,
                               model_name=h.llm.model,
                               reasoning_content=resp.get("reasoning"),
                               metrics={"prompt_tokens": _turn["prompt_tokens"],
                                        "completion_tokens": _turn["completion_tokens"],
                                        "cached_tokens": _turn["cached_tokens"]},
                               extra={"turn": dict(_turn)})
            _r = resp.get("reasoning") or ""
            st.reasoning_chars += len(_r)
            messages.append(_assistant(last_text or None, _r, assistant_calls))
            messages.extend(tool_msgs)
            if done:
                st.stop_reason = "finish_call"
                break

        # Nothing harness-authored may outlive the loop: the next role sharing this
        # conversation, and the report call below, must see only the model's own turns.
        _sweep_transients(messages, st)

        # THE REPORT CALL. The loop no longer ends on a `finish` tool, so the structured
        # output the pipeline needs (PATCH: summary/changed_files; VERIFY: verdict/
        # test_command/evidence -- the retry loop EXITS on those) is obtained by one extra
        # call with the schema forced. This keeps termination free (end_turn) while keeping
        # the structured handoff guaranteed.
        report_ok = False
        if result is None:
            try:
                # The report exchange is HARNESS bookkeeping, not agent work. Left in the
                # conversation it leaked into the next role sharing it: VERIFY.1 on katex read
                # the BASELINE role's "Report what you did by calling finish" as its own
                # instruction, re-emitted the baseline report, and ran nothing.
                _n_before_report = len(messages)
                messages.append({"role": "user", "content": self.report_prompt()})
                resp = await h.llm.chat(
                    messages, tools=[self.finish_schema()],
                    # Named forcing, verified on the live server: returns the `finish`
                    # call every time; "auto" let the model answer in prose instead.
                    tool_choice={"type": "function", "function": {"name": "finish"}})
                # #7: the report call is a real LLM turn -- record it, and read its
                # finish_reason. Before this it was invisible in every artifact, so its
                # three failures in run 5 could not be diagnosed at all.
                rec.n_llm_calls += 1
                _fr = resp.get("finish_reason")
                rec.add_agent_step(resp.get("content") or "(report call)", llm_call_count=1,
                                   model_name=h.llm.model,
                                   reasoning_content=resp.get("reasoning"),
                                   extra={"report_call": True, "finish_reason": _fr,
                                          "tool_calls": [c["name"] for c in (resp.get("tool_calls") or [])]})
                if _fr == "length":
                    st.truncations += 1
                    h.logger.warning("role %s: REPORT CALL truncated (finish_reason=length, "
                                     "reasoning=%d chars)", self.name,
                                     len(resp.get("reasoning") or ""))
                for c in (resp.get("tool_calls") or []):
                    if c["name"] == "finish":
                        args = c["arguments"] or {}
                        err = self.validate(args)
                        if not err:
                            result, report_ok = dict(args), True
                        else:
                            h.logger.warning("role %s report call rejected: %s",
                                             self.name, err)
                        break
                if result is None and (resp.get("content") or "").strip():
                    assistant_texts.append(resp["content"])
            except Exception as e:  # noqa: BLE001 -- never let the report call kill the role
                h.logger.warning("role %s report call failed (%s: %s)",
                                 self.name, type(e).__name__, e)

        # Drop the report prompt (and anything the report call appended after it) so the
        # conversation ends at the role's last real turn.
        if "_n_before_report" in locals():
            del messages[_n_before_report:]

        if result is None:
            # Report call failed or was rejected. NEVER hand the next role a bare truncated
            # message -- that is what made 70% of handoffs useless. Reconstruct the declared
            # output fields from what the role ACTUALLY DID.
            result = self._synthesize(assistant_texts, touched, h)
            rec.finish("incomplete", json.dumps(result, default=str)[:300])
        else:
            rec.finish("ok", json.dumps(result, default=str)[:300])

        # Every v8 mechanism emits a counter here. Anything that only touches `messages` is
        # invisible to every artifact -- that shape has already produced two unfalsifiable
        # criteria in this project, so end_turn/compactions/report_ok are all counted.
        # The system prompt the model ACTUALLY had on its last request, hashed -- proves from
        # the artifact which role's prompt this run was governed by (the A0 bug was invisible
        # to every artifact because system messages are never recorded in the trajectory).
        _sys_now = (messages[0].get("content") if messages and messages[0].get("role") == "system" else "") or ""
        _sys_sha_now = hashlib.sha256(str(_sys_now).encode()).hexdigest()[:8]
        if _sys_sha_now != sys_sha:
            h.logger.error("role %s SYSTEM PROMPT MISMATCH: started %s, ended %s -- a shared "
                           "conversation was re-seeded by another role", self.name, sys_sha, _sys_sha_now)
        h.logger.warning("role %s END edits=%d source_writes=%d refused=%d steps=%d wall=%.0fs "
                         "stop=%s end_turn=%s compactions=%d report_ok=%s touched=%d ~tok=%d "
                         "reasoning_chars=%d truncations=%d nudges=%d pushes=%d transients_deleted=%d "
                         "cap_hits=%d sys_sha=%s",
                         self.name, st.edits, st.source_writes, st.source_writes_refused, step + 1,
                         time.monotonic() - started, st.stop_reason, end_turn, st.compactions,
                         report_ok, len(touched), compact.est_tokens(messages),
                         st.reasoning_chars, st.truncations, st.nudges, st.pushes,
                         st.transients_deleted, st.cap_hits, _sys_sha_now)

        h.traj.dispatch(self.name, rec, ctx[:200],
                        json.dumps(result, default=str)[:8000])
        _elapsed = time.monotonic() - started
        _fr = [t.get("finish_reason") for t in st.turns]
        bb.record_stats(self.name, {
            "role": self.name, "attempt": bb.attempt, "steps": step + 1,
            "started_utc": _started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "planned_wall_sec": round(budget["max_wall_sec"]),
            "wall_sec": round(_elapsed), "sec_per_step": round(_elapsed / max(step + 1, 1), 1),
            "stop_reason": st.stop_reason,
            "end_turn": end_turn, "wall_terminated": (not end_turn) and _elapsed >= budget["max_wall_sec"],
            "edits": st.edits, "zero_edit": self.mutates_expected and st.edits == 0,
            "touched": len(touched), "source_writes": st.source_writes,
            "source_writes_refused": st.source_writes_refused,
            "compactions": st.compact_stats["compacted"], "stub_rejections": st.compact_stats["rejected"],
            "stub_attempts": st.compact_stats["stub_attempts"], "truncations": st.truncations,
            "reasoning_chars": st.reasoning_chars, "report_ok": report_ok,
            "net_fetch": st.net_fetch, "fork_fetch": st.fork_fetch, "tests_run": st.tests_run,
            "nudges": st.nudges, "pushes": st.pushes, "transients_deleted": st.transients_deleted,
            "tool_calls": st.tool_calls, "cap_hits": st.cap_hits,
            "reads_without_limit": st.reads_without_limit,
            "sys_sha": sys_sha, "sys_sha_end": _sys_sha_now, "prompt_file": self.prompt_file,
            "llm_calls": rec.n_llm_calls,
            "llm_sec": round(sum(t.get("llm_s", 0) for t in st.turns), 1),
            "tool_sec": round(sum(t.get("tool_s", 0) for t in st.turns), 1),
            "prompt_tokens": sum(t.get("prompt_tokens", 0) for t in st.turns),
            "completion_tokens": sum(t.get("completion_tokens", 0) for t in st.turns),
            # None when the server never reported prompt_tokens_details (v8.3) -- a 0 here
            # was a false zero for every run before EXP-027's audit.
            "cached_tokens": (sum(t["cached_tokens"] for t in st.turns if t.get("cached_tokens") is not None)
                              if any(t.get("cached_tokens") is not None for t in st.turns) else None),
            "compact_backoffs": st.compact_backoffs,
            "finish_length": sum(1 for f in _fr if f == "length"),
            "finish_stop": sum(1 for f in _fr if f == "stop"),
            "finish_tool_calls": sum(1 for f in _fr if f == "tool_calls"),
            "conv_msgs_end": len(messages), "conv_tokens_end": compact.est_tokens(messages),
        })
        bb.record(self.name, result)
        # CONVERSATION SNAPSHOT: exactly what the model was shown, post-sweep. The only
        # artifact that can prove what a role received rather than what the harness meant.
        if h.snapshot is not None:
            try:
                h.snapshot(self.name, bb.attempt, messages, {"sys_sha": sys_sha, "stop_reason": st.stop_reason,
                                                             "turns": st.turns})
            except Exception:  # noqa: BLE001
                h.logger.exception("conversation snapshot failed (non-fatal)")
        if h.on_progress is not None:
            try:
                h.on_progress()
            except Exception:  # noqa: BLE001 -- observability must never break a role
                h.logger.exception("on_progress failed (non-fatal)")
        return result


    def _synthesize(self, texts: list, touched: list, h) -> dict:
        """Build the role's declared output from observed behaviour, not a truncated tail.

        A role that ran out of steps still did real work: it read files, edited files, and
        said things. Passing `{"_incomplete": true, "summary": "<last 2000 chars>"}` throws
        all of that away. Here every declared output key gets a best-effort value of the
        right TYPE, so the next role receives something it can act on.
        """
        prose = "\n\n".join(t for t in texts[-4:] if t)[:4000]
        out: dict = {"_incomplete": True}
        for key, spec in self.output.items():
            t = spec.get("type", "string")
            if t == "string_list":
                out[key] = touched[:25] if touched else []
            elif t == "enum":
                vals = list(spec.get("values", []))
                # An unfinished verification is NOT a pass -- same rule as the gate.
                out[key] = ("fail" if "fail" in vals else (vals[0] if vals else ""))
            else:
                out[key] = prose or "(role was cut off before reporting)"
        out["_note"] = (f"{self.name} ended without a usable report (wall, deadline, or a report call that returned no tool call); "
                        f"fields reconstructed from {len(touched)} touched file(s) and its "
                        f"last messages.")
        h.logger.warning("role %s INCOMPLETE -- synthesized %s from %d touched files",
                         self.name, list(self.output), len(touched))
        return out


# --------------------------------------------------------------------------------------
# THE ROLES. This is the multi-agent architecture's data half -- add or remove instances.
# --------------------------------------------------------------------------------------
PATCH = Role(
    name="patch", prompt_file="patch.md",
    tools=["Bash", "Read", "Edit", "Write"],
    output={"summary": {"type": "string", "description": "What you changed and why."},
            "changed_files": {"type": "string_list", "description": "Files you edited."},
            },
)

VERIFY = Role(
    name="verify", prompt_file="verify.md", readonly=False, flag_source_writes=True,
    # v6: VERIFY WRITES ITS OWN TESTS. Rationale, from the complete mini-swe run (n=112):
    # p2p mean 0.9978 with only 8/112 regressions, while f2p shortfalls cost 94 trials --
    # so the pre-existing suite, which is all VERIFY could measure before, is the one signal
    # that almost never fails. The graded f2p tests do NOT exist in this container
    # (`test.patch` creates them at grade time, verified: no /tests, no test.patch on the
    # agent filesystem), so the only honest proxy is a test written from `instruction.md`.
    #
    # readonly=False is LESS dangerous than it looks: `readonly` only strips tools flagged
    # mutates=True (write_file/edit_file). VERIFY already had `bash` and could always write
    # files -- through the UNGUARDED path. write_file/edit_file run `denied_path` host-side
    # first, which blocks conftest.py / pytest.ini / tox.ini / lockfiles / test.sh, i.e.
    # DeepSWE's own anti-cheat tripwires. So this CHANNELS writes through the enforced path.
    tools=["Bash", "Read", "Edit", "Write"],
    # EVIDENCE IS STRUCTURAL, NOT REQUESTED. There is no hard-coded gate behind this role
    # any more, so these fields are the only thing keeping a verdict honest. Both mechanisms
    # have lied: the deterministic gate false-passed a tree with 230 broken tests (4.3s,
    # rc=0), and in smoke #3 this role passed a patch breaking 11 tests after two steps.
    # ITEM 5 -- ADVERSARIAL VERIFY. Measured over 148 graded tasks: when VERIFY said `pass` the
    # grader failed 74-75% of the time, and 52% of those were near-misses (f2p >= 0.9) -- the test
    # was real but shallow. EXP-024's first graded task reproduced it: katex f2p 0.947 after a
    # VERIFY that re-ran the suite, wrote no test, and passed. So VERIFY now ENUMERATES the
    # behaviours the task requires and reports which its OWN test exercised; the pipeline
    # accepts a pass only when every one is covered.
    output={"verdict": {"type": "enum", "values": ["pass", "fail"],
                        "description": "pass ONLY if EVERY behaviour in `behaviours` appears in "
                                       "`behaviours_tested` (demonstrated by a test YOU wrote and ran) "
                                       "AND the suite shows no regression vs your baseline. Any "
                                       "untested behaviour means fail."},
            "behaviours": {"type": "string_list",
                           "description": "EVERY distinct behaviour and edge case the task statement "
                                          "requires, one per entry, specific (inputs, outputs, errors, "
                                          "names). Required."},
            "behaviours_tested": {"type": "string_list",
                                  "description": "The entries of `behaviours` that your OWN test in "
                                                 "/tmp/seedling/scratch/ exercised and that PASSED. Copy "
                                                 "the wording from `behaviours`. Required."},
            "build_command": {"type": "string",
                              "description": "The exact build command you ran, or 'none' if "
                                             "this project has no build step."},
            "test_command": {"type": "string",
                             "description": "The exact test command you ran. Required."},
            "evidence": {"type": "string",
                         "description": "Verbatim excerpt of the test output: the pass/fail "
                                        "counts and any failure lines. Required."},
            "issues": {"type": "string_list",
                       "description": "Concrete failures for the patcher to fix."}},
)

BASELINE = Role(
    name="baseline", prompt_file="baseline.md", tools=["Bash", "Read"], readonly=True,
    conversation_key="verify",
    # Runs ONCE per task before PATCH attempt 1, on the unmodified repository. Why: VERIFY
    # could not tell pre-existing failures from regressions (dateutil run 8 burned an attempt
    # on 229 failures that predated the patch) and rediscovered a different test scope every
    # attempt (566 vs 2,031 tests on one repo). Every task differs, so the AGENT finds the
    # command -- a heuristic cannot -- and the pipeline validates what it found.
    output={"test_command": {"type": "string",
                             "description": "The exact canonical FULL-suite test command you ran. Required."},
            "tests_collected": {"type": "string", "description": "Number of tests collected/run, as a number."},
            "passed": {"type": "string", "description": "Number passed, as a number."},
            "failed": {"type": "string", "description": "Number failed or errored, as a number."},
            "failing_tests": {"type": "string_list",
                              "description": "Names/ids of the tests that FAIL on the unmodified repo."},
            "build_command": {"type": "string", "description": "Build command run, or 'none'."},
            "build_ok": {"type": "enum", "values": ["yes", "no", "none"],
                         "description": "Did the build succeed? 'none' if no build step."},
            "duration_sec": {"type": "string", "description": "Wall seconds the test command took, as a number."},
            "notes": {"type": "string", "description": "Anything VERIFY must know to run this suite again."}},
)

SOLO = Role(
    name="solo", prompt_file="solo.md",
    tools=["Bash", "Read", "Edit", "Write"],
    output={"summary": {"type": "string", "description": "What you changed."},
            "changed_files": {"type": "string_list", "description": "Files you edited."}},
)

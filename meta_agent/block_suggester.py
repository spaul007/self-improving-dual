"""Per-EXPAND block-level improvement suggestion.

Companion to failure_summarizer.py / behavior_summarizer.py: same
philosophy (cheap deterministic framing in code, one LLM call to produce
grounded free text), but scoped to exactly one "block" -- a category of
editable surface within the MAS (individual_subagent / collaboration_workflow
/ foundation_capability / verifiers) -- rather than the whole agent or the
whole failure set.

The suggestion is NOT a file edit -- it is fed as extra steering context
into AgentEditor.apply's ``context`` argument (see
hgm.py::HGMManager._render_expand_context / ._expand). AgentEditor is
unchanged and still does all the actual diagnosis-that-writes-code, with
its own full view of the current mutable surface; this module only narrows
WHICH PART of the MAS this EXPAND should focus on and front-loads a
well-grounded diagnosis for that slice. Steer, don't fence (see
tier_based_hgm.md): the block is described by role/interface in the prompt
text, never enforced by file/region fencing, and AgentEditor is never told
what block was sampled -- it just receives this suggestion as one more
paragraph of context, same as the failure summary or behavior memory.

Persists block_suggestion_prompt.txt / block_suggestion.md into the CHILD
round's out_dir (the EXPAND being steered), not the parent -- this is a
per-EXPAND-attempt artifact, unlike failure/behavior summaries which are
per-node evaluation artifacts.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from . import source_context
from .models import AgentFeedback
from .registry import register

_SYSTEM_PREAMBLE = (
    "You are the block-suggestion module of a self-evolving multi-agent "
    "system (MAS). Your job is NOT to write code and NOT to produce a file "
    "edit -- a separate self-improvement editor does that, using your "
    "suggestion as one input among several. Your only job is to diagnose a "
    "real, evidence-grounded problem scoped to exactly ONE block of the MAS "
    "(described below) and propose ONE concrete, scoped improvement to it.\n\n"
    "You are given: the MAS's full current source code, (when available) "
    "its immutable tool implementations and database schema, and a "
    "feedback digest from the most recent evaluation of the agent you are "
    "diagnosing (score, tool error rates, project metrics, runtime "
    "exceptions, and -- when available -- an LLM-synthesized cross-case "
    "failure summary).\n\n"
    "Ground every claim in what you are actually shown. Before you assert "
    "a root cause (e.g. \"stage X never checks Y\", \"the prompt for role Z "
    "doesn't mention W\"), re-read the specific function, prompt string, or "
    "feedback field you are about to cite and confirm it really says what "
    "you are about to claim -- do not invent a plausible-sounding diagnosis "
    "you have not verified against the source or feedback shown to you. If "
    "a claim doesn't hold up on re-reading, drop it or hedge it (e.g. "
    "\"possibly\", \"this may contribute\") rather than stating it as "
    "settled fact. A correct, narrowly-scoped suggestion with an honest, "
    "hedged rationale is better than a confident but unverified one -- the "
    "downstream editor will trust what you say without re-checking it "
    "itself."
)

_SYSTEM_CLOSING = (
    "\n\nOutput plain markdown text only -- no tool call, no `files` "
    "payload, no code. Stay under 400 words. If the evidence available to "
    "you is too thin to support a specific diagnosis for this block (e.g. "
    "no failing case actually touches it yet), say so explicitly rather "
    "than inventing one; a hedged 'no strong signal for this block yet' is "
    "a valid and useful answer.\n\n"
    "If sibling edits already tried off this same parent are shown below, "
    "your suggestion must differ from them -- but this NEVER means "
    "leaving the block described above. Siblings may have targeted a "
    "DIFFERENT block entirely (e.g. one sibling touched "
    "individual_subagent while you are assigned collaboration_workflow) "
    "-- in that case it already differs from yours by construction and "
    "does not need to change anything about your diagnosis; do not let "
    "it pull you toward that other block's territory. When a sibling DID "
    "target this SAME block, differentiate within it: a different aspect "
    "of the diagnosis, a different failing case, or a different specific "
    "bug -- never by switching to a different block just to be "
    "different. The downstream editor is expected to implement your "
    "suggestion largely as given, not to independently resolve a "
    "conflict between 'follow the suggestion' and 'be different from "
    "siblings' -- that reconciliation is your job, not its."
)

_BLOCK_BODIES: dict[str, str] = {
    "individual_subagent": (
        "## Block: individual_subagent\n\n"
        "Scope: ONE individual role/sub-agent's own code -- both what it "
        "is told (its system-prompt wording/instructions) and how it acts "
        "(which tools it is allowed to call, its control flow, its "
        "retry/budget handling). In this codebase's convention, one "
        "sub-agent's prompt and behavioral logic typically live together "
        "in one file/module (e.g. a per-role file defining both a "
        "`<ROLE>_SYSTEM_PROMPT` string and a `run_<role>_stage(...)` "
        "function implementing that role's tool-calling loop). Identify a "
        "stage function with this shape -- it owns a distinct role, a "
        "distinct (possibly overlapping) subset of the available tools, "
        "and returns its result to the rest of the pipeline via the "
        "standard inter-stage message contract.\n\n"
        "Pick exactly ONE such role. Diagnose a real, specific problem "
        "with THAT role's own performance on its specific task -- a "
        "misleading or incomplete instruction in its prompt, a wrong "
        "tool selected or omitted, a missing retry when its output "
        "doesn't meet the expected shape, a control-flow bug specific to "
        "it, or any other harness change that makes THIS role better at "
        "its own job. Any part of this role's own harness is fair game, "
        "including a substantial rewrite, AS LONG AS the goal is that "
        "role's own task performance.\n\n"
        "If, as a side effect of fixing this role's own performance, the "
        "content it sends to another role also changes (e.g. you "
        "rewrite its output format as part of fixing how it reasons "
        "about the task), that is fine and does not disqualify the "
        "suggestion. What is NOT in scope here: a change whose goal is "
        "EXCLUSIVELY about collaboration/messaging -- where this role's "
        "own task performance is untouched and the only thing being "
        "fixed is what it sends to or receives from another stage -- "
        "that belongs entirely in collaboration_workflow, not here. "
        "Likewise, do NOT propose changes to logic that is shared "
        "verbatim across every role (that is foundation_capability) -- "
        "if your diagnosis turns out to actually be about shared "
        "plumbing or cross-role sequencing with no bearing on this "
        "role's own task performance, say so explicitly and stop rather "
        "than force-fitting it into this block.\n\n"
        "Output a short markdown suggestion with:\n"
        "  - **Target**: which role/stage you are diagnosing, and how you "
        "identified it as this block (its prompt + behavioral-logic "
        "file/function).\n"
        "  - **Diagnosis**: the specific, evidence-grounded problem (cite "
        "the case_id, tool name, prompt text, or feedback field you're "
        "relying on).\n"
        "  - **Proposed change**: what to change -- prompt wording, tool "
        "subset, control flow, or retry/budget handling -- concrete "
        "enough that an editor could implement it, described in prose, "
        "not code."
    ),
    "collaboration_workflow": (
        "## Block: collaboration_workflow\n\n"
        "Scope: the code that calls each stage/role function in SEQUENCE "
        "and decides what each stage receives from upstream stages -- "
        "i.e. how the roles collaborate, not what any one role does on "
        "its own. In this codebase's convention, stage functions share a "
        "standard interface: each takes an `inbox` list of messages "
        "produced by upstream stages it depends on, and returns exactly "
        "one message of its own to hand downstream (look for a shared "
        "message/record type passed this way -- typically with a sender "
        "identifier and content field -- and a lookup helper that finds "
        "one upstream stage's message in another stage's inbox by sender "
        "name). The collaboration_workflow block is the code that "
        "SEQUENCES these calls: which stage runs after which, and which "
        "upstream messages populate each stage's inbox argument at each "
        "call site. It is identifiable by this call-site pattern (stage "
        "functions invoked in order, each fed a list of prior stages' "
        "outputs), not by any particular filename -- find it by looking "
        "for where these stage functions are actually invoked together, "
        "wherever that lives in this codebase.\n\n"
        "Your proposed change MUST center on this message interface -- "
        "what a stage SENDS (how it constructs its outgoing message: its "
        "content, its status/ok field, or equivalent) and what a stage "
        "RECEIVES (which upstream messages populate its inbox, and what it "
        "reads out of them via the sender-lookup helper). Editing the "
        "logic of how an agent sends or receives its message IS the "
        "primary edit surface for this block. It is fine, and often "
        "necessary, for the concrete implementation to also touch one or "
        "more individual stage files -- e.g. changing what a stage reads "
        "from its inbox, or what fields it writes into its outgoing "
        "message so a new downstream consumer can use them -- but that "
        "must be a CONSEQUENCE of a message/interface change, not the "
        "goal itself. If your proposed change doesn't touch how any "
        "message is constructed, sent, or consumed between stages -- e.g. "
        "it's really about one stage's own prompt wording, tool "
        "selection, or retry logic with no interface impact -- that "
        "belongs in individual_subagent instead, not here.\n\n"
        "One common, in-scope case worth naming explicitly: if you find "
        "that the content one stage sends is missing something a "
        "downstream stage needs, or is simply wrong, it is fine -- and "
        "often the correct fix -- to change the SENDING stage's own "
        "prompt or logic so it actually produces the missing/correct "
        "content. This stays in scope for this block as long as the "
        "change is about content headed into another stage's inbox (what "
        "gets sent), not about that stage's unrelated internal behavior. "
        "State plainly which stage sends and which stage receives the "
        "content you're fixing.\n\n"
        "Diagnose a real, specific problem with this sequencing or "
        "handoff structure -- e.g. a stage that would benefit from seeing "
        "an upstream stage's output but currently doesn't (its inbox is "
        "missing something it needs), a stage that runs before "
        "information it depends on is ready, or (notably) the ABSENCE of "
        "a step that would close the loop -- e.g. no stage re-reads the "
        "fully assembled result and routes it back to an earlier stage if "
        "something is wrong. You may propose ADDING a new handoff or step "
        "(a stage that reviews the combined output and can route work "
        "back to an earlier stage), not only editing an existing call "
        "site -- this block is about the SHAPE of collaboration, which "
        "can include a structural addition, not just a parameter tweak. "
        "Do NOT propose changing what happens INSIDE any one stage's own "
        "logic with no bearing on its message interface (that is "
        "individual_subagent) and do NOT propose changing plumbing shared "
        "verbatim by every stage (that is foundation_capability).\n\n"
        "Output a short markdown suggestion with:\n"
        "  - **Target**: which sequencing/handoff code you are "
        "diagnosing, and how you identified it (the inbox/message-passing "
        "call pattern).\n"
        "  - **Diagnosis**: the specific, evidence-grounded gap or bug in "
        "how stages hand off work (cite what you actually read in the "
        "sequencing code or the feedback).\n"
        "  - **Proposed change**: what to change about the message "
        "interface -- what a stage sends, what a stage receives, "
        "reordering, an added/changed inbox dependency, or a new "
        "handoff/step -- concrete enough to implement, described in "
        "prose, not code. Name any sub-agent files the interface change "
        "requires touching, and say why (e.g. \"flight.py's outgoing "
        "message must add field X so sightseeing.py can read it\")."
    ),
    "foundation_capability": (
        "## Block: foundation_capability\n\n"
        "Scope: logic SHARED across every (or nearly every) role, rather "
        "than any one role's own behavior. Look for: rule text that "
        "multiple roles' prompts include verbatim or near-verbatim (a "
        "shared constants/rules block imported by every role rather than "
        "restated per-file), a shared bounded call-and-tool-execution loop "
        "that multiple roles run through (message accumulation, "
        "iteration/step budget, tool-error surfacing), any shared "
        "tool-routing/schema-filtering plumbing, and iteration/budget "
        "constants that apply system-wide rather than to one role. This "
        "is the substrate every role's own logic is built on top of -- a "
        "bug or gap here affects every role at once, so it is usually "
        "higher-leverage than a single-role fix, but also the most "
        "consequential to get right, since a bad change here can degrade "
        "every role simultaneously.\n\n"
        "Concrete examples of what belongs here: improving the "
        "RELIABILITY of tool calling itself -- e.g. adding a retry "
        "wrapper around tool calls that currently fail silently or "
        "aren't retried at all, so a transient tool/API error doesn't "
        "just lose that turn for whichever role happened to hit it; "
        "adding a retry around the shared LLM-call function (e.g. "
        "retrying once on a malformed/empty/truncated response before "
        "giving up, instead of every role's own tool loop independently "
        "failing the same way); adding a timeout to a tool or LLM call "
        "that can currently hang with none; and improving shared "
        "ERROR-LOG PARSING -- e.g. better classification/extraction of "
        "what actually went wrong from a tool result or exception so "
        "every role (and the failure/behavior summarizers reading the "
        "same trace) gets a more useful signal, instead of a generic "
        "'Error' string. These are 'foundational' in the literal sense: "
        "general-purpose reliability/observability primitives (retry, "
        "timeout, backoff, error classification) that every role's calls "
        "route through, not a domain-specific fix for one role's own "
        "reasoning.\n\n"
        "Diagnose a real, specific problem in this shared substrate -- "
        "e.g. shared rule text that is ambiguous or contradicts what a "
        "scored dimension actually requires, an iteration/step budget "
        "that is measurably too tight or too loose (cite the evidence), a "
        "class of tool-call or tool-error handling that silently swallows "
        "a failure every role then inherits, or model-specific "
        "brittleness in the shared call loop (e.g. a reasoning/thinking-"
        "mode setting known to break tool calling for the model in use -- "
        "check the feedback/runtime-exception evidence for this before "
        "claiming it, don't assume it). Do NOT propose a change scoped to "
        "only one role's own prompt or control flow (that is "
        "individual_subagent) and do NOT propose a change to how roles "
        "sequence or hand off work to each other (that is "
        "collaboration_workflow).\n\n"
        "Output a short markdown suggestion with:\n"
        "  - **Target**: which shared component you are diagnosing (name "
        "the shared constant, function, or plumbing file).\n"
        "  - **Diagnosis**: the specific, evidence-grounded problem, and "
        "why it plausibly affects multiple roles, not just one (cite what "
        "you read).\n"
        "  - **Proposed change**: what to change about this shared "
        "substrate -- concrete enough to implement, described in prose, "
        "not code. Flag explicitly if the fix is broad enough to deserve "
        "extra caution (e.g. touches every role's call loop)."
    ),
    "verifiers": (
        "## Block: verifiers\n\n"
        "Scope: ANY code or check whose job is to ENFORCE QUALITY -- to "
        "verify that something holds, not to accomplish the task itself. "
        "Be explicit about this: a verifier can check that a specific "
        "CONSTRAINT holds (e.g. a required field is present, a value is "
        "within bounds, a business/domain rule is satisfied), and it can "
        "check this against an INTERMEDIATE output (what one stage "
        "produces along the way) just as much as the FINAL output the MAS "
        "returns -- both are equally in scope. A verifier can be a small "
        "inline check inside an existing function, a dedicated helper "
        "called at a decision point, or a larger validation pass over a "
        "fuller output. What makes something a verifier is its job -- "
        "CHECK/ENFORCE a constraint -- not where in the codebase it "
        "physically lives.\n\n"
        "First check: does a check for the specific constraint or failure "
        "pattern you're diagnosing already exist anywhere in this "
        "codebase, on any stage's intermediate or final output? If none "
        "exists, say so explicitly and propose ADDING one -- this is a "
        "legitimate, common outcome for this block, not a failure to find "
        "something. If evaluation/scoring-rubric material is shown to you "
        "below, ground your proposed verifier in the ACTUAL scored "
        "dimensions and hard constraints it defines -- do not invent "
        "generic sanity checks (e.g. \"check for empty output\") that "
        "don't correspond to anything the rubric actually scores. If a "
        "relevant check already exists, diagnose a specific, "
        "evidence-grounded gap in its coverage or reliability instead of "
        "proposing a duplicate.\n\n"
        "DO NOT propose changes to any other (any other subagent's) "
        "role's task-performance logic that has nothing to do with "
        "checking/enforcing a constraint (that is individual_subagent) "
        "or to shared plumbing unrelated to checking/enforcing "
        "constraints (that is foundation_capability).\n\n"
        "Output a short markdown suggestion with:\n"
        "  - **Target**: whether a relevant check already exists (name "
        "it) or does not (say so), and whether it applies to an "
        "intermediate or the final output.\n"
        "  - **Diagnosis**: which specific scored dimension(s), hard "
        "constraint(s), or other quality requirement is least likely to "
        "be caught today, and why (cite rubric material and/or failure "
        "evidence you were shown).\n"
        "  - **Proposed change**: the check to add or fix -- what "
        "constraint it enforces, where it runs (which stage's "
        "intermediate or final output, and at what point), and what it "
        "does on failure (block/patch/flag) -- concrete enough to "
        "implement, described in prose, not code."
    ),
}


@register("block_suggester", "default")
class BlockSuggester:
    """LLM-synthesized, block-scoped improvement suggestion for one EXPAND.

    Args mirror FailureSummarizer/BehaviorSummarizer: llm_caller is the
    same callable injected into AgentEditor; tools_source/db_schema/
    scorer_source/mutable_exclude mirror AgentEditor's own project-context
    injections (see meta_agent/config.py::build_components) so the
    suggestion is grounded in the same reference material the editor sees.
    """

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        domain_label: Optional[str] = None,
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
        mutable_exclude: Optional[list[str]] = None,
        # Caps runaway generation -- a real failure mode observed live: a
        # reasoning-model call degenerated into a ~320x repetition loop and
        # produced a 60KB suggestion before this cap existed. 16384 matches
        # the task_agent default. None disables the cap.
        max_output_tokens: Optional[int] = 16384,
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.tools_source = tools_source
        self.db_schema = db_schema
        self.scorer_source = scorer_source
        self.mutable_exclude = mutable_exclude
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def suggest(
        self,
        *,
        block: str,
        agent_dir: Path,
        out_dir: Path,
        node_id: int,
        feedback: Optional[AgentFeedback] = None,
        failure_summary: Optional[str] = None,
        siblings: Optional[list[tuple[Optional[str], str]]] = None,
    ) -> Optional[str]:
        """Produce a block-scoped suggestion, persisted into ``out_dir``.

        ``agent_dir`` is the PARENT's task_agent/ (the current code to
        diagnose); ``out_dir`` is the CHILD round dir this EXPAND is
        writing into (where the suggestion artifacts are persisted, since
        this is a per-EXPAND-attempt output, not a per-node-evaluation
        one). ``siblings`` is a list of ``(block, optimization_goal)`` for
        every prior edit already branched off this same parent -- the
        sibling-differentiation instruction (see ``_SYSTEM_CLOSING``) is
        owned entirely by this component, not the caller; the manager no
        longer shows the editor a separate, potentially-conflicting
        differentiation directive when a suggester is configured (see
        ``hgm.py::_render_expand_context``). Returns the suggestion text,
        or None on error/empty response/unknown block -- callers should
        treat this the same as the other summarizers: skip silently, never
        fail the round.
        """
        if block not in _BLOCK_BODIES:
            print(f"[block_suggester] unknown block {block!r} — skipped", flush=True)
            return None

        try:
            sources = source_context.read_mutable_sources(
                agent_dir, mutable_exclude=self.mutable_exclude
            )
        except OSError as exc:
            print(f"[block_suggester] failed to read sources: {exc!r}", flush=True)
            return None

        system = _SYSTEM_PREAMBLE + "\n\n" + _BLOCK_BODIES[block] + _SYSTEM_CLOSING

        user_parts: list[str] = source_context.format_project_context(
            tools_source=self.tools_source,
            db_schema=self.db_schema,
            scorer_source=self.scorer_source,
        )
        user_parts.append(source_context.format_current_sources(sources))
        user_parts.append(self._format_feedback_digest(feedback, failure_summary))
        if siblings:
            user_parts.append(self._format_siblings(block, siblings))

        prompt_user = "\n".join(p for p in user_parts if p)

        try:
            (out_dir / "block_suggestion_prompt.txt").write_text(
                f"### SYSTEM\n{system}\n\n### USER\n{prompt_user}", encoding="utf-8"
            )
        except OSError:
            pass

        try:
            response = self._call_llm(system, prompt_user)
        except Exception:
            print(
                f"[block_suggester] LLM call failed for node {node_id} "
                f"(block={block}):\n" + traceback.format_exc(limit=3),
                flush=True,
            )
            return None

        text = (response or "").strip()
        if not text:
            print(
                f"[block_suggester] empty suggestion for node {node_id} "
                f"(block={block}) — skipped",
                flush=True,
            )
            return None

        try:
            (out_dir / "block_suggestion.md").write_text(text, encoding="utf-8")
        except OSError:
            pass
        print(
            f"[block_suggester] node {node_id}: wrote block_suggestion.md "
            f"(block={block}, {len(text)} chars)",
            flush=True,
        )
        return text

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _format_feedback_digest(
        self, feedback: Optional[AgentFeedback], failure_summary: Optional[str]
    ) -> str:
        lines = ["## Feedback digest (most recent evaluation of this agent)"]
        if feedback is None:
            lines.append("(no prior evaluation — this is the tree root / an unevaluated node.)")
        else:
            ev = feedback.eval_result
            lines.append(f"score={ev.score:.3f}  passed={ev.passed}  failed={ev.failed}")
            if feedback.tool_error_rate:
                ranked = sorted(
                    ((n, r) for n, r in feedback.tool_error_rate.items() if r > 0),
                    key=lambda kv: -kv[1],
                )
                if ranked:
                    lines.append(
                        "tool error rates: "
                        + ", ".join(f"{n}={r:.2f}" for n, r in ranked[:5])
                    )
            if feedback.runtime_exceptions:
                lines.append("runtime_exceptions:")
                for exc in feedback.runtime_exceptions[:3]:
                    lines.append(f"  - {exc[:200]}")
        if failure_summary:
            lines.append("\n## Cross-case failure summary (LLM-synthesized)")
            lines.append(failure_summary)
        return "\n".join(lines)

    def _format_siblings(
        self, block: str, siblings: list[tuple[Optional[str], str]]
    ) -> str:
        lines = [
            f"## {len(siblings)} sibling edit(s) already branched off this "
            "same parent"
        ]
        for sib_block, goal in siblings[:8]:
            if sib_block == block:
                tag = f"SAME block ({block}) — differentiate within it"
            elif sib_block:
                tag = f"different block ({sib_block}) — does not constrain you"
            else:
                tag = "block unknown — treat cautiously as same-block"
            lines.append(f"  - [{tag}] {goal[:160]}")
        return "\n".join(lines)

    def _call_llm(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model:
            kwargs["model"] = self.model
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = 0.2
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        response = self.llm(**kwargs)
        return getattr(response, "content", None) or ""

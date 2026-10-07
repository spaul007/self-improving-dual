"""TextGrad optimization of the memory generator's instruction addendum.

Opt-in replacement for the one-call instruction updater
(``edit_memory.config.instruction_optimizer: "textgrad"``). The agentic
instruction curator still gathers the evidence (``q.md``); what changes is
how ``I_k -> I_{k+1}`` is computed. For every memory version ``B_v`` that
the audited editors read and that was generated under the current addendum,
one TextGrad graph is built:

    addendum I_k (trainable) ─┐
                              ├─ compose ─> generator system prompt ─┐
    fixed core (constant) ────┘                                      ├─ LLMCall ─> B_v ─┐
    generator input (previous memory + curation, constant) ──────────┘                  ├─ critic ─> loss_v
                     evidence (audit q.md, curation, previous memory) ──────────────────┘

The LLMCall is *replayed*: ``B_v`` is the memory the editors actually read
(recorded inputs in ``window_NNN/generation_inputs.json``), so no generation
call is spent and the critique is about the document whose use the audit
observed. ``loss_v.backward()`` sends the critic's feedback through the
generator to the addendum (one backward call per hop through an LLM; the
compose step passes it through with a note that only the addendum can
change; items the critic attributed to the generator's inputs, [INPUT],
are stripped from the critique first); the losses' gradients accumulate on ``I_k`` and one
``TextualGradientDescent`` step writes ``I_{k+1}``.

Three deviations from stock textgrad 0.1.8, all local:

* the addendum is shown to the optimizer in full (stock TGD shows a
  20-word excerpt of the variable it rewrites);
* earlier steps' addendum feedback is passed to the optimizer as momentum
  (stock ``gradient_memory`` is stored but never rendered into the prompt);
  it is read from disk, so it survives a resume;
* every model call goes through the repo's ``call_llm`` (provider pinning,
  retries, key handling) via ``CallLLMEngine``, and is logged in full to
  ``calls.jsonl``.

Importing this module imports textgrad, whose own import creates a log
directory (``TEXTGRAD_LOG_DIR``, default ``./logs``); a temp dir is set
here unless the caller chose one, and ``redirect_logs`` moves textgrad's
log file into the run.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

if "textgrad" not in sys.modules:
    os.environ.setdefault("TEXTGRAD_LOG_DIR", os.path.join(tempfile.gettempdir(), "textgrad_logs"))

import textgrad as tg  # noqa: E402
from textgrad.autograd import FormattedLLMCall, LLMCall  # noqa: E402
from textgrad.autograd.function import BackwardContext, Function  # noqa: E402
from textgrad.optimizer.optimizer import get_gradient_and_context_text  # noqa: E402
from textgrad.optimizer.optimizer_prompts import (  # noqa: E402
    CONSTRAINT_PROMPT_ADDITION,
    TGD_PROMPT_PREFIX,
    TGD_PROMPT_SUFFIX,
)

from . import prompts as P  # noqa: E402
from .generator import LLMSpec, validate_addendum  # noqa: E402

TEXTGRAD_DIR = "textgrad"                      # under instruction_update_NNN/
GENERATION_INPUTS_FILE = "generation_inputs.json"   # under window_NNN/
ADDENDUM_FEEDBACK_FILE = "addendum_feedback.md"
CALLS_FILE = "calls.jsonl"
SUMMARY_FILE = "summary.json"
FALLBACKS = ("updater", "keep")


@dataclass
class TextGradConfig:
    # Memory versions critiqued per step (newest first); each costs one
    # critic call and two backward calls.
    max_versions: int = 2
    # Earlier steps' addendum feedback shown to the optimizer as momentum.
    past_feedback: int = 2
    # What to do when the TextGrad step yields no addendum: run the classic
    # one-call updater on the same audit, or keep the current addendum.
    fallback: str = "updater"

    def __post_init__(self) -> None:
        if self.max_versions < 1:
            raise ValueError("edit_memory.textgrad.max_versions must be >= 1")
        if self.past_feedback < 0:
            raise ValueError("edit_memory.textgrad.past_feedback must be >= 0")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"edit_memory.textgrad.fallback must be one of {FALLBACKS}, "
                             f"got {self.fallback!r}")


@dataclass
class CritiqueSample:
    """One memory version to critique, with everything its graph needs."""
    memory_version: int
    memory: str                 # B_v, exactly as editors read it
    previous_memory: str        # B_{v-1} ("" for the first memory)
    curation: str               # the window review B_v was generated from
    system_prompt: str          # the generator's system prompt (core + I_k)
    user_prompt: str            # the generator's input (window header, B_{v-1}, curation)
    usage: str                  # which audited nodes read B_v


# ---------------------------------------------------------------------- #
# Role descriptions (textgrad puts these into every backward / optimizer
# prompt; being specific matters)
# ---------------------------------------------------------------------- #

ROLE_ADDENDUM = (
    "learned addendum to the system prompt of the edit-memory generator: a short "
    "list of concrete guidance bullets telling the generator what the edit memory "
    "must additionally contain, at what granularity and in what form. It is "
    "appended after a fixed core instruction that cannot change"
)
ROLE_SYSTEM = (
    "system prompt of the edit-memory generator: a fixed core instruction (not "
    "editable) followed by the learned addendum (the only editable part)"
)
ROLE_GENERATOR_INPUT = (
    "input of the edit-memory generator: the window header, the previous edit "
    "memory and the curator's review of the newest nodes (fixed evidence)"
)
ROLE_MEMORY = (
    "edit memory: the document a code-editing meta-agent (the editor) reads before "
    "it edits a task agent, consolidating evidence about which code edits helped, "
    "why, and whether the task agent could use them"
)
ROLE_CRITIQUE = "critic's evaluation of the edit memory (the loss of the instruction optimization)"
ROLE_CONTEXT = "evidence for the critic (not optimized)"


# ---------------------------------------------------------------------- #
# The critic
# ---------------------------------------------------------------------- #

CRITIC_SYSTEM = (
    "You are the critic in an optimization loop that tunes the instruction under "
    "which a self-evolving agent's EDIT MEMORY is written. You judge one version "
    "of that memory against the evidence it was built from and the evidence of "
    "how it was used. Your critique is the loss: it is propagated back through "
    "the generator to the editable addendum of its instruction, so it must be "
    "specific, evidence-backed and attributable — say what is wrong, why it "
    "matters to the memory's reader, which part of the pipeline caused it, and "
    "what already works and must not be lost. You never rewrite the memory or "
    "the instruction yourself, and you never judge by predicted scores."
)

# Rendered once per sample with FormattedLLMCall; the six {fields} are the
# graph's inputs. No other braces may appear in this text.
CRITIC_FORMAT = (
    "# How the edit memory is used\n"
    "A meta-agent (the editor) produces the next version of a task agent's code. "
    "Before it edits, it may read the edit memory. Its TARGET comes from its "
    "parent's failing cases and from the block it was assigned; the memory is "
    "guidance for HOW: which mechanisms worked for a task agent like its own, "
    "what failed and why, whether the task agent could actually use an edit as "
    "delivered (deterministic process, proper linking into the agent, a clear "
    "instruction), which edits conflict, and what is still untried. The editor "
    "opens the code of the nodes the memory cites, so citations must lead "
    "somewhere. The memory's unit is the EDIT (one mechanism, labelled "
    "E<node>.<k> by the curator), never the node.\n\n"
    "A good memory lets an editor working under ANY assignment find, quickly and "
    "reliably, the evidence-backed lessons that apply to its parent, and verify "
    "them in the cited code. A bad memory is one the editor cannot navigate, "
    "misreads, cannot verify, or that states things the evidence does not "
    "support.\n\n"
    "# Your inputs\n"
    "  <GENERATOR_INSTRUCTION>: the instruction the memory was written under "
    "(fixed core, then the learned addendum).\n"
    "  <PREVIOUS_MEMORY>: the version this one replaced.\n"
    "  <CURATION>: the curator's review of the newest window — the evidence this "
    "version had to consolidate, ending with a gradient section (which entries of "
    "the previous memory were confirmed / contradicted / too coarse / missing).\n"
    "  <AUDIT>: an auditor's report on how editors used the memory and whether "
    "the edits it guided worked when the task agent ran. Its 'Proposed "
    "instruction changes' are the auditor's hypothesis: adopt only what its "
    "evidence supports.\n"
    "  <USAGE>: which audited nodes were produced by an editor that had THIS "
    "version.\n"
    "  <MEMORY>: the version you judge.\n\n"
    "# Criteria (judge each from evidence; 'no issue' is a valid finding)\n"
    "C1 Faithfulness. Every claim traces to the curation or the previous memory; "
    "verdicts (helped / unclear / hurt) agree with the curator's per-edit "
    "verdicts; nothing the curation contradicts survives; no mechanism or node is "
    "invented or misattributed.\n"
    "C2 Consolidation. Every item of the curation's gradient section is "
    "answered; entries of the previous memory are not silently dropped or "
    "renamed; an edit that recurs across nodes is ONE entry listing all of them; "
    "no duplicates; the change from the previous memory is minimal for what the "
    "evidence requires.\n"
    "C3 Usability by the editor. Using the audit: what editors looked for and did "
    "not find, misread, or ignored, and why. Can an editor under a given "
    "assignment find what applies to it? Are citations (node ids, edit labels, "
    "files / functions) precise enough to open the code?\n"
    "C4 Task-agent utilization and compatibility. For the entries that matter, "
    "does the memory say whether the task agent could use the edit and, if not, "
    "what would let it? Are conflicts and dependencies between edits recorded "
    "where the curation or the audit shows them?\n"
    "C5 Guidance. Several directions with evidence, each marked tried / untried, "
    "plus what to avoid; not a single work order; nothing that tells an editor to "
    "work outside its assignment.\n"
    "C6 Form. Granularity, ordering, redundancy, length relative to information: "
    "padding and repetition are defects, and so is a line too terse to act on.\n"
    "Hard violations (always report, under C1): any predicted score, expected "
    "gain or number forecasting an evaluation; a ranking of nodes instead of "
    "edits; a missing required section; benchmark-case-specific values (train or "
    "flight numbers, hotel / restaurant / attraction names) copied from logs.\n\n"
    "# Attribution (label every issue)\n"
    "  [INSTRUCTION] the instruction does not ask for this, or asks ambiguously; "
    "a change to the addendum would fix it.\n"
    "  [COMPLIANCE] the instruction already requires it and the generator did not "
    "do it; a sharper, checkable formulation might fix it.\n"
    "  [INPUT] the curation or audit lacked the evidence; no instruction change "
    "can fix it. Report it, briefly, so it is not mistaken for an instruction "
    "problem.\n\n"
    "# Rules for the critique\n"
    "  - Cite evidence for every issue: quote the memory line, or name the "
    "curation / audit section and the node ids or edit labels.\n"
    "  - State each issue as a rule that will hold for future windows of this "
    "run (the KIND of information, structure or granularity the memory needs). "
    "A fact from this run may illustrate the rule, but must not stand in for "
    "it: the facts themselves belong in the memory, where the curator re-checks "
    "them every window.\n"
    "  - Judge the memory only against what its generator could know: never "
    "fault it for lacking something absent from <PREVIOUS_MEMORY> and "
    "<CURATION> (e.g. a mechanism a later node discovered) — that is [INPUT] — "
    "and never use such a fact as the example in an [INSTRUCTION] or "
    "[COMPLIANCE] issue.\n"
    "  - Rank issues by their consequence for the editor, not by how easy they "
    "are to name. If the audit shows editors did not read or use a part of the "
    "memory, ask why before calling it a defect of that part.\n"
    "  - Do not reward length or ask for more text by default: an addition must "
    "replace a gap the evidence shows.\n"
    "  - Never ask for scores, predictions, expected gains or rankings by mean "
    "score.\n"
    "  - If the memory serves its reader well, say so and keep the issue list "
    "short; do not invent problems.\n\n"
    "# Output (at most about 1200 words)\n"
    "## Verdict\n"
    "One paragraph: how well this version serves the editor, and the one to "
    "three issues that matter most.\n"
    "## Issues\n"
    "Most important first, at most eight, each one line or two: `- [SOURCE] Cn — "
    "what is wrong (evidence: ...) — consequence for the editor — the general "
    "rule the memory should follow instead`.\n"
    "## Keep\n"
    "What works and must be preserved when the instruction changes.\n"
    "## Not attributable to the instruction\n"
    "The [INPUT] items, one line each (or 'none').\n\n"
    "<GENERATOR_INSTRUCTION>\n{generator_instruction}\n</GENERATOR_INSTRUCTION>\n\n"
    "<PREVIOUS_MEMORY>\n{previous_memory}\n</PREVIOUS_MEMORY>\n\n"
    "<CURATION>\n{curation}\n</CURATION>\n\n"
    "<AUDIT>\n{audit}\n</AUDIT>\n\n"
    "<USAGE>\n{usage}\n</USAGE>\n\n"
    "<MEMORY>\n{memory}\n</MEMORY>\n"
)
CRITIC_FIELDS = ("generator_instruction", "previous_memory", "curation", "audit", "usage", "memory")
INPUT_SECTION = "## Not attributable to the instruction"
_INPUT_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+[*_`]*\[INPUT\]")
_ITEM_START_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def strip_input_items(critique: str) -> str:
    """The critique as it is back-propagated: without its 'Not attributable
    to the instruction' section and without any list item labelled [INPUT]
    (with the item's continuation lines). Those describe gaps in the
    generator's evidence; sent backward, they would be blamed on the
    instruction (textgrad's backward prompt knows nothing of the labels)."""
    out: list[str] = []
    in_input_section = dropping = False
    for line in critique.splitlines():
        if line.startswith("## "):
            in_input_section = line.strip().startswith(INPUT_SECTION)
            dropping = False
            if in_input_section:
                continue
        if in_input_section:
            continue
        if _ITEM_START_RE.match(line):
            dropping = bool(_INPUT_ITEM_RE.match(line))
        elif not line.strip():
            dropping = False
        if not dropping:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"

# What the compose step tells the addendum about feedback aimed at the
# whole system prompt.
COMPOSE_PASS_THROUGH = (
    "The feedback below was computed for the generator's whole system prompt, "
    "which is a fixed core instruction (cannot change) followed by this addendum. "
    "Apply it to the addendum only. Where it says the generator ignored a rule the "
    "fixed core already states, an addendum bullet may make that rule concrete and "
    "checkable, but must not restate the core. Ignore parts that concern the "
    "generator's inputs (the curator's review) rather than its instruction, and "
    "parts the critic marked [INPUT].\n<FEEDBACK>{feedback}</FEEDBACK>"
)

# ---------------------------------------------------------------------- #
# The optimizer prompt (stock TGD pieces, full variable, momentum, core)
# ---------------------------------------------------------------------- #

OPTIMIZER_CORE_BLOCK = (
    "For context, the fixed core the addendum is appended to (it cannot change "
    "and must never be repeated in the addendum):\n"
    "<FIXED_CORE>\n{core}\n</FIXED_CORE>\n\n"
)
OPTIMIZER_PAST_FEEDBACK_BLOCK = (
    "Feedback from earlier optimization steps of this addendum (older first; for "
    "momentum, already applied — do not re-apply it, but if the same problem "
    "recurs in the current feedback, the earlier change was insufficient and a "
    "more decisive change is warranted):\n<PAST_FEEDBACK>\n{past}\n</PAST_FEEDBACK>\n\n"
)
EMPTY_ADDENDUM = "(empty: the generator currently runs on the fixed core alone)"


def optimizer_constraints(*, max_chars: int, forbid_case_values: bool) -> list[str]:
    out = [
        "Output only the addendum: a short list of concrete guidance bullets for the "
        "memory generator, no preamble, no headings, no code fences.",
        f"Stay within {max_chars} characters.",
        "Never repeat, paraphrase wholesale or contradict the fixed core, and never "
        f"include its sentinel line ({P.CORE_SENTINEL}) or 'END OF FIXED CORE'.",
        "Change the addendum minimally: keep every bullet the feedback does not "
        "refute, apply only changes the feedback's evidence justifies, drop bullets "
        "it shows are harmful or ignored for a good reason. Each change must trace "
        "to the feedback.",
        "Bullets state rules about what the memory contains, at what granularity "
        "and in what form. A fact from this run may illustrate a rule, but must not "
        "stand in for one: the facts themselves belong in the memory, where the "
        "curator re-checks them every window.",
        "Every requirement must be satisfiable together with the others and with "
        "the memory's size cap and fixed sections; do not invent character budgets "
        "or formats the content cannot fit.",
        "Never ask the memory to predict scores, expected gains, or any number that "
        "forecasts an evaluation, and never ask it to rank nodes.",
        "Ignore feedback the critic attributed to the generator's inputs ([INPUT]).",
    ]
    if forbid_case_values:
        out.append("Never ask the memory to record benchmark-case values (train or flight "
                   "numbers, hotel, restaurant or attraction names).")
    return out


# ---------------------------------------------------------------------- #
# textgrad plumbing
# ---------------------------------------------------------------------- #

def redirect_logs(log_dir: Path) -> None:
    """Point textgrad's file logger into ``log_dir`` (the run's
    ``edit_memory/textgrad_logs``) instead of where its import put it."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("textgrad")
    formatter = None
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            formatter = h.formatter
            logger.removeHandler(h)
            h.close()
    handler = logging.FileHandler(log_dir / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    if formatter is not None:
        handler.setFormatter(formatter)
    logger.addHandler(handler)


class CallLLMEngine(tg.EngineLM):
    """A textgrad engine over the repo's ``call_llm``: one system + one user
    message, the layer's model kwargs, every call logged in full."""

    def __init__(self, llm: Callable[..., Any], spec: LLMSpec, log_path: Path) -> None:
        self.llm = llm
        self.spec = spec
        self.model_string = spec.model or "call_llm"
        self.log_path = Path(log_path)
        self.phase = ""
        self.n_calls = 0

    def generate(self, prompt: Any, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        if isinstance(prompt, list):  # multimodal container; only text is used here
            prompt = "\n".join(p for p in prompt if isinstance(p, str))
        messages = [{"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt}]
        t0 = time.time()
        self.n_calls += 1
        record: dict[str, Any] = {"n": self.n_calls, "phase": self.phase, "t": round(t0, 3),
                                  "system": messages[0]["content"], "prompt": prompt}
        try:
            response = self.llm(messages=messages, **self.spec.kwargs())
            text = (getattr(response, "content", None) or "").strip()
            if not text:
                raise RuntimeError("empty response")
            record["response"] = text
            return text
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_s"] = round(time.time() - t0, 3)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __call__(self, prompt: Any, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


class _FullVariable(tg.Variable):
    """Shown in full wherever textgrad would show a 20-word excerpt."""

    def get_short_value(self, n_words_offset: int = 10) -> str:
        return self.value


class _ComposeInstruction(Function):
    """``system prompt = fixed core + addendum``; backward hands the system
    prompt's feedback to the addendum (no model call)."""

    def forward(self, system_prompt: str, addendum: tg.Variable) -> tg.Variable:
        composed = tg.Variable(system_prompt, predecessors=[addendum], requires_grad=True,
                               role_description=ROLE_SYSTEM)
        composed.set_grad_fn(BackwardContext(backward_fn=self.backward,
                                             composed=composed, addendum=addendum))
        return composed

    def backward(self, composed: tg.Variable, addendum: tg.Variable, backward_engine: Any) -> None:
        feedback = composed.get_gradient_text().strip()
        if not feedback:
            return
        g = tg.Variable(COMPOSE_PASS_THROUGH.format(feedback=feedback), requires_grad=False,
                        role_description=f"feedback to {addendum.get_role_description()}")
        addendum.gradients.add(g)
        addendum.gradients_context[g] = None


def _replayed_generation(engine: CallLLMEngine, system: tg.Variable, user_prompt: str,
                         memory: str) -> tg.Variable:
    """The generator's LLMCall with its recorded output: the response node
    and backward function ``LLMCall.forward`` would build, without the call."""
    call = LLMCall(engine, system_prompt=system)
    gen_input = tg.Variable(user_prompt, requires_grad=False, role_description=ROLE_GENERATOR_INPUT)
    response = tg.Variable(memory, predecessors=[system, gen_input], requires_grad=True,
                           role_description=ROLE_MEMORY)
    response.set_grad_fn(BackwardContext(backward_fn=call.backward, response=response,
                                         prompt=user_prompt, system_prompt=system.value))
    return response


class _AddendumTGD(tg.TGD):
    """TGD whose prompt shows the whole addendum, the fixed core, and the
    earlier steps' feedback."""

    def __init__(self, *, core: str, past_feedback: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.core = core
        self.past_feedback = list(past_feedback)

    def _update_prompt(self, variable: tg.Variable) -> str:
        grad = get_gradient_and_context_text(variable)
        if not isinstance(grad, str):
            grad = "\n".join(p for p in grad if isinstance(p, str))
        prompt = TGD_PROMPT_PREFIX.format(
            variable_desc=variable.get_role_description(),
            variable_short=variable.value.strip() or EMPTY_ADDENDUM,
            variable_grad=grad,
        )
        prompt += OPTIMIZER_CORE_BLOCK.format(core=self.core)
        if self.past_feedback:
            past = "\n\n".join(f"<FEEDBACK-{i}>\n{fb.strip()}\n</FEEDBACK-{i}>"
                               for i, fb in enumerate(self.past_feedback, start=1))
            prompt += OPTIMIZER_PAST_FEEDBACK_BLOCK.format(past=past)
        if self.constraints:
            prompt += CONSTRAINT_PROMPT_ADDITION.format(constraint_text=self.constraint_text)
        prompt += TGD_PROMPT_SUFFIX.format(new_variable_start_tag=self.new_variable_tags[0],
                                           new_variable_end_tag=self.new_variable_tags[1])
        return prompt


# ---------------------------------------------------------------------- #
# One optimization step
# ---------------------------------------------------------------------- #

def optimize_addendum(
    llm: Callable[..., Any],
    spec: LLMSpec,
    *,
    addendum: str,
    samples: list[CritiqueSample],
    audit: str,
    core: str,
    past_feedback: list[str],
    max_chars: int,
    out_dir: Path,
    forbid_case_values: bool = False,
) -> tuple[Optional[str], list[str], dict[str, Any]]:
    """``(I_{k+1}, remaining_errors, record)``. The text is ``None`` when
    no addendum could be produced (a model call failed, or the optimizer's
    reply could not be parsed twice); a draft that still fails
    ``validate_addendum`` after one retry is returned with its findings
    (the layer's never-discard policy). With no feedback at all the
    current addendum is returned unchanged."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = CallLLMEngine(llm, spec, out_dir / CALLS_FILE)
    record: dict[str, Any] = {"samples": [s.memory_version for s in samples], "attempts": []}

    param = _FullVariable(addendum, requires_grad=True, role_description=ROLE_ADDENDUM)
    critic_system = tg.Variable(CRITIC_SYSTEM, requires_grad=False,
                                role_description="system prompt of the critic")
    critic = FormattedLLMCall(engine=engine, format_string=CRITIC_FORMAT,
                              fields={k: None for k in CRITIC_FIELDS}, system_prompt=critic_system)
    try:
        for s in samples:
            system = _ComposeInstruction()(s.system_prompt, param)
            memory = _replayed_generation(engine, system, s.user_prompt, s.memory)

            def ctx(value: str) -> tg.Variable:
                return tg.Variable(value.strip() or "(none)", requires_grad=False,
                                   role_description=ROLE_CONTEXT)

            engine.phase = f"critic_v{s.memory_version:03d}"
            loss = critic(inputs={
                "generator_instruction": ctx(s.system_prompt),
                "previous_memory": ctx(s.previous_memory or "(none: this was the first memory)"),
                "curation": ctx(s.curation), "audit": ctx(audit), "usage": ctx(s.usage),
                "memory": memory,
            }, response_role_description=ROLE_CRITIQUE)
            (out_dir / f"critique_v{s.memory_version:03d}.md").write_text(loss.value + "\n", encoding="utf-8")
            propagated = strip_input_items(loss.value)
            if propagated.strip() != loss.value.strip():
                loss.set_value(propagated.rstrip())
                (out_dir / f"critique_v{s.memory_version:03d}_propagated.md").write_text(
                    propagated, encoding="utf-8")
            engine.phase = f"backward_v{s.memory_version:03d}"
            loss.backward(engine)
            (out_dir / f"memory_feedback_v{s.memory_version:03d}.md").write_text(
                memory.get_gradient_text() + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - call_llm already retried
        record["error"] = f"{type(exc).__name__}: {exc}"
        _write_summary(out_dir, record, engine)
        return None, [f"textgrad forward/backward failed: {record['error']}"], record

    feedback = param.get_gradient_text().strip()
    (out_dir / ADDENDUM_FEEDBACK_FILE).write_text(feedback + "\n", encoding="utf-8")
    if not feedback:
        record["result"] = "no feedback; addendum unchanged"
        _write_summary(out_dir, record, engine)
        return addendum, validate_addendum(addendum, max_chars=max_chars), record

    base_constraints = optimizer_constraints(max_chars=max_chars, forbid_case_values=forbid_case_values)
    optimizer = _AddendumTGD(parameters=[param], engine=engine, constraints=base_constraints,
                             core=core, past_feedback=past_feedback)
    new_text: Optional[str] = None
    errors: list[str] = ["no draft produced"]
    for attempt in (1, 2):
        param.set_value(addendum)
        engine.phase = f"optimizer_{attempt}"
        try:
            optimizer.step()
        except IndexError:
            errors = ["the optimizer's reply had no <IMPROVED_VARIABLE> span"]
            record["attempts"].append({"attempt": attempt, "errors": errors})
            optimizer.constraints = base_constraints + [
                "Your previous reply was rejected: put the improved addendum between the tags."]
            continue
        except Exception as exc:  # noqa: BLE001
            errors = [f"optimizer call failed: {type(exc).__name__}: {exc}"]
            record["attempts"].append({"attempt": attempt, "errors": errors})
            break
        new_text = param.value.strip()
        if new_text == EMPTY_ADDENDUM:
            new_text = ""
        errors = validate_addendum(new_text, max_chars=max_chars)
        record["attempts"].append({"attempt": attempt, "chars": len(new_text), "errors": errors})
        if not errors:
            break
        optimizer.constraints = base_constraints + [
            f"Your previous draft was rejected: {e}" for e in errors]
    record["result"] = "written" if new_text is not None else "failed"
    _write_summary(out_dir, record, engine)
    if new_text is None:
        return None, errors, record
    return new_text + ("\n" if new_text and not new_text.endswith("\n") else ""), errors, record


def _write_summary(out_dir: Path, record: dict[str, Any], engine: CallLLMEngine) -> None:
    record["n_llm_calls"] = engine.n_calls
    (out_dir / SUMMARY_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

"""Per-EXPAND implementation-strategy steering text.

Orthogonal to block_suggester.py's "block" axis: block says WHICH PART of
the MAS an EXPAND should target (one sub-agent, the collaboration wiring,
shared foundations, verifiers, the LLM backbone config); this module says
HOW the fix should be implemented once a target is chosen -- how much of it
should be prompt/LLM work vs. deterministic code. Both axes' text lands in
the same steering context together (see hgm.py::_render_expand_context),
neither replacing the other.

Unlike block_suggester.py, there is no LLM-synthesized diagnosis step here
-- just a fixed instruction body per strategy value, selected by
HGMManager._select_implementation_strategy and spliced straight into
context. Steer, don't fence: nothing mechanically prevents the editor from
deviating; each body asks the editor to say so explicitly in its rationale
if it genuinely can't follow the assigned bias, mirroring curriculum.py's
own "ESCAPE HATCH" pattern.
"""
from __future__ import annotations

_IMPLEMENTATION_STRATEGY_BODIES: dict[str, str] = {
    "llm_heavy": (
        "## Implementation strategy for this EXPAND: llm_heavy\n\n"
        "Implement this EXPAND's fix almost ENTIRELY through prompt/LLM "
        "changes -- rewrite or extend a role's system prompt, its "
        "reasoning instructions, or what it is asked to decide -- rather "
        "than adding new deterministic code, parsing logic, or hard-coded "
        "control flow. Prefer teaching the model to reason its way to the "
        "correct behavior over hand-coding the behavior for it. Bring in "
        "deterministic code only for something a prompt genuinely cannot "
        "do at all (e.g. wiring a new field through an existing call "
        "site) -- and even then keep it to the minimum plumbing needed to "
        "let the prompt-level change take effect, never as the actual "
        "fix itself. If, after diagnosing the failure, you conclude a "
        "prompt change truly cannot address it, say so explicitly in your "
        "rationale and explain why, rather than silently defaulting to a "
        "code-heavy fix."
    ),
    "mixed": (
        "## Implementation strategy for this EXPAND: mixed\n\n"
        "Implement this EXPAND's fix with a roughly even balance of "
        "prompt/LLM changes and deterministic code/harness changes -- use "
        "whichever of the two actually fits each part of the diagnosis, "
        "rather than defaulting entirely to one. A common shape: teach "
        "the model what to decide via prompt changes, and have code "
        "deterministically handle the mechanical, unambiguous parts "
        "(formatting, threading a value between stages, extraction of an "
        "already-decided value). State briefly in your rationale which "
        "part of the fix is prompt-driven and which is code-driven."
    ),
    "harness_heavy": (
        "## Implementation strategy for this EXPAND: harness_heavy\n\n"
        "Implement most of this EXPAND's fix DETERMINISTICALLY in code -- "
        "validation, retries, formatting, scheduling/control-flow logic, "
        "and any check whose outcome should not depend on model judgment. "
        "Reserve the LLM for narrow tasks it is actually needed for (e.g. "
        "extracting/parsing a value from free text, or a genuinely "
        "open-ended judgment call) -- not for anything that can be "
        "computed or enforced directly. When you do touch a prompt, keep "
        "the change small and mechanical (e.g. asking the model to emit "
        "one more structured field code will then process) rather than "
        "rewriting its reasoning instructions. If the failure genuinely "
        "cannot be fixed deterministically (it requires real judgment), "
        "say so explicitly and use the LLM for exactly that narrow "
        "piece, nothing more."
    ),
}

"""LLM-based unit selection for meta_agent/unit_curriculum.py's UnitCurriculum.

Ported from experiment_harness_staged.py's choose_unit() LLM branch: one
lightweight call_llm invocation, given the per-candidate failure summaries
(already rendered by unit_curriculum.py's describe_unit_from_counts) and
asked to pick the best unit to target next via a ``propose_unit`` tool
call -- reasoning about root-cause commonality, fix difficulty, and
expected gain, NOT just raw failure count (see that function's own
docstring for why raw count is a poor proxy).

Deliberately split from the fallback logic in
unit_curriculum.py::choose_unit(): this class's ``choose()`` either returns
a valid unit name or None/raises -- it never itself falls back to the
mechanical "most failures" heuristic. That discipline lives one layer up
so each half stays independently testable (this class: does the LLM call
behave correctly given a fake llm_caller; choose_unit(): does the fallback
trigger correctly on any kind of failure from this class).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .registry import register


@register("unit_selector", "default")
class UnitSelector:
    """Args mirror BlockSuggester's own injection convention: llm_caller is
    the same callable injected into AgentEditor/BlockSuggester
    (platform_core.llm_wrapper.call_llm in production, a fake in tests)."""

    def __init__(
        self,
        llm_caller: Callable[..., object],
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        max_output_tokens: Optional[int] = 4096,
    ) -> None:
        self.llm = llm_caller
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens

    def choose(
        self, unit_counts: dict[str, int], descriptions: dict[str, str],
        attempts: dict[str, int], max_attempts: int,
    ) -> Optional[str]:
        """``unit_counts``/``descriptions`` are already narrowed to exactly
        the CANDIDATE units (attempts < max_attempts, count > 0) by
        unit_curriculum.py::choose_unit() -- this method never re-derives
        candidacy itself. Returns the chosen unit name, or None if the LLM
        call produced no valid ``propose_unit`` tool call naming one of the
        given candidates (the caller's fallback handles that, and any
        exception this method raises, identically)."""
        candidates = sorted(unit_counts, key=lambda u: -unit_counts[u])
        lines = [
            "Candidates for the NEXT improvement round to target, with real failure data from the CURRENT best "
            "agent's evaluation (most failing instances first):"
        ]
        for u in candidates:
            lines.append(f"\n## {u} ({unit_counts[u]} failing check instances)")
            lines.append(descriptions.get(u, ""))
        user = "\n".join(lines)
        system = (
            "You are choosing which unit (a group of related failing checks/constraints) an agent's next "
            "improvement round should target, from the candidates given. For EACH one: identify the likely root "
            "cause(s), estimate how hard a fix looks (a narrow, mechanical, code-level fix for one clear shared "
            "cause is easy; something needing broad prompt/schedule changes across many unrelated causes is "
            "harder), and estimate the likely gain if fixed (do most of its failures look like the SAME root "
            "cause, or many different one-off causes? a fix for one shared bug explaining most failures is worth "
            "more than a unit with more failures but no single actionable diagnosis). Raw failure count is NOT "
            "the same as best value -- a unit with fewer failures but one clean, high-confidence, narrow root "
            "cause is often the better pick over a unit with more failures and no clear single fix. Also "
            "consider DEPENDENCIES between candidates: prefer a unit whose fix stands on its own over one whose "
            "real fix would require another candidate to be fixed first (e.g. if unit X's failures are actually "
            "downstream of unit Y's -- fixing Y would likely resolve some of X's failures too, but not the "
            "reverse) -- pick Y in that case, not X, since a fix attempted on the downstream unit while the "
            "upstream cause is still broken is more likely to be shallow, fragile, or to get undone by a later "
            "fix to the real cause. Judge from the failure examples given -- you have no tools here, so reason "
            "directly from what's in front of you rather than guessing at anything not shown. Call propose_unit "
            "with your choice (the unit name EXACTLY as given) and a short rationale."
        )
        propose_tool = {
            "name": "propose_unit", "description": "Choose the next unit for this round to target.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "enum": candidates},
                    "rationale": {"type": "string"},
                },
                "required": ["unit", "rationale"],
            },
        }
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [propose_tool],
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
        calls = getattr(response, "tool_calls", None) or []
        valid = set(candidates)
        for call in calls:
            if call.name != "propose_unit":
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            chosen = args.get("unit")
            if chosen in valid:
                return chosen
        return None

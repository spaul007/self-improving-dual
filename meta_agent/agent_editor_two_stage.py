"""Two-stage agent editor: propose -> retrieve -> edit.

Stage 1 (one LLM call) reads the steering context (in belief mode: the
objective, the judge's line for the parent, the scope rule, the lineage, one
judge-first line per sibling, and the belief document), the feedback digest,
and the parent's mutable sources, and produces: 1-3 tentative edits, a
falsifiable PREDICTION naming the belief that justifies the edit plus the
judge verdict and targeted checks it expects, and an explicit MEMORY QUERY —
which past nodes / categories / keywords it wants to inspect. The query is resolved
deterministically (``edit_archive``) into record + code slices, and stage 2 is
the ordinary single-call self-improvement with the advisory proposal and the
retrieved memory appended to its context.

The split earns its cost through retrieval direction, not delegation: the
proposal is ADVISORY (stage 2 sees the full code and may override), which is
what distinguishes this from the older two-call design whose lossy hand-off
the single-call editor replaced. Any stage-1 failure degrades byte-for-byte to
the parent class's single-call behavior.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import edit_archive, verbose_log
from .agent_editor import AgentEditor, Validator
from .edit_beliefs import PREDICTION_NAME
from .models import AgentFeedback, EditResult
from .perf_text import judge_summary, perf_summary, regressions_summary
from .registry import register

PROPOSAL_TOOL: dict[str, Any] = {
    "name": "submit_edit_proposal",
    "description": (
        "Submit tentative edit proposals, a prediction naming the belief that "
        "justifies them, and the memory query for the records/code you want "
        "to see before writing the final edit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "object", "properties": {
                    "goal": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "strategy": {"type": "string"},
                    "area": {"type": "string"}},
                    "required": ["goal", "mechanism"]},
            },
            "prediction": {"type": "object", "properties": {
                "belief_id": {"type": "string"},
                "expected_effect": {"type": "string",
                                    "enum": ["improved", "no_effect", "regressed"]},
                "expected_targets": {"type": "array",
                                     "items": {"type": "string"}},
                # Legacy (pre-judge) fields: still accepted and persisted.
                "expected_direction": {"type": "string",
                                       "enum": ["up", "down", "neutral"]},
                "expected_delta": {"type": "number"},
                "why": {"type": "string"}},
                "required": ["belief_id", "expected_effect"]},
            "memory_query": {"type": "object", "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}},
                "strategies": {"type": "array", "items": {"type": "string"}},
                "areas": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "include_code": {"type": "boolean"}}},
        },
        "required": ["edits", "memory_query"],
    },
}

PROPOSE_SYSTEM = (
    "You are the planning pass of a self-improving agent's editor. A second "
    "call with the same context and the full code writes the actual edit and "
    "may override you. Do three things. (1) Memory query: name the past nodes, "
    "strategy/area ids, or keywords whose records and implementation the "
    "editing call should see — they are retrieved verbatim (each node's "
    "record plus the lines its edit added, per definition). (2) Sketch 1-3 "
    "candidate edits (goal + mechanism), tagged with the strategy/area ids "
    "used in the belief document where they fit. (3) Prediction: the "
    "`belief:<slug>` from the belief document that your main edit relies on "
    "(omit when none applies), the checks you expect the judge to see move "
    "(`expected_targets`, names as in the failure analysis), and the verdict "
    "you expect the judge to give the main mechanism (`expected_effect`: "
    "improved / no_effect / regressed). The score Δ is context, never the "
    "target. Prefer evidence not already seen: the registry block lists the "
    "measured nodes and which nodes were retrieved for the parent's own edit. "
    "Call `submit_edit_proposal`."
)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@register("editor", "two_stage")
class TwoStageEditor(AgentEditor):
    """``AgentEditor`` with a propose+retrieve planning pass in front.

    The ctor re-declares every injectable kwarg explicitly: config.py's
    ``_build_with_injection`` matches injections against named signature
    parameters, so a bare ``**kwargs`` would silently receive none of them.
    """

    def __init__(
        self,
        llm_caller: Callable[..., object],
        validators: Iterable[Validator],
        *,
        max_attempts: int = 2,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        tools_source: Optional[str] = None,
        db_schema: Optional[str] = None,
        scorer_source: Optional[str] = None,
        propose_model: Optional[str] = None,
        propose_reasoning_effort: Optional[str] = None,
        retrieval_char_budget: int = edit_archive.DEFAULT_CHAR_BUDGET,
        max_retrieved_nodes: int = edit_archive.DEFAULT_MAX_NODES,
        propose_enabled: bool = True,
        objective: str = "score",
    ) -> None:
        super().__init__(
            llm_caller, validators, max_attempts=max_attempts, model=model,
            reasoning_effort=reasoning_effort, base_url=base_url,
            tools_source=tools_source, db_schema=db_schema,
            scorer_source=scorer_source, objective=objective,
        )
        self.propose_model = propose_model
        self.propose_reasoning_effort = propose_reasoning_effort
        self.retrieval_char_budget = int(retrieval_char_budget)
        self.max_retrieved_nodes = int(max_retrieved_nodes)
        self.propose_enabled = bool(propose_enabled)

    # ------------------------------------------------------------------ #
    def apply(
        self,
        feedback: Optional[AgentFeedback],
        base_dir: Path,
        out_dir: Path,
        *,
        context: Optional[str] = None,
    ) -> EditResult:
        """Propose -> retrieve -> single-call edit. Every stage-1 failure
        falls through to the parent's behavior with the original context."""
        if not self.propose_enabled:
            return super().apply(feedback, base_dir, out_dir, context=context)
        try:
            proposal = self._propose(feedback=feedback, context=context,
                                     base_dir=Path(base_dir), out_dir=Path(out_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[editor:two_stage] propose failed: {exc!r}; falling back "
                  "to single-call edit", flush=True)
            proposal = None
        if not proposal:
            return super().apply(feedback, base_dir, out_dir, context=context)

        context2 = context or ""
        try:
            self._write_prediction(Path(out_dir), proposal)
        except Exception as exc:  # noqa: BLE001
            print(f"[editor:two_stage] prediction write failed: {exc!r}",
                  flush=True)
        try:
            result = edit_archive.resolve_query(
                Path(base_dir).parent, proposal.get("memory_query") or {},
                char_budget=self.retrieval_char_budget,
                max_nodes=self.max_retrieved_nodes)
            edit_archive.write_manifest(Path(out_dir), result)
            retrieved = edit_archive.render_retrieved(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[editor:two_stage] retrieval failed: {exc!r}", flush=True)
            retrieved = ""

        context2 += ("\n\n## Planning-pass proposal (advisory — override it "
                     "if the code says otherwise)\n"
                     + self._render_proposal(proposal))
        if retrieved:
            context2 += ("\n\n## Retrieved records and implementations "
                         "(what the planning pass asked for)\n" + retrieved)
        return super().apply(feedback, base_dir, out_dir, context=context2)

    # ------------------------------------------------------------------ #
    def _propose(self, *, feedback: Optional[AgentFeedback],
                 context: Optional[str], base_dir: Path,
                 out_dir: Path) -> Optional[dict]:
        parts: list[str] = []
        if context:
            parts.append(f"## Steering context\n{context}\n")
        # The registry ids the memory query can name (data for retrieval,
        # not steering): without them the planner invents ids that match
        # nothing and category retrieval silently returns empty.
        registry_block = self._render_registry_ids(Path(base_dir).parent,
                                                   base_dir=Path(base_dir))
        if registry_block:
            parts.append("## Registry ids for the memory query (strategy / "
                         "area ids with the nodes that used them)\n"
                         + registry_block + "\n")
        if feedback is not None:
            parts.append(self._format_feedback(feedback))
        sources = self._read_mutable_sources(base_dir / "task_agent")
        parts.append(self._format_current_sources(sources))
        user = "\n".join(parts)

        kwargs: dict[str, Any] = {
            "messages": [{"role": "system", "content": PROPOSE_SYSTEM},
                         {"role": "user", "content": user}],
            "tools": [PROPOSAL_TOOL],
        }
        model = self.propose_model or self.model
        effort = self.propose_reasoning_effort or self.reasoning_effort
        if model:
            kwargs["model"] = model
        if effort:
            kwargs["reasoning_effort"] = effort
        else:
            kwargs["temperature"] = 0.2
        if self.base_url:
            kwargs["base_url"] = self.base_url
        response = self.llm(**kwargs)

        if verbose_log.is_enabled():
            verbose_log.write_text(out_dir, "editor_propose_system.txt",
                                   PROPOSE_SYSTEM)
            verbose_log.write_text(out_dir, "editor_propose_user.txt", user)
            verbose_log.write_json(out_dir, "editor_propose_response.json", {
                "content": getattr(response, "content", None),
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments}
                    for c in (getattr(response, "tool_calls", []) or [])
                ],
            })

        for call in getattr(response, "tool_calls", []) or []:
            if call.name == "submit_edit_proposal":
                args = call.arguments
                if isinstance(args, dict) and args.get("edits"):
                    return args
        print("[editor:two_stage] warning: model did not call "
              "submit_edit_proposal; falling back to single-call edit",
              flush=True)
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_registry_ids(experiment_dir: Path, *, base_dir: Optional[Path] = None,
                             cap: int = 40) -> str:
        """Registry ids + node lists, the measured nodes with their Δ and
        verdict (so the query can target evidence, not just ids), and the
        nodes already retrieved for the parent's own edit (so successive
        expands do not re-read the same four records)."""
        from .edit_memory_render import _load_records
        registry = edit_archive._load_registry(experiment_dir)
        lines: list[str] = []
        for axis, label in (("strategies", "strategy"), ("areas", "area")):
            rows = []
            for cid, entry in (registry.get(axis) or {}).items():
                nodes = sorted({r.get("node") for r in (entry.get("edits") or [])
                                if isinstance(r.get("node"), int)})
                if nodes:
                    rows.append((-len(nodes), cid, nodes))
            rows.sort()
            for _, cid, nodes in rows[:cap]:
                lines.append(f"- {label} `{cid}` — nodes "
                             + ", ".join(str(n) for n in nodes))
        try:
            records = _load_records(experiment_dir)
        except Exception:  # noqa: BLE001
            records = {}
        measured = [(n, r) for n, r in sorted(records.items())
                    if r.get("delta") is not None or r.get("delta_all") is not None
                    or r.get("effect")]
        if measured:
            lines.append("measured nodes (judge verdict for the primary mechanism "
                         "with the checks it targeted · regressions · "
                         "implementation verdict · score as context: paired Δ over "
                         "shared cases, else unpaired Δ ± SE):")
            for n, r in measured[-cap:]:
                v = ("implementation sound" if r.get("impl_sound") else
                     "implementation unsound" if r.get("impl_sound") is False
                     else "implementation not judged")
                what = next((t.get("what") for t in (r.get("tags") or [])
                             if t.get("what")), "")
                bits = [f"judge {judge_summary(r) or 'not judged yet'}"]
                reg = regressions_summary(r)
                if reg:
                    bits.append(f"regressions: {reg}")
                bits += [v, f"score {perf_summary(r)}"]
                lines.append(f"- node {n}: " + " · ".join(bits)
                             + (f' · "{what[:90]}"' if what else ""))
        if base_dir is not None:
            try:
                m = json.loads((Path(base_dir) / edit_archive.RETRIEVAL_MANIFEST)
                               .read_text(encoding="utf-8"))
                prev = [x.get("node") for x in m.get("selected", [])
                        if isinstance(x.get("node"), int)]
            except (OSError, json.JSONDecodeError, AttributeError):
                prev = []
            if prev:
                lines.append("already retrieved for the parent's own edit "
                             "(prefer others unless needed again): "
                             + ", ".join(str(n) for n in prev))
        return "\n".join(lines)

    @staticmethod
    def _render_proposal(proposal: dict) -> str:
        lines: list[str] = []
        for i, e in enumerate(proposal.get("edits") or [], 1):
            if not isinstance(e, dict):
                continue
            tag = " / ".join(str(e[k]) for k in ("strategy", "area") if e.get(k))
            tag = f" [{tag}]" if tag else ""
            lines.append(f"{i}. {str(e.get('goal', ''))[:300]}{tag}")
            if e.get("mechanism"):
                lines.append(f"   mechanism: {str(e['mechanism'])[:500]}")
        pred = proposal.get("prediction") or {}
        if isinstance(pred, dict) and pred.get("belief_id"):
            eff = str(pred.get("expected_effect") or "").strip()
            targets = [str(t).strip() for t in (pred.get("expected_targets") or [])
                       if str(t).strip()][:8]
            if eff:
                # Judge-first: what verdict, on which checks. The score Δ is
                # never rendered into the editing call's context.
                line = f"prediction: belief:{pred['belief_id']} -> judge expected {eff}"
                if targets:
                    line += " on " + ", ".join(targets)
            else:
                line = (f"prediction: belief:{pred['belief_id']} -> expected "
                        f"{pred.get('expected_direction', '?')}")
            lines.append(line)
        return "\n".join(lines) or "(empty proposal)"

    def _write_prediction(self, out_dir: Path, proposal: dict) -> None:
        pred = proposal.get("prediction") or {}
        if not isinstance(pred, dict):
            pred = {}
        # Models sometimes echo the "belief:" anchor prefix into the id;
        # store the bare slug so joins/credit key consistently.
        belief_id = str(pred.get("belief_id") or "")
        if belief_id.lower().startswith("belief:"):
            belief_id = belief_id[len("belief:"):]
        payload = {
            "version": 2,
            "round_dir": out_dir.name,
            "belief_id": belief_id,
            # v2: the judge verdict and targeted checks the planner expects.
            "expected_effect": str(pred.get("expected_effect") or ""),
            "expected_targets": [str(t)[:80] for t in
                                 (pred.get("expected_targets") or [])
                                 if str(t).strip()][:8],
            # Legacy (pre-judge) prediction fields, kept for old readers.
            "expected_direction": str(pred.get("expected_direction") or ""),
            "expected_delta": pred.get("expected_delta"),
            "why": str(pred.get("why") or "")[:500],
            "proposal_goals": [
                str(e.get("goal", ""))[:300]
                for e in (proposal.get("edits") or []) if isinstance(e, dict)
            ],
            "query": proposal.get("memory_query") or {},
        }
        _atomic_write(out_dir / PREDICTION_NAME,
                      json.dumps(payload, indent=2) + "\n")

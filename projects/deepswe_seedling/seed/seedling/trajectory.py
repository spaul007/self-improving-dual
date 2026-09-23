"""ATIF v1.7 trajectory emission, with per-role attribution.

WHY THE SHAPE IS WHAT IT IS
---------------------------
ATIF v1.7 has native multi-agent support (Trajectory.subagent_trajectories +
ObservationResult.subagent_trajectory_ref), so we do NOT invent attribution.

But `pier view` renders ROOT steps only -- grepping pier/viewer/ for "subagent" returns
nothing. So attribution has to live in two places at once:
  * structurally, in subagent_trajectories (one per role run), and
  * visibly, as extra["role"] on every root step.

Embedded (not file-ref) subagents are the primary form because the whole document is then
ONE atomic write. A file-ref scheme needs N+1 writes, and the moment they can be
interrupted is exactly the moment they matter: a half-written set leaves a trajectory_path
pointing at a file that does not exist.

VALIDATOR COMPLIANCE IS STRUCTURAL, NOT DISCIPLINED
---------------------------------------------------
Each of these is enforced by the API shape rather than by remembering:
  step_id sequential from 1  -> the builder owns the counter; callers cannot pass an id
  steps min_length 1         -> every builder is SEEDED at construction
  agent-only fields          -> add_user_step/add_system_step simply LACK those parameters
  source_call_id must match  -> results are built FROM the calls, ids generated inside
  llm_call_count==0          -> drops metrics/reasoning rather than raising
  unique subagent ids        -> central registry, auto-suffixed on collision
  extra="forbid"             -> custom data only ever goes in the `extra` dicts

The Trajectory object is constructed NOWHERE except write(), which falls back to a minimal
document. A pydantic ValidationError must never reach run()'s outer handler -- that would
skip the collect hooks and zero the task.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA_VERSION = "ATIF-v1.7"
MAX_OBS_CHARS = 20_000  # keep the doc small: this is a sync write on a shared event loop


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(s: str, n: int = MAX_OBS_CHARS) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n...[{len(s) - n} chars elided]...\n" + s[-n // 2 :]


@dataclass
class Call:
    """One tool call and its result, kept together so the ids can never disagree."""

    function_name: str
    arguments: dict
    result: str = ""
    subagent_id: str | None = None
    extra: dict | None = None


class _StepList:
    """Owns the step counter. This is why step_id is always sequential from 1."""

    def __init__(self) -> None:
        self._steps: list[dict] = []

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> list[dict]:
        return self._steps

    def _append(self, source: str, message: str, **fields) -> dict:
        step = {
            "step_id": len(self._steps) + 1,  # callers cannot influence this
            "timestamp": _now(),
            "source": source,
            "message": _clip(message),
        }
        step.update({k: v for k, v in fields.items() if v is not None})
        self._steps.append(step)
        return step

    # -- non-agent steps: these methods deliberately have NO agent-only parameters ----
    def add_user_step(self, message: str, extra: dict | None = None) -> dict:
        return self._append("user", message, extra=extra)

    def add_system_step(self, message: str, extra: dict | None = None) -> dict:
        return self._append("system", message, extra=extra)

    # -- agent steps ------------------------------------------------------------------
    def add_agent_step(
        self,
        message: str,
        *,
        calls: list[Call] | None = None,
        llm_call_count: int = 1,
        model_name: str | None = None,
        reasoning_content: str | None = None,
        metrics: dict | None = None,
        extra: dict | None = None,
    ) -> dict:
        # v1.7: llm_call_count==0 means deterministic dispatch and forbids these.
        if llm_call_count == 0:
            metrics = None
            reasoning_content = None

        tool_calls = None
        observation = None
        if calls:
            tool_calls, results = [], []
            for c in calls:
                cid = f"tc_{uuid.uuid4().hex[:12]}"
                tc = {"tool_call_id": cid, "function_name": c.function_name,
                      "arguments": c.arguments}
                if c.extra:
                    tc["extra"] = c.extra
                tool_calls.append(tc)
                # Built FROM the call, so source_call_id can never dangle.
                res = {"source_call_id": cid, "content": _clip(c.result)}
                if c.subagent_id:
                    res["subagent_trajectory_ref"] = [{"trajectory_id": c.subagent_id}]
                results.append(res)
            observation = {"results": results}

        return self._append(
            "agent",
            message,
            model_name=model_name,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            llm_call_count=llm_call_count,
            extra=extra,
        )


class RoleRecorder(_StepList):
    """One role run == one embedded subagent trajectory, with its own counter from 1."""

    def __init__(self, trajectory_id: str, role: str, agent_name: str,
                 agent_version: str, session_id: str, brief: str,
                 extra: dict | None = None) -> None:
        super().__init__()
        self.trajectory_id = trajectory_id
        self.role = role
        self._agent = {"name": f"{agent_name}/{role}", "version": agent_version}
        self._session_id = session_id
        self.outcome: str = "running"
        self.summary: str = ""
        self.n_llm_calls = 0
        self.cost_usd = 0.0
        # SEEDED so `steps` is never empty even if the role dies on its first await.
        # `brief` is the ACTUAL context the role was handed -- see roles.py. Recording the
        # brief instead was an observability hole: for an agent whose entire premise is the
        # inter-role handoff, the handoff content was unauditable after the fact. A future
        # self-evolving algorithm reading these trajectories would be equally blind.
        self.add_user_step(brief or f"role: {role}", extra=extra)

    def finish(self, outcome: str, summary: str = "") -> None:
        self.outcome = outcome
        self.summary = summary
        self.add_system_step(f"role {self.role} finished: {outcome}")

    def payload(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "session_id": self._session_id,
            "agent": self._agent,
            "steps": self.steps,
        }


class TrajectoryBuilder(_StepList):
    """The root trajectory: the orchestrator's decision log."""

    def __init__(self, agent_name: str, agent_version: str, session_id: str,
                 model_name: str | None = None) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.session_id = session_id
        self.model_name = model_name
        self._subagents: list[RoleRecorder] = []
        self._ids: set[str] = set()
        self._role_counts: dict[str, int] = {}
        self.notes: str = ""
        self.extra: dict = {}
        self.totals = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0}

    def role(self, name: str, brief: str = "", extra: dict | None = None) -> RoleRecorder:
        n = self._role_counts.get(name, 0) + 1
        self._role_counts[name] = n
        tid = f"{self.session_id}.{name}.{n}"
        while tid in self._ids:  # collision guard; ids must be unique
            tid += "x"
        self._ids.add(tid)
        rec = RoleRecorder(tid, name, self.agent_name, self.agent_version,
                           self.session_id, brief, extra=extra)
        self._subagents.append(rec)
        return rec

    def dispatch(self, role: str, rec: RoleRecorder, brief: str, result: str) -> None:
        """Root-stream record of one role run. llm_call_count=0: deterministic dispatch."""
        self.add_agent_step(
            f"dispatch {role}",
            calls=[Call("run_role", {"role": role, "attempt": rec.trajectory_id.split(".")[-1]},
                        result=result, subagent_id=rec.trajectory_id,
                        extra={"outcome": rec.outcome, "llm_calls": rec.n_llm_calls})],
            llm_call_count=0,
            extra={"role": role},  # pier view only renders root steps -- keep it visible
        )

    def set_tokens(self, prompt: int = 0, completion: int = 0, cached: int = 0,
                   cost: float = 0.0) -> None:
        """Overwrite the totals from an authoritative counter (the LLM client).

        SET, not add: this is called on every observability flush as well as at the end, and
        an accumulating version would multiply the totals by the number of flushes.

        Why the totals are sourced from the LLM rather than recorded per call site: v8 has
        THREE places that hit the endpoint -- the role loop, the report call, and compaction --
        and `add_tokens` was never wired to any of them, so `final_metrics` reported all
        zeros and pier's own `n_input_tokens` came back None for every trial. Counting at the
        client means a new call site is included automatically instead of being forgotten,
        and compaction (the most expensive single call in v8) is counted for free.
        """
        self.totals["prompt"] = int(prompt or 0)
        self.totals["completion"] = int(completion or 0)
        self.totals["cached"] = int(cached or 0)
        self.totals["cost"] = float(cost or 0.0)

    def add_tokens(self, prompt: int = 0, completion: int = 0, cached: int = 0,
                   cost: float = 0.0) -> None:
        self.totals["prompt"] += int(prompt or 0)
        self.totals["completion"] += int(completion or 0)
        self.totals["cached"] += int(cached or 0)
        self.totals["cost"] += float(cost or 0.0)

    def all_steps(self) -> list[dict]:
        out = list(self.steps)
        for s in self._subagents:
            out.extend(s.steps)
        return out

    # -- payloads ---------------------------------------------------------------------
    def _root(self, steps: list[dict], subagents: list[dict] | None) -> dict:
        n_root = len(steps)
        total = n_root + sum(len(s["steps"]) for s in (subagents or []))
        doc = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "agent": {"name": self.agent_name, "version": self.agent_version,
                      **({"model_name": self.model_name} if self.model_name else {})},
            "steps": steps,
            # The schema permits total_steps != len(steps) only if documented in notes.
            "notes": (self.notes + " " if self.notes else "")
                     + f"total_steps counts root + subagent steps; root steps = {n_root}.",
            "final_metrics": {
                "total_prompt_tokens": self.totals["prompt"],
                "total_completion_tokens": self.totals["completion"],
                "total_cached_tokens": self.totals["cached"],
                "total_cost_usd": round(self.totals["cost"], 6),
                "total_steps": total,
            },
        }
        if subagents:
            doc["subagent_trajectories"] = subagents
        if self.extra:
            doc["extra"] = self.extra
        return doc

    def payload(self) -> dict:
        return self._root(self.steps, [s.payload() for s in self._subagents])

    def payload_minimal(self) -> dict:
        """Built only from data guaranteed to exist, so it is guaranteed to validate."""
        seed = self.steps[0] if self.steps else {
            "step_id": 1, "timestamp": _now(), "source": "user", "message": "(no steps)"}
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "agent": {"name": self.agent_name, "version": self.agent_version},
            "steps": [seed],
            "notes": "degraded: full trajectory failed to build",
        }

    # -- the write ---------------------------------------------------------------------
    def write(self, path) -> str:
        """Synchronous, atomic, and NEVER raises. Safe to call after CancelledError."""
        import pathlib

        path = pathlib.Path(path)
        payload, status = None, "full"
        for candidate, label in ((self.payload, "full"), (self.payload_minimal, "minimal")):
            try:
                doc = candidate()
                _validate(doc)
                payload, status = json.dumps(doc, indent=2, default=str), label
                break
            except Exception:  # noqa: BLE001
                continue
        if payload is None:
            status = "fallback"
            payload = json.dumps({
                "schema_version": SCHEMA_VERSION,
                "agent": {"name": self.agent_name, "version": self.agent_version},
                "steps": [{"step_id": 1, "timestamp": _now(), "source": "system",
                           "message": "trajectory build failed"}],
            }, indent=2)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.part")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # atomic
        except Exception:  # noqa: BLE001
            status += "+write-failed"
        return status


def _validate(doc: dict) -> None:
    """Validate against the real ATIF model when pier is importable; skip if not."""
    try:
        from pier.models.trajectories.trajectory import Trajectory
    except Exception:  # noqa: BLE001 -- unit tests may run without pier
        return
    Trajectory.model_validate(doc)

"""``EditMemoryLayer`` — the manager-facing object of the edit-memory layer.

Hooks (called by ``HGMManager``):

* ``setup(experiment_dir)`` once per run;
* ``choose_arm(tree) -> (arm, memory_version, memory_path)`` before every
  expansion — Thompson sampling over the two arms' pooled tallies;
* ``on_event(event, tree, node_id=...)`` next to every tree snapshot:
  ``expand_eval`` of a successful child advances the window; when the
  window holds ``window_size`` nodes the memory curator runs, then — every
  ``instruction_every`` memory versions — the instruction curator + updater
  revise the addendum, and finally the generator writes the next memory
  version under the (possibly just revised) instruction.

Everything is written under ``<experiment_dir>/edit_memory/`` (see the
package docstring for the layout); ``state.json`` is the audit trail.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..agentic.policy import MEMORY_DIR_NAME
from ..edit_diff import changed_mutable_files
from ..registry import register
from . import generator as G
from . import prompts as P
from .curator import CuratorConfig, run_curator
from .policy import build_curator_policy

ARM_WITH = "with"
ARM_WITHOUT = "without"
ARM_NONE = "none"
SELECTIONS = ("bandit", "always", "never")

MEMORY_FILE = "edit_memory.md"
INSTRUCTION_FILE = "instruction.md"
STATE_FILE = "state.json"


def memory_version_name(j: int) -> str:
    return f"edit_memory_v{j:03d}.md"


def instruction_version_name(k: int) -> str:
    return f"instruction_v{k:03d}.md"


@register("edit_memory", "agentic")
class EditMemoryLayer:
    def __init__(
        self,
        llm_caller: Callable[..., Any],
        *,
        window_size: int = 4,
        instruction_every: int = 2,
        selection: str = "bandit",
        arm_min_pulls: int = 2,
        beta_prior: float = 1.0,
        seed: int = 42,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        llm_timeout_s: Optional[float] = None,
        extra_body: Optional[dict[str, Any]] = None,
        curator: Optional[dict[str, Any]] = None,
        memory_max_chars: int = 40000,
        instruction_addendum_max_chars: int = 10000,
        project_root: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        if selection not in SELECTIONS:
            raise ValueError(f"edit_memory.selection must be one of {SELECTIONS}, got {selection!r}")
        if window_size < 1:
            raise ValueError("edit_memory.window_size must be >= 1")
        if instruction_every < 1:
            raise ValueError("edit_memory.instruction_every must be >= 1")
        self.llm = llm_caller
        self.window_size = int(window_size)
        self.instruction_every = int(instruction_every)
        self.selection = selection
        self.arm_min_pulls = int(arm_min_pulls)
        self.beta_prior = float(beta_prior)
        self.seed = int(seed)
        self.spec = G.LLMSpec(model=model, reasoning_effort=reasoning_effort, base_url=base_url,
                              api_key_env=api_key_env, llm_timeout_s=llm_timeout_s,
                              extra_body=extra_body)
        self.curator_cfg = CuratorConfig(**(curator or {}))
        self.memory_max_chars = int(memory_max_chars)
        self.instruction_addendum_max_chars = int(instruction_addendum_max_chars)
        self.project_root = Path(project_root) if project_root else None
        if repo_root is None:
            from ..agentic.policy import REPO_ROOT
            repo_root = REPO_ROOT
        self.repo_root = Path(repo_root)

        # Per-run state (setup()).
        self.dir: Optional[Path] = None
        self.memory_version = 0            # j: 0 = no memory yet
        self.instruction_version = 0       # k: 0 = empty addendum
        self.window: list[int] = []        # successful node ids in the open window
        self.window_failed: list[int] = []
        self.window_index = 0              # windows closed so far
        self.versions_since_instruction = 0
        self.with_nodes_since_instruction: list[int] = []
        self.node_arms: dict[int, tuple[str, Optional[int]]] = {}
        self.pulls: dict[str, int] = {ARM_WITH: 0, ARM_WITHOUT: 0}
        self.rng = random.Random(self.seed)
        self.rng_draws = 0
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Setup / files
    # ------------------------------------------------------------------ #

    def setup(self, experiment_dir: Path) -> None:
        self.dir = Path(experiment_dir) / MEMORY_DIR_NAME
        self.dir.mkdir(parents=True, exist_ok=True)
        state_path = self.dir / STATE_FILE
        if state_path.exists():
            self._load_state(json.loads(state_path.read_text(encoding="utf-8")))
        else:
            (self.dir / instruction_version_name(0)).write_text("", encoding="utf-8")
            (self.dir / INSTRUCTION_FILE).write_text("", encoding="utf-8")
            self._save_state("setup")

    @property
    def memory_path(self) -> Optional[Path]:
        """The current memory's versioned file (what a with-arm editor
        reads), or ``None`` before the first memory exists."""
        if self.dir is None or self.memory_version == 0:
            return None
        return self.dir / memory_version_name(self.memory_version)

    def _addendum(self) -> str:
        if self.dir is None:
            return ""
        p = self.dir / INSTRUCTION_FILE
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _memory_text(self) -> str:
        p = self.memory_path
        return p.read_text(encoding="utf-8") if p is not None and p.exists() else ""

    # ------------------------------------------------------------------ #
    # Bandit
    # ------------------------------------------------------------------ #

    def arm_tallies(self, tree: Any) -> dict[str, dict[str, float]]:
        """Pooled HGM tallies per arm over every node expanded under that
        arm. Only nodes that were actually a pull -- ``with`` or ``without``
        -- count; the pre-memory ``none`` nodes (expanded before the first
        window closed, when no memory existed to withhold) and the seed
        root are excluded, so the posteriors start updating only once the
        bandit is live. Edit-failed nodes carry no mass."""
        out = {ARM_WITH: {"S": 0.0, "F": 0.0, "n_nodes": 0},
               ARM_WITHOUT: {"S": 0.0, "F": 0.0, "n_nodes": 0}}
        for node in tree.nodes.values():
            if node.parent_id is None or node.memory_arm not in out:
                continue
            arm = node.memory_arm
            out[arm]["S"] += float(node.n_success)
            out[arm]["F"] += float(node.n_failure)
            out[arm]["n_nodes"] += 1
        return out

    def choose_arm(self, tree: Any) -> tuple[str, Optional[int], Optional[Path]]:
        """``(arm, memory_version, memory_path)`` for the expansion about to
        run. Records the pull; the node id is attached by ``on_event``."""
        path = self.memory_path
        if path is None:
            return ARM_NONE, None, None
        if self.selection == "never":
            self.pulls[ARM_WITHOUT] += 1
            return ARM_WITHOUT, self.memory_version, None
        if self.selection == "always":
            arm = ARM_WITH
        elif self.pulls[ARM_WITH] < self.arm_min_pulls:
            arm = ARM_WITH
        elif self.pulls[ARM_WITHOUT] < self.arm_min_pulls:
            arm = ARM_WITHOUT
        else:
            tallies = self.arm_tallies(tree)
            theta = {}
            for a in (ARM_WITH, ARM_WITHOUT):
                theta[a] = self.rng.betavariate(tallies[a]["S"] + self.beta_prior,
                                                tallies[a]["F"] + self.beta_prior)
                self.rng_draws += 1
            arm = max(theta, key=theta.get)
        self.pulls[arm] += 1
        return arm, self.memory_version, (path if arm == ARM_WITH else None)

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def on_event(self, event: str, tree: Any, *, node_id: Optional[int] = None) -> None:
        if self.dir is None:
            raise RuntimeError("EditMemoryLayer.setup(experiment_dir) was not called")
        if event in ("expand", "expand_eval") and node_id is not None:
            node = tree[node_id]
            self.node_arms[node_id] = (node.memory_arm, node.memory_version)
            if node.memory_arm == ARM_WITH and node_id not in self.with_nodes_since_instruction:
                self.with_nodes_since_instruction.append(node_id)
        if event == "expand" and node_id is not None and tree[node_id].edit_failed:
            self.window_failed.append(node_id)
            self._save_state("edit_failed", node_id=node_id)
            return
        if event == "expand_eval" and node_id is not None:
            if node_id not in self.window:
                self.window.append(node_id)
            self._save_state("window_add", node_id=node_id)
            if len(self.window) >= self.window_size:
                self._close_window(tree)
            return
        if event == "finalize":
            self._save_state("finalize")

    # ------------------------------------------------------------------ #
    # Window → curation → memory
    # ------------------------------------------------------------------ #

    def _node_meta(self, tree: Any, node_id: int) -> dict[str, Any]:
        n = tree[node_id]
        return {
            "node_id": n.node_id, "parent_id": n.parent_id,
            "memory_arm": n.memory_arm, "memory_version": n.memory_version,
            "n_evals": n.n_evals, "mean_utility": float(n.mean_utility),
            "edit_failed": bool(n.edit_failed), "round_dir": str(n.round_dir),
            "changed_files": (changed_mutable_files(tree[n.parent_id].round_dir, n.round_dir)
                              if n.parent_id is not None and not n.edit_failed else []),
        }

    def _close_window(self, tree: Any) -> None:
        assert self.dir is not None
        self.window_index += 1
        j = self.window_index
        workspace = self.dir / f"window_{j:03d}"
        workspace.mkdir(parents=True, exist_ok=True)
        live_ids, failed_ids = list(self.window), list(self.window_failed)
        self.window, self.window_failed = [], []
        nodes_meta = [self._node_meta(tree, nid) for nid in live_ids + failed_ids]
        (workspace / "window.json").write_text(json.dumps({
            "window_index": j, "nodes": nodes_meta, "memory_version_before": self.memory_version,
            "instruction_version": self.instruction_version, "t": time.time(),
        }, indent=2), encoding="utf-8")
        print(f"[edit_memory] window {j}: curating nodes {live_ids} "
              f"(failed edits: {failed_ids or 'none'})", flush=True)

        # 1. memory curator (agentic)
        policy = build_curator_policy(
            workspace=workspace, memory_dir=self.dir,
            node_dirs=[tree[nid].round_dir for nid in live_ids],
            parent_dirs=[tree[tree[nid].parent_id].round_dir if tree[nid].parent_id is not None else None
                         for nid in live_ids],
            repo_root=self.repo_root, project_root=self.project_root,
        )
        instruction = P.render_memory_curation_instruction(
            nodes=nodes_meta, previous_memory_exists=self.memory_version > 0,
            addendum=self._addendum(), max_llm_calls=self.curator_cfg.max_llm_calls,
            timeout_s=self.curator_cfg.timeout_s, max_attempts=self.curator_cfg.max_attempts,
        )
        try:
            result = run_curator(
                self.llm, policy=policy, system_prompt=P.MEMORY_CURATOR_SYSTEM,
                instruction=instruction, output_file=P.CURATION_FILE,
                validate=lambda text: G.validate_curation(text, node_ids=live_ids),
                salvage=lambda text: G.salvage_curation(text, node_ids=live_ids),
                cfg=self.curator_cfg, llm_kwargs=self.spec.kwargs_plain(),
            )
        except Exception as exc:  # noqa: BLE001 - never cost the run a window
            print(f"[edit_memory] window {j}: curator crashed: {exc!r}", flush=True)
            self._save_state("window_failed", window=j, error=repr(exc))
            return
        if not result.success:
            print(f"[edit_memory] window {j}: curation not accepted "
                  f"({result.end_reason}: {result.errors[:1]})", flush=True)
            self._save_state("window_failed", window=j, end_reason=result.end_reason)
            return
        curation = result.output_path.read_text(encoding="utf-8")

        # 2. instruction update BEFORE the generation it should govern: every
        # ``instruction_every`` memory versions, audit the nodes expanded with
        # the memory since the last update and revise the addendum, so that
        # B_{j+1} is generated under the instruction learned from B_j's use.
        # (Never fires at the first window: no memory version exists yet.)
        if self.versions_since_instruction >= self.instruction_every:
            self._update_instruction(tree)

        # 3. memory generator (one call), under the current addendum
        new_memory, memory_errors = G.generate_memory(
            self.llm, self.spec, previous_memory=self._memory_text(), curation=curation,
            addendum=self._addendum(),
            window_meta={"window_index": j, "nodes": [
                {"node_id": m["node_id"], "memory_arm": m["memory_arm"],
                 "memory_version": m["memory_version"]} for m in nodes_meta if not m["edit_failed"]]},
            max_chars=self.memory_max_chars, record_path=workspace / "memory_call.json",
            verbose_dir=workspace,
        )
        if new_memory is None:
            print(f"[edit_memory] window {j}: memory generation failed ({memory_errors[:1]}); "
                  f"keeping v{self.memory_version:03d}", flush=True)
            self._save_state("memory_failed", window=j, errors=memory_errors)
            return
        if memory_errors:
            # Never discard generated content: the final draft is used and the
            # check's remaining findings are recorded here and in state.json.
            print(f"[edit_memory] window {j}: memory accepted with check findings: "
                  f"{memory_errors[:2]}", flush=True)
        self.memory_version += 1
        (self.dir / memory_version_name(self.memory_version)).write_text(new_memory, encoding="utf-8")
        shutil.copyfile(self.dir / memory_version_name(self.memory_version), self.dir / MEMORY_FILE)
        self.versions_since_instruction += 1
        print(f"[edit_memory] window {j}: wrote {memory_version_name(self.memory_version)} "
              f"({len(new_memory)} chars)", flush=True)
        self._save_state("memory_written", window=j, memory_version=self.memory_version,
                         check_errors=memory_errors)

    # ------------------------------------------------------------------ #
    # Instruction update
    # ------------------------------------------------------------------ #

    def _update_instruction(self, tree: Any) -> None:
        assert self.dir is not None
        self.versions_since_instruction = 0
        with_ids = [nid for nid in self.with_nodes_since_instruction
                    if nid in tree.nodes and not tree[nid].edit_failed]
        if not with_ids:
            print("[edit_memory] instruction update skipped: no node was expanded "
                  "with the memory since the last update", flush=True)
            self._save_state("instruction_skipped")
            return
        k = self.instruction_version + 1
        workspace = self.dir / f"instruction_update_{k:03d}"
        workspace.mkdir(parents=True, exist_ok=True)
        nodes_meta = [self._node_meta(tree, nid) for nid in with_ids]
        (workspace / "nodes.json").write_text(json.dumps(nodes_meta, indent=2), encoding="utf-8")
        print(f"[edit_memory] instruction update {k}: auditing with-memory nodes {with_ids}", flush=True)
        policy = build_curator_policy(
            workspace=workspace, memory_dir=self.dir,
            node_dirs=[tree[nid].round_dir for nid in with_ids],
            parent_dirs=[tree[tree[nid].parent_id].round_dir for nid in with_ids],
            repo_root=self.repo_root, project_root=self.project_root,
        )
        instruction = P.render_instruction_curation_instruction(
            nodes=nodes_meta, addendum=self._addendum(),
            max_llm_calls=self.curator_cfg.max_llm_calls, timeout_s=self.curator_cfg.timeout_s,
            max_attempts=self.curator_cfg.max_attempts,
        )
        try:
            result = run_curator(
                self.llm, policy=policy, system_prompt=P.INSTRUCTION_CURATOR_SYSTEM,
                instruction=instruction, output_file=P.Q_FILE, validate=G.validate_q,
                salvage=G.salvage_q,
                cfg=self.curator_cfg, llm_kwargs=self.spec.kwargs_plain(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[edit_memory] instruction update {k}: curator crashed: {exc!r}", flush=True)
            self._save_state("instruction_failed", update=k, error=repr(exc))
            return
        if not result.success:
            print(f"[edit_memory] instruction update {k}: audit not accepted "
                  f"({result.end_reason})", flush=True)
            self._save_state("instruction_failed", update=k, end_reason=result.end_reason)
            return
        q = result.output_path.read_text(encoding="utf-8")
        previous_q = []
        for i in range(1, k):
            pq = self.dir / f"instruction_update_{i:03d}" / P.Q_FILE
            if pq.exists():
                previous_q.append(pq.read_text(encoding="utf-8"))
        new_addendum, addendum_errors = G.update_instruction(
            self.llm, self.spec, addendum=self._addendum(), q=q, previous_q=previous_q,
            max_chars=self.instruction_addendum_max_chars,
            record_path=workspace / "update_call.json", verbose_dir=workspace,
        )
        if new_addendum is None:
            print(f"[edit_memory] instruction update {k}: failed ({addendum_errors[:1]}); keeping "
                  f"v{self.instruction_version:03d}", flush=True)
            self._save_state("instruction_failed", update=k, errors=addendum_errors)
            return
        if addendum_errors:
            print(f"[edit_memory] instruction update {k}: addendum accepted with check findings: "
                  f"{addendum_errors[:2]}", flush=True)
        self.instruction_version = k
        (self.dir / instruction_version_name(k)).write_text(new_addendum, encoding="utf-8")
        (self.dir / INSTRUCTION_FILE).write_text(new_addendum, encoding="utf-8")
        self.with_nodes_since_instruction = []
        print(f"[edit_memory] instruction update {k}: wrote {instruction_version_name(k)} "
              f"({len(new_addendum)} chars)", flush=True)
        self._save_state("instruction_written", update=k, check_errors=addendum_errors)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def state(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "instruction_version": self.instruction_version,
            "window_index": self.window_index,
            "window": list(self.window),
            "window_failed": list(self.window_failed),
            "versions_since_instruction": self.versions_since_instruction,
            "with_nodes_since_instruction": list(self.with_nodes_since_instruction),
            "node_arms": {str(k): list(v) for k, v in self.node_arms.items()},
            "pulls": dict(self.pulls),
            "rng_draws": self.rng_draws,
            "config": {
                "window_size": self.window_size, "instruction_every": self.instruction_every,
                "selection": self.selection, "arm_min_pulls": self.arm_min_pulls,
                "beta_prior": self.beta_prior, "seed": self.seed,
            },
            "events": self.events,
        }

    def _save_state(self, event: str, **fields: Any) -> None:
        assert self.dir is not None
        self.events.append({"t": round(time.time(), 3), "event": event, **fields})
        (self.dir / STATE_FILE).write_text(json.dumps(self.state(), indent=2), encoding="utf-8")

    def _load_state(self, st: dict[str, Any]) -> None:
        self.memory_version = int(st.get("memory_version", 0))
        self.instruction_version = int(st.get("instruction_version", 0))
        self.window_index = int(st.get("window_index", 0))
        self.window = list(st.get("window", []))
        self.window_failed = list(st.get("window_failed", []))
        self.versions_since_instruction = int(st.get("versions_since_instruction", 0))
        self.with_nodes_since_instruction = list(st.get("with_nodes_since_instruction", []))
        self.node_arms = {int(k): (v[0], v[1]) for k, v in st.get("node_arms", {}).items()}
        self.pulls = {ARM_WITH: 0, ARM_WITHOUT: 0, **st.get("pulls", {})}
        # betavariate consumes a variable number of underlying draws, so the
        # stream cannot be replayed exactly; re-seed deterministically from
        # the draw count instead (state.json is an audit trail, not a resume).
        self.rng_draws = int(st.get("rng_draws", 0))
        self.rng = random.Random(self.seed + 1_000_003 * self.rng_draws)
        self.events = list(st.get("events", []))

"""EditMemoryLayer: bandit over pooled arm tallies, window cadence, files,
and the end-to-end HGM run with the layer attached (scripted LLM playing
the curators and the two single calls; sandbox "none")."""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent.edit_memory import prompts as P
from meta_agent.edit_memory.layer import (
    ARM_NONE,
    ARM_WITH,
    ARM_WITHOUT,
    EditMemoryLayer,
    memory_version_name,
)
from meta_agent.managers.hgm_tree import HGMNode, HGMTree
from meta_agent.models import CaseResult

# --------------------------------------------------------------------------- #
# Scripted model
# --------------------------------------------------------------------------- #


@dataclass
class _Call:
    id: str
    name: str
    arguments: dict


@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    raw: object = None


def _memory_doc(tag: str) -> str:
    return "".join(h + f"\n- {tag}: node 1 did X (helped)\n" for h in P.MEMORY_SECTIONS)


def _curation_doc(node_ids) -> str:
    out = ""
    for nid in node_ids:
        out += P.NODE_SECTION_HEADING.format(node_id=nid) + "\n"
        out += "".join(f"### {s}\nevidence for node {nid}\n" for s in P.NODE_SUBSECTIONS)
    out += P.CURATION_CROSS_HEADING + "\npatterns\n" + P.CURATION_GRADIENT_HEADING + "\nmissing: x\n"
    return out


def _q_doc() -> str:
    return "".join(h + "\nfinding\n" for h in P.Q_SECTIONS)


class _ScriptedLLM:
    """Plays every role by looking at the tools it is offered: a curator
    session (has `submit_curation`) writes its document with one editor
    call then submits; a plain call (no tools) returns the next scripted
    document. Records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.plain_docs: list[str] = []
        self.session_n = 0

    def __call__(self, **kw):
        self.calls.append({k: v for k, v in kw.items() if k != "messages"} | {"messages": list(kw["messages"])})
        tools = kw.get("tools") or []
        names = [t["name"] for t in tools]
        if P.SUBMIT_CURATION_NAME in names:
            instr = kw["messages"][1]["content"]
            # Which document does this session owe?
            if f"$WORK_DIR/{P.Q_FILE}" in instr:
                doc, fname = _q_doc(), P.Q_FILE
            else:
                ids = [int(t.split()[0]) for t in
                       [line.strip().lstrip("$NODE_").split(None, 1)[1] for line in instr.splitlines()
                        if line.strip().startswith("$NODE_")]]
                doc, fname = _curation_doc(ids), P.CURATION_FILE
            if any(isinstance(m, dict) and m.get("type") == "function_call" for m in kw["messages"]):
                return _Resp(tool_calls=[_Call(f"s{self.session_n}", P.SUBMIT_CURATION_NAME, {"summary": "done"})])
            self.session_n += 1
            return _Resp(tool_calls=[_Call(f"c{self.session_n}", "editor",
                                           {"command": "create", "path": f"$WORK_DIR/{fname}", "file_text": doc})])
        # plain call: memory generation or instruction update
        sys_msg = kw["messages"][0]["content"]
        if sys_msg.startswith(P.CORE_SENTINEL):
            self.plain_docs.append("memory")
            return _Resp(_memory_doc(f"v{self.plain_docs.count('memory')}"))
        self.plain_docs.append("instruction")
        return _Resp(f"- addendum v{self.plain_docs.count('instruction')}: cite failing checks")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tree_with(nodes: list[tuple[int, int | None, str, list[float]]]) -> HGMTree:
    tree = HGMTree(rng=random.Random(0))
    for nid, pid, arm, scores in nodes:
        n = HGMNode(node_id=nid, parent_id=pid, round_dir=Path(f"/r/round_{nid:03d}"), memory_arm=arm)
        for i, s in enumerate(scores):
            n.record(CaseResult(case_id=f"c{i}", passed=s >= 1, score=s))
        tree.add(n)
    return tree


class LayerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="editmem_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.exp = self.tmp / "exp"
        self.exp.mkdir()
        (self.exp / "config.snapshot.yaml").write_text("")

    def layer(self, **kw) -> EditMemoryLayer:
        opts = dict(window_size=2, instruction_every=2, arm_min_pulls=1, seed=3,
                    curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60})
        opts.update(kw)
        lay = EditMemoryLayer(_ScriptedLLM(), repo_root=self.tmp / "repo", **opts)
        lay.setup(self.exp)
        return lay


# --------------------------------------------------------------------------- #
# Bandit
# --------------------------------------------------------------------------- #


class TestBandit(LayerBase):
    def test_none_until_a_memory_exists(self) -> None:
        lay = self.layer()
        tree = _tree_with([(0, None, "none", [0.5]), (1, 0, "none", [0.6])])
        self.assertEqual(lay.choose_arm(tree), (ARM_NONE, None, None))
        self.assertEqual(lay.pulls, {ARM_WITH: 0, ARM_WITHOUT: 0})

    def test_tallies_pool_by_arm_and_exclude_root_none_and_failed(self) -> None:
        lay = self.layer()
        tree = _tree_with([
            (0, None, "none", [1.0, 1.0]),          # root — excluded
            (1, 0, "none", [0.5, 0.5]),             # pre-memory — excluded (not a pull)
            (2, 0, "without", [0.25]),
            (3, 1, "with", [0.75, 0.75, 0.75]),
        ])
        failed = HGMNode(node_id=4, parent_id=1, round_dir=Path("/r/4"), memory_arm="with", edit_failed=True)
        tree.add(failed)
        t = lay.arm_tallies(tree)
        self.assertAlmostEqual(t[ARM_WITHOUT]["S"], 0.25)
        self.assertAlmostEqual(t[ARM_WITHOUT]["F"], 0.75)
        self.assertEqual(t[ARM_WITHOUT]["n_nodes"], 1)
        self.assertAlmostEqual(t[ARM_WITH]["S"], 2.25)
        self.assertAlmostEqual(t[ARM_WITH]["F"], 0.75)
        self.assertEqual(t[ARM_WITH]["n_nodes"], 2)        # the failed node is listed, adds no mass

    def test_thompson_prefers_the_better_arm(self) -> None:
        lay = self.layer(arm_min_pulls=1)
        lay.memory_version = 1
        (lay.dir / memory_version_name(1)).write_text("M")
        # Forced pulls first: with (0 pulls) then without (0 pulls).
        tree = _tree_with([(0, None, "none", [0.5])])
        self.assertEqual(lay.choose_arm(tree)[0], ARM_WITH)
        self.assertEqual(lay.choose_arm(tree)[0], ARM_WITHOUT)
        # Lopsided tallies: with-arm nodes score ~0.9, without ~0.1.
        tree = _tree_with([(0, None, "none", [0.5])] +
                          [(i, 0, "with", [0.9] * 20) for i in range(1, 4)] +
                          [(i, 0, "without", [0.1] * 20) for i in range(4, 7)])
        picks = [lay.choose_arm(tree)[0] for _ in range(30)]
        self.assertGreater(picks.count(ARM_WITH), 27)
        arm, version, path = lay.choose_arm(tree)
        self.assertEqual((version, path), (1, lay.dir / memory_version_name(1)))
        self.assertEqual(lay.rng_draws, 2 * 31)
        # Reverse the evidence → the other arm.
        tree = _tree_with([(0, None, "none", [0.5])] +
                          [(i, 0, "with", [0.1] * 20) for i in range(1, 4)] +
                          [(i, 0, "without", [0.9] * 20) for i in range(4, 7)])
        picks = [lay.choose_arm(tree)[0] for _ in range(30)]
        self.assertGreater(picks.count(ARM_WITHOUT), 27)
        self.assertIsNone(lay.choose_arm(tree)[2] if picks[-1] == ARM_WITHOUT else None)

    def test_always_and_never(self) -> None:
        tree = _tree_with([(0, None, "none", [0.5])])
        lay = self.layer(selection="always")
        lay.memory_version = 2
        (lay.dir / memory_version_name(2)).write_text("M")
        self.assertEqual(lay.choose_arm(tree), (ARM_WITH, 2, lay.dir / memory_version_name(2)))
        lay = self.layer(selection="never")
        lay.memory_version = 2
        (lay.dir / memory_version_name(2)).write_text("M")
        self.assertEqual(lay.choose_arm(tree), (ARM_WITHOUT, 2, None))
        with self.assertRaises(ValueError):
            self.layer(selection="random")


# --------------------------------------------------------------------------- #
# Cadence + files (layer driven directly)
# --------------------------------------------------------------------------- #


def _round_dir(exp: Path, nid: int, code: str = "def run_task(task):\n    return None\n") -> Path:
    rd = exp / f"round_{nid:03d}"
    (rd / "task_agent" / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (rd / "task_agent" / "workflow.py").write_text(code)
    (rd / "task_agent" / "tool_wrapper.py").write_text("")
    (rd / "task_agent" / "tools_schema.json").write_text("[]")
    (rd / "logs").mkdir(exist_ok=True)
    (rd / "strategy.json").write_text("{}")
    (rd / "feedback.json").write_text("{}")
    return rd


class TestCadence(LayerBase):
    def _tree(self) -> HGMTree:
        tree = HGMTree(rng=random.Random(0))
        root = HGMNode(node_id=0, parent_id=None, round_dir=_round_dir(self.exp, 0))
        root.record(CaseResult(case_id="a", passed=True, score=1.0))
        tree.add(root)
        return tree

    def _child(self, tree: HGMTree, nid: int, arm: str = "none", version=None, *, failed=False,
               score: float = 0.5) -> HGMNode:
        n = HGMNode(node_id=nid, parent_id=0, round_dir=_round_dir(self.exp, nid, f"# node {nid}\n"),
                    memory_arm=arm, memory_version=version, edit_failed=failed)
        if not failed:
            n.record(CaseResult(case_id="a", passed=score >= 1, score=score))
        tree.add(n)
        return n

    def test_window_closes_after_m_successful_nodes(self) -> None:
        lay = self.layer(window_size=2, instruction_every=2)
        llm: _ScriptedLLM = lay.llm
        tree = self._tree()
        lay.on_event("seed", tree)
        self._child(tree, 1)
        lay.on_event("expand", tree, node_id=1)
        lay.on_event("expand_eval", tree, node_id=1)
        self.assertEqual(lay.window, [1])
        self.assertEqual(lay.memory_version, 0)
        # A failed edit is listed but does not count.
        self._child(tree, 2, failed=True)
        lay.on_event("expand", tree, node_id=2)
        self.assertEqual(lay.window_failed, [2])
        self.assertEqual(lay.window, [1])
        self._child(tree, 3)
        lay.on_event("expand", tree, node_id=3)
        lay.on_event("expand_eval", tree, node_id=3)
        # Window closed → curation + generation.
        self.assertEqual(lay.window, [])
        self.assertEqual(lay.window_index, 1)
        self.assertEqual(lay.memory_version, 1)
        w = lay.dir / "window_001"
        self.assertTrue((w / P.CURATION_FILE).exists())
        self.assertTrue((w / "agentic" / "transcript.jsonl").exists())
        self.assertTrue((w / "memory_call.json").exists())
        meta = json.loads((w / "window.json").read_text())
        self.assertEqual([n["node_id"] for n in meta["nodes"]], [1, 3, 2])
        self.assertTrue(meta["nodes"][2]["edit_failed"])
        self.assertEqual(meta["nodes"][0]["changed_files"], ["workflow.py"])
        self.assertEqual((lay.dir / memory_version_name(1)).read_text(), _memory_doc("v1"))
        self.assertEqual((lay.dir / "edit_memory.md").read_text(), _memory_doc("v1"))
        self.assertEqual(lay.memory_path, lay.dir / memory_version_name(1))
        # The curator session saw the right roots and the generation call the curation.
        sess = json.loads((w / "agentic" / "session.json").read_text())
        self.assertEqual(sess["end_reason"], "submitted")
        self.assertEqual(set(sess["roots"]), {"WORK_DIR", "MEMORY_DIR", "NODE_1", "PARENT_1", "NODE_2", "PARENT_2", "REPO_DIR"})
        gen_calls = [c for c in llm.calls if not c.get("tools")]
        self.assertEqual(len(gen_calls), 1)
        self.assertIn("evidence for node 3", gen_calls[0]["messages"][1]["content"])
        self.assertIn("(none yet", gen_calls[0]["messages"][1]["content"])
        # No instruction update yet (n = 2).
        self.assertEqual(lay.instruction_version, 0)
        self.assertEqual(lay.versions_since_instruction, 1)

    def test_instruction_update_runs_before_the_generation_it_governs(self) -> None:
        """window 1 -> v1 (no audit possible yet); window 2 -> audit the
        with-arm nodes of window 2 -> I1 -> v2 generated UNDER I1; window 3
        -> I2 -> v3 under I2 (instruction_every 1)."""
        lay = self.layer(window_size=2, instruction_every=1)
        tree = self._tree()
        gen_calls = lambda: [c for c in lay.llm.calls if not c.get("tools")
                             and c["messages"][0]["content"].startswith(P.CORE_SENTINEL)]
        # Window 1: two pre-memory nodes.
        for nid in (1, 2):
            self._child(tree, nid)
            lay.on_event("expand", tree, node_id=nid); lay.on_event("expand_eval", tree, node_id=nid)
        self.assertEqual((lay.memory_version, lay.instruction_version), (1, 0))
        self.assertNotIn("instruction_skipped", [e["event"] for e in lay.events])   # gate never opened
        self.assertNotIn("ADDENDUM", gen_calls()[0]["messages"][0]["content"])
        # Window 2: one with-arm node, one without.
        self._child(tree, 3, arm="with", version=1)
        lay.on_event("expand", tree, node_id=3); lay.on_event("expand_eval", tree, node_id=3)
        self._child(tree, 4, arm="without", version=1)
        lay.on_event("expand", tree, node_id=4); lay.on_event("expand_eval", tree, node_id=4)
        self.assertEqual((lay.memory_version, lay.instruction_version), (2, 1))
        events = [e["event"] for e in lay.events]
        mem_writes = [i for i, e in enumerate(events) if e == "memory_written"]
        self.assertLess(mem_writes[0], events.index("instruction_written"))   # v1 first
        self.assertLess(events.index("instruction_written"), mem_writes[1])   # then I1, then v2
        u = lay.dir / "instruction_update_001"
        self.assertEqual([n["node_id"] for n in json.loads((u / "nodes.json").read_text())], [3])
        self.assertEqual((lay.dir / "instruction.md").read_text(), "- addendum v1: cite failing checks\n")
        # v2 was generated under I1 ...
        self.assertIn("- addendum v1: cite failing checks", gen_calls()[1]["messages"][0]["content"])
        # ... and the audit ran after the window's curation (it may read it).
        self.assertTrue((lay.dir / "window_002" / P.CURATION_FILE).exists())
        self.assertEqual(lay.with_nodes_since_instruction, [])
        # Window 3: audit node 5 (with v2) -> I2 -> v3 under I2.
        self._child(tree, 5, arm="with", version=2)
        lay.on_event("expand", tree, node_id=5); lay.on_event("expand_eval", tree, node_id=5)
        self._child(tree, 6, arm="with", version=2)
        lay.on_event("expand", tree, node_id=6); lay.on_event("expand_eval", tree, node_id=6)
        self.assertEqual((lay.memory_version, lay.instruction_version), (3, 2))
        self.assertIn("- addendum v2: cite failing checks", gen_calls()[2]["messages"][0]["content"])
        self.assertEqual([n["node_id"] for n in json.loads((lay.dir / "instruction_update_002" / "nodes.json").read_text())], [5, 6])
        st = json.loads((lay.dir / "state.json").read_text())
        self.assertEqual((st["memory_version"], st["instruction_version"]), (3, 2))
        self.assertEqual(st["node_arms"]["3"], ["with", 1])

    def test_instruction_update_skipped_when_no_with_arm_nodes(self) -> None:
        lay = self.layer(window_size=1, instruction_every=1)
        tree = self._tree()
        self._child(tree, 1)
        lay.on_event("expand", tree, node_id=1); lay.on_event("expand_eval", tree, node_id=1)
        self.assertEqual(lay.memory_version, 1)
        # Window 2 with a without-arm node only: gate opens, audit skipped, v2 still written.
        self._child(tree, 2, arm="without", version=1)
        lay.on_event("expand", tree, node_id=2); lay.on_event("expand_eval", tree, node_id=2)
        self.assertEqual((lay.memory_version, lay.instruction_version), (2, 0))
        self.assertIn("instruction_skipped", [e["event"] for e in lay.events])
        self.assertEqual(lay.versions_since_instruction, 1)
        # instruction_every 2 (fresh run dir): the gate opens at the third window.
        self.exp = self.tmp / "exp2"
        self.exp.mkdir()
        (self.exp / "config.snapshot.yaml").write_text("")
        lay2 = self.layer(window_size=1, instruction_every=2)
        tree2 = self._tree()
        for nid, arm, v in ((1, "none", None), (2, "with", 1), (3, "with", 2)):
            self._child(tree2, nid, arm=arm, version=v)
            lay2.on_event("expand", tree2, node_id=nid); lay2.on_event("expand_eval", tree2, node_id=nid)
        self.assertEqual((lay2.memory_version, lay2.instruction_version), (3, 1))
        self.assertEqual([n["node_id"] for n in json.loads((lay2.dir / "instruction_update_001" / "nodes.json").read_text())], [2, 3])

    def test_failed_curation_closes_the_window_without_a_memory(self) -> None:
        lay = self.layer(window_size=1)

        def dead(**kw):
            raise RuntimeError("provider down")
        lay.llm = dead
        tree = self._tree()
        self._child(tree, 1)
        lay.on_event("expand", tree, node_id=1)
        lay.on_event("expand_eval", tree, node_id=1)
        self.assertEqual(lay.memory_version, 0)
        self.assertEqual(lay.window, [])
        self.assertEqual(lay.window_index, 1)
        self.assertEqual(lay.events[-1]["event"], "window_failed")
        self.assertIsNone(lay.memory_path)

    def test_generation_with_findings_is_still_written(self) -> None:
        """A memory that fails the check after the retry is used anyway; the
        findings are recorded in state.json and memory_call.json."""
        lay = self.layer(window_size=1)
        tree = self._tree()
        self._child(tree, 1)
        lay.on_event("expand", tree, node_id=1)
        lay.on_event("expand_eval", tree, node_id=1)
        self.assertEqual(lay.memory_version, 1)
        orig = lay.llm

        class Bad(_ScriptedLLM):
            def __call__(self, **kw):
                if not kw.get("tools"):
                    self.calls.append(kw)
                    return _Resp(_memory_doc("v2") + "\nexpected score 0.95\n")
                return orig(**kw)
        lay.llm = Bad()
        self._child(tree, 2)
        lay.on_event("expand", tree, node_id=2)
        lay.on_event("expand_eval", tree, node_id=2)
        self.assertEqual(lay.memory_version, 2)
        self.assertEqual(len(lay.llm.calls), 2)                       # one retry
        self.assertIn("expected score 0.95", (lay.dir / memory_version_name(2)).read_text())
        ev = [e for e in lay.events if e["event"] == "memory_written"][-1]
        self.assertTrue(any("score prediction" in x for x in ev["check_errors"]))
        rec = json.loads((lay.dir / "window_002" / "memory_call.json").read_text())
        self.assertFalse(rec["accepted"])

    def test_llm_failure_keeps_previous_memory(self) -> None:
        lay = self.layer(window_size=1)
        tree = self._tree()
        self._child(tree, 1)
        lay.on_event("expand", tree, node_id=1)
        lay.on_event("expand_eval", tree, node_id=1)
        orig = lay.llm

        class Dead(_ScriptedLLM):
            def __call__(self, **kw):
                if not kw.get("tools"):
                    raise RuntimeError("provider down")
                return orig(**kw)
        lay.llm = Dead()
        self._child(tree, 2)
        lay.on_event("expand", tree, node_id=2)
        lay.on_event("expand_eval", tree, node_id=2)
        self.assertEqual(lay.memory_version, 1)
        self.assertEqual(lay.events[-1]["event"], "memory_failed")


# --------------------------------------------------------------------------- #
# End to end through HGMManager with a stub editor
# --------------------------------------------------------------------------- #


class TestHGMIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="editmem_hgm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        (self.seed / "mutable_tools").mkdir(parents=True)
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        (self.seed / "tool_wrapper.py").write_text("")
        (self.seed / "tools_schema.json").write_text("[]")
        self.exp = self.tmp / "exp"
        self.exp.mkdir()
        (self.exp / "config.snapshot.yaml").write_text("")

    def _run(self, *, expand_eval_size=4, selection="bandit", window_size=2, instruction_every=2):
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        manager = HGMManager(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6,
                             seed=7, finalize_top_k=0, expand_eval_size=expand_eval_size)
        lay = EditMemoryLayer(_ScriptedLLM(), repo_root=self.tmp / "repo", window_size=window_size,
                              instruction_every=instruction_every, selection=selection,
                              arm_min_pulls=1, seed=3,
                              curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60})
        editor = _StubEditor()
        outcome = manager.evolve(
            editor=editor, evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed, benchmark_dir=self.tmp / "bench", experiment_dir=self.exp,
            max_rounds=30, score_target=None, train_case_ids=[f"c{i}" for i in range(20)],
            eval_case_ids=None, edit_memory=lay,
        )
        return manager, lay, editor, outcome

    def test_layer_runs_inside_the_loop(self) -> None:
        manager, lay, editor, _ = self._run()
        nodes = [n for n in manager._tree.nodes.values() if n.parent_id is not None]
        self.assertGreaterEqual(len(nodes), 4)
        self.assertGreaterEqual(lay.memory_version, 1)
        self.assertEqual(lay.window_index, len(nodes) // 2)
        # Every expansion after the first memory chose an arm; before it, none.
        arms = [n.memory_arm for n in sorted(nodes, key=lambda n: n.node_id)]
        self.assertEqual(arms[:2], ["none", "none"])
        self.assertTrue(set(arms[2:]) <= {"with", "without"} and arms[2:])
        self.assertIn("with", arms)
        # The editor received the memory path exactly on the with-arm expansions.
        for n, mp in zip(sorted(nodes, key=lambda n: n.node_id), editor.memory_paths):
            if n.memory_arm == "with":
                self.assertEqual(mp, lay.dir / memory_version_name(n.memory_version))
            else:
                self.assertIsNone(mp)
        # Sidecars and snapshots carry the arm.
        side = json.loads((nodes[-1].round_dir / "hgm_node.json").read_text())
        self.assertIn("memory_arm", side)
        self.assertIn("memory_version", side)
        st = json.loads((lay.dir / "state.json").read_text())
        self.assertEqual(st["events"][-1]["event"], "finalize")
        self.assertEqual(sum(st["pulls"].values()), len([a for a in arms if a != "none"]))

    def test_layer_requires_paired_evaluation(self) -> None:
        with self.assertRaises(ValueError):
            self._run(expand_eval_size=0)

    def test_never_mode_writes_memory_nobody_reads(self) -> None:
        manager, lay, editor, _ = self._run(selection="never")
        self.assertGreaterEqual(lay.memory_version, 1)
        self.assertTrue(all(mp is None for mp in editor.memory_paths))
        arms = {n.memory_arm for n in manager._tree.nodes.values() if n.parent_id is not None}
        self.assertEqual(arms, {"none", "without"})


if __name__ == "__main__":
    unittest.main()

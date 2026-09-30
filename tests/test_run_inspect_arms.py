"""Dashboard loaders for agentic / edit-memory runs (``meta_agent.run_inspect``
arm helpers, ``run_inspect_agentic.load_verbose_prompts``).

The main check runs a small block-HGM search with the real agentic editor
and the real memory layer (scripted LLMs, stub evaluator) and asserts that
the dashboard's pooled arm tallies, recomputed from the round dirs, equal
the layer's own ``arm_tallies`` -- what the bandit actually sampled from.

    PYTHONPATH=. python3 -m unittest tests.test_run_inspect_arms
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from meta_agent import run_inspect as ri
from meta_agent import run_inspect_agentic as ra

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "projects" / "travel_mas_refactored"
EXCL = ["workflow.py", "benchmark/", "agents/immutable/"]


def _router():
    from tests.test_agentic_editor import _Call, _Resp
    from tests.test_edit_memory_layer import _ScriptedLLM

    curators = _ScriptedLLM()

    def llm(**kw):
        names = [t["name"] for t in (kw.get("tools") or [])]
        if "submit_self_improvement" in names:
            done = sum(1 for m in kw["messages"]
                       if isinstance(m, dict) and m.get("type") == "function_call_output")
            if done == 0:
                return _Resp(tool_calls=[_Call("e1", "editor", {
                    "command": "create", "path": f"agents/check_{uuid.uuid4().hex[:6]}.py",
                    "file_text": "def check(plan):\n    return bool(plan)\n"})])
            return _Resp(tool_calls=[_Call("s1", "submit_self_improvement", {
                "optimization_goal": "add a verifier", "proposed_changes": "p", "rationale": "r"})])
        return curators(**kw)

    return llm


class AgenticRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from meta_agent.agent_editor_agentic import AgenticEditor
        from meta_agent.edit_memory.layer import EditMemoryLayer
        from meta_agent.editor_validators import ImmutableFilesValidator, SyntaxValidator
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm_block_tagged import BlockTaggedHGMManager
        from tests.test_hgm_smoke import _StubEvaluator

        cls.tmp = Path(tempfile.mkdtemp(prefix="inspect_arms_"))
        cls.exp = cls.tmp / "run"
        cls.exp.mkdir()
        (cls.exp / "config.snapshot.yaml").write_text(
            "edit_memory:\n  type: agentic\n  config: {window_size: 2, beta_prior: 1.0}\n")
        llm = _router()
        editor = AgenticEditor(llm, [SyntaxValidator(), ImmutableFilesValidator(mutable_exclude=EXCL)],
                               mutable_exclude=EXCL, project_root=PROJECT, sandbox="none",
                               max_llm_calls=6)
        cls.layer = EditMemoryLayer(llm, window_size=2, instruction_every=1, arm_min_pulls=1, seed=3,
                                    curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60},
                                    mutable_exclude=EXCL, project_root=PROJECT)
        cls.manager = BlockTaggedHGMManager(
            eval_budget=40, init_expansions=1, eval_batch_size=4, alpha=0.6, seed=5,
            snapshot_tree=True, finalize_top_k=0, expand_eval_size=4,
            block_selection_strategy="adaptive")
        cls.manager.evolve(
            editor=editor, evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
            seed_dir=PROJECT / "seed", benchmark_dir=PROJECT / "benchmark", experiment_dir=cls.exp,
            max_rounds=20, score_target=None, train_case_ids=[f"c{i}" for i in range(10)],
            eval_case_ids=None, edit_memory=cls.layer)
        cls.rounds = ri.discover_rounds(cls.exp)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dashboard_tallies_equal_the_bandits(self) -> None:
        self.assertEqual(ri.arm_beta_tallies(self.rounds), self.layer.arm_tallies(self.manager._tree))

    def test_round_fields(self) -> None:
        by_id = {r.node_id: r for r in self.rounds}
        for nid, node in self.manager._tree.nodes.items():
            r = by_id[nid]
            self.assertEqual(r.memory_arm, node.memory_arm)
            if nid:
                self.assertEqual(r.assignment["block"], self.manager._feedback[nid].strategy.block)
                self.assertEqual(r.agentic_session["end_reason"], "submitted")
        self.assertTrue(by_id[0].is_root)
        self.assertIsNone(by_id[0].assignment)

    def test_summary_and_config(self) -> None:
        summ = ri.arm_utility_summary(self.rounds)
        n_children = len(self.rounds) - 1
        self.assertEqual(sum(e["n_nodes"] for e in summ.values()), n_children)
        cfg = ri.load_config_snapshot(self.exp)
        self.assertEqual(ri.edit_memory_config(cfg), {"window_size": 2, "beta_prior": 1.0})
        self.assertIsNone(ri.edit_memory_config({}))

    def test_edit_memory_loader_and_prompts(self) -> None:
        em = ra.load_edit_memory(self.exp)
        self.assertIsNotNone(em)
        self.assertEqual(len(em.memory_versions), self.layer.memory_version)
        prompts = ra.load_verbose_prompts(self.rounds[1].round_dir)
        self.assertIn("## Selected block for this EXPAND", prompts["instruction"])
        self.assertIn("You are the self-improvement module", prompts["system"])


class HelperTests(unittest.TestCase):
    def test_state_arms_for_rounds_without_a_sidecar(self) -> None:
        r = ri.RoundInfo(round_dir=Path("round_004"), node_id=4,
                         feedback={"base_round": 0, "edit_errors": ["x"]})
        self.assertEqual(r.memory_arm, "none")
        ri.attach_state_arms([r], {"node_arms": {"4": ["without", 2]}})
        self.assertEqual((r.memory_arm, r.memory_version), ("without", 2))

    def test_prob_beta_greater(self) -> None:
        self.assertAlmostEqual(ri.prob_beta_greater(3, 3, 3, 3), 0.5, places=2)
        self.assertGreater(ri.prob_beta_greater(20, 2, 2, 20), 0.99)
        s = ri.beta_summary(2, 2)
        self.assertAlmostEqual(s["mean"], 0.5)
        self.assertLess(s["lo90"], 0.5)
        self.assertGreater(s["hi90"], 0.5)


if __name__ == "__main__":
    unittest.main()

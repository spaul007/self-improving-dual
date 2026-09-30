"""Assignment-aware memory prompts (block HGM).

When the window's nodes carry an assignment (the ``block`` in their
strategy.json), the memory curator tags every edit with the assignment it
was made under and says whether the node stayed within its own, and the
generator (and the instruction updater, which sees the generator's fixed
core) gets the assignment rule. Without assignments the texts are sep18's.

    PYTHONPATH=. python3 -m unittest tests.test_edit_memory_assignments
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.edit_memory import prompts as P


def _nodes(block=None, impl=None):
    out = []
    for nid in (9, 10):
        n = {"node_id": nid, "parent_id": 4, "memory_arm": "with", "memory_version": 2,
             "n_evals": 32, "mean_utility": 0.4, "edit_failed": False}
        if block:
            n.update(block=block, implementation_strategy=impl)
        out.append(n)
    return out


def _curation(nodes):
    return P.render_memory_curation_instruction(nodes=nodes, previous_memory_exists=True, addendum="",
                                                max_llm_calls=60, timeout_s=2400, max_attempts=2)


def _generation(nodes, **kw):
    return P.render_memory_generation_messages(previous_memory="B", curation="Z", addendum="",
                                               window_meta={"window_index": 3, "nodes": nodes},
                                               max_chars=30000, **kw)


class CuratorTests(unittest.TestCase):
    def test_edits_are_tagged_and_intent_covers_the_assignment(self) -> None:
        text = _curation(_nodes("verifiers"))
        self.assertIn("— files: <paths> — assignment: <block> — new | inherited from", text)
        self.assertIn(P.CURATION_ASSIGNMENT_NOTES, text)
        self.assertIn("verifiers", text)                     # the node table column

    def test_no_assignment_no_extra_text(self) -> None:
        text = _curation(_nodes())
        self.assertNotIn("assignment", text)
        self.assertIn("— files: <paths> — new | inherited from", text)


class GeneratorTests(unittest.TestCase):
    def test_rule_added_with_assignments(self) -> None:
        system, user = (m["content"] for m in _generation(_nodes("verifiers", "harness_heavy")))
        self.assertIn(P.ASSIGNMENT_RULE + " Stay within 30000 characters.", system)
        self.assertIn("block verifiers / harness_heavy", user)

    def test_rule_order_with_case_values(self) -> None:
        system = _generation(_nodes("mixed"), forbid_case_values=True)[0]["content"]
        self.assertIn(P.ASSIGNMENT_RULE + " " + P.CASE_VALUES_RULE + " Stay within 30000", system)

    def test_no_assignment_is_the_plain_core(self) -> None:
        system = _generation(_nodes())[0]["content"]
        self.assertEqual(system, P.EDIT_MEMORY_CORE_INSTRUCTION.format(max_chars=30000))

    def test_updater_sees_the_same_rules(self) -> None:
        user = P.render_instruction_update_messages(addendum="", q="Q", previous_q=[], max_chars=8000,
                                                    assignments=True, forbid_case_values=True)[1]["content"]
        self.assertIn(P.memory_core_instruction("N", assignments=True, forbid_case_values=True), user)
        plain = P.render_instruction_update_messages(addendum="", q="Q", previous_q=[], max_chars=8000)
        self.assertNotIn(P.ASSIGNMENT_RULE, plain[1]["content"])


class LayerTests(unittest.TestCase):
    """Through the real layer inside an HGM run: every node gets a block from
    the manager's block selection, so the curator and the generator get the
    assignment texts."""

    def test_block_hgm_run_uses_the_assignment_texts(self) -> None:
        from meta_agent.edit_memory.layer import EditMemoryLayer
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_edit_memory_layer import _ScriptedLLM, _memory_stub_editor
        from tests.test_hgm_smoke import _StubEvaluator

        tmp = Path(tempfile.mkdtemp(prefix="editmem_assign_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        seed = tmp / "seed"
        (seed / "mutable_tools").mkdir(parents=True)
        (seed / "workflow.py").write_text("def run_task(task):\n    return None\n")
        exp = tmp / "exp"
        exp.mkdir()
        (exp / "config.snapshot.yaml").write_text("")
        llm = _ScriptedLLM()
        layer = EditMemoryLayer(llm, repo_root=tmp / "repo", window_size=2, instruction_every=1,
                                arm_min_pulls=1, seed=3,
                                curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60})
        HGMManager(eval_budget=40, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7,
                   finalize_top_k=0, expand_eval_size=4, block_selection_strategy="adaptive").evolve(
            editor=_memory_stub_editor(), evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
            seed_dir=seed, benchmark_dir=tmp / "b", experiment_dir=exp, max_rounds=30,
            score_target=None, train_case_ids=[f"c{i}" for i in range(20)], eval_case_ids=None,
            edit_memory=layer)
        self.assertGreaterEqual(layer.memory_version, 2)
        curator_instr = [c["messages"][1]["content"] for c in llm.calls
                         if any(t["name"] == P.SUBMIT_CURATION_NAME for t in c.get("tools") or [])
                         and f"$WORK_DIR/{P.CURATION_FILE}" in c["messages"][1]["content"]]
        gen_systems = [c["messages"][0]["content"] for c in llm.calls
                       if not c.get("tools") and P.CORE_SENTINEL in c["messages"][0]["content"]]
        updates = [c["messages"][1]["content"] for c in llm.calls
                   if not c.get("tools") and "# Fixed core (context only, immutable)" in c["messages"][1]["content"]]
        self.assertTrue(curator_instr and gen_systems and updates)
        self.assertTrue(all("assignment: <block> —" in t for t in curator_instr))
        self.assertTrue(all(P.ASSIGNMENT_RULE in s for s in gen_systems))
        self.assertTrue(all(P.ASSIGNMENT_RULE in u for u in updates))


if __name__ == "__main__":
    unittest.main()

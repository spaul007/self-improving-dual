"""The editor's objective sentence: "score" (legacy, byte-identical prompts
for the `full` steering mode and the no-edit-memory control) vs "judge"
(selected automatically under belief-mode steering).

    PYTHONPATH=. python3 -m unittest tests.test_editor_objective
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent.agent_editor import OBJECTIVE_JUDGE, OBJECTIVE_SCORE, AgentEditor
from meta_agent.agent_editor_two_stage import TwoStageEditor


class _Stop(Exception):
    pass


def _system_prompt(editor: AgentEditor) -> str:
    captured: dict = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    editor.llm = fake_llm
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "round_001"
        (out_dir / "task_agent" / "mutable_tools").mkdir(parents=True)
        (out_dir / "task_agent" / "workflow.py").write_text(
            "def run_task(task):\n    return None\n")
        (out_dir / "task_agent" / "tool_wrapper.py").write_text("")
        (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
        try:
            editor._self_improve(out_dir=out_dir, feedback=None, context=None,
                                 prior_errors=[], attempt=1)
        except _Stop:
            pass
    return next(m["content"] for m in captured["messages"] if m["role"] == "system")


class TestObjectiveSentence(unittest.TestCase):
    def test_default_is_the_legacy_score_sentence(self):
        prompt = _system_prompt(AgentEditor(lambda **k: None, []))
        self.assertIn("graded. " + OBJECTIVE_SCORE + "\n", prompt)
        self.assertNotIn("judge named", prompt)

    def test_judge_objective_replaces_only_that_sentence(self):
        score = _system_prompt(AgentEditor(lambda **k: None, []))
        judge = _system_prompt(AgentEditor(lambda **k: None, [], objective="judge"))
        self.assertIn("graded. " + OBJECTIVE_JUDGE + "\n", judge)
        self.assertNotIn("most affect the score", judge)
        self.assertIn("The benchmark score is context, not the objective.", judge)
        # Everything else — the hard rules included — is byte-identical.
        self.assertEqual(score.replace(OBJECTIVE_SCORE, OBJECTIVE_JUDGE), judge)

    def test_invalid_objective_raises(self):
        with self.assertRaises(ValueError):
            AgentEditor(lambda **k: None, [], objective="vibes")

    def test_two_stage_forwards_the_objective(self):
        self.assertEqual(TwoStageEditor(lambda **k: None, [], objective="judge")
                         .objective, "judge")
        self.assertEqual(TwoStageEditor(lambda **k: None, []).objective, "score")


class TestConfigSelection(unittest.TestCase):
    """``editor_objective`` reads only ``cfg.edit_memory.config``; a
    namespace stands in for the full FrameworkConfig, plus one real YAML."""

    @staticmethod
    def _cfg(edit_memory):
        from types import SimpleNamespace
        return SimpleNamespace(edit_memory=(
            SimpleNamespace(config=edit_memory) if edit_memory is not None else None))

    def test_belief_steering_selects_judge(self):
        self.assertEqual(cfg_mod.editor_objective(
            self._cfg({"steering_mode": "belief"})), "judge")

    def test_full_mode_and_no_edit_memory_keep_score(self):
        self.assertEqual(cfg_mod.editor_objective(
            self._cfg({"steering_mode": "full"})), "score")
        self.assertEqual(cfg_mod.editor_objective(self._cfg({})), "score")
        self.assertEqual(cfg_mod.editor_objective(self._cfg(None)), "score")

    def test_real_belief_config_selects_judge(self):
        repo = Path(__file__).resolve().parents[1]
        cfg = cfg_mod.load(repo / "configs" / "hgm_travel_1000_qwen122b_gpt54_beliefs2stage.yaml")
        self.assertEqual(cfg_mod.editor_objective(cfg), "judge")


if __name__ == "__main__":
    unittest.main()

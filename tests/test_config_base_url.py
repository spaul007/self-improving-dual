"""Tests for ``base_url`` flowing from YAML through config / runtime_env
into the editor + manager constructors and into the task-agent
subprocess env.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_config_base_url
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from meta_agent import config as cfg_mod  # noqa: E402
from meta_agent import runtime_env  # noqa: E402


class LLMSpecBaseUrlTests(unittest.TestCase):
    def test_base_url_defaults_to_none(self) -> None:
        spec = cfg_mod.LLMSpec()
        self.assertIsNone(spec.base_url)

    def test_base_url_round_trips_on_LLMSpec(self) -> None:
        spec = cfg_mod.LLMSpec(base_url="http://vllm:8000/v1")
        self.assertEqual(spec.base_url, "http://vllm:8000/v1")

    def test_base_url_round_trips_on_TaskAgentSpec(self) -> None:
        spec = cfg_mod.TaskAgentSpec(
            model="local-model",
            reasoning_effort="medium",
            base_url="http://vllm:8000/v1",
        )
        self.assertEqual(spec.base_url, "http://vllm:8000/v1")
        self.assertEqual(spec.model, "local-model")
        self.assertEqual(spec.reasoning_effort, "medium")


class ApplyTaskAgentEnvBaseUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snap = {
            k: os.environ.get(k)
            for k in ("LLM_MODEL", "LLM_REASONING_EFFORT",
                      "LLM_BASE_URL", "OPENAI_API_KEY")
        }
        for k in ("LLM_MODEL", "LLM_REASONING_EFFORT",
                  "LLM_BASE_URL", "OPENAI_API_KEY"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_base_url_unset_leaves_env_alone(self) -> None:
        runtime_env.apply_task_agent_env(cfg_mod.TaskAgentSpec())
        self.assertNotIn("LLM_BASE_URL", os.environ)
        # We must NOT seed OPENAI_API_KEY when base_url is unset — that
        # would silently mask a missing-key bug for the OpenAI path.
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_base_url_set_exports_env(self) -> None:
        runtime_env.apply_task_agent_env(
            cfg_mod.TaskAgentSpec(base_url="http://vllm:8000/v1")
        )
        self.assertEqual(
            os.environ["LLM_BASE_URL"], "http://vllm:8000/v1"
        )
        # base_url is set → wrapper would have raised on missing key, so
        # we pre-fill EMPTY for the subprocess's benefit.
        self.assertEqual(os.environ["OPENAI_API_KEY"], "EMPTY")

    def test_existing_api_key_is_not_overwritten(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-real"
        runtime_env.apply_task_agent_env(
            cfg_mod.TaskAgentSpec(base_url="http://vllm:8000/v1")
        )
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-real")

    def test_model_and_reasoning_still_export(self) -> None:
        runtime_env.apply_task_agent_env(
            cfg_mod.TaskAgentSpec(
                model="local-7b",
                reasoning_effort="low",
                base_url="http://vllm:8000/v1",
            )
        )
        self.assertEqual(os.environ["LLM_MODEL"], "local-7b")
        self.assertEqual(os.environ["LLM_REASONING_EFFORT"], "low")
        self.assertEqual(os.environ["LLM_BASE_URL"], "http://vllm:8000/v1")


class EditorAndManagerBaseUrlTests(unittest.TestCase):
    """The editor + manager must accept ``base_url`` / ``strategy_base_url``
    via constructor and pass it through to the LLM call."""

    def test_editor_constructor_accepts_base_url(self) -> None:
        from meta_agent.agent_editor import AgentEditor

        captured: dict = {}

        def fake_llm(**kwargs):
            captured["base_url"] = kwargs.get("base_url")
            # Don't actually invoke the tool path — just record the call.
            raise RuntimeError("fake_llm intentionally stops here")

        editor = AgentEditor(
            llm_caller=fake_llm,
            validators=[],
            base_url="http://editor-local:8000/v1",
        )
        self.assertEqual(editor.base_url, "http://editor-local:8000/v1")

    def test_manager_constructor_accepts_strategy_base_url(self) -> None:
        from meta_agent.managers.hill_climbing import HillClimbingManager

        mgr = HillClimbingManager(
            strategy_base_url="http://mgr-local:8000/v1"
        )
        self.assertEqual(
            mgr.strategy_base_url, "http://mgr-local:8000/v1"
        )

    def test_editor_threads_base_url_into_llm_kwargs(self) -> None:
        """End-to-end: editor passes base_url= through to the llm callable
        when its constructor was given one."""
        from meta_agent.agent_editor import AgentEditor

        captured: dict = {}

        class _Stop(Exception):
            pass

        def fake_llm(**kwargs):
            captured.update(kwargs)
            raise _Stop()

        editor = AgentEditor(
            llm_caller=fake_llm,
            validators=[],
            base_url="http://editor-local:8000/v1",
        )

        # Call the private proposer directly with minimal scaffolding.
        # We don't care about the result — only that base_url was forwarded.
        import tempfile
        from meta_agent.models import EvolutionStrategy

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "round_001"
            (out_dir / "task_agent" / "mutable_tools").mkdir(parents=True)
            (out_dir / "task_agent" / "workflow.py").write_text(
                "def run_task(task):\n    return None\n"
            )
            (out_dir / "task_agent" / "tool_wrapper.py").write_text("")
            (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
            strategy = EvolutionStrategy(
                target_files=["workflow.py"],
                optimization_goal="test",
                proposed_changes="test",
            )
            with self.assertRaises(_Stop):
                editor._propose_edits(
                    out_dir=out_dir,
                    strategy=strategy,
                    feedback=None,
                    prior_errors=[],
                    attempt=1,
                )

        self.assertEqual(
            captured.get("base_url"), "http://editor-local:8000/v1"
        )


if __name__ == "__main__":
    unittest.main()

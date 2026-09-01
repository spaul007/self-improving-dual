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
    """The editor must accept ``base_url`` via constructor and thread it
    through to the LLM call (the single self-improvement step)."""

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

        # Call the single self-improvement step directly with minimal
        # scaffolding. We don't care about the result — only that
        # base_url was forwarded to the llm callable.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "round_001"
            (out_dir / "task_agent" / "mutable_tools").mkdir(parents=True)
            (out_dir / "task_agent" / "workflow.py").write_text(
                "def run_task(task):\n    return None\n"
            )
            (out_dir / "task_agent" / "tool_wrapper.py").write_text("")
            (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
            with self.assertRaises(_Stop):
                editor._self_improve(
                    out_dir=out_dir,
                    feedback=None,
                    context=None,
                    prior_errors=[],
                    attempt=1,
                )

        self.assertEqual(
            captured.get("base_url"), "http://editor-local:8000/v1"
        )


class TaskAgentTemperatureTests(unittest.TestCase):
    """Task-agent-only greedy temperature: config default, env isolation,
    evaluator child-env plumbing, and the injection path used by
    ``build_components``."""

    def setUp(self) -> None:
        self._snap = {
            k: os.environ.get(k)
            for k in ("LLM_MODEL", "LLM_REASONING_EFFORT", "LLM_BASE_URL",
                      "LLM_TEMPERATURE", "OPENAI_API_KEY")
        }
        for k in self._snap:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ----- config schema -------------------------------------------------

    def test_task_agent_spec_defaults_to_low_variance(self) -> None:
        self.assertEqual(cfg_mod.TaskAgentSpec().temperature, 0.2)

    def test_task_agent_spec_null_round_trips(self) -> None:
        self.assertIsNone(
            cfg_mod.TaskAgentSpec(temperature=None).temperature
        )
        # YAML `temperature: null` shape through pydantic validation.
        spec = cfg_mod.TaskAgentSpec.model_validate({"temperature": None})
        self.assertIsNone(spec.temperature)

    def test_llm_spec_has_no_temperature_field(self) -> None:
        # Guard: editor/summarizer/edit_memory specs must not silently
        # grow a config-driven temperature.
        self.assertNotIn("temperature", cfg_mod.LLMSpec.model_fields)

    # ----- parent-process env isolation ----------------------------------

    def test_apply_task_agent_env_never_exports_temperature(self) -> None:
        runtime_env.apply_task_agent_env(
            cfg_mod.TaskAgentSpec(
                model="local-7b",
                reasoning_effort="medium",
                base_url="http://vllm:8000/v1",
            )
        )
        self.assertNotIn("LLM_TEMPERATURE", os.environ)

    # ----- evaluator child env -------------------------------------------

    def test_child_env_carries_temperature_without_touching_os_environ(
        self,
    ) -> None:
        import tempfile

        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator(task_agent_temperature=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            env = ev._child_env(Path(tmp) / "trace.jsonl")
        self.assertEqual(env.get("LLM_TEMPERATURE"), "0.0")
        self.assertNotIn("LLM_TEMPERATURE", os.environ)

    def test_child_env_omits_temperature_when_none(self) -> None:
        import tempfile

        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator()  # constructor default: None
        with tempfile.TemporaryDirectory() as tmp:
            env = ev._child_env(Path(tmp) / "trace.jsonl")
        self.assertNotIn("LLM_TEMPERATURE", env)

    # ----- build_components injection path -------------------------------

    def test_injection_threads_temperature_into_evaluator(self) -> None:
        # Mirrors the exact call build_components makes (signature-filtered
        # injection with setdefault semantics).
        cfg_mod._ensure_builtins_loaded()
        spec = cfg_mod.ComponentSpec(type="subprocess", config={})
        ev = cfg_mod._build_with_injection(
            spec, "evaluator",
            {"scorer": None, "task_agent_temperature": 0.0},
        )
        self.assertEqual(ev.task_agent_temperature, 0.0)

        ev_null = cfg_mod._build_with_injection(
            spec, "evaluator",
            {"scorer": None, "task_agent_temperature": None},
        )
        self.assertIsNone(ev_null.task_agent_temperature)

    def test_yaml_evaluator_config_wins_over_injection(self) -> None:
        cfg_mod._ensure_builtins_loaded()
        spec = cfg_mod.ComponentSpec(
            type="subprocess", config={"task_agent_temperature": 0.6}
        )
        ev = cfg_mod._build_with_injection(
            spec, "evaluator",
            {"scorer": None, "task_agent_temperature": 0.0},
        )
        self.assertEqual(ev.task_agent_temperature, 0.6)

    # ----- meta-agent invariant ------------------------------------------

    def test_editor_kwargs_unchanged_by_global_effort(self) -> None:
        # With a global reasoning effort exported (the task-agent setting)
        # and no editor-level effort, the editor still passes its explicit
        # temperature=0.2 — which call_llm drops in the reasoning branch
        # (covered by the wrapper tests). No env-sourced temperature can
        # reach it because LLM_TEMPERATURE is never in the parent process.
        from meta_agent.agent_editor import AgentEditor

        os.environ["LLM_REASONING_EFFORT"] = "medium"

        captured: dict = {}

        class _Stop(Exception):
            pass

        def fake_llm(**kwargs):
            captured.update(kwargs)
            raise _Stop()

        editor = AgentEditor(llm_caller=fake_llm, validators=[])

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "round_001"
            (out_dir / "task_agent" / "mutable_tools").mkdir(parents=True)
            (out_dir / "task_agent" / "workflow.py").write_text(
                "def run_task(task):\n    return None\n"
            )
            (out_dir / "task_agent" / "tool_wrapper.py").write_text("")
            (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
            with self.assertRaises(_Stop):
                editor._self_improve(
                    out_dir=out_dir,
                    feedback=None,
                    context=None,
                    prior_errors=[],
                    attempt=1,
                )

        self.assertEqual(captured.get("temperature"), 0.2)
        self.assertNotIn("LLM_TEMPERATURE", os.environ)


if __name__ == "__main__":
    unittest.main()

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

    # ----- task_agent.max_output_tokens (same child-env-only plumbing) ----

    def test_max_output_tokens_spec_default_none(self) -> None:
        spec = cfg_mod.TaskAgentSpec()
        self.assertIsNone(spec.max_output_tokens)
        self.assertEqual(
            cfg_mod.TaskAgentSpec(max_output_tokens=65536).max_output_tokens, 65536
        )

    def test_child_env_carries_max_output_tokens_without_touching_os_environ(
        self,
    ) -> None:
        import tempfile

        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator(task_agent_max_output_tokens=65536)
        with tempfile.TemporaryDirectory() as tmp:
            env = ev._child_env(Path(tmp) / "trace.jsonl")
        self.assertEqual(env.get("LLM_MAX_OUTPUT_TOKENS"), "65536")
        self.assertNotIn("LLM_MAX_OUTPUT_TOKENS", os.environ)

        ev_none = SubprocessEvaluator()
        with tempfile.TemporaryDirectory() as tmp:
            env = ev_none._child_env(Path(tmp) / "trace.jsonl")
        self.assertNotIn("LLM_MAX_OUTPUT_TOKENS", env)

    def test_injection_threads_max_output_tokens_into_evaluator(self) -> None:
        cfg_mod._ensure_builtins_loaded()
        spec = cfg_mod.ComponentSpec(type="subprocess", config={})
        ev = cfg_mod._build_with_injection(
            spec, "evaluator",
            {"scorer": None, "task_agent_max_output_tokens": 65536},
        )
        self.assertEqual(ev.task_agent_max_output_tokens, 65536)
        ev_null = cfg_mod._build_with_injection(
            spec, "evaluator",
            {"scorer": None, "task_agent_max_output_tokens": None},
        )
        self.assertIsNone(ev_null.task_agent_max_output_tokens)

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


class MetaBaseUrlWarningTests(unittest.TestCase):
    """A meta component naming a model but no base_url inherits the task
    agent's LLM_BASE_URL (runtime_env exports it) — warn at build time."""

    def _cfg(self, **over):
        base = dict(
            experiment_name="t", project="math",
            task_agent={"model": "local", "reasoning_effort": "low",
                        "base_url": "http://vllm:8000/v1"},
            editor={"type": "default", "config": {"model": "gpt-5.4",
                                                  "reasoning_effort": "medium"}},
            manager={"type": "hill_climbing", "config": {}},
            evaluator={"type": "subprocess", "config": {}},
            gatherer={"type": "default", "config": {}},
            validators=[], loop={"max_rounds": 1})
        base.update(over)
        return cfg_mod.FrameworkConfig(**base)

    def test_warns_for_meta_component_without_base_url(self) -> None:
        got = cfg_mod.meta_base_url_warnings(self._cfg())
        self.assertEqual(len(got), 1)
        self.assertIn("editor names model 'gpt-5.4' but no base_url", got[0])
        self.assertIn("http://vllm:8000/v1", got[0])

    def test_silent_when_base_url_set_or_no_task_base_url(self) -> None:
        pinned = self._cfg(editor={"type": "default", "config": {
            "model": "gpt-5.4", "base_url": "https://api.openai.com/v1"}})
        self.assertEqual(cfg_mod.meta_base_url_warnings(pinned), [])
        no_task = self._cfg(task_agent={"model": "gpt-5.4-mini",
                                        "reasoning_effort": "low"})
        self.assertEqual(cfg_mod.meta_base_url_warnings(no_task), [])
        # no model on the component: it inherits LLM_MODEL too -> consistent
        inherit = self._cfg(editor={"type": "default", "config": {}})
        self.assertEqual(cfg_mod.meta_base_url_warnings(inherit), [])

    def test_edit_memory_and_summarizer_are_covered(self) -> None:
        got = cfg_mod.meta_base_url_warnings(self._cfg(
            edit_memory={"type": "default", "config": {"model": "gpt-5.4"}},
            summarizer={"type": "default", "config": {"model": "gpt-5.4"}}))
        self.assertEqual(sorted(w.split()[2] for w in got),
                         ["edit_memory", "editor", "summarizer"])


class TaskAgentStallGuardTests(unittest.TestCase):
    """`task_agent.timeout_s` / `.max_output_tokens` exist to stop one stalled
    or runaway generation from costing a case its whole score. Both must stay
    task-agent-only: a global export would cap the meta agents' own calls."""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in
                       ("LLM_TIMEOUT_S", "LLM_MAX_OUTPUT_TOKENS")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults_are_unset(self) -> None:
        spec = cfg_mod.TaskAgentSpec()
        self.assertIsNone(spec.timeout_s)
        self.assertIsNone(spec.max_output_tokens)

    def test_apply_task_agent_env_never_exports_them(self) -> None:
        runtime_env.apply_task_agent_env(cfg_mod.TaskAgentSpec(
            model="m", timeout_s=600, max_output_tokens=32768))
        self.assertNotIn("LLM_TIMEOUT_S", os.environ)
        self.assertNotIn("LLM_MAX_OUTPUT_TOKENS", os.environ)

    def test_child_env_carries_them_and_os_environ_does_not(self) -> None:
        from meta_agent.evaluator import SubprocessEvaluator
        ev = SubprocessEvaluator(wall_time_s_per_case=1800,
                                 task_agent_timeout_s=600,
                                 task_agent_max_output_tokens=32768)
        env = ev._child_env(Path("/tmp/trace.jsonl"))
        self.assertEqual(env["LLM_TIMEOUT_S"], "600.0")
        self.assertEqual(env["LLM_MAX_OUTPUT_TOKENS"], "32768")
        self.assertNotIn("LLM_TIMEOUT_S", os.environ)
        self.assertNotIn("LLM_MAX_OUTPUT_TOKENS", os.environ)

    def test_unset_child_env_is_unchanged(self) -> None:
        from meta_agent.evaluator import SubprocessEvaluator
        env = SubprocessEvaluator()._child_env(Path("/tmp/trace.jsonl"))
        self.assertNotIn("LLM_TIMEOUT_S", env)
        self.assertNotIn("LLM_MAX_OUTPUT_TOKENS", env)

    def test_timeout_at_or_above_the_case_limit_warns(self) -> None:
        import contextlib
        import io
        from meta_agent.evaluator import SubprocessEvaluator
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SubprocessEvaluator(wall_time_s_per_case=600, task_agent_timeout_s=600)
        self.assertIn("not below wall_time_s_per_case", buf.getvalue())
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            SubprocessEvaluator(wall_time_s_per_case=1800, task_agent_timeout_s=600)
        self.assertEqual(buf2.getvalue(), "")

    def test_wrapper_resolves_the_output_cap_from_env(self) -> None:
        from platform_core.llm_wrapper import _env_default_max_output_tokens as f
        self.assertIsNone(f())                       # unset = uncapped, as before
        for raw, want in (("32768", 32768), ("0", None), ("-5", None),
                          ("junk", None), ("", None), ("1024.0", 1024)):
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = raw
            self.assertEqual(f(), want, raw)

    def test_belief_configs_guard_against_stalls(self) -> None:
        for name in ("hgm_travel_1000_qwen122b_gpt54_beliefs2stage",
                     "hgm_travel_1000_qwen122b_dsv4pro_beliefs2stage",
                     "hgm_travel_1000_qwen122b_node5_editmem_beliefs2stage",
                     "hgm_travel_1000_local_qwen122b_medium_beliefs2stage",
                     "hgm_travel_smoke_beliefs2stage"):
            cfg = cfg_mod.load(Path("configs") / f"{name}.yaml")
            wall = float(cfg.evaluator.config["wall_time_s_per_case"])
            self.assertIsNotNone(cfg.task_agent.timeout_s, name)
            self.assertLess(cfg.task_agent.timeout_s, wall, name)
            self.assertGreaterEqual(cfg.task_agent.max_output_tokens, 8192, name)

"""Tests for TaskAgentSpec.provider / runtime_env.apply_task_agent_env's
LLM_PROVIDER_PREFERENCE export -- the OpenRouter provider-routing
preference (order/ignore/quantizations/allow_fallbacks) that steers
task-agent calls away from a specific provider or quantization level.
Confirmed live this session against the real OpenRouter API (a test call
with ignore=["DeepInfra"] was genuinely served by Venice instead) and at
production scale (0 terminal failures across 3511 real calls) -- see
openrouter_failure_report.md. platform_core/llm_wrapper.py::call_llm's
own retry/provider logic has its own dedicated tests
(test_llm_wrapper_status_failed_retry.py); this file covers only the
config -> env var plumbing.

    PYTHONPATH=. python3 -m unittest tests.test_runtime_env_provider
"""
from __future__ import annotations

import os
import unittest

from meta_agent import config as cfg_mod
from meta_agent import runtime_env


# apply_task_agent_env can set any of these -- snapshot/restore all of
# them around every test in this file, not just LLM_PROVIDER_PREFERENCE,
# so a test here can never leak an env var (e.g. LLM_BASE_URL pointed at
# OpenRouter) into an unrelated test elsewhere in the suite. Confirmed
# live this was a real bug: an earlier version of this file only
# restored LLM_PROVIDER_PREFERENCE, leaving LLM_BASE_URL set to
# OpenRouter's URL after RealGemmaConfigWiringTests ran, which made some
# later, unrelated test's call_llm() actually hit the real OpenRouter
# endpoint with no API key -- a real 401 AuthenticationError, retried 30
# times (~4 minutes) before failing.
_ENV_VARS_TOUCHED = (
    "LLM_MODEL", "LLM_REASONING_EFFORT", "LLM_TEMPERATURE",
    "LLM_MAX_OUTPUT_TOKENS", "LLM_BASE_URL", "LLM_PROVIDER_PREFERENCE",
    "OPENAI_API_KEY",
)


def _snapshot_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_VARS_TOUCHED}


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TaskAgentProviderEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = _snapshot_env()

    def tearDown(self) -> None:
        _restore_env(self._snapshot)

    def test_none_provider_leaves_env_var_unset(self) -> None:
        spec = cfg_mod.TaskAgentSpec()
        self.assertIsNone(spec.provider)
        runtime_env.apply_task_agent_env(spec)
        self.assertNotIn("LLM_PROVIDER_PREFERENCE", os.environ)

    def test_provider_dict_exported_as_json(self) -> None:
        spec = cfg_mod.TaskAgentSpec(
            provider={"ignore": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks": False}
        )
        runtime_env.apply_task_agent_env(spec)
        raw = os.environ.get("LLM_PROVIDER_PREFERENCE")
        self.assertIsNotNone(raw)
        import json
        self.assertEqual(
            json.loads(raw),
            {"ignore": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks": False},
        )

    def test_empty_dict_provider_does_not_export(self) -> None:
        # {} is falsy -- same "nothing to opt into" convention as every
        # other optional field here (model, base_url, etc. use truthiness
        # checks, not `is not None`).
        spec = cfg_mod.TaskAgentSpec(provider={})
        runtime_env.apply_task_agent_env(spec)
        self.assertNotIn("LLM_PROVIDER_PREFERENCE", os.environ)


class RealGemmaConfigWiringTests(unittest.TestCase):
    """End-to-end: loading the actual production Gemma configs and
    running apply_all really does set LLM_PROVIDER_PREFERENCE -- not
    just a synthetic TaskAgentSpec. Restores every env var
    apply_task_agent_env can touch (see _ENV_VARS_TOUCHED above) --
    these configs set LLM_BASE_URL to the real OpenRouter endpoint,
    which must never leak into a later, unrelated test."""

    def setUp(self) -> None:
        self._snapshot = _snapshot_env()

    def tearDown(self) -> None:
        _restore_env(self._snapshot)

    def _check_config(self, path: str) -> None:
        cfg = cfg_mod.load(path)
        # No quantizations filter -- dropped 2026-09-14 after discovering
        # live that it 404s every llm_backbone_selection catalog model
        # ("No endpoints found for the request with quantization: bf16"),
        # re-validated for reliability without it (0 terminal failures /
        # 1075 real calls, deepinfra_only_3x_32/) before this change.
        self.assertEqual(
            cfg.task_agent.provider,
            {"ignore": ["DeepInfra"], "allow_fallbacks": False},
        )
        runtime_env.apply_task_agent_env(cfg.task_agent)
        import json
        self.assertEqual(
            json.loads(os.environ["LLM_PROVIDER_PREFERENCE"]),
            {"ignore": ["DeepInfra"], "allow_fallbacks": False},
        )

    def test_backbone_selection_on_config(self) -> None:
        self._check_config(
            "configs/hgm_travel_gemma_full_scale_block_tagged_X100Y180.yaml"
        )

    def test_backbone_selection_off_config(self) -> None:
        self._check_config(
            "configs/hgm_travel_gemma_no_backbone_selection_X100Y180.yaml"
        )


if __name__ == "__main__":
    unittest.main()

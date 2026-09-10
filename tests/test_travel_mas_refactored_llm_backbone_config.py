"""Tests for the per-agent backbone LLM config loader,
projects/travel_mas_refactored/seed/agents/llm_backbone.py.

Real gap this closes: get_backbone_config() is new code with no direct
test coverage -- its merge semantics (per-agent field overrides default,
missing agent falls back to default, all-null round-trips to "use the
LLM_* env var defaults") are exactly what makes the seed refactor a
provable no-op today and a real lever once an llm_backbone_selection
EXPAND (see meta_agent/block_suggester.py's _BLOCK_BODIES entry) writes
real values into mas_llm_backbone.yaml.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_llm_backbone_config
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "projects" / "travel_mas_refactored" / "seed"

_ENV_VAR = "MAS_LLM_BACKBONE_CFG"
_ALL_FIELDS = {"model", "base_url", "temperature", "max_output_tokens", "reasoning_effort"}


class LlmBackboneConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(SEED_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SEED_DIR)))
        sys.modules.pop("agents.llm_backbone", None)
        sys.modules.pop("agents", None)
        self.addCleanup(lambda: sys.modules.pop("agents.llm_backbone", None))
        self.addCleanup(lambda: sys.modules.pop("agents", None))

        self._prior_env = os.environ.get(_ENV_VAR)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self._prior_env is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = self._prior_env

    def _write_cfg(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        os.environ[_ENV_VAR] = str(path)
        return path

    def _import_fresh(self):
        from agents import llm_backbone
        llm_backbone._load.cache_clear()
        return llm_backbone

    def test_seed_shipped_config_is_all_null(self) -> None:
        # No env override -- exercises the real shipped
        # projects/travel_mas_refactored/seed/mas_llm_backbone.yaml.
        os.environ.pop(_ENV_VAR, None)
        llm_backbone = self._import_fresh()
        for agent in ("flight", "train", "sightseeing", "accounting"):
            cfg = llm_backbone.get_backbone_config(agent)
            self.assertEqual(set(cfg), _ALL_FIELDS)
            self.assertTrue(all(v is None for v in cfg.values()), (agent, cfg))

    def test_missing_config_file_returns_all_null(self) -> None:
        os.environ[_ENV_VAR] = "/nonexistent/path/mas_llm_backbone.yaml"
        llm_backbone = self._import_fresh()
        cfg = llm_backbone.get_backbone_config("flight")
        self.assertTrue(all(v is None for v in cfg.values()))

    def test_per_agent_field_overrides_default_field(self) -> None:
        self._write_cfg(
            "default:\n"
            "  model: default-model\n"
            "  base_url: http://default\n"
            "  temperature: 0.5\n"
            "  max_output_tokens: 1000\n"
            "  reasoning_effort: null\n"
            "agents:\n"
            "  sightseeing:\n"
            "    model: sightseeing-model\n"
            "    base_url: http://sightseeing\n"
        )
        llm_backbone = self._import_fresh()
        cfg = llm_backbone.get_backbone_config("sightseeing")
        self.assertEqual(cfg["model"], "sightseeing-model")
        self.assertEqual(cfg["base_url"], "http://sightseeing")
        # Fields not overridden fall back to default.
        self.assertEqual(cfg["temperature"], 0.5)
        self.assertEqual(cfg["max_output_tokens"], 1000)
        self.assertIsNone(cfg["reasoning_effort"])

    def test_empty_agent_section_inherits_default_entirely(self) -> None:
        self._write_cfg(
            "default:\n"
            "  model: default-model\n"
            "  base_url: http://default\n"
            "  temperature: 0.2\n"
            "  max_output_tokens: null\n"
            "  reasoning_effort: null\n"
            "agents:\n"
            "  flight: {}\n"
        )
        llm_backbone = self._import_fresh()
        cfg = llm_backbone.get_backbone_config("flight")
        self.assertEqual(cfg["model"], "default-model")
        self.assertEqual(cfg["base_url"], "http://default")
        self.assertEqual(cfg["temperature"], 0.2)

    def test_unlisted_agent_name_falls_back_to_default(self) -> None:
        self._write_cfg(
            "default:\n"
            "  model: default-model\n"
            "  base_url: null\n"
            "  temperature: null\n"
            "  max_output_tokens: null\n"
            "  reasoning_effort: null\n"
            "agents: {}\n"
        )
        llm_backbone = self._import_fresh()
        cfg = llm_backbone.get_backbone_config("some_agent_not_in_yaml")
        self.assertEqual(cfg["model"], "default-model")

    def test_missing_default_and_agents_sections_returns_all_null(self) -> None:
        self._write_cfg("{}\n")
        llm_backbone = self._import_fresh()
        cfg = llm_backbone.get_backbone_config("accounting")
        self.assertTrue(all(v is None for v in cfg.values()))

if __name__ == "__main__":
    unittest.main()

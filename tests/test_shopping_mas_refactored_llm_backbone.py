"""Tests for projects/shopping_mas_refactored/shopping_mas/agents/llm_backbone.py
and its wiring into agents/base.py::call_agent -- the per-agent backbone
LLM override mechanism mirrored from travel_mas_refactored's own
agents/llm_backbone.py/mas_llm_backbone.yaml, added here on request so
shopping_mas_refactored has the same mechanism (e.g. to run one agent on
a different model/endpoint, or force thinking off for a nothink
baseline run -- see configs/eval_local_shopping_mas_refactored_qwen27b_nothink.yaml).

Real gap this closes: the shipped mas_llm_backbone.yaml has every field
null (zero behavior change by default), so a wiring bug here would be
silent -- every call would just keep working off the single global
MASConfig.server/temperature/max_tokens, and nobody would notice the
override path was broken until the day someone actually tries to use
it. These tests exercise both ends: get_backbone_config's merge
semantics, and call_agent actually forwarding (or not forwarding, when
unset) the resolved values into llm.chat_json.

    PYTHONPATH=. python3 -m unittest tests.test_shopping_mas_refactored_llm_backbone
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEED_DIR = REPO_ROOT / "projects" / "shopping_mas_refactored" / "shopping_mas"

_SEED_MODULE_NAMES = [
    "agents", "agents.base", "agents.llm_backbone",
    "config", "llm_client", "mas_prompt_cfg",
]


class LlmBackboneTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in _SEED_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SEED_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SEED_DIR)))
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in _SEED_MODULE_NAMES])
        self.addCleanup(lambda: os.environ.pop("MAS_LLM_BACKBONE_CFG", None))

    def _write_yaml(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, text.encode())
        os.close(fd)
        self.addCleanup(lambda: os.unlink(path))
        return path

    def test_shipped_default_yaml_resolves_every_field_to_none(self) -> None:
        from agents.llm_backbone import get_backbone_config

        backbone = get_backbone_config("cart_executor")
        self.assertEqual(
            backbone,
            {"model": None, "base_url": None, "temperature": None,
             "max_tokens": None, "enable_thinking": None},
        )

    def test_missing_config_file_resolves_to_all_none_not_an_error(self) -> None:
        os.environ["MAS_LLM_BACKBONE_CFG"] = "/nonexistent/path/does-not-exist.yaml"
        from agents.llm_backbone import get_backbone_config

        self.assertEqual(get_backbone_config("cart_executor")["model"], None)

    def test_per_agent_overrides_default_field_by_field(self) -> None:
        os.environ["MAS_LLM_BACKBONE_CFG"] = self._write_yaml("""
default:
  model: "default-model"
  base_url: null
  temperature: 0.2
  max_tokens: null
  enable_thinking: null
agents:
  cart_optimizer:
    model: "optimizer-only-model"
    temperature: 0.0
""")
        from agents.llm_backbone import get_backbone_config

        optimizer = get_backbone_config("cart_optimizer")
        self.assertEqual(optimizer["model"], "optimizer-only-model")  # per-agent wins
        self.assertEqual(optimizer["temperature"], 0.0)               # per-agent wins
        self.assertIsNone(optimizer["base_url"])                      # falls through to default (null)

        scout = get_backbone_config("product_scout")                  # no per-agent section at all
        self.assertEqual(scout["model"], "default-model")             # falls through to default
        self.assertEqual(scout["temperature"], 0.2)

    def test_call_agent_requires_agent_name_and_forwards_resolved_backbone(self) -> None:
        from agents import base

        base.build_system_prompt = lambda *a, **k: "SYSTEM"
        fake_llm = MagicMock()
        fake_llm.chat_json.return_value = {"ok": True}

        with self.assertRaises(TypeError):
            base.call_agent(fake_llm, None, 1, "hi")  # type: ignore[call-arg]

        base.call_agent(fake_llm, None, 1, "hi", agent_name="cart_executor", thinking=False)
        _, kwargs = fake_llm.chat_json.call_args
        self.assertIsNone(kwargs["model"])
        self.assertIsNone(kwargs["base_url"])
        self.assertIsNone(kwargs["temperature"])
        self.assertIsNone(kwargs["max_tokens"])
        self.assertFalse(kwargs["thinking"])  # unset backbone leaves the call site's own thinking alone

    def test_call_agent_backbone_enable_thinking_overrides_call_sites_own_default(self) -> None:
        from agents import base

        os.environ["MAS_LLM_BACKBONE_CFG"] = self._write_yaml("""
default: {}
agents:
  product_scout:
    enable_thinking: false
""")
        base.build_system_prompt = lambda *a, **k: "SYSTEM"
        fake_llm = MagicMock()
        fake_llm.chat_json.return_value = {"ok": True}

        # product_scout's own call sites pass thinking=True (the module-level
        # default) -- the backbone's explicit False must win.
        base.call_agent(fake_llm, None, 1, "hi", agent_name="product_scout", thinking=True)
        _, kwargs = fake_llm.chat_json.call_args
        self.assertFalse(kwargs["thinking"])


if __name__ == "__main__":
    unittest.main()

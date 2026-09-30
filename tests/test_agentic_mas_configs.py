"""The agentic block-HGM configs and the config-level wiring of the agentic
editor + edit memory.

    PYTHONPATH=. python3 -m unittest tests.test_agentic_mas_configs
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from meta_agent import config as C

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NO_MEM = CONFIGS / "hgm_travel_mas_agentic_no_editmem_X100Y180.yaml"
MEM = CONFIGS / "hgm_travel_mas_agentic_editmem_X100Y180.yaml"
SANITY = CONFIGS / "hgm_travel_mas_agentic_editmem_sanity.yaml"


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class PairTests(unittest.TestCase):
    def test_pair_differs_only_in_edit_memory_and_name(self) -> None:
        a, b = _raw(NO_MEM), _raw(MEM)
        self.assertNotIn("edit_memory", a)
        self.assertIn("edit_memory", b)
        for d in (a, b):
            d.pop("edit_memory", None)
            d.pop("experiment_name")
        self.assertEqual(a, b)

    def test_decided_settings(self) -> None:
        for path in (NO_MEM, MEM, SANITY):
            with self.subTest(config=path.name):
                raw = _raw(path)
                m = raw["manager"]["config"]
                self.assertEqual(m["expand_eval_size"], m["eval_batch_size"])
                self.assertFalse(m["curriculum_enabled"])
                self.assertIsNone(m["implementation_strategy_selection_strategy"])
                self.assertNotIn("llm_backbone_selection", m["active_blocks"])
                self.assertEqual(len(m["active_blocks"]), 5)
                self.assertNotIn("block_suggester", raw)
                self.assertEqual(raw["editor"]["type"], "agentic")
                self.assertEqual(raw["editor"]["config"]["steering"], "assignment")
                self.assertEqual(raw["editor"]["config"]["model"], "deepseek/deepseek-v4-pro-0813")
                self.assertEqual(raw["task_agent"]["model"], "Qwen/Qwen3.8-27B")
                self.assertEqual(raw["seed_dir_name"], "seed_qwen27b_nothink")
                self.assertEqual(raw["env"]["TRAVEL_CONVERT_MODEL"], "Qwen/Qwen3.8-27B")
                self.assertEqual(raw["env"]["TRAVEL_CONVERT_BASE_URL"], raw["task_agent"]["base_url"])
                self.assertEqual(raw["env"]["TRAVEL_CONVERT_ENABLE_THINKING"], "false")
                names = [v["type"] for v in raw["validators"]]
                self.assertIn("hardcoded_answers", names)
                self.assertEqual(names[-1], "smoke_test")

    def test_meta_side_calls_are_deepseek_medium_and_no_behavior_summarizer(self) -> None:
        model = "deepseek/deepseek-v4-pro-0813"
        for path in (NO_MEM, MEM, SANITY):
            with self.subTest(config=path.name):
                raw = _raw(path)
                self.assertNotIn("summarizer", raw)
                sections = [raw["editor"]["config"], raw["failure_summarizer"]["config"]]
                if "edit_memory" in raw:
                    sections.append(raw["edit_memory"]["config"])
                for conf in sections:
                    self.assertEqual((conf["model"], conf["reasoning_effort"]), (model, "medium"))
                g = raw["gatherer"]["config"]
                self.assertEqual((g["error_bucket_model"], g["error_bucket_reasoning_effort"]),
                                 (model, "medium"))
                # Every OpenRouter call names the OpenRouter key and the editor's provider pin.
                pin = raw["editor"]["config"]["extra_body"]
                for conf in sections:
                    self.assertEqual((conf["api_key_env"], conf["extra_body"]),
                                     ("OpenRouter_API_KEY", pin))
                self.assertEqual((g["error_bucket_api_key_env"], g["error_bucket_extra_body"]),
                                 ("OpenRouter_API_KEY", pin))
                fw = C.build_components(C.load(path))
                self.assertIsNone(fw.summarizer)
                self.assertEqual((fw.failure_summarizer.model, fw.failure_summarizer.reasoning_effort),
                                 (model, "medium"))
                self.assertEqual((fw.gatherer.error_bucket_model, fw.gatherer.error_bucket_reasoning_effort),
                                 (model, "medium"))
                self.assertEqual(fw.gatherer.error_bucket_api_key_env, "OpenRouter_API_KEY")
                self.assertEqual(fw.failure_summarizer.api_key_env, "OpenRouter_API_KEY")

    def test_all_build(self) -> None:
        for path in (NO_MEM, MEM, SANITY):
            with self.subTest(config=path.name):
                fw = C.build_components(C.load(path))
                self.assertEqual(type(fw.editor).__name__, "AgenticEditor")
                self.assertTrue(fw.editor.hardcode_rule)
                self.assertEqual(fw.editor.validate_skipped, ("smoke_test",))
                self.assertIsNone(fw.block_suggester)
                self.assertEqual(fw.edit_memory is not None, path != NO_MEM)
                if fw.edit_memory is not None:
                    self.assertEqual(fw.edit_memory.mutable_exclude, fw.config.mutable_exclude)
                    self.assertTrue(fw.edit_memory.forbid_case_values)


class ConfigWiringTests(unittest.TestCase):
    def cfg(self, mutate) -> C.FrameworkConfig:
        raw = copy.deepcopy(_raw(SANITY))
        mutate(raw)
        return C.FrameworkConfig.model_validate(raw)

    def test_registry_bucket_and_field(self) -> None:
        from meta_agent import registry

        C._ensure_builtins_loaded()
        self.assertIn("agentic", registry.available("edit_memory"))
        self.assertIn("edit_memory", C.FrameworkConfig.model_fields)

    def test_edit_memory_needs_the_agentic_editor(self) -> None:
        def mutate(raw):
            raw["editor"] = {"type": "default", "config": {"model": "m", "base_url": "http://x/v1"}}
        with self.assertRaisesRegex(ValueError, "memory_path"):
            C.build_components(self.cfg(mutate))

    def test_edit_memory_needs_paired_evaluation(self) -> None:
        def mutate(raw):
            raw["manager"]["config"]["expand_eval_size"] = 0
        with self.assertRaisesRegex(ValueError, "expand_eval_size"):
            C.build_components(self.cfg(mutate))

    def test_meta_model_must_be_explicit(self) -> None:
        for key in ("model", "base_url"):
            for section in ("editor", "edit_memory"):
                with self.subTest(section=section, key=key):
                    def mutate(raw, s=section, k=key):
                        raw[s]["config"].pop(k)
                    with self.assertRaisesRegex(ValueError, f"{section}.config.{key}"):
                        C.check_meta_llm_config(self.cfg(mutate))

    def test_output_cap_warning(self) -> None:
        def mutate(raw):
            raw["editor"]["config"].pop("max_output_tokens")
        warnings = C.check_meta_llm_config(self.cfg(mutate))
        self.assertEqual(len(warnings), 1)
        self.assertIn("editor.config.max_output_tokens", warnings[0])
        self.assertEqual(C.check_meta_llm_config(self.cfg(lambda raw: None)), [])

    def test_default_editor_configs_are_not_checked(self) -> None:
        cfg = C.load(CONFIGS / "hgm_travel_full_scale_block_tagged_X100Y180.yaml")
        self.assertEqual(C.check_meta_llm_config(cfg), [])

    def test_preflight_names_a_missing_key(self) -> None:
        import os
        from unittest import mock

        from main_loop import _preflight_keys

        cfg = C.load(SANITY)
        env = {k: v for k, v in os.environ.items() if k != "OpenRouter_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OpenRouter_API_KEY"):
                _preflight_keys(cfg)
        with mock.patch.dict(os.environ, {**env, "OpenRouter_API_KEY": "k"}, clear=True):
            _preflight_keys(cfg)


if __name__ == "__main__":
    unittest.main()

"""Tests for LLMBackboneConfigValidator (meta_agent/editor_validators.py)
-- structural checks on a task agent's mas_llm_backbone.yaml, catching a
malformed edit (bad YAML, a model with no base_url or vice versa, a bad
type on any field) before it burns a full evaluation batch.

    PYTHONPATH=. python3 -m unittest tests.test_llm_backbone_config_validator
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.editor_validators import LLMBackboneConfigValidator


class LLMBackboneConfigValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="llm_backbone_config_validator_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out_dir = self.tmp / "out"
        self.base_dir = self.tmp / "base"
        (self.out_dir / "task_agent").mkdir(parents=True)
        (self.base_dir / "task_agent").mkdir(parents=True)
        self.validator = LLMBackboneConfigValidator()

    def _write(self, text: str) -> None:
        (self.out_dir / "task_agent" / "mas_llm_backbone.yaml").write_text(text, encoding="utf-8")

    def _validate(self) -> list[str]:
        return self.validator.validate(self.out_dir, self.base_dir)

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(self._validate(), [])

    def test_all_null_shipped_shape_is_valid(self) -> None:
        self._write(
            "default:\n"
            "  model: null\n"
            "  base_url: null\n"
            "  temperature: null\n"
            "  max_output_tokens: null\n"
            "  reasoning_effort: null\n"
            "agents:\n"
            "  flight: {}\n"
            "  train: {}\n"
            "  sightseeing: {}\n"
            "  accounting: {}\n"
        )
        self.assertEqual(self._validate(), [])

    def test_valid_paired_model_and_base_url_is_valid(self) -> None:
        self._write(
            "default:\n"
            "  model: null\n"
            "  base_url: null\n"
            "agents:\n"
            "  sightseeing:\n"
            "    model: \"qwen/qwen3.5-122b-a10b\"\n"
            "    base_url: \"https://openrouter.ai/api/v1\"\n"
            "    temperature: 0.2\n"
            "    max_output_tokens: 16384\n"
            "    reasoning_effort: \"medium\"\n"
        )
        self.assertEqual(self._validate(), [])

    def test_invalid_yaml_is_rejected(self) -> None:
        self._write("default:\n  model: [unterminated\n")
        errors = self._validate()
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid YAML", errors[0])

    def test_model_without_base_url_is_rejected(self) -> None:
        self._write(
            "default: {}\n"
            "agents:\n"
            "  flight:\n"
            "    model: \"qwen/qwen3.5-35b-a3b\"\n"
        )
        errors = self._validate()
        self.assertTrue(any("but no base_url" in e for e in errors), errors)

    def test_base_url_without_model_is_rejected(self) -> None:
        self._write(
            "default: {}\n"
            "agents:\n"
            "  train:\n"
            "    base_url: \"https://openrouter.ai/api/v1\"\n"
        )
        errors = self._validate()
        self.assertTrue(any("but no model" in e for e in errors), errors)

    def test_base_url_not_a_url_is_rejected(self) -> None:
        self._write(
            "default: {}\n"
            "agents:\n"
            "  flight:\n"
            "    model: \"qwen/qwen3.5-35b-a3b\"\n"
            "    base_url: \"not-a-url\"\n"
        )
        errors = self._validate()
        self.assertTrue(any("does not look like a URL" in e for e in errors), errors)

    def test_non_numeric_temperature_is_rejected(self) -> None:
        self._write("default:\n  temperature: \"warm\"\n")
        errors = self._validate()
        self.assertTrue(any("temperature must be a number" in e for e in errors), errors)

    def test_negative_max_output_tokens_is_rejected(self) -> None:
        self._write("default:\n  max_output_tokens: -5\n")
        errors = self._validate()
        self.assertTrue(any("max_output_tokens must be a positive integer" in e for e in errors), errors)

    def test_non_integer_max_output_tokens_is_rejected(self) -> None:
        self._write("default:\n  max_output_tokens: 12.5\n")
        errors = self._validate()
        self.assertTrue(any("max_output_tokens must be a positive integer" in e for e in errors), errors)

    def test_non_string_reasoning_effort_is_rejected(self) -> None:
        self._write("default:\n  reasoning_effort: 3\n")
        errors = self._validate()
        self.assertTrue(any("reasoning_effort must be a string" in e for e in errors), errors)

    def test_unrecognized_field_is_rejected(self) -> None:
        self._write("default:\n  modle: \"typo\"\n")
        errors = self._validate()
        self.assertTrue(any("unrecognized field" in e for e in errors), errors)

    def test_non_mapping_agent_entry_is_rejected(self) -> None:
        self._write("default: {}\nagents:\n  flight: \"oops\"\n")
        errors = self._validate()
        self.assertTrue(any("must be a mapping" in e for e in errors), errors)

    def test_non_mapping_agents_section_is_rejected(self) -> None:
        self._write("default: {}\nagents: \"oops\"\n")
        errors = self._validate()
        self.assertTrue(any("'agents' must be a mapping" in e for e in errors), errors)

    def test_empty_agent_entry_is_valid(self) -> None:
        self._write("default: {}\nagents:\n  flight: {}\n")
        self.assertEqual(self._validate(), [])

    def test_null_agent_entry_is_valid(self) -> None:
        self._write("default: {}\nagents:\n  flight: null\n")
        self.assertEqual(self._validate(), [])


if __name__ == "__main__":
    unittest.main()

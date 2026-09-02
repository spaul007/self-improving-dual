"""End-to-end test that config.py::build_components actually injects a
real evaluator + benchmark_dir into SmokeTestValidator when it's listed
in validators:, using a real project's config (travel_mas_refactored) --
no network/LLM calls, since building components never runs anything.

Regression guard for a real bug found while wiring this up: benchmark_dir
was referenced before its own assignment in build_components (a
NameError, the same "name assigned later in the function shadows an
earlier use" class of bug as agents/flight.py's crash earlier this
session) -- fixed by moving path resolution earlier in the function.

    PYTHONPATH=. python3 -m unittest tests.test_smoke_test_validator_wiring
"""
from __future__ import annotations

import unittest
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent.editor_validators import SmokeTestValidator
from meta_agent.evaluator import SubprocessEvaluator


class SmokeTestValidatorWiringTests(unittest.TestCase):
    def test_smoke_test_validator_receives_the_real_evaluator_and_benchmark_dir(
        self,
    ) -> None:
        cfg = cfg_mod.load("configs/hgm_travel_full_scale_adaptive_X100Y180.yaml")
        cfg.validators = list(cfg.validators) + [
            cfg_mod.ComponentSpec(type="smoke_test", config={})
        ]
        fw = cfg_mod.build_components(cfg)

        smoke_validators = [
            v for v in fw.validators if isinstance(v, SmokeTestValidator)
        ]
        self.assertEqual(len(smoke_validators), 1)
        v = smoke_validators[0]
        self.assertIsInstance(v.evaluator, SubprocessEvaluator)
        self.assertIsInstance(v.benchmark_dir, Path)
        self.assertTrue(v.benchmark_dir.is_dir())
        self.assertTrue((v.benchmark_dir / "cases.jsonl").is_file())

    def test_other_validators_are_unaffected_by_the_new_injections(self) -> None:
        # Every other validator's __init__ doesn't declare evaluator/
        # benchmark_dir, so _build_with_injection's own rule (only inject
        # params the constructor actually accepts) must leave them
        # untouched -- this is the regression check for existing configs.
        cfg = cfg_mod.load("configs/hgm_travel_full_scale_adaptive_X100Y180.yaml")
        fw = cfg_mod.build_components(cfg)
        self.assertGreater(len(fw.validators), 0)
        for v in fw.validators:
            self.assertNotIsInstance(v, SmokeTestValidator)


if __name__ == "__main__":
    unittest.main()

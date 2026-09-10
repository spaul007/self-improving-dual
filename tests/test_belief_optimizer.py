"""Unit tests for the online guidance optimizer (belief_optimizer).

Fake LLM adapter, no network.

    PYTHONPATH=. python3 -m unittest tests.test_belief_optimizer
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.belief_optimizer import (
    INSTRUCTION_ARCHIVE_DIR,
    INSTRUCTION_NAME,
    SEED_INSTRUCTION,
    InstructionOptimizer,
)
from meta_agent.belief_scoring import Scored
from tests.test_belief_scoring import _rec


class _Call:
    def __init__(self, instruction="Weigh node counts; never trust one Δ.",
                 responses=None):
        self.calls: list[tuple[str, str, str]] = []
        self.responses = list(responses) if responses is not None else None
        self.instruction = instruction

    def __call__(self, system, user, tool, tag):
        self.calls.append((system, user, tag))
        if self.responses:
            return self.responses.pop(0)
        return {"critique": "too confident", "instruction": self.instruction}


def _scored(node, iv, brier_p, y=1, label_source="judge", n_shared=8):
    judge = label_source == "judge"
    return Scored(node=node, kind="strategy", slug="s", p=brier_p, y=y,
                  brier=(brier_p - y) ** 2, belief_version=1,
                  instruction_version=iv, resolved_at_update=1,
                  n_shared=n_shared, delta=0.05,
                  impl_reason="gate fired on case a",
                  label_source=label_source,
                  effect=("improved" if y else "no_effect") if judge else "",
                  evidence="strong" if judge else "",
                  effect_reason="gate fixed hours on 5 cases" if judge else "",
                  matched_edit=1)


class OptimizerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = {"n_updates": 3}

    def _preds(self, nodes):
        return {n: {"strategy": {"slug": "s", "p": 0.7,
                                 "section": f"### belief:s — t\n- predict: p=0.7 (node {n})"},
                    "implementation": None} for n in nodes}


class TestSeedAndGating(OptimizerBase):
    def test_seed_files_and_state(self):
        opt = InstructionOptimizer(_Call())
        opt.ensure_seed(self.tmp, self.state)
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(),
                         SEED_INSTRUCTION)
        self.assertTrue((self.tmp / INSTRUCTION_ARCHIVE_DIR / "v000.md").exists())
        self.assertEqual(self.state["instruction"]["version"], 0)
        self.assertEqual(opt.current_text(self.tmp), SEED_INSTRUCTION)
        # idempotent
        self.state["instruction"]["version"] = 0
        opt.ensure_seed(self.tmp, self.state)
        self.assertEqual(len(self.state["instruction"]["versions"]), 1)

    def test_no_step_below_every(self):
        call = _Call()
        opt = InstructionOptimizer(call, every=4)
        opt.ensure_seed(self.tmp, self.state)
        self.state["scored_since_step"] = 3
        self.assertFalse(opt.maybe_step(self.tmp, self.state, scored=[],
                                        predictions={}, records={},
                                        calibration_report="r", n_updates=3))
        self.assertEqual(call.calls, [])

    def test_disabled(self):
        call = _Call()
        opt = InstructionOptimizer(call, enabled=False, every=1)
        self.state["scored_since_step"] = 5
        self.assertFalse(opt.maybe_step(self.tmp, self.state, scored=[],
                                        predictions={}, records={},
                                        calibration_report="r", n_updates=3))
        self.assertEqual(call.calls, [])


class TestStep(OptimizerBase):
    def test_step_writes_version_and_prompt(self):
        call = _Call()
        opt = InstructionOptimizer(call, every=2, min_scored=2)
        opt.ensure_seed(self.tmp, self.state)
        scored = [_scored(1, 0, 0.7, 1), _scored(2, 0, 0.7, 0)]
        self.state["scored_since_step"] = 2
        records = {1: _rec(0.05, 8, True), 2: _rec(-0.05, 8, True)}
        self.assertTrue(opt.maybe_step(self.tmp, self.state, scored=scored,
                                       predictions=self._preds([1, 2]),
                                       records=records,
                                       calibration_report="## Calibration report\n- x",
                                       n_updates=5))
        info = self.state["instruction"]
        self.assertEqual(info["version"], 1)
        self.assertEqual(self.state["scored_since_step"], 0)
        self.assertEqual([e["event"] for e in info["events"]], ["step"])
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(),
                         "Weigh node counts; never trust one Δ.")
        self.assertTrue((self.tmp / INSTRUCTION_ARCHIVE_DIR / "v001.md").exists())
        prompt = (self.tmp / INSTRUCTION_ARCHIVE_DIR / "step_001_prompt.txt").read_text()
        self.assertIn("## Current guidance (v0", prompt)
        self.assertIn("window Brier", prompt)
        self.assertIn("(node 1)", prompt)              # belief text at registration
        # Judge-first: the verdict and its reason are the outcome; a Δ over
        # 8 shared cases is not well measured and is not quoted.
        self.assertIn('- outcome: judge says improved (strong evidence) — '
                      '"gate fixed hours on 5 cases"', prompt)
        self.assertNotIn("Δ+0.0500", prompt)
        self.assertNotIn("score context", prompt)
        self.assertIn("## Calibration report", prompt)
        system = call.calls[0][0]
        self.assertIn("its verdicts are the labels", system)
        self.assertNotIn("rarely decides", system)
        resp = json.loads((self.tmp / INSTRUCTION_ARCHIVE_DIR
                           / "step_001_response.json").read_text())
        self.assertEqual(resp["critique"], "too confident")
        self.assertEqual(info["versions"][1]["parent"], 0)

    def test_score_is_quoted_only_when_well_measured_and_delta_rows_unchanged(self):
        call = _Call()
        opt = InstructionOptimizer(call, every=2, min_scored=2)
        opt.ensure_seed(self.tmp, self.state)
        scored = [_scored(1, 0, 0.7, 1, n_shared=16),          # paired, well measured
                  _scored(2, 0, 0.7, 0, label_source="delta")]  # ablation row
        self.state["scored_since_step"] = 2
        records = {1: _rec(0.05, 16, True), 2: _rec(-0.05, 8, True)}
        records[1]["targets_by_edit"] = {1: ["opening_hours"]}
        self.assertTrue(opt.maybe_step(self.tmp, self.state, scored=scored,
                                       predictions=self._preds([1, 2]),
                                       records=records,
                                       calibration_report="r", n_updates=5))
        prompt = (self.tmp / INSTRUCTION_ARCHIVE_DIR / "step_001_prompt.txt").read_text()
        self.assertIn("- outcome: judge says improved (strong evidence; targets: "
                      "opening_hours)", prompt)
        self.assertIn("- score (well measured): helped Δ+0.0500/16 shared", prompt)
        self.assertIn("- outcome: Δ+0.0500 over 8 shared (helped = Δ ≥ threshold)",
                      prompt)

    def test_over_cap_retries_then_rejects(self):
        call = _Call(instruction="x" * 500)
        opt = InstructionOptimizer(call, every=1, char_cap=200)
        opt.ensure_seed(self.tmp, self.state)
        self.state["scored_since_step"] = 1
        self.assertFalse(opt.maybe_step(self.tmp, self.state,
                                        scored=[_scored(1, 0, 0.7, 1)],
                                        predictions=self._preds([1]),
                                        records={1: _rec(0.05, 8, True)},
                                        calibration_report="r", n_updates=4))
        self.assertEqual(len(call.calls), 2)
        self.assertIn("rejected", call.calls[1][1])
        self.assertIn("the cap is 200", call.calls[1][1])
        info = self.state["instruction"]
        self.assertEqual(info["version"], 0)
        self.assertEqual([e["event"] for e in info["events"]], ["rejected"])
        self.assertEqual(self.state["scored_since_step"], 0)
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(),
                         SEED_INSTRUCTION)

    def test_retry_fixes_over_cap(self):
        call = _Call(responses=[{"instruction": "y" * 500},
                                {"instruction": "short and sweet"}])
        opt = InstructionOptimizer(call, every=1, char_cap=200)
        opt.ensure_seed(self.tmp, self.state)
        self.state["scored_since_step"] = 1
        self.assertTrue(opt.maybe_step(self.tmp, self.state,
                                       scored=[_scored(1, 0, 0.7, 1)],
                                       predictions=self._preds([1]),
                                       records={1: _rec(0.05, 8, True)},
                                       calibration_report="r", n_updates=4))
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(),
                         "short and sweet")


class TestRollback(OptimizerBase):
    def test_worse_version_is_reverted_before_the_step(self):
        call = _Call(instruction="v2 text")
        opt = InstructionOptimizer(call, every=2, min_scored=3, rollback_margin=0.02)
        opt.ensure_seed(self.tmp, self.state)
        # A hand-made v1 that is currently live.
        (self.tmp / INSTRUCTION_ARCHIVE_DIR / "v001.md").write_text("v1 text\n")
        (self.tmp / INSTRUCTION_NAME).write_text("v1 text\n")
        info = self.state["instruction"]
        info["versions"].append({"version": 1, "file": "v001.md",
                                 "created_at_update": 2, "parent": 0,
                                 "critique": "", "chars": 7})
        info["version"] = 1
        scored = ([_scored(n, 0, 0.9, 1) for n in (1, 2, 3)]        # v0: Brier 0.01
                  + [_scored(n, 1, 0.9, 0) for n in (4, 5, 6)])     # v1: Brier 0.81
        self.state["scored_since_step"] = 2
        self.assertTrue(opt.maybe_step(self.tmp, self.state, scored=scored,
                                       predictions=self._preds([1, 2, 3, 4, 5, 6]),
                                       records={n: _rec(0.05, 8, True)
                                                for n in range(1, 7)},
                                       calibration_report="r", n_updates=6))
        events = [e["event"] for e in info["events"]]
        self.assertEqual(events, ["revert", "step"])
        self.assertEqual(info["events"][0]["to"], 0)
        # The step started from the reverted (v0) text and produced v2.
        self.assertEqual(info["version"], 2)
        self.assertEqual(info["versions"][2]["parent"], 0)
        self.assertIn(SEED_INSTRUCTION, call.calls[0][1])
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(), "v2 text")


if __name__ == "__main__":
    unittest.main()

"""The ``hardcoded_answers`` validator and the travel answer extractor.

    PYTHONPATH=. python3 -m unittest tests.test_hardcoded_answer_validator
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from meta_agent.editor_validators import HardcodedAnswerValidator

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "projects" / "travel_mas_refactored"
EXCLUDE = ["agents/immutable/", "benchmark/", "workflow.py"]
EXTRACTOR = "projects.travel_mas_refactored.adapter.answer_literals:answer_literals"

# A tiny extractor registered as an importable module for the unit tests.
_stub = types.ModuleType("_hardcode_stub")
_stub.lits = lambda case: case.get("answers", [])
sys.modules["_hardcode_stub"] = _stub


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.bench = root / "bench"
        self.bench.mkdir()
        cases = [{"id": "1", "input": "q", "answers": ["G729", "Ji Hotel Dalian", "ab"]}]
        (self.bench / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
        self.base, self.out = root / "base", root / "out"
        for d in (self.base, self.out):
            (d / "task_agent" / "agents" / "immutable").mkdir(parents=True)
        (self.base / "task_agent" / "agents" / "flight.py").write_text("A = 1\nKNOWN = 'G729'\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def v(self, **kw) -> HardcodedAnswerValidator:
        return HardcodedAnswerValidator(extractor="_hardcode_stub:lits",
                                        benchmark_dir=self.bench, mutable_exclude=EXCLUDE, **kw)

    def write(self, rel: str, text: str) -> None:
        p = self.out / "task_agent" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_added_literal_is_rejected_without_naming_a_case(self) -> None:
        self.write("agents/train.py", "PICK = 'g729'  # latest train\n")
        errors = self.v().validate(self.out, self.base)
        self.assertEqual(len(errors), 1)
        self.assertIn("agents/train.py: adds a value that belongs to specific benchmark cases", errors[0])
        self.assertNotIn("case 1", errors[0])

    def test_literal_the_parent_already_has_is_allowed(self) -> None:
        self.write("agents/flight.py", "A = 2\nKNOWN = 'G729'\n")
        self.assertEqual(self.v().validate(self.out, self.base), [])

    def test_word_boundaries_and_min_length(self) -> None:
        self.write("agents/x.py", "a = 'G7290'\nb = 'XG729'\nc = 'ab'\n")
        self.assertEqual(self.v().validate(self.out, self.base), [])
        self.write("agents/y.py", "hotel = 'Ji Hotel Dalian'\n")
        self.assertEqual(len(self.v().validate(self.out, self.base)), 1)

    def test_python_comment_lines_are_skipped_strings_are_not(self) -> None:
        self.write("agents/c.py", "    # e.g. 'flight G729, A - B'\nPROMPT = 'take G729'\n")
        errors = self.v().validate(self.out, self.base)
        self.assertEqual(len(errors), 1)
        self.write("agents/c.py", "    # e.g. 'flight G729, A - B'\n")
        self.assertEqual(self.v().validate(self.out, self.base), [])
        self.write("notes.md", "# G729\n")          # not Python: a '#' line still counts
        self.assertEqual(len(self.v().validate(self.out, self.base)), 1)

    def test_excluded_and_generated_paths_are_ignored(self) -> None:
        self.write("agents/immutable/m.py", "X = 'G729'\n")
        self.write("__pycache__/x.py", "X = 'G729'\n")
        self.assertEqual(self.v().validate(self.out, self.base), [])

    def test_forbidden_patterns(self) -> None:
        self.write("agents/z.py", "open('projects/x/benchmark/cases.jsonl')\n")
        errors = self.v(forbidden_patterns=[r"cases\.jsonl"]).validate(self.out, self.base)
        self.assertEqual(len(errors), 1)
        self.assertIn("references evaluation data", errors[0])

    def test_max_reported(self) -> None:
        for i in range(8):
            self.write(f"agents/f{i}.py", "X = 'G729'\n")
        errors = self.v(max_reported=3).validate(self.out, self.base)
        self.assertEqual(len(errors), 4)
        self.assertEqual(errors[-1], "(+5 more hard-coded value(s))")

    def test_bad_extractor_spec(self) -> None:
        with self.assertRaises(ValueError):
            HardcodedAnswerValidator(extractor="no_colon")

    def test_flag_read_by_the_agentic_editor(self) -> None:
        self.assertTrue(HardcodedAnswerValidator.rejects_hardcoded_answers)


class TravelExtractorTests(unittest.TestCase):
    def test_only_values_absent_from_the_query(self) -> None:
        from projects.travel_mas_refactored.adapter.answer_literals import answer_literals

        case = {"input": "Near 'Youhao Square' please.", "meta_info": {"hard_constraints": {
            "a": {"inbound_train_no": "G729", "required_tag": "Outdoor", "hotel_price": 257.0},
            "b": {"attraction_name": "Youhao Square", "restaurant_name": "Haishi Tower",
                  "attraction_names": ["Deji Plaza", "G729"]},
        }}}
        self.assertEqual(answer_literals(case), ["G729", "Haishi Tower", "Deji Plaza"])

    def test_real_benchmark_every_case_has_answers(self) -> None:
        from projects.travel_mas_refactored.adapter.answer_literals import answer_literals

        rows = [json.loads(l) for l in (PROJECT / "benchmark" / "cases.jsonl").read_text().splitlines()]
        self.assertTrue(all(answer_literals(r) for r in rows))

    def test_zero_hits_on_the_seed(self) -> None:
        """Every line of the seed counted as added: nothing may match."""
        v = HardcodedAnswerValidator(extractor=EXTRACTOR, benchmark_dir=PROJECT / "benchmark",
                                     mutable_exclude=EXCLUDE,
                                     forbidden_patterns=[r"cases\.jsonl", r"hard_constraints"])
        with tempfile.TemporaryDirectory() as d:
            out, base = Path(d) / "out", Path(d) / "base"
            shutil.copytree(PROJECT / "seed", out / "task_agent")
            (base / "task_agent").mkdir(parents=True)
            self.assertEqual(v.validate(out, base), [])


class InjectionTests(unittest.TestCase):
    def test_build_injects_benchmark_dir_and_exclude(self) -> None:
        from meta_agent.config import ComponentSpec, _build_with_injection, _ensure_builtins_loaded

        _ensure_builtins_loaded()
        v = _build_with_injection(
            ComponentSpec(type="hardcoded_answers", config={"extractor": EXTRACTOR}),
            "validator", {"mutable_exclude": EXCLUDE, "benchmark_dir": PROJECT / "benchmark",
                          "evaluator": object()},
        )
        self.assertEqual(v.benchmark_dir, PROJECT / "benchmark")
        self.assertEqual(v.mutable_exclude, EXCLUDE)


if __name__ == "__main__":
    unittest.main()

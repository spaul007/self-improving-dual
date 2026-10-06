"""deepswe_seedling reflection hooks: sessions from pier agent/conv, grading outcome
from verifier/, hidden test names returned as redact terms (and so absent from the
lessons_only rendering). Synthetic trial dir; no docker, no network.

    PYTHONPATH=. python3 -m pytest tests/test_deepswe_reflection_hooks.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from meta_agent.models import CaseResult, EvaluationResult

FAIL = "[f2p] pkg/widgets/parser_test.go TestParseWidgetRoundtrip"


class DeepSWEReflectionHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="dsw_refl_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.trial = self.root / "job" / "round_000" / "task" / "sid-x" / "task__abc"
        conv = self.trial / "agent" / "conv"
        conv.mkdir(parents=True)
        (self.trial / "result.json").write_text("{}")
        for name, text in (("patch.1.json", "old"), ("patch.2.json", "newest patch"),
                           ("patch.10.json", "attempt ten"), ("verify.1.json", "verify")):
            (conv / name).write_text(json.dumps({"role": name.split(".")[0], "messages": [
                {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": text}]}))
        (conv / "baseline.0.json").write_text(json.dumps({"messages": [{"role": "user", "content": "b"}]}))
        ver = self.trial / "verifier"
        ver.mkdir()
        (ver / "ctrf.json").write_text(json.dumps({"results": {"tests": [
            {"name": FAIL, "status": "failed"}, {"name": "[p2p] TestOld", "status": "passed"}]}}))
        (ver / "test-stdout.txt").write_text("... --- FAIL: TestParseWidgetRoundtrip expected 3 got 2\n")
        self.env = mock.patch.dict(os.environ, {"SID_PIER_JOBS_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        from projects.deepswe_seedling.adapter.scorer_impl import DeepSWESeedlingScorer

        self.s = DeepSWESeedlingScorer()
        self.case = CaseResult(case_id="task", passed=False, score=0.5, details={
            "agent_metadata": {"trial_dir": str(self.trial)}, "f2p_passed": 1, "f2p_total": 2,
            "p2p_passed": 1, "p2p_total": 1})

    def test_sessions_are_last_attempt_per_role(self) -> None:
        sess = self.s.reflection_sessions(self.case, self.root)
        self.assertEqual(sorted(sess), ["patch", "verify"])
        self.assertEqual(sess["patch"]["messages"][-1]["content"], "attempt ten")
        self.assertIn("PATCH role", sess["patch"]["preamble"])

    def test_untrusted_trial_gives_nothing(self) -> None:
        bad = CaseResult(case_id="t", passed=False, score=0.0,
                         details={"agent_metadata": {"trial_dir": "/etc"}})
        self.assertEqual(self.s.reflection_sessions(bad, self.root), {})
        self.assertNotIn("FAILED", self.s.grading_outcome(bad, "full")["text"])

    def test_grading_full_vs_numeric_and_redact(self) -> None:
        full = self.s.grading_outcome(self.case, "full")
        self.assertIn("TestParseWidgetRoundtrip", full["text"])
        self.assertIn("expected 3 got 2", full["text"])
        self.assertIn("TestParseWidgetRoundtrip", full["redact"])
        self.assertIn("pkg/widgets/parser_test.go", full["redact"])

    def test_go_qualified_name_redacts_unqualified(self) -> None:
        ctrf = self.trial / "verifier" / "ctrf.json"
        ctrf.write_text(json.dumps({"results": {"tests": [
            {"name": "[f2p] github.com/acme/lint.TestPinningMixedRefs", "status": "failed"},
            {"name": "[f2p] tests/test_api.py::TestCls::test_roundtrip_unicode", "status": "failed"}]}}))
        red = self.s.grading_outcome(self.case, "full")["redact"]
        for term in ("TestPinningMixedRefs", "test_roundtrip_unicode", "TestCls"):
            self.assertIn(term, red)

    def test_pytest_parametrised_name_with_spaces(self) -> None:
        ctrf = self.trial / "verifier" / "ctrf.json"
        n = ("[f2p] tests.models.test_cookie_store.test_cookie_store_supports_combined_set_cookie_header"
             "[a=1; Expires=Wed, 21 Oct 2099 07:28:00 GMT, b=2-a=1; b=2]")
        ctrf.write_text(json.dumps({"results": {"tests": [{"name": n, "status": "failed"}]}}))
        red = self.s.grading_outcome(self.case, "full")["redact"]
        self.assertIn("test_cookie_store_supports_combined_set_cookie_header", red)
        self.assertIn("tests.models.test_cookie_store.test_cookie_store_supports_combined_set_cookie_header", red)
        self.assertIn("test_cookie_store", red)
        self.assertFalse(any(x in red for x in ("b=2]", "GMT", "Expires=Wed")))
        num = self.s.grading_outcome(self.case, "numeric")
        self.assertIn("1/2", num["text"])
        self.assertNotIn("TestParseWidget", num["text"])

    def test_end_to_end_lessons_only_hides_test_names(self) -> None:
        from meta_agent.reflector import Reflector, render_reflections

        seen = []

        def chat(msgs, max_tokens, **kw):
            seen.append(msgs[-1]["content"])
            return {"content": "1. WHERE: x\n2. WHY: y\n3. WHAT WOULD HAVE CAUGHT IT: run "
                               "TestParseWidgetRoundtrip-like checks\n4. GENERAL LESSON: check the "
                               "round-trip in pkg/widgets/parser_test.go style tests", "finish_reason": "stop"}

        rd = self.root / "round_001"
        r = Reflector(scorer=self.s, chat_caller=chat, modes=["post_grading"])
        st = r.reflect(rd, EvaluationResult(score=0.5, passed=0, failed=1, per_case=[self.case]))
        self.assertEqual((st["calls"], st["ok"]), (2, 2))
        self.assertTrue(all("TestParseWidgetRoundtrip" in s for s in seen))
        text = render_reflections(rd, "lessons_only")
        self.assertIn("check the round-trip", text)
        self.assertNotIn("TestParseWidgetRoundtrip", text)
        self.assertNotIn("parser_test.go", text)


if __name__ == "__main__":
    unittest.main()

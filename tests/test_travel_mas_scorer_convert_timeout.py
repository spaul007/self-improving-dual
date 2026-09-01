"""Tests for the overall wall-clock deadline in
projects/travel_mas/adapter/scorer_impl.py::_convert_plan_to_json.

Same bug class as tests/test_travel_scorer_convert_timeout.py (the
travel_mas_refactored sibling): DEFAULT_RETRIES=31 with only a per-attempt
HTTP timeout can still legitimately sum to hours. This project's version
was unpatched until 2026-08-31 (found while scoping an unattended
overnight reval chain that uses it) -- CONVERT_OVERALL_TIMEOUT_S enforces
a hard ceiling on the WHOLE retry loop, independent of the per-attempt
timeout or how many attempts have run.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_scorer_convert_timeout
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from projects.travel_mas.adapter import scorer_impl


class ConvertPlanOverallDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(
            "os.environ",
            {"TRAVEL_CONVERT_BASE_URL": "http://fake-host:8000/v1"},
            clear=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_always_failing_client_stops_at_the_overall_deadline_not_31_retries(
        self,
    ) -> None:
        call_count = 0

        def _always_raise(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("simulated hang/failure")

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _always_raise

        with patch("openai.OpenAI", return_value=fake_client), patch.object(
            scorer_impl, "CONVERT_OVERALL_TIMEOUT_S", 2.5
        ):
            started = time.monotonic()
            result, err = scorer_impl._convert_plan_to_json("a real plan")
            wall_time = time.monotonic() - started

        self.assertIsNone(result)
        self.assertIn("timed out after", err)
        self.assertLess(wall_time, 10.0)
        self.assertLess(call_count, scorer_impl.DEFAULT_RETRIES)
        self.assertGreater(call_count, 0)

    def test_success_on_first_attempt_is_unaffected(self) -> None:
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '<JSON>{"ok": true}</JSON>'

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            result, err = scorer_impl._convert_plan_to_json("a real plan")

        self.assertEqual(result, {"ok": True})
        self.assertIsNone(err)
        fake_client.chat.completions.create.assert_called_once()

    def test_empty_plan_text_short_circuits_before_any_client_call(self) -> None:
        with patch("openai.OpenAI") as mock_openai_cls:
            result, err = scorer_impl._convert_plan_to_json("   ")
        self.assertIsNone(result)
        self.assertEqual(err, "agent produced no plan")
        mock_openai_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()

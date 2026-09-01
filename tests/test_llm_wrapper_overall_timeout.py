"""Tests for the overall wall-clock deadline in
platform_core.llm_wrapper.call_llm's retry loop.

Real bug this guards against (same class as
projects/travel_mas_refactored/adapter/scorer_impl.py's
CONVERT_OVERALL_TIMEOUT_S, confirmed live there: a real run stalled 6.6+
hours despite a per-attempt timeout): call_llm's only protection used to be
a per-attempt HTTP timeout around up to DEFAULT_API_MAX_RETRIES=30
attempts -- which can still legitimately sum to hours even when each
individual timeout fires correctly. DEFAULT_OVERALL_TIMEOUT_S enforces a
hard ceiling on the WHOLE retry loop, independent of how many attempts
have run.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_llm_wrapper_overall_timeout
"""
from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock


class _AlwaysFailingResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("simulated hang/failure")


class _SucceedingResponses:
    def create(self, **kwargs):
        return SimpleNamespace(
            output=[], output_text="ok", status="completed", usage=None,
        )


class _FakeOpenAI:
    def __init__(self, responses_impl, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = responses_impl


class OverallTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = {
            k: os.environ.get(k)
            for k in ("OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        }
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["LLM_MODEL"] = "gpt-5.4-mini"
        os.environ.pop("LLM_BASE_URL", None)

    def tearDown(self) -> None:
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_always_failing_client_stops_at_overall_deadline_not_30_retries(
        self,
    ) -> None:
        from platform_core import llm_wrapper

        failing = _AlwaysFailingResponses()

        # Real (unmocked) fixed backoff (DEFAULT_API_BACKOFF_S=1.5s) between
        # attempts -- elapsed time genuinely crosses a small overall
        # deadline after just 2 failed attempts, well before anywhere near
        # DEFAULT_API_MAX_RETRIES=30.
        with mock.patch(
            "openai.OpenAI",
            new=lambda **kw: _FakeOpenAI(failing, **kw),
        ), mock.patch.object(llm_wrapper, "DEFAULT_OVERALL_TIMEOUT_S", 2.5):
            started = time.monotonic()
            with self.assertRaises(TimeoutError) as ctx:
                llm_wrapper.call_llm(messages=[{"role": "user", "content": "hi"}])
            wall_time = time.monotonic() - started

        self.assertIn("timed out after", str(ctx.exception))
        # The whole point of the fix: bounded by the overall deadline, NOT
        # by DEFAULT_API_MAX_RETRIES=30 real attempts.
        self.assertLess(wall_time, 10.0)
        self.assertLess(failing.calls, llm_wrapper.DEFAULT_API_MAX_RETRIES)
        self.assertGreater(failing.calls, 0)

    def test_success_on_first_attempt_is_unaffected_by_the_deadline_check(
        self,
    ) -> None:
        from platform_core import llm_wrapper

        succeeding = _SucceedingResponses()

        with mock.patch(
            "openai.OpenAI",
            new=lambda **kw: _FakeOpenAI(succeeding, **kw),
        ):
            response = llm_wrapper.call_llm(
                messages=[{"role": "user", "content": "hi"}]
            )

        self.assertEqual(response.content, "ok")


if __name__ == "__main__":
    unittest.main()

"""Tests for platform_core.llm_wrapper.call_llm's handling of a
response.status == "failed" result.

Real bug this guards against (confirmed live 2026-09-11 against the real
OpenRouter API, google/gemma-4-31b-it): client.responses.create() can
return WITHOUT raising an exception, yet report its own generation as
failed (status="failed", empty content). Before this fix, call_llm's
retry loop only retried on a thrown exception -- a "successful" call
whose own status says it failed was accepted immediately and handed
downstream unchanged, masquerading as the task agent's own output-quality
problem instead of a transient provider issue. Traced live to a
43.75%-vs-1.7% no_plan_rate spike in one HGM evaluation round.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_llm_wrapper_status_failed_retry
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock


class _FakeOpenAI:
    def __init__(self, responses_impl, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = responses_impl


class _FailsNTimesThenSucceeds:
    """Returns a status="failed", empty-content response for the first
    ``n_failures`` calls (no exception raised, matching the real API
    behavior this guards against), then a normal successful response."""

    def __init__(self, n_failures: int) -> None:
        self.n_failures = n_failures
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.n_failures:
            return SimpleNamespace(output=[], status="failed", usage=None)
        return SimpleNamespace(
            output=[], output_text="recovered", status="completed", usage=None,
        )


class _AlwaysFailedStatus:
    """Every call returns status="failed" -- never raises, never recovers."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(output=[], status="failed", usage=None)


class _AlwaysFailedWithErrorInfo:
    """Every call returns status="failed" AND a populated error field (code
    + message) -- the real Responses-API shape this test guards against
    call_llm actually surfacing, instead of discarding it."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            output=[],
            status="failed",
            usage=None,
            error=SimpleNamespace(code="server_error", message="internal error"),
        )


class StatusFailedRetryTests(unittest.TestCase):
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

    def test_transient_failed_status_is_retried_and_recovers(self) -> None:
        from platform_core import llm_wrapper

        flaky = _FailsNTimesThenSucceeds(n_failures=2)

        with mock.patch(
            "openai.OpenAI", new=lambda **kw: _FakeOpenAI(flaky, **kw)
        ), mock.patch.object(llm_wrapper, "DEFAULT_API_BACKOFF_S", 0.0):
            response = llm_wrapper.call_llm(
                messages=[{"role": "user", "content": "hi"}]
            )

        # The whole point of the fix: a status="failed" response (no
        # exception) is retried just like an exception would be, and the
        # eventual successful attempt's content is what callers see.
        self.assertEqual(response.content, "recovered")
        self.assertEqual(response.stop_reason, "completed")
        self.assertEqual(flaky.calls, 3)

    def test_persistent_failed_status_falls_through_unchanged_on_last_attempt(
        self,
    ) -> None:
        """Exhausting retries on a status="failed" response must NOT raise
        -- that would be a behavior change from before this fix, where a
        status="failed" response was never treated as retriable at all
        and simply passed through as-is. The final attempt's (still
        failed/empty) response is returned normally."""
        from platform_core import llm_wrapper

        always_failed = _AlwaysFailedStatus()

        with mock.patch(
            "openai.OpenAI", new=lambda **kw: _FakeOpenAI(always_failed, **kw)
        ), mock.patch.object(llm_wrapper, "DEFAULT_API_MAX_RETRIES", 3), mock.patch.object(
            llm_wrapper, "DEFAULT_API_BACKOFF_S", 0.0
        ):
            response = llm_wrapper.call_llm(
                messages=[{"role": "user", "content": "hi"}]
            )

        self.assertIsNone(response.content)
        self.assertEqual(response.stop_reason, "failed")
        self.assertEqual(always_failed.calls, 3)

    def test_response_error_code_and_message_are_surfaced_not_discarded(
        self,
    ) -> None:
        """Real bug: the OpenAI Responses API's own response.error field
        (code/message, populated by the API when status=="failed") was
        never read anywhere in call_llm -- a failure's actual reason
        (rate limit? server error? content filter?) was silently thrown
        away. Confirm it now reaches both the retry trace event and the
        final llm_response trace event."""
        from platform_core import llm_wrapper

        always_failed = _AlwaysFailedWithErrorInfo()
        emitted: list[tuple[str, dict]] = []

        def _capture_emit(kind, payload):
            emitted.append((kind, payload))

        with mock.patch(
            "openai.OpenAI", new=lambda **kw: _FakeOpenAI(always_failed, **kw)
        ), mock.patch.object(
            llm_wrapper, "DEFAULT_API_MAX_RETRIES", 2
        ), mock.patch.object(
            llm_wrapper, "DEFAULT_API_BACKOFF_S", 0.0
        ), mock.patch.object(
            llm_wrapper.trace, "emit", side_effect=_capture_emit
        ):
            response = llm_wrapper.call_llm(
                messages=[{"role": "user", "content": "hi"}]
            )

        retry_events = [p for k, p in emitted if k == "llm_call_retry"]
        response_events = [p for k, p in emitted if k == "llm_response"]

        self.assertTrue(retry_events, "expected at least one retry event")
        self.assertEqual(retry_events[0]["response_error_code"], "server_error")
        self.assertEqual(retry_events[0]["response_error_message"], "internal error")

        self.assertEqual(len(response_events), 1)
        self.assertEqual(response_events[0]["response_error_code"], "server_error")
        self.assertEqual(response_events[0]["response_error_message"], "internal error")
        self.assertEqual(response.stop_reason, "failed")

    def test_successful_first_attempt_is_not_retried(self) -> None:
        from platform_core import llm_wrapper

        flaky = _FailsNTimesThenSucceeds(n_failures=0)

        with mock.patch(
            "openai.OpenAI", new=lambda **kw: _FakeOpenAI(flaky, **kw)
        ):
            response = llm_wrapper.call_llm(
                messages=[{"role": "user", "content": "hi"}]
            )

        self.assertEqual(response.content, "recovered")
        self.assertEqual(flaky.calls, 1)


if __name__ == "__main__":
    unittest.main()

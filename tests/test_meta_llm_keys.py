"""Per-call key and request body for the failure summarizer and the
gatherer's error-bucket classification (``api_key_env`` / ``extra_body``;
for the gatherer ``error_bucket_api_key_env`` / ``error_bucket_extra_body``).

Driven through the real ``call_llm`` with a recorded OpenAI client, so the
tests show which key and body actually reach the provider, and that
components configured without these options send exactly what they did
before.

    PYTHONPATH=. python3 -m unittest tests.test_meta_llm_keys
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from platform_core.llm_wrapper import call_llm

PIN = {"provider": {"order": ["StreamLake", "Alibaba"], "allow_fallbacks": False}}
ENV = {"OPENAI_API_KEY": "openai-key", "OpenRouter_API_KEY": "openrouter-key"}
OPENROUTER = "https://openrouter.ai/api/v1"


class _Recorder:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.requests: list[dict] = []

    def factory(self, **kw):
        self.keys.append(kw.get("api_key"))
        rec = self

        class _Responses:
            def create(self, **req):
                rec.requests.append(req)
                return SimpleNamespace(output=[], output_text="summary", status="completed", usage=None)

        return SimpleNamespace(responses=_Responses())


def _recording(fn):
    rec = _Recorder()
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("LLM_") and k not in ENV}
    with mock.patch.dict(os.environ, {**base, **ENV}, clear=True), \
            mock.patch("openai.OpenAI", side_effect=rec.factory):
        fn()
    return rec


def _failing_case():
    from meta_agent.models import CaseResult, EvaluationResult

    case = CaseResult(case_id="c1", passed=False, score=0.2,
                      details={"query": "trip", "raw_result": "plan", "failed_checks": ["budget"]})
    return EvaluationResult(score=0.2, passed=0, failed=1, per_case=[case])


class FailureSummarizerKeyTests(unittest.TestCase):
    def summarize(self, **kw):
        from meta_agent.failure_summarizer import FailureSummarizer

        fs = FailureSummarizer(call_llm, model="deepseek/deepseek-v4-pro-0813",
                               reasoning_effort="medium", base_url=OPENROUTER, **kw)
        return _recording(lambda: fs._call_llm("system", "user"))

    def test_named_key_and_pin_reach_the_provider(self) -> None:
        rec = self.summarize(api_key_env="OpenRouter_API_KEY", extra_body=PIN)
        self.assertEqual(rec.keys, ["openrouter-key"])
        self.assertEqual(rec.requests[0]["extra_body"], PIN)

    def test_default_is_unchanged(self) -> None:
        rec = self.summarize()
        self.assertEqual(rec.keys, ["openai-key"])
        self.assertNotIn("extra_body", rec.requests[0])


class GathererKeyTests(unittest.TestCase):
    def classify(self, **kw):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer

        g = DefaultFeedbackGatherer(llm_caller=call_llm,
                                    error_bucket_model="deepseek/deepseek-v4-pro-0813",
                                    error_bucket_base_url=OPENROUTER,
                                    error_bucket_reasoning_effort="medium", **kw)
        with tempfile.TemporaryDirectory() as d:
            return _recording(lambda: g._error_bucket_prevalence(_failing_case(), [], Path(d)))

    def test_named_key_and_pin_reach_the_provider(self) -> None:
        rec = self.classify(error_bucket_api_key_env="OpenRouter_API_KEY",
                            error_bucket_extra_body=PIN)
        self.assertEqual(rec.keys, ["openrouter-key"])
        self.assertEqual(rec.requests[0]["extra_body"], PIN)
        self.assertEqual(rec.requests[0]["model"], "deepseek/deepseek-v4-pro-0813")

    def test_default_is_unchanged(self) -> None:
        rec = self.classify()
        self.assertEqual(rec.keys, ["openai-key"])
        self.assertNotIn("extra_body", rec.requests[0])


class AnalyzerKwargsTests(unittest.TestCase):
    def test_kwargs_only_when_set(self) -> None:
        from meta_agent.error_bucket_analyzer import classify_batches

        from meta_agent.error_bucket_analyzer import build_case_digest

        digest = [build_case_digest({"case_id": "c1", "passed": False, "score": 0.2,
                                     "details": {"query": "q", "raw_result": "p"}}, [])]
        calls: list[dict] = []

        def fake(**kw):
            calls.append(kw)
            return SimpleNamespace(content="")

        classify_batches(digest, llm_caller=fake, model="m", base_url=None,
                         reasoning_effort=None, batch_size=4)
        classify_batches(digest, llm_caller=fake, model="m", base_url=None,
                         reasoning_effort=None, batch_size=4,
                         api_key_env="K", extra_body=PIN)
        self.assertNotIn("api_key_env", calls[0])
        self.assertNotIn("extra_body", calls[0])
        self.assertEqual((calls[1]["api_key_env"], calls[1]["extra_body"]), ("K", PIN))


if __name__ == "__main__":
    unittest.main()

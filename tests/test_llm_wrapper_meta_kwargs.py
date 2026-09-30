"""call_llm's per-call options for meta-agent callers: ``api_key_env``,
``timeout_s`` and ``extra_body`` (used by the agentic editor and the
edit-memory curators, which talk to a different provider than the task
agent). Without them the request must be exactly what it was before.

    PYTHONPATH=. python3 -m unittest tests.test_llm_wrapper_meta_kwargs
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

_CLEAN_ENV = {
    k: v for k, v in os.environ.items()
    if not k.startswith("LLM_") and k not in ("OPENAI_API_KEY", "MY_META_KEY")
}


class _Recorder:
    def __init__(self) -> None:
        self.client_kwargs: list[dict] = []
        self.requests: list[dict] = []

    def factory(self, **kwargs):
        self.client_kwargs.append(kwargs)
        rec = self

        class _Responses:
            def create(self, **req):
                rec.requests.append(req)
                return SimpleNamespace(output=[], output_text="ok", status="completed", usage=None)

        return SimpleNamespace(responses=_Responses())


def _call(env: dict, **kwargs):
    from platform_core import llm_wrapper

    rec = _Recorder()
    with mock.patch.dict(os.environ, {**_CLEAN_ENV, **env}, clear=True), \
            mock.patch("openai.OpenAI", side_effect=rec.factory):
        llm_wrapper.call_llm([{"role": "user", "content": "hi"}], **kwargs)
    return rec


class DefaultRequestUnchangedTests(unittest.TestCase):
    def test_no_new_kwargs_same_request_and_client(self) -> None:
        rec = _call({"OPENAI_API_KEY": "k"}, model="m", base_url="http://x/v1")
        self.assertEqual(rec.client_kwargs, [{
            "api_key": "k", "timeout": 300.0, "max_retries": 0, "base_url": "http://x/v1",
        }])
        self.assertEqual(rec.requests, [{
            "model": "m", "input": [{"role": "user", "content": "hi"}], "temperature": 1.0,
        }])

    def test_env_provider_and_thinking_still_apply_without_extra_body(self) -> None:
        rec = _call(
            {"OPENAI_API_KEY": "k", "LLM_PROVIDER_PREFERENCE": '{"order": ["A"]}',
             "LLM_ENABLE_THINKING": "false"},
            model="m",
        )
        self.assertEqual(rec.requests[0]["extra_body"], {
            "provider": {"order": ["A"]}, "chat_template_kwargs": {"enable_thinking": False},
        })


class ApiKeyEnvTests(unittest.TestCase):
    def test_key_taken_from_named_var(self) -> None:
        rec = _call({"OPENAI_API_KEY": "global", "MY_META_KEY": "meta"}, model="m",
                    api_key_env="MY_META_KEY")
        self.assertEqual(rec.client_kwargs[0]["api_key"], "meta")

    def test_unset_named_var_raises_even_with_base_url(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            _call({"OPENAI_API_KEY": "global"}, model="m", base_url="http://x/v1",
                  api_key_env="MY_META_KEY")
        self.assertIn("MY_META_KEY", str(cm.exception))


class TimeoutTests(unittest.TestCase):
    def test_per_attempt_timeout(self) -> None:
        rec = _call({"OPENAI_API_KEY": "k"}, model="m", timeout_s=42)
        self.assertEqual(rec.client_kwargs[0]["timeout"], 42.0)

    @staticmethod
    def _attempts(**kwargs) -> int:
        """Attempts made when every attempt fails and each takes 700 s
        (clock: start=0, then the loop's pre-attempt checks at 0, 700, 1400...)."""
        from platform_core import llm_wrapper

        attempts = []

        class _Failing:
            def create(self, **req):
                attempts.append(1)
                raise RuntimeError("boom")

        clock = iter([0.0, 0.0] + [float(i * 700) for i in range(1, 10)])
        with mock.patch.dict(os.environ, {**_CLEAN_ENV, "OPENAI_API_KEY": "k"}, clear=True), \
                mock.patch("openai.OpenAI", return_value=SimpleNamespace(responses=_Failing())), \
                mock.patch.object(llm_wrapper.time, "time", side_effect=lambda: next(clock)), \
                mock.patch.object(llm_wrapper.time, "sleep"):
            try:
                llm_wrapper.call_llm([{"role": "user", "content": "x"}], model="m", **kwargs)
            except TimeoutError:
                return len(attempts)
        raise AssertionError("call_llm did not time out")

    def test_overall_ceiling_fits_two_attempts(self) -> None:
        # Default ceiling 600 s: the attempt at t=0 runs, the check at t=700 stops.
        self.assertEqual(self._attempts(), 1)
        # timeout_s=600 -> ceiling max(600, 1200) = 1200 s: t=0 and t=700 run.
        self.assertEqual(self._attempts(timeout_s=600), 2)


class ExtraBodyTests(unittest.TestCase):
    def test_extra_body_merged_and_its_provider_beats_env(self) -> None:
        rec = _call(
            {"OPENAI_API_KEY": "k", "LLM_PROVIDER_PREFERENCE": '{"order": ["TaskAgentPin"]}'},
            model="m",
            extra_body={"provider": {"order": ["StreamLake"], "allow_fallbacks": False},
                        "custom": 1},
        )
        self.assertEqual(rec.requests[0]["extra_body"], {
            "custom": 1, "provider": {"order": ["StreamLake"], "allow_fallbacks": False},
        })

    def test_explicit_args_beat_extra_body(self) -> None:
        rec = _call(
            {"OPENAI_API_KEY": "k"}, model="m",
            provider={"order": ["Explicit"]}, enable_thinking=True,
            extra_body={"provider": {"order": ["Body"]},
                        "chat_template_kwargs": {"enable_thinking": False, "other": 2}},
        )
        self.assertEqual(rec.requests[0]["extra_body"], {
            "provider": {"order": ["Explicit"]},
            "chat_template_kwargs": {"enable_thinking": True, "other": 2},
        })

    def test_extra_body_thinking_beats_env(self) -> None:
        rec = _call({"OPENAI_API_KEY": "k", "LLM_ENABLE_THINKING": "true"}, model="m",
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(rec.requests[0]["extra_body"],
                         {"chat_template_kwargs": {"enable_thinking": False}})


if __name__ == "__main__":
    unittest.main()

"""Travel scorer conversion-call controls: reasoning effort, thinking off, env timeouts.

Unset env must send nothing and keep the module-default timeouts (old behaviour).
"""
import os
import unittest
from unittest import mock

from projects.travel_mas_refactored.adapter import scorer_impl as S

_KEYS = ("TRAVEL_CONVERT_REASONING_EFFORT", "TRAVEL_CONVERT_ENABLE_THINKING",
         "TRAVEL_CONVERT_PER_ATTEMPT_TIMEOUT_S", "TRAVEL_CONVERT_OVERALL_TIMEOUT_S")


class ConvertControlsTest(unittest.TestCase):
    def _capture(self, env):
        """Run one conversion attempt against a fake client; return (client kwargs, request kwargs)."""
        seen = {"client": {}, "req": {}}

        class FakeCompletions:
            def create(self, **kw):
                seen["req"].update(kw)
                raise RuntimeError("stop after capturing the request")

        class FakeClient:
            def __init__(self, **kw):
                seen["client"].update(kw)
                self.chat = mock.Mock(completions=FakeCompletions())

        base = {k: "" for k in _KEYS}
        base["LLM_BASE_URL"] = "http://x/v1"
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=False), \
                mock.patch("openai.OpenAI", FakeClient), \
                mock.patch.object(S.time, "sleep", lambda *_: None):
            S._convert_plan_to_json("Day 1: x", retries=1)
        return seen["client"], seen["req"]

    def test_unset_sends_nothing_and_keeps_default_timeout(self):
        client, req = self._capture({})
        self.assertNotIn("extra_body", req)
        self.assertEqual(client["timeout"], S.CONVERT_PER_ATTEMPT_TIMEOUT_S)

    def test_effort_only(self):
        _, req = self._capture({"TRAVEL_CONVERT_REASONING_EFFORT": "low"})
        self.assertEqual(req["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "low"}})

    def test_thinking_off(self):
        for v in ("0", "false", "OFF"):
            _, req = self._capture({"TRAVEL_CONVERT_ENABLE_THINKING": v})
            self.assertEqual(req["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}}, v)

    def test_thinking_on_value_sends_nothing(self):
        _, req = self._capture({"TRAVEL_CONVERT_ENABLE_THINKING": "1"})
        self.assertNotIn("extra_body", req)

    def test_both_combine_into_one_kwargs(self):
        _, req = self._capture({"TRAVEL_CONVERT_REASONING_EFFORT": "low",
                                "TRAVEL_CONVERT_ENABLE_THINKING": "0"})
        self.assertEqual(req["extra_body"]["chat_template_kwargs"],
                         {"reasoning_effort": "low", "enable_thinking": False})

    def test_env_timeouts_reach_client_and_ceiling(self):
        env = {"TRAVEL_CONVERT_PER_ATTEMPT_TIMEOUT_S": "600", "TRAVEL_CONVERT_OVERALL_TIMEOUT_S": "1500"}
        client, _ = self._capture(env)
        self.assertEqual(client["timeout"], 600.0)
        with mock.patch.dict(os.environ, env):
            self.assertEqual(S._convert_timeouts(), (600.0, 1500.0))

    def test_bad_values_fall_back(self):
        with mock.patch.dict(os.environ, {"TRAVEL_CONVERT_PER_ATTEMPT_TIMEOUT_S": "abc",
                                          "TRAVEL_CONVERT_OVERALL_TIMEOUT_S": "-5"}):
            self.assertEqual(S._convert_timeouts(),
                             (S.CONVERT_PER_ATTEMPT_TIMEOUT_S, S.CONVERT_OVERALL_TIMEOUT_S))

    def test_per_attempt_clamped_below_overall(self):
        with mock.patch.dict(os.environ, {"TRAVEL_CONVERT_PER_ATTEMPT_TIMEOUT_S": "900",
                                          "TRAVEL_CONVERT_OVERALL_TIMEOUT_S": "800"}):
            self.assertEqual(S._convert_timeouts(), (400.0, 800.0))


if __name__ == "__main__":
    unittest.main()

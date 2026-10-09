"""platform_core.session_log (opt-in per-case LLM session log) and the travel /
shopping reflection hooks built on it. No network: the OpenAI clients are faked.

    PYTHONPATH=. python3 -m pytest tests/test_reflection_sessions.py -q
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from meta_agent.models import CaseResult

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sesslog_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.round_dir = self.tmp / "round_001"
        self.scratch = self.round_dir / "logs" / "scratch"
        self.scratch.mkdir(parents=True)

    def _env(self, on=True):
        env = {"META_AGENT_SCRATCH_DIR": str(self.scratch)}
        if on:
            env["META_AGENT_SESSION_LOG"] = "1"
        return mock.patch.dict(os.environ, env)


class SessionLogTests(_Tmp):
    def test_off_by_default_writes_nothing(self) -> None:
        from platform_core import session_log, trace

        with self._env(on=False), trace.case_scope("c1"):
            os.environ.pop("META_AGENT_SESSION_LOG", None)
            self.assertIsNone(session_log.log_session([{"role": "user", "content": "x"}], [], fmt="chat"))
        self.assertFalse((self.scratch / "sessions").exists())

    def test_outside_case_scope_writes_nothing(self) -> None:
        from platform_core import session_log

        with self._env():
            self.assertIsNone(session_log.log_session([{"role": "user", "content": "x"}], [], fmt="chat"))

    def test_same_opening_overwrites_last_write_is_whole_session(self) -> None:
        from platform_core import session_log, trace

        base = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        with self._env(), trace.case_scope("case/1"):
            session_log.log_session(base, [{"role": "assistant", "content": "a1"}], fmt="chat")
            session_log.log_session(base + [{"role": "assistant", "content": "a1"},
                                            {"role": "user", "content": "more"}],
                                    [{"role": "assistant", "content": "a2"}], fmt="chat")
            session_log.log_session([{"role": "system", "content": "OTHER"},
                                     {"role": "user", "content": "U"}], [], fmt="chat")
        recs = session_log.load_sessions(self.scratch, "case/1")
        self.assertEqual(len(recs), 2)
        whole = [r for r in recs if r["messages"][0]["content"] == "S"][0]
        self.assertEqual([m["content"] for m in whole["messages"]][-1], "a2")
        self.assertEqual(len(whole["messages"]), 5)

    def test_call_llm_logs_responses_session_without_reasoning(self) -> None:
        from platform_core import llm_wrapper, session_log, trace

        class Item(SimpleNamespace):
            def model_dump(self, exclude_none=True):
                return {k: v for k, v in vars(self).items() if v is not None}

        out = [Item(type="reasoning", summary=[]),
               Item(type="message", role="assistant",
                    content=[{"type": "output_text", "text": "the plan"}])]
        resp = SimpleNamespace(output=out, status="completed", usage=None, output_text="the plan")
        client = mock.MagicMock()
        client.responses.create.return_value = resp
        fake_openai = SimpleNamespace(OpenAI=lambda **kw: client)
        msgs = [{"role": "system", "content": "You are an airline booking specialist"},
                {"role": "user", "content": "trip"}]
        with self._env(), mock.patch.dict(sys.modules, {"openai": fake_openai}), \
                trace.case_scope("t1"):
            llm_wrapper.call_llm(msgs, model="m", base_url="http://x/v1")
        recs = session_log.load_sessions(self.scratch, "t1")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["format"], "responses")
        types = [m.get("type") for m in recs[0]["messages"]]
        self.assertNotIn("reasoning", types)
        self.assertEqual(recs[0]["messages"][-1]["content"][0]["text"], "the plan")
        self.assertEqual(recs[0]["meta"]["model"], "m")


class TravelHookTests(_Tmp):
    def _scorer(self):
        sys.path.insert(0, str(REPO_ROOT / "projects" / "travel_mas_refactored" / "benchmark"))
        try:
            from projects.travel_mas_refactored.adapter.scorer_impl import TravelCompositeScorer
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"travel scorer not importable here: {exc!r}")
        return TravelCompositeScorer()

    def test_sessions_mapped_by_role_and_grading_text(self) -> None:
        from platform_core import session_log, trace

        with self._env(), trace.case_scope("tc1"):
            for sys_prompt in ("You are an airline booking specialist, one role",
                               "You are a rail booking specialist, one role",
                               "You are the sightseeing and logistics planner. Uses flight/train results.",
                               "You are something new"):
                session_log.log_session([{"role": "system", "content": sys_prompt},
                                         {"role": "user", "content": "q"}], [], fmt="responses",
                                        meta={"model": "m1", "base_url": "http://b"})
        s = self._scorer()
        self.assertTrue(s.needs_session_log)
        case = CaseResult(case_id="tc1", passed=False, score=0.4,
                          details={"failed_checks": ["hard:budget"], "commonsense_score": 0.5,
                                   "hard_score": 0.0})
        sess = s.reflection_sessions(case, self.round_dir)
        self.assertEqual(sorted(sess), ["agent4", "flight", "sightseeing", "train"])
        self.assertEqual(sess["flight"]["format"], "responses")
        self.assertEqual(sess["flight"]["model"], "m1")
        self.assertIn("hard:budget", s.grading_outcome(case, "full")["text"])
        self.assertNotIn("hard:budget", s.grading_outcome(case, "numeric")["text"])
        # A grader-side conversion failure is not the agents' fault: nothing to reflect on.
        infra = CaseResult(case_id="tc1", passed=False, score=0.0,
                           details={"conversion_error_type": "timeout"})
        self.assertEqual(s.reflection_sessions(infra, self.round_dir), {})


class ShoppingHookTests(_Tmp):
    def test_chat_json_logs_and_scorer_reads_with_redaction(self) -> None:
        from platform_core import trace

        seed = REPO_ROOT / "projects" / "shopping_mas" / "shopping_mas"
        sys.path.insert(0, str(seed))
        self.addCleanup(lambda: sys.path.remove(str(seed)))
        before = set(sys.modules)
        # Drop the seed's top-level modules (llm_client, config, ...) afterwards so
        # they cannot shadow another project's same-named modules in later tests.
        self.addCleanup(lambda: [sys.modules.pop(m, None) for m in set(sys.modules) - before])
        sys.modules.pop("llm_client", None)
        import llm_client  # the project's own immutable client

        cfg = SimpleNamespace(max_tool_rounds=2, json_retries=0,
                              server=SimpleNamespace(served_model_name="qm", url="http://s/v1"))
        client = llm_client.LLMClient.__new__(llm_client.LLMClient)
        client.cfg = cfg
        client.chat_raw = lambda *a, **k: (SimpleNamespace(content='{"ok": 1}', tool_calls=None), "stop")
        with self._env(), trace.case_scope("s1"):
            self.assertEqual(client.chat_json("You are the Product-Scout Agent. Find.", "item"), {"ok": 1})

        from projects.shopping_mas.adapter.scorer_impl import ShoppingMasScorer

        s = ShoppingMasScorer()
        case = CaseResult(case_id="s1", passed=False, score=0.5,
                          details={"missing_products": ["P-12345"], "extra_products": [],
                                   "matched_products": [], "error_log": "missing P-12345",
                                   "matched_count": 1, "expected_count": 2})
        sess = s.reflection_sessions(case, self.round_dir)
        self.assertEqual(list(sess), ["product_scout"])
        self.assertEqual(sess["product_scout"]["format"], "chat")
        self.assertEqual(sess["product_scout"]["base_url"], "http://s/v1")
        self.assertEqual(sess["product_scout"]["messages"][-1]["content"], '{"ok": 1}')
        g = s.grading_outcome(case, "full")
        self.assertIn("P-12345", g["text"])
        self.assertIn("P-12345", g["redact"])


class ReflectorEnvTests(unittest.TestCase):
    def test_reflector_exports_session_log_env_only_when_needed(self) -> None:
        from meta_agent.reflector import Reflector

        class S:
            needs_session_log = True

            def reflection_sessions(self, case, round_dir):
                return {}

            def grading_outcome(self, case, detail):
                return {}

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("META_AGENT_SESSION_LOG", None)
            Reflector(scorer=object())
            self.assertNotIn("META_AGENT_SESSION_LOG", os.environ)
            Reflector(scorer=S())
            self.assertEqual(os.environ.get("META_AGENT_SESSION_LOG"), "1")


if __name__ == "__main__":
    unittest.main()

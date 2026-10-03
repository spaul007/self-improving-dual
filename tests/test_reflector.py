"""Task-agent reflection (meta_agent/reflector.py): prompt construction per mode,
parsing, redaction under ``exposure``, storage outside logs/, consumer gating, and
the HGM wiring end to end with stub components and a fake LLM (no network).

    PYTHONPATH=. python3 -m pytest tests/test_reflector.py -q
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.models import CaseResult, EvaluationResult

SECRET_TEST = "test_parse_widget_roundtrip"
SECRET_PATH = "pkg/widgets/parser_test.go"

POST_ANSWER = f"""1. WHERE: "I'll assume the default" -- I skipped the edge case.
2. WHY: the handoff said tests passed.
3. WHAT WOULD HAVE CAUGHT IT: VERIFY should run {SECRET_TEST} style checks on the interface.
4. GENERAL LESSON: Re-read the stated behaviour list before finishing; {SECRET_PATH} showed it."""

UNSURE_ANSWER = """1. UNSURE PARTS:
- [40] empty-input handling of the new flag
- [85] error message wording
- [20] behaviour when the config file is missing
2. OVERALL_CONFIDENCE: 55
3. FIRST CHECK: run the CLI with no args."""

ESSENTIAL_ANSWER = """1. ESSENTIAL: running the full suite before finishing.
2. CLOSE CALLS: nearly missed the rename.
3. KEEP: always run the project's own tests after the last edit."""


class _FakeScorer:
    def __init__(self) -> None:
        self.outcome_calls: list[str] = []

    def reflection_sessions(self, case, round_dir):
        return {
            "PATCH": {"messages": [{"role": "system", "content": "you are PATCH"},
                                   {"role": "user", "content": "do the task"},
                                   {"role": "assistant", "content": None,
                                    "tool_calls": [{"id": "1", "type": "function",
                                                    "function": {"name": "bash", "arguments": "{}"}}],
                                    "extra_junk": 1},
                                   {"role": "tool", "tool_call_id": "1", "content": "ok"}],
                      "format": "chat", "preamble": "You are PATCH."},
        }

    def grading_outcome(self, case, detail):
        self.outcome_calls.append(detail)
        return {"text": f"Failing hidden test: {SECRET_TEST} in {SECRET_PATH}",
                "redact": [SECRET_TEST, SECRET_PATH]}


def _fake_chat(seen: list):
    def call(msgs, max_tokens, **kw):
        seen.append(msgs)
        turn = msgs[-1]["content"]
        if "post-grading" in turn:
            return {"content": POST_ANSWER, "finish_reason": "stop"}
        if "self-assessment" in turn:
            return {"content": UNSURE_ANSWER, "finish_reason": "stop"}
        return {"content": ESSENTIAL_ANSWER, "finish_reason": "stop"}
    return call


def _batch():
    return EvaluationResult(score=0.5, passed=1, failed=2, per_case=[
        CaseResult(case_id="fail-near", passed=False, score=0.9),
        CaseResult(case_id="fail-far", passed=False, score=0.1),
        CaseResult(case_id="ok", passed=True, score=1.0),
        CaseResult(case_id="infra", passed=False, score=0.0, details={"excluded": True}),
    ])


class ReflectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="reflector_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seen: list = []

    def _make(self, **kw):
        from meta_agent.reflector import Reflector

        return Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat(self.seen), **kw)

    def test_modes_per_outcome_and_storage_outside_logs(self) -> None:
        r = self._make()
        st = r.reflect(self.tmp, _batch())
        # 3 usable cases (infra excluded): 2 failed x (unsure+post) + 1 ok x (unsure+essential)
        self.assertEqual(st["cases"], 3)
        self.assertEqual(st["calls"], 6)
        self.assertEqual(st["ok"], 6)
        files = sorted(p.name for p in (self.tmp / "reflections").glob("*.json"))
        self.assertIn("fail-near.PATCH.post_grading.json", files)
        self.assertIn("ok.PATCH.essential.json", files)
        self.assertNotIn("infra.PATCH.unsure.json", files)
        self.assertFalse((self.tmp / "logs").exists())

    def test_session_replayed_with_one_turn_and_cleaned(self) -> None:
        r = self._make(modes=["post_grading"])
        r.reflect(self.tmp, _batch())
        msgs = self.seen[0]
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn(SECRET_TEST, msgs[-1]["content"])  # full grading goes IN
        asst = msgs[2]
        self.assertEqual(asst["content"], "")
        self.assertNotIn("extra_junk", asst)

    def test_unsure_is_blind(self) -> None:
        r = self._make(modes=["unsure"])
        r.reflect(self.tmp, _batch())
        for msgs in self.seen:
            self.assertNotIn(SECRET_TEST, msgs[-1]["content"])
            self.assertNotIn("NOT solved", msgs[-1]["content"])

    def test_grading_detail_none_skips_post_grading(self) -> None:
        r = self._make(grading_detail="none")
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["calls"], 4)  # unsure x3 + essential x1
        # Still asked for redact terms (numeric detail), but no grading text in any question.
        self.assertEqual(set(r.scorer.outcome_calls), {"numeric"})
        self.assertFalse(any(SECRET_TEST in m[-1]["content"] for m in self.seen))

    def test_lessons_only_redacts_and_hides_raw(self) -> None:
        from meta_agent.reflector import render_reflections

        self._make().reflect(self.tmp, _batch())
        text = render_reflections(self.tmp, "lessons_only")
        self.assertIn("Re-read the stated behaviour list", text)
        self.assertIn("always run the project's own tests", text)
        self.assertIn("behaviour when the config file is missing", text)  # unsure < 70
        self.assertNotIn("error message wording", text)                   # 85: confident
        self.assertNotIn(SECRET_TEST, text)
        self.assertNotIn(SECRET_PATH, text)
        self.assertNotIn("WHERE", text)  # raw answers never shown
        self.assertEqual(render_reflections(self.tmp, "off"), "")
        full = render_reflections(self.tmp, "full")
        self.assertIn("WHERE", full)
        self.assertNotIn(SECRET_TEST, full)  # redact applies under full too

    def test_parse_unsure(self) -> None:
        from meta_agent.reflector import parse_reflection

        p = parse_reflection("unsure", UNSURE_ANSWER)
        self.assertEqual(p["overall_confidence"], 55)
        self.assertEqual([i["confidence"] for i in p["items"]], [40, 85, 20])

    def test_selection_cap_prefers_near_misses(self) -> None:
        r = self._make(modes=["post_grading"], max_cases_per_batch=1)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["cases"], 1)
        self.assertTrue((self.tmp / "reflections" / "fail-near.PATCH.post_grading.json").exists())

    def test_length_retry_then_errors_recorded(self) -> None:
        from meta_agent.reflector import Reflector

        budgets = []

        def trunc(msgs, max_tokens, **kw):
            budgets.append(max_tokens)
            return {"content": "", "finish_reason": "length"}

        r = Reflector(scorer=_FakeScorer(), chat_caller=trunc, modes=["unsure"],
                      max_cases_per_batch=1, max_output_tokens=1000)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["errors"], 1)
        self.assertEqual(budgets, [1000, 2000])

    def test_context_guard_skips(self) -> None:
        r = self._make(modes=["unsure"], max_cases_per_batch=1, ctx_tokens=100,
                       max_output_tokens=90)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["skipped"], 1)
        self.assertEqual(self.seen, [])

    def test_unsupported_project_is_a_noop(self) -> None:
        from meta_agent.reflector import Reflector

        r = Reflector(scorer=object(), chat_caller=_fake_chat(self.seen))
        self.assertEqual(r.reflect(self.tmp, _batch())["calls"], 0)
        self.assertFalse((self.tmp / "reflections").exists())

    def test_llm_exception_never_raises(self) -> None:
        from meta_agent.reflector import Reflector

        def boom(msgs, max_tokens, **kw):
            raise RuntimeError("server down")

        r = Reflector(scorer=_FakeScorer(), chat_caller=boom, modes=["unsure"])
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["errors"], 3)
        rec = json.loads((self.tmp / "reflections" / "ok.PATCH.unsure.json").read_text())
        self.assertIn("server down", rec["error"])

    def test_phase_gating_and_bad_config(self) -> None:
        r = self._make()
        self.assertEqual(r.reflect(self.tmp, _batch(), phase="root")["calls"], 0)
        from meta_agent.reflector import Reflector

        for bad in ({"modes": ["nope"]}, {"exposure": "x"}, {"grading_detail": "x"},
                    {"consumers": ["x"]}, {"phases": ["x"]}):
            with self.assertRaises(ValueError):
                Reflector(**bad)

    def test_defaults_come_from_task_agent(self) -> None:
        from meta_agent.config import TaskAgentSpec
        from meta_agent.reflector import Reflector

        ta = TaskAgentSpec(model="m-x", base_url="http://h:1/v1", reasoning_effort="medium")
        r = Reflector(task_agent=ta)
        self.assertEqual((r.model, r.base_url, r.reasoning_effort), ("m-x", "http://h:1/v1", "medium"))


    def test_parse_json_shaped_answers(self) -> None:
        from meta_agent.reflector import parse_reflection

        ess = '\n\n{"status": "ok", "essential": ["a"], "close_calls": ["b"],\n "keep_rules": ["Always re-read the cart", "Never guess ids"]}'
        self.assertEqual(parse_reflection("essential", ess)["keep"], "Always re-read the cart Never guess ids")
        post = '{"where": "x", "why": "y", "what_would_have_caught_it": "a diff check", "general_lesson": ["L1", "L2"]}'
        p = parse_reflection("post_grading", post)
        self.assertEqual((p["lesson"], p["catch"]), ("L1 L2", "a diff check"))
        uns = '{"unsure_parts": [{"item": "budget edge", "confidence": 40}], "overall_confidence": 60}'
        u = parse_reflection("unsure", uns)
        self.assertEqual((u["items"], u["overall_confidence"]), ([{"confidence": 40, "item": "item: budget edge"}], 60))

    def test_bold_heading_with_numbered_list_inside(self) -> None:
        from meta_agent.reflector import parse_reflection

        text = ("**1. WHERE**\nx\n\n**3. WHAT WOULD HAVE CAUGHT IT**\n1. a test\n2. a diff\n\n"
                "**4. GENERAL LESSON**\n\n1. Anchor formats to the codebase's own API.\n2. Re-validate.")
        p = parse_reflection("post_grading", text)
        self.assertIn("Anchor formats", p["lesson"])
        self.assertIn("Re-validate", p["lesson"])
        self.assertIn("a diff", p["catch"])
        self.assertNotIn("GENERAL LESSON", p["catch"])

    def test_off_task_reply_is_reasked_then_counted(self) -> None:
        from meta_agent.reflector import Reflector

        replies = iter(["Let me fix it. I'll edit the parser now.", POST_ANSWER,
                        "Let me fix it.", "I will now continue editing."])
        asked = []

        def chat(msgs, max_tokens, **kw):
            asked.append(msgs[-1]["content"])
            return {"content": next(replies), "finish_reason": "stop"}

        r = Reflector(scorer=_FakeScorer(), chat_caller=chat, modes=["post_grading"],
                      max_cases_per_batch=1)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual((st["ok"], st["off_task"]), (1, 0))
        self.assertIn("continued the task", asked[1])
        rec = json.loads((self.tmp / "reflections" / "fail-near.PATCH.post_grading.json").read_text())
        self.assertIn("off_task_first_reply", rec)
        self.assertIn("Re-read the stated behaviour list", rec["parsed"]["lesson"])
        st = r.reflect(self.tmp / "b", _batch())
        self.assertEqual((st["ok"], st["off_task"]), (0, 1))

    def test_heading_must_start_a_line(self) -> None:
        from meta_agent.reflector import parse_reflection

        text = "1. ESSENTIAL: we keep the tests green\n2. CLOSE CALLS: none\n3. KEEP: run the suite last"
        self.assertEqual(parse_reflection("essential", text)["keep"], "run the suite last")

    def test_solved_case_answers_are_redacted_too(self) -> None:
        from meta_agent.reflector import Reflector, render_reflections

        def chat(msgs, max_tokens, **kw):
            return {"content": f"1. ESSENTIAL: x\n2. CLOSE CALLS: y\n3. KEEP: always run {SECRET_TEST}",
                    "finish_reason": "stop"}

        Reflector(scorer=_FakeScorer(), chat_caller=chat, modes=["essential"]).reflect(self.tmp, _batch())
        text = render_reflections(self.tmp, "lessons_only")
        self.assertIn("always run [redacted]", text)

class ReflectorConfigTests(unittest.TestCase):
    def test_registered_and_off_by_default(self) -> None:
        from meta_agent import registry
        from meta_agent.config import FrameworkConfig, _ensure_builtins_loaded

        _ensure_builtins_loaded()
        self.assertIsNotNone(registry.get("reflector", "default"))
        self.assertIsNone(FrameworkConfig.model_fields["reflector"].default)


class ReflectorHGMWiringTests(unittest.TestCase):
    """Stub evolve: reflections are written for train batches and reach the editor
    steering only when the 'editor' consumer is enabled; none of it without a reflector."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="reflector_hgm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")

    def _run(self, reflector, tag):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

        contexts: list[str] = []

        class Editor(_StubEditor):
            def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False, **kw):
                contexts.append(context or "")
                return super().apply(feedback, base_dir, out_dir, context=context,
                                     has_suggestion=has_suggestion)

        exp = self.tmp / tag
        exp.mkdir()
        m = HGMManager(eval_budget=16, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7)
        m.evolve(editor=Editor(), evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "bench", experiment_dir=exp,
                 max_rounds=10, score_target=None, train_case_ids=[f"c{i}" for i in range(8)],
                 eval_case_ids=None, reflector=reflector)
        return m, contexts

    def test_reflections_flow_to_editor(self) -> None:
        from meta_agent.reflector import Reflector

        seen: list = []
        r = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat(seen),
                      phases=["root", "expand"], consumers=["editor"])
        m, contexts = self._run(r, "on")
        root_refl = list((m._tree[0].round_dir / "reflections").glob("*.json"))
        self.assertTrue(root_refl)
        self.assertTrue(any("Task-agent reflections" in c for c in contexts))
        self.assertFalse(any(SECRET_TEST in c for c in contexts))

    def test_consumer_not_listed_means_no_steering(self) -> None:
        from meta_agent.reflector import Reflector

        r = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat([]),
                      phases=["root", "expand"], consumers=["failure_summarizer"])
        _, contexts = self._run(r, "noeditor")
        self.assertFalse(any("Task-agent reflections" in c for c in contexts))

    def test_no_reflector_is_legacy(self) -> None:
        m, contexts = self._run(None, "off")
        self.assertFalse(any((n.round_dir / "reflections").exists()
                             for n in m._tree.nodes.values()))
        self.assertFalse(any("Task-agent reflections" in c for c in contexts))


class FailureSummarizerConsumerTests(unittest.TestCase):
    def _summ(self, refl):
        from types import SimpleNamespace

        from meta_agent.failure_summarizer import FailureSummarizer

        tmp = Path(tempfile.mkdtemp(prefix="fs_refl_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        prompts = []

        def llm(messages, **kw):
            prompts.append(messages[-1]["content"])
            return SimpleNamespace(content="## Main failure patterns\n- x", tool_calls=[],
                                   stop_reason="stop", raw=None)

        kw = {"reflections": refl} if refl is not None else {}
        FailureSummarizer(llm).summarize(eval_result=_batch(), round_dir=tmp, node_id=1, **kw)
        return prompts[0]

    def test_reflections_appended_only_when_given(self) -> None:
        self.assertIn("own reflections", self._summ("- lesson A"))
        self.assertIn("- lesson A", self._summ("- lesson A"))
        self.assertNotIn("own reflections", self._summ(None))
        self.assertEqual(self._summ(None), self._summ(""))


if __name__ == "__main__":
    unittest.main()

"""Task-agent reflection (meta_agent/reflector.py), two-turn design: one session per
(node, case, role, evaluation) -- a BLIND turn, then a GRADED turn (failed: where/why/
catch/lesson; solved: essential/close calls/keep) plus the node's probe questions --
parsing, per-evaluation records, redaction under ``exposure``, consumer gating, and the
HGM wiring end to end with stub components and a fake LLM (no network).

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

PROBE_ANSWERS = "\n5. PROBE 1: yes, I ran the new checklist before finishing.\n6. PROBE 2: it showed nothing new."


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
        probes = PROBE_ANSWERS if "PROBE 1" in turn else ""
        if "post-grading" in turn:
            return {"content": POST_ANSWER + probes, "finish_reason": "stop"}
        if "NEW TURN: self-assessment" in turn:
            return {"content": UNSURE_ANSWER, "finish_reason": "stop"}
        return {"content": ESSENTIAL_ANSWER + probes, "finish_reason": "stop"}
    return call


def _batch():
    return EvaluationResult(score=0.5, passed=1, failed=2, per_case=[
        CaseResult(case_id="fail-near", passed=False, score=0.9),
        CaseResult(case_id="fail-far", passed=False, score=0.1),
        CaseResult(case_id="ok", passed=True, score=1.0),
        CaseResult(case_id="infra", passed=False, score=0.0, details={"excluded": True}),
        CaseResult(case_id="crashed", passed=False, score=0.0, error="docker died"),
    ])


def _rec(root: Path, name: str) -> dict:
    return json.loads((root / "reflections" / name).read_text())


class ReflectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="reflector_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seen: list = []

    def _make(self, **kw):
        from meta_agent.reflector import Reflector

        return Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat(self.seen), **kw)

    def test_every_usable_case_one_two_turn_session(self) -> None:
        st = self._make().reflect(self.tmp, _batch(), node_id=3, parent_id=1)
        # infra-excluded and errored cases are never reflected on; passed AND failed are.
        self.assertEqual(st["cases"], 3)
        self.assertEqual((st["calls"], st["ok"], st["llm_calls"]), (3, 3, 6))
        files = sorted(p.name for p in (self.tmp / "reflections").glob("*.json"))
        self.assertEqual(files, ["fail-far.PATCH.e1.json", "fail-near.PATCH.e1.json", "ok.PATCH.e1.json"])
        self.assertFalse((self.tmp / "logs").exists())
        rec = _rec(self.tmp, "fail-near.PATCH.e1.json")
        self.assertEqual((rec["node_id"], rec["parent_id"], rec["eval_index"], rec["passed"]), (3, 1, 1, False))
        self.assertEqual([t["name"] for t in rec["turns"]], ["blind", "graded"])
        p = rec["parsed"]
        self.assertEqual(p["overall_confidence"], 55)
        self.assertIn("Re-read the stated behaviour list", p["lesson"])
        self.assertIn("handoff said tests passed", p["why"])
        ok = _rec(self.tmp, "ok.PATCH.e1.json")["parsed"]
        self.assertIn("always run the project's own tests", ok["keep"])
        self.assertIn("nearly missed the rename", ok["close_calls"])
        self.assertNotIn("lesson", ok)

    def test_turn1_blind_turn2_graded_in_the_same_conversation(self) -> None:
        self._make().reflect(self.tmp, _batch())
        failed = [m for m in self.seen if "post-grading" in m[-1]["content"]]
        blind = [m for m in self.seen if "NEW TURN: self-assessment" in m[-1]["content"]]
        self.assertEqual((len(blind), len(failed)), (3, 2))
        for msgs in blind:
            self.assertNotIn(SECRET_TEST, msgs[-1]["content"])
            self.assertNotIn("NOT solved", msgs[-1]["content"])
            self.assertNotIn("PASSED", msgs[-1]["content"])
        g = failed[0]
        # replayed session (4) + blind question + blind answer + graded question
        self.assertEqual(len(g), 7)
        self.assertEqual(g[5], {"role": "assistant", "content": UNSURE_ANSWER})
        self.assertIn(SECRET_TEST, g[-1]["content"])          # full grading goes IN for a failure
        self.assertEqual(g[2]["content"], "")                  # cleaned assistant turn
        self.assertNotIn("extra_junk", g[2])
        passed = [m for m in self.seen if "PASSED" in m[-1]["content"]]
        self.assertEqual(len(passed), 1)
        self.assertNotIn(SECRET_TEST, passed[0][-1]["content"])  # no grader text for a pass

    def test_re_evaluation_gets_a_new_index_never_overwrites(self) -> None:
        r = self._make()
        r.reflect(self.tmp, _batch())
        r.reflect(self.tmp, _batch())
        names = sorted(p.name for p in (self.tmp / "reflections").glob("ok.*"))
        self.assertEqual(names, ["ok.PATCH.e1.json", "ok.PATCH.e2.json"])
        self.assertEqual(_rec(self.tmp, "ok.PATCH.e2.json")["eval_index"], 2)

    def test_probe_questions_asked_in_graded_turn_and_parsed(self) -> None:
        probes = ["Did you run the new checklist?", "What did it show?", "did you run the new checklist?",
                  "x" * 999, "a fourth one"]
        self._make().reflect(self.tmp, _batch(), probes=probes)
        graded = [m[-1]["content"] for m in self.seen if "NEW TURN: self-assessment" not in m[-1]["content"]]
        self.assertTrue(all("PROBE 1: Did you run the new checklist?" in t for t in graded))
        self.assertTrue(all("PROBE 3: " + "x" * 300 + "\n" not in t for t in graded))
        rec = _rec(self.tmp, "fail-near.PATCH.e1.json")
        self.assertEqual(len(rec["probe_questions"]), 3)       # deduped (case-insensitive), capped at 3
        self.assertEqual(len(rec["probe_questions"][2]), 300)  # each capped at 300 chars
        self.assertEqual(rec["parsed"]["probes"][0],
                         {"q": "Did you run the new checklist?", "a": "yes, I ran the new checklist before finishing."})
        for t in [m[-1]["content"] for m in self.seen if "NEW TURN: self-assessment" in m[-1]["content"]]:
            self.assertNotIn("PROBE", t)                          # probes never leak into the blind turn

    def test_probes_off_by_config(self) -> None:
        self._make(probe_questions=False).reflect(self.tmp, _batch(), probes=["Did you X?"])
        self.assertFalse(any("PROBE" in m[-1]["content"] for m in self.seen))

    def test_grading_detail_none_reveals_verdict_only(self) -> None:
        r = self._make(grading_detail="none")
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["ok"], 3)
        self.assertEqual(set(r.scorer.outcome_calls), {"numeric"})  # still fetched for redact terms
        self.assertFalse(any(SECRET_TEST in m[-1]["content"] for m in self.seen))
        self.assertTrue(any("NOT solved" in m[-1]["content"] for m in self.seen))

    def test_modes_select_turns(self) -> None:
        st = self._make(modes=["post_grading"]).reflect(self.tmp, _batch())
        # failed cases: graded turn only; the solved case has nothing to ask -> skipped
        self.assertEqual((st["ok"], st["skipped"], st["llm_calls"]), (2, 1, 2))
        self.assertFalse(any("NEW TURN: self-assessment" in m[-1]["content"] for m in self.seen))
        # without the blind turn, the graded turn carries the role preamble
        self.assertTrue(all(m[-1]["content"].startswith("You are PATCH.") for m in self.seen))

    def test_lessons_only_redacts_and_hides_raw(self) -> None:
        from meta_agent.reflector import render_reflections

        self._make().reflect(self.tmp, _batch(), probes=["Did you run the new checklist?"])
        text = render_reflections(self.tmp, "lessons_only")
        self.assertIn("Re-read the stated behaviour list", text)
        self.assertIn("always run the project's own tests", text)
        self.assertIn("behaviour when the config file is missing", text)  # unsure < 70
        self.assertNotIn("error message wording", text)                   # 85: confident
        self.assertIn("ran the new checklist", text)                      # probe answers
        self.assertNotIn(SECRET_TEST, text)
        self.assertNotIn(SECRET_PATH, text)
        self.assertNotIn("WHERE", text)  # raw answers never shown
        self.assertEqual(render_reflections(self.tmp, "off"), "")
        full = render_reflections(self.tmp, "full")
        self.assertIn("WHERE", full)
        self.assertNotIn(SECRET_TEST, full)  # redact applies under full too

    def test_legacy_mode_records_still_render(self) -> None:
        from meta_agent.reflector import render_reflections

        d = self.tmp / "reflections"
        d.mkdir()
        (d / "a.PATCH.post_grading.json").write_text(json.dumps({
            "case_id": "a", "role": "PATCH", "mode": "post_grading", "passed": False,
            "parsed": {"lesson": f"old lesson about {SECRET_TEST}", "catch": "c"},
            "redact_terms": [SECRET_TEST], "response": {"content": "x"}}))
        text = render_reflections(self.tmp, "lessons_only")
        self.assertIn("old lesson about [redacted]", text)

    def test_parse_unsure_and_probe_sections(self) -> None:
        from meta_agent.reflector import parse_reflection

        p = parse_reflection("unsure", UNSURE_ANSWER)
        self.assertEqual(p["overall_confidence"], 55)
        self.assertEqual([i["confidence"] for i in p["items"]], [40, 85, 20])
        q = parse_reflection("essential", ESSENTIAL_ANSWER + PROBE_ANSWERS, 2)
        self.assertEqual(q["probe_answers"], ["yes, I ran the new checklist before finishing.",
                                              "it showed nothing new."])
        self.assertIn("own tests after the last edit", q["keep"])  # KEEP ends at PROBE 1
        j = parse_reflection("post_grading", '{"general_lesson": "L", "probe_1": "A1"}', 1)
        self.assertEqual((j["lesson"], j["probe_answers"]), ("L", ["A1"]))

    def test_selection_cap_prefers_near_misses(self) -> None:
        r = self._make(modes=["post_grading"], max_cases_per_batch=1)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["cases"], 1)
        self.assertTrue((self.tmp / "reflections" / "fail-near.PATCH.e1.json").exists())

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
        r = self._make(max_cases_per_batch=1, ctx_tokens=100, max_output_tokens=90)
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

        r = Reflector(scorer=_FakeScorer(), chat_caller=boom)
        st = r.reflect(self.tmp, _batch())
        self.assertEqual(st["errors"], 3)
        rec = _rec(self.tmp, "ok.PATCH.e1.json")
        self.assertIn("server down", rec["error"])
        self.assertEqual(rec["status"], "errors")

    def test_phase_gating_and_bad_config(self) -> None:
        r = self._make()
        self.assertEqual(r.reflect(self.tmp, _batch(), phase="root")["calls"], 0)
        from meta_agent.reflector import Reflector

        for bad in ({"modes": ["nope"]}, {"exposure": "x"}, {"grading_detail": "x"},
                    {"consumers": ["x"]}, {"phases": ["x"]}, {"max_cases_per_batch": 0}):
            with self.assertRaises(ValueError):
                Reflector(**bad)

    def test_defaults_come_from_task_agent(self) -> None:
        from meta_agent.config import TaskAgentSpec
        from meta_agent.reflector import Reflector

        ta = TaskAgentSpec(model="m-x", base_url="http://h:1/v1", reasoning_effort="medium")
        r = Reflector(task_agent=ta)
        self.assertEqual((r.model, r.base_url, r.reasoning_effort), ("m-x", "http://h:1/v1", "medium"))
        self.assertIsNone(r.max_cases_per_batch)  # all cases by default

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
        self.assertEqual((st["ok"], st["off_task"], st["llm_calls"]), (1, 0, 2))
        self.assertIn("continued the task", asked[1])
        rec = _rec(self.tmp, "fail-near.PATCH.e1.json")
        self.assertIn("off_task_first_reply", rec["turns"][0]["response"])
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


class ProbeQuestionEditorTests(unittest.TestCase):
    def test_strategy_field_and_both_editor_parse_paths(self) -> None:
        from meta_agent.agent_editor import (AGENTIC_SUBMIT_SUMMARY_TOOL, SELF_IMPROVEMENT_TOOL,
                                             AgentEditor)
        from meta_agent.models import EvolutionStrategy

        self.assertEqual(EvolutionStrategy(optimization_goal="g", proposed_changes="p").probe_questions, [])
        for tool in (SELF_IMPROVEMENT_TOOL, AGENTIC_SUBMIT_SUMMARY_TOOL):
            props = tool["input_schema"]["properties"]
            self.assertEqual(props["probe_questions"]["type"], "array")
            self.assertNotIn("probe_questions", tool["input_schema"]["required"])
        ed = AgentEditor.__new__(AgentEditor)
        ed.mutable_exclude = []
        strat, _ = ed._parse_self_improvement({
            "optimization_goal": "g", "proposed_changes": "p", "files": [],
            "probe_questions": ["Q1?", "  Q1?  ", "Q2?", "Q3?", "Q4?"]})
        self.assertEqual(strat.probe_questions, ["Q1?", "Q2?", "Q3?"])
        strat, _ = ed._parse_self_improvement({"optimization_goal": "g", "proposed_changes": "p",
                                               "files": [], "probe_questions": "only one?"})
        self.assertEqual(strat.probe_questions, ["only one?"])
        strat, _ = ed._parse_self_improvement({"optimization_goal": "g", "proposed_changes": "p",
                                               "files": [], "probe_questions": {"bad": 1}})
        self.assertEqual(strat.probe_questions, [])


class ReflectorConfigTests(unittest.TestCase):
    def test_registered_and_off_by_default(self) -> None:
        from meta_agent import registry
        from meta_agent.config import FrameworkConfig, _ensure_builtins_loaded

        _ensure_builtins_loaded()
        self.assertIsNotNone(registry.get("reflector", "default"))
        self.assertIsNone(FrameworkConfig.model_fields["reflector"].default)


class ReflectorHGMWiringTests(unittest.TestCase):
    """Stub evolve: reflections are written for train batches, linked to their node, the
    per-test-case files are built, and the editor sees reflections only when the 'editor'
    consumer is enabled; none of it without a reflector."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="reflector_hgm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.seed = self.tmp / "seed"
        self.seed.mkdir()
        (self.seed / "workflow.py").write_text("def run_task(task):\n    return None\n")

    def _run(self, reflector, tag, probes=None, failure_summarizer=None):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager
        from tests.test_hgm_smoke import _StubEditor, _StubEvaluator

        contexts: list[str] = []

        class Editor(_StubEditor):
            def apply(self, feedback, base_dir, out_dir, *, context=None, has_suggestion=False, **kw):
                contexts.append(context or "")
                res = super().apply(feedback, base_dir, out_dir, context=context,
                                    has_suggestion=has_suggestion)
                if probes and res.strategy is not None:
                    res.strategy.probe_questions = list(probes)
                return res

        exp = self.tmp / tag
        exp.mkdir()
        m = HGMManager(eval_budget=16, init_expansions=2, eval_batch_size=4, alpha=0.6, seed=7)
        m.evolve(editor=Editor(), evaluator=_StubEvaluator(), gatherer=DefaultFeedbackGatherer(),
                 seed_dir=self.seed, benchmark_dir=self.tmp / "bench", experiment_dir=exp,
                 max_rounds=10, score_target=None, train_case_ids=[f"c{i}" for i in range(8)],
                 eval_case_ids=None, reflector=reflector,
                 **({"failure_summarizer": failure_summarizer} if failure_summarizer else {}))
        return m, contexts, exp

    def test_reflections_flow_to_editor_and_case_files(self) -> None:
        from meta_agent.reflector import Reflector

        seen: list = []
        r = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat(seen),
                      phases=["root", "expand"], consumers=["editor"])
        m, contexts, exp = self._run(r, "on", probes=["Did you run the new checklist?"])
        root_refl = list((m._tree[0].round_dir / "reflections").glob("*.json"))
        self.assertTrue(root_refl)
        self.assertTrue(any("Task-agent reflections" in c for c in contexts))
        self.assertTrue(any("probe_questions" in c for c in contexts))   # editor told it may write probes
        self.assertFalse(any(SECRET_TEST in c for c in contexts))
        # every record is linked to the node whose round dir holds it
        for n in m._tree.nodes.values():
            for f in (n.round_dir / "reflections").glob("*.json"):
                rec = json.loads(f.read_text())
                self.assertEqual((rec["node_id"], rec["parent_id"]), (n.node_id, n.parent_id))
                if n.node_id != 0:
                    self.assertEqual(rec["probe_questions"], ["Did you run the new checklist?"])
                else:
                    self.assertEqual(rec["probe_questions"], [])
        cases = exp / "case_reflections"
        self.assertTrue((cases / "INDEX.md").is_file())
        texts = {f.name: f.read_text() for f in cases.glob("*.md")}
        self.assertFalse(any(SECRET_TEST in t or SECRET_PATH in t for t in texts.values()))
        multi = [t for k, t in texts.items() if k != "INDEX.md" and t.count("### node ") >= 2]
        self.assertTrue(multi, "some case was reflected on by more than one node")

    def test_failure_summarizer_gets_cross_node_pass_rates(self) -> None:
        from meta_agent.reflector import Reflector

        got: list[str] = []

        class FS:
            def summarize(self, *, eval_result, round_dir, node_id, reflections=""):
                got.append(reflections)

        r = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat([]), phases=["root", "expand"],
                      consumers=["failure_summarizer"])
        self._run(r, "fs", failure_summarizer=FS())
        cross = [g for g in got if "across ALL nodes" in g]
        self.assertTrue(cross)
        self.assertTrue(any("pass rate:" in g for g in cross))
        self.assertFalse(any(SECRET_TEST in g for g in got))

    def test_consumer_not_listed_means_no_steering(self) -> None:
        from meta_agent.reflector import Reflector

        r = Reflector(scorer=_FakeScorer(), chat_caller=_fake_chat([]),
                      phases=["root", "expand"], consumers=["failure_summarizer"])
        _, contexts, _ = self._run(r, "noeditor")
        self.assertFalse(any("Task-agent reflections" in c for c in contexts))

    def test_no_reflector_is_legacy(self) -> None:
        m, contexts, exp = self._run(None, "off")
        self.assertFalse(any((n.round_dir / "reflections").exists()
                             for n in m._tree.nodes.values()))
        self.assertFalse(any("Task-agent reflections" in c for c in contexts))
        self.assertFalse(any("probe_questions" in c for c in contexts))
        self.assertFalse((exp / "case_reflections").exists())


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

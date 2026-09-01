"""Unit tests for the deterministic usage-capture layer (edit_usage.py):
surface detection, idempotent trace consumption, and the no-data-vs-zero
rendering rule. No LLM, no network.

    PYTHONPATH=. python3 -m unittest tests.test_edit_usage
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.edit_usage import (
    _dyn_eq, added_surface, agreement_counts, analysis_sig,
    build_analysis_prompt, consume_trace, ensure_store, load_store,
    render_analysis, usage_lines,
)


def _agent(round_dir: Path, workflow: str, tools_schema: str = "[]") -> None:
    a = round_dir / "task_agent"
    (a / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (a / "workflow.py").write_text(workflow, encoding="utf-8")
    (a / "tool_wrapper.py").write_text("def x(): return None\n", encoding="utf-8")
    (a / "tools_schema.json").write_text(tools_schema, encoding="utf-8")
    (a / "mutable_tools" / "__init__.py").write_text("", encoding="utf-8")


def _trace_line(kind: str, **payload) -> str:
    return json.dumps({"kind": kind, "payload": payload}) + "\n"


class TestSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p, self.c = self.tmp / "round_000", self.tmp / "round_001"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_tool_file_and_schema_entry_detected(self):
        _agent(self.p, "x = 1\n")
        _agent(self.c, "x = 2\n",
               tools_schema='[{"function": {"name": "select_route"}}]')
        (self.c / "task_agent" / "mutable_tools" / "helper.py").write_text(
            "def run():\n    return 1\n", encoding="utf-8")
        s = added_surface(self.p, self.c)
        self.assertEqual(s["tools"], ["helper", "select_route"])
        self.assertEqual(s["removed_tools"], [])

    def test_aliased_import_and_label_kwarg_are_found(self):
        # Editors routinely alias the logger and pass label= as a kwarg —
        # both were missed by a bare `log('...` regex in the first retrofit.
        _agent(self.p, "x = 1\n")
        _agent(self.c,
               "from platform_core.trace import log as trace_log\n"
               "def run_task(t):\n"
               "    trace_log(\n"
               "        label='plan_check',\n"
               "        verdict='pass',\n"
               "        name='overall',\n"
               "    )\n")
        s = added_surface(self.p, self.c)
        self.assertEqual(s["labels"],
                         [{"label": "plan_check", "name": "overall"}])

    def test_inherited_pair_excluded_but_new_name_under_old_label_counts(self):
        base = ("from platform_core.trace import log\n"
                "def run_task(t):\n"
                "    log('check', verdict='pass', name='old')\n")
        _agent(self.p, base)
        _agent(self.c, base + "    log('check', verdict='pass', name='new')\n")
        s = added_surface(self.p, self.c)
        self.assertEqual(s["labels"], [{"label": "check", "name": "new"}])
        self.assertNotIn({"label": "check", "name": "old"}, s["labels"])

    def test_fstring_name_matches_runtime_values(self):
        self.assertTrue(_dyn_eq("tool_{tc.name}", "tool_query_route"))
        self.assertFalse(_dyn_eq("tool_{tc.name}_error", "tool_query_route"))
        self.assertTrue(_dyn_eq("plain", "plain"))
        self.assertFalse(_dyn_eq("plain", "other"))
        self.assertTrue(_dyn_eq(None, "anything"))


class TestConsume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p, self.c = self.tmp / "round_000", self.tmp / "round_001"
        _agent(self.p, "x = 1\n")
        _agent(self.c,
               "from platform_core.trace import log\n"
               "def run_task(t):\n"
               "    log('check', verdict='pass', name='main')\n")
        (self.c / "task_agent" / "mutable_tools" / "unused_tool.py").write_text(
            "def run():\n    return 1\n", encoding="utf-8")
        self.store = ensure_store(self.c, self.p, 1, 0)
        self.trace = self.c / "logs" / "trace.jsonl"
        self.trace.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_trace_consumed_once(self):
        self.trace.write_text(
            _trace_line("mutable_log", label="check", verdict="pass",
                        name="main", case_id="a")
            + _trace_line("tool_call", name="query_route", case_id="a"),
            encoding="utf-8")
        self.assertTrue(consume_trace(self.store, self.trace))
        self.assertFalse(consume_trace(self.store, self.trace))  # same bytes
        self.assertEqual(len(self.store["batches"]), 1)
        self.assertEqual(self.store["tools"]["query_route"]["calls"], 1)
        self.assertEqual(len(self.store["events"]), 1)

    def test_distinct_batches_accumulate(self):
        self.trace.write_text(
            _trace_line("mutable_log", label="check", verdict="pass",
                        name="main", case_id="a"), encoding="utf-8")
        consume_trace(self.store, self.trace)
        self.trace.write_text(
            _trace_line("mutable_log", label="check", verdict="fail",
                        name="main", case_id="b"), encoding="utf-8")
        consume_trace(self.store, self.trace)
        self.assertEqual(len(self.store["batches"]), 2)
        self.assertEqual(len(self.store["events"]), 2)
        self.assertEqual(self.store["case_ids"], ["a", "b"])

    def test_missing_or_empty_trace_is_a_noop(self):
        self.assertFalse(consume_trace(self.store, self.trace))  # absent
        self.trace.write_text("", encoding="utf-8")
        self.assertFalse(consume_trace(self.store, self.trace))  # empty
        self.assertEqual(self.store["batches"], [])

    def test_event_cap_keeps_verdict_diversity_and_flags(self):
        lines = [_trace_line("mutable_log", label="chatty", verdict="pass",
                             name="x", case_id=str(i)) for i in range(50)]
        lines.append(_trace_line("mutable_log", label="rare", verdict="fail",
                                 name="y", case_id="z"))
        self.trace.write_text("".join(lines), encoding="utf-8")
        consume_trace(self.store, self.trace, max_events=10)
        self.assertTrue(self.store["events_truncated"])
        self.assertLessEqual(len(self.store["events"]), 10)
        # the rare group survives the chatty one
        self.assertTrue(any(e["label"] == "rare" for e in self.store["events"]))

    def test_usage_lines_no_data_vs_zero(self):
        # No consumed batch -> NOTHING rendered (absence of data is not "0").
        self.assertEqual(usage_lines(self.store), [])
        # A consumed batch where the surface tool never ran -> explicit 0.
        self.trace.write_text(
            _trace_line("mutable_log", label="check", verdict="pass",
                        name="main", case_id="a"), encoding="utf-8")
        consume_trace(self.store, self.trace)
        lines = "\n".join(usage_lines(self.store))
        self.assertIn("`unused_tool` **0 calls**", lines)
        self.assertIn("new log point `check/main`**: fired 1x (pass 1)", lines)

    def test_store_roundtrips_through_disk(self):
        self.trace.write_text(
            _trace_line("tool_call", name="q", case_id="a"), encoding="utf-8")
        consume_trace(self.store, self.trace)
        from meta_agent.edit_usage import save_store
        save_store(self.c, self.store)
        again = load_store(self.c)
        self.assertEqual(again["tools"], self.store["tools"])
        self.assertEqual(again["batches"], self.store["batches"])


class TestAnalysisHelpers(unittest.TestCase):
    def test_analysis_sig_tracks_own_evidence_only(self):
        a = analysis_sig("sig1", ["b1"])
        self.assertEqual(a, analysis_sig("sig1", ["b1"]))
        self.assertNotEqual(a, analysis_sig("sig1", ["b1", "b2"]))
        self.assertNotEqual(a, analysis_sig("sig2", ["b1"]))
        # bumping the question-set version invalidates every cached analysis
        # exactly once (ANALYSIS_VERSION is salted into the hash)
        self.assertNotEqual(a, analysis_sig("sig1", ["b1"], version=99))

    def test_render_analysis_is_evidence_only(self):
        md = render_analysis({
            "components": [{"component": "check/main", "activated": "3x",
                            "verdict_behavior": "pass 2 / fail 1",
                            "agreement": "case a: pass but scorer failed x",
                            "cause": "lookup misses venue names with commas"}],
            "targeted_constraints": [
                {"constraint": "x", "remaining_failures": "12/58",
                 "was": "19/58", "evidence": "cases a, b"},
                # model stuffed the context into the field itself -> the
                # renderer must not append a second "(was ...)"
                {"constraint": "y", "remaining_failures": "5/9 (was 9/9, +4)",
                 "was": "9/9"}],
            "collateral": "z 0->2 fails (-2)"})
        self.assertTrue(md.startswith("## Analysis"))
        self.assertIn("`check/main`", md)
        self.assertIn("likely cause: lookup misses", md)
        self.assertIn("remaining 12/58 (was 19/58, +7) — cases a, b", md)
        self.assertNotIn("(was 9/9, +4) (was", md)
        self.assertIn("**collateral**: z 0->2 fails (-2)", md)
        # no judgment labels anywhere
        self.assertNotIn("assessment", md)
        self.assertNotIn("confidence", md)
        self.assertNotIn("**overall**", md)
        # pre-v2 cached payloads still render without raising
        md2 = render_analysis({"components": [{"component": "c",
                                               "activated": "1x",
                                               "verdict_behavior": "pass",
                                               "agreement": "?",
                                               "assessment": "ineffective"}],
                               "constraint_effect": "helped x"})
        self.assertIn("likely cause: ineffective", md2)
        self.assertIn("**collateral**: helped x", md2)
        # malformed component entries are skipped, never raise
        md3 = render_analysis({"components": [{}, "junk"]})
        self.assertIn("**collateral**: ?", md3)

    def test_build_analysis_prompt_has_absolute_table_and_diff(self):
        from types import SimpleNamespace
        oc = SimpleNamespace(per_check={"a:x": (19, 12)}, n_check_cases=58,
                             n_shared=60, child_mean_shared=0.72,
                             parent_mean_shared=0.63, delta_shared=0.09,
                             per_check_delta={"a:x": 7})
        prompt = build_analysis_prompt(
            node_id=7, record_body="## Edit 1\n- **what**: w",
            outcome=oc, store={"events": [], "surface": {}}, u_lines=[],
            parent_cases=[], child_cases=[], recipe=None,
            code_diff="--- parent/workflow.py\n+++ child/workflow.py\n+x = 1")
        self.assertIn("`a:x` 19->12 fails / 58 cases (+7)", prompt)
        self.assertIn("# Code diff vs parent", prompt)
        self.assertIn("+x = 1", prompt)
        self.assertIn("child score: 0.7200 over 60 evaluated cases; vs parent on 60 shared", prompt)

    def test_uninstrumented_edit_guards(self):
        from types import SimpleNamespace
        oc = SimpleNamespace(per_check={"a:x": (2, 1)}, n_check_cases=2,
                             n_shared=2, child_mean_shared=0.8,
                             parent_mean_shared=0.6, delta_shared=0.2,
                             per_check_delta={"a:x": 1})
        recipe = {"mode": "list", "path": "details.failed_checks"}
        parent = [{"case_id": "a", "score": 0.5, "passed": False,
                   "details": {"failed_checks": ["a:x"]}},
                  {"case_id": "b", "score": 0.7, "passed": False,
                   "details": {"failed_checks": ["a:x"]}}]
        child = [{"case_id": "a", "score": 1.0, "passed": True,
                  "details": {"failed_checks": []}},
                 {"case_id": "b", "score": 0.7, "passed": False,
                  "details": {"failed_checks": ["a:x"]}}]
        prompt = build_analysis_prompt(
            node_id=4, record_body="## Edit 1\n- **what**: w",
            outcome=oc, store={"events": [], "surface": {}}, u_lines=[],
            parent_cases=parent, child_cases=child, recipe=recipe,
            code_diff="+def _verify(): ...")
        # guard 1: explicit no-inference instruction when no events exist
        self.assertIn("report every component's \"agreement\" as unmeasured",
                      prompt)
        # guard 2: ground truth falls back to cases where checks moved —
        # case "a" (x fixed) is included, unmoved case "b" is not
        self.assertIn("fallback: this edit produced no runtime events", prompt)
        self.assertIn('"case_id": "a"', prompt)
        self.assertNotIn('"case_id": "b"', prompt)
        # with events present, neither guard text appears
        ev = [{"label": "chk", "name": "m", "verdict": "pass", "case_id": "a"}]
        prompt2 = build_analysis_prompt(
            node_id=4, record_body="x", outcome=oc,
            store={"events": ev, "surface": {}}, u_lines=[],
            parent_cases=parent, child_cases=child, recipe=recipe)
        self.assertNotIn("do not infer firing behavior", prompt2)
        self.assertIn("cases the components touched", prompt2)


if __name__ == "__main__":
    unittest.main()


class TestRoundThree(unittest.TestCase):
    CASES = [{"case_id": "a", "passed": True, "score": 1.0},
             {"case_id": "b", "passed": False, "score": 0.4},
             {"case_id": "c", "passed": False, "score": 0.2}]

    def test_agreement_counts_joins_and_ignores_unknown(self):
        ev = [{"label": "g", "verdict": "pass", "case_id": "a"},
              {"label": "g", "verdict": "pass", "case_id": "b"},
              {"label": "g", "verdict": "pass", "case_id": "b"},   # dedup case
              {"label": "g", "verdict": "fail", "case_id": "c"},
              {"label": "g", "verdict": "pass", "case_id": "zz"},  # unknown
              {"label": "g", "verdict": "pass"}]                   # no case
        got = agreement_counts(ev, self.CASES)
        self.assertEqual(got["pass"], (2, 1, 1))
        self.assertEqual(got["fail"], (1, 0, 1))

    def test_usage_lines_cross_tab_and_suspect(self):
        store = {"batches": ["x"], "case_ids": ["a", "b", "c"],
                 "surface": {"tools": [], "labels": [
                     {"label": "gate", "name": None}]},
                 "tools": {},
                 "events": [
                     {"label": "gate", "verdict": "pass", "case_id": "b"},
                     {"label": "gate", "verdict": "pass", "case_id": "c"}]}
        plain = usage_lines(store)
        self.assertNotIn("scorer on those cases", "\n".join(plain))
        rich = "\n".join(usage_lines(store, child_cases=self.CASES))
        self.assertIn("-> scorer on those cases: 0 pass / 2 fail", rich)
        self.assertIn("SUSPECT VERIFIER", rich)  # passes land on scorer fails
        # majority scorer-pass -> no flag
        store["events"] = [{"label": "gate", "verdict": "pass", "case_id": "a"}]
        rich2 = "\n".join(usage_lines(store, child_cases=self.CASES))
        self.assertIn("1 pass / 0 fail", rich2)
        self.assertNotIn("SUSPECT", rich2)

    def test_render_analysis_role_multiline_no_generalization(self):
        # A cached v3 payload still carries `generalization`; it must render
        # without that line — the seen-vs-unseen split lives only in Outcome.
        md = render_analysis({
            "components": [{"component": "v", "role": "detector",
                            "activated": "5x", "verdict_behavior": "fail 5",
                            "agreement": "cases a,b confirmed",
                            "cause": "line1\nline2\nline3\nline4-dropped"}],
            "targeted_constraints": [],
            "collateral": "none observed",
            "generalization": "gains concentrated on seen cases (diff line 12)"})
        self.assertIn("(detector)", md)
        self.assertEqual(md.count("likely cause:"), 3)   # capped at 3
        self.assertNotIn("line4-dropped", md)
        self.assertNotIn("generalization", md)

    def test_prompt_has_no_generalization_section(self):
        from types import SimpleNamespace
        oc = SimpleNamespace(per_check={}, n_check_cases=0, n_shared=10,
                             child_mean_shared=0.7, parent_mean_shared=0.6,
                             delta_shared=0.1, per_check_delta={},
                             child_mean_all=0.7, child_n_all=10)
        base = dict(node_id=1, record_body="b", outcome=oc,
                    store={"events": [], "surface": {}}, u_lines=[],
                    parent_cases=[], child_cases=[], recipe=None)
        self.assertNotIn("# Generalization", build_analysis_prompt(**base))

    def test_ensure_store_captures_seen_only_at_record_time(self):
        import json as _json
        tmp = Path(tempfile.mkdtemp())
        try:
            par, ch = tmp / "round_000", tmp / "round_001"
            for d in (par, ch):
                (d / "task_agent" / "mutable_tools").mkdir(parents=True)
                (d / "task_agent" / "workflow.py").write_text("x = 1\n")
                (d / "task_agent" / "tools_schema.json").write_text("[]")
            (par / "eval_result.json").write_text(_json.dumps(
                {"per_case": [{"case_id": "7"}, {"case_id": "3"}]}))
            st = ensure_store(ch, par, 1, 0)
            self.assertEqual(st["seen_case_ids"], ["3", "7"])
            # lazy path (resumed run) must not fabricate a seen set
            ch2 = tmp / "round_002"
            (ch2 / "task_agent").mkdir(parents=True)
            (ch2 / "task_agent" / "workflow.py").write_text("x = 2\n")
            (ch2 / "task_agent" / "tools_schema.json").write_text("[]")
            st2 = ensure_store(ch2, par, 2, 0, capture_seen=False)
            self.assertEqual(st2["seen_case_ids"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

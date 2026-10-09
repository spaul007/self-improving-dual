"""Per-test-case reflection files (meta_agent/case_reflections.py) and their read path
(``cases/`` in meta_agent/log_access.py, the editor and the block suggester).

    PYTHONPATH=. python3 -m pytest tests/test_case_reflections.py -q
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.managers.hgm_tree import HGMNode
from meta_agent.models import CaseResult

SECRET = "test_hidden_grader_name"


def _rec(node, parent, case, role, k, passed, score, *, lesson="", keep="", conf=None, probes=(),
         terms=(), ts=0.0):
    parsed = {"overall_confidence": conf, "unsure_items": [{"confidence": 30, "item": f"edge of {case}"}]}
    if lesson:
        parsed["lesson"] = lesson
    if keep:
        parsed["keep"] = keep
    if probes:
        parsed["probes"] = [{"q": q, "a": a} for q, a in probes]
    return {"version": 2, "node_id": node, "parent_id": parent, "case_id": case, "role": role,
            "eval_index": k, "passed": passed, "score": score, "ts": ts, "status": "ok",
            "redact_terms": list(terms), "parsed": parsed, "probe_questions": [q for q, _ in probes],
            "turns": [{"name": "blind", "question": "q", "response": {"content": f"FULL blind {SECRET}"}},
                      {"name": "graded", "question": "q", "response": {"content": "FULL graded"}}]}


class CaseFileBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = Path(tempfile.mkdtemp(prefix="casefiles_"))
        self.addCleanup(shutil.rmtree, self.run, True)
        self.nodes = []
        for nid, parent, goal in ((0, None, "Seed agent."), (1, 0, "add a checklist"), (2, 1, f"fix {SECRET} | pipe")):
            rd = self.run / f"round_{nid:03d}"
            (rd / "reflections").mkdir(parents=True)
            (rd / "strategy.json").write_text(json.dumps(
                {"optimization_goal": goal, "proposed_changes": "", "block": None if nid == 0 else "prompts"}))
            self.nodes.append(HGMNode(node_id=nid, parent_id=parent, round_dir=rd))
        n0, n1, n2 = self.nodes
        # case A: fails on root, passes on 1, evaluated twice on 2 (fail then pass), plus an
        # infra-excluded eval on 1 that must NOT count.
        for node, cr in ((n0, CaseResult(case_id="A", passed=False, score=0.4)),
                         (n1, CaseResult(case_id="A", passed=True, score=1.0)),
                         (n1, CaseResult(case_id="A", passed=False, score=0.0, details={"excluded": True})),
                         (n2, CaseResult(case_id="A", passed=False, score=0.6)),
                         (n2, CaseResult(case_id="A", passed=True, score=1.0)),
                         (n0, CaseResult(case_id="B/x y", passed=True, score=1.0)),
                         (n0, CaseResult(case_id="C", passed=False, score=0.0, error="docker died"))):
            node.case_results.append(cr)
        recs = [
            (n0, _rec(0, None, "A", "PATCH", 1, False, 0.4, lesson=f"never trust {SECRET}", conf=80,
                      terms=[SECRET], ts=1)),
            (n0, _rec(0, None, "A", "VERIFY", 1, False, 0.4, lesson="check the brief", conf=90, ts=1)),
            (n1, _rec(1, 0, "A", "PATCH", 1, True, 1.0, keep="run the suite", conf=70,
                      probes=[("Did you use the checklist?", "yes, step 3")], ts=2)),
            (n2, _rec(2, 1, "A", "PATCH", 1, False, 0.6, lesson="read the interface", conf=60, ts=3)),
            (n2, _rec(2, 1, "A", "PATCH", 2, True, 1.0, keep="keep the interface read", conf=75, ts=4)),
            (n0, _rec(0, None, "B/x y", "PATCH", 1, True, 1.0, keep="quote the spec", conf=95, ts=1)),
        ]
        for node, r in recs:
            name = f"{r['case_id'].replace('/', '_').replace(' ', '_')}.{r['role']}.e{r['eval_index']}.json"
            (node.round_dir / "reflections" / name).write_text(json.dumps(r))

    def _build(self, **kw):
        from meta_agent.case_reflections import build_case_files

        return build_case_files(self.run, self.nodes, **kw)

    def test_one_file_per_case_linked_to_nodes_with_pass_rates(self) -> None:
        from meta_agent.case_reflections import case_file_name

        summary = self._build()
        out = self.run / "case_reflections"
        self.assertEqual(sorted(p.name for p in out.glob("*.md")),
                         sorted(["INDEX.md", "A.md", case_file_name("B/x y"), "C.md"]))
        self.assertEqual((summary["A"]["evals"], summary["A"]["passes"], summary["A"]["nodes"]), (4, 2, 3))
        self.assertEqual(summary["C"]["evals"], 0)  # errored eval: not a model zero
        a = (out / "A.md").read_text()
        self.assertIn("pass rate: 2/4 evaluations (50%) across 3 node(s)", a)
        self.assertIn("calibration: mean blind confidence 75/100 over 5 reflection(s) vs actual pass rate 50%", a)
        # node table: parent, depth, block, goal; pipes in a goal cannot break the table
        self.assertIn("| 2 | 1 | 2 | prompts | fix [redacted] / pipe | 2 | 1 | 0.60, 1.00 |", a)
        self.assertIn("| 1 | 0 | 1 | prompts | add a checklist | 1 | 1 | 1.00 |", a)
        # both outcomes, each block linked to its node and eval index, roles grouped
        self.assertIn("### node 0 (parent -) · eval 1 · score 0.40 · FAILED\n**PATCH**", a)
        self.assertIn("**VERIFY**", a)
        self.assertIn("### node 1 (parent 0) · eval 1 · score 1.00 · PASSED", a)
        self.assertIn("### node 2 (parent 1) · eval 2 · score 1.00 · PASSED", a)
        self.assertIn("keep: keep the interface read", a)
        self.assertIn("probe questions (written by the editor for this node's edit): [1] Did you use the checklist?", a)
        self.assertIn("probe 1: yes, step 3", a)
        self.assertLess(a.index("### node 0"), a.index("### node 1"))       # oldest first
        self.assertLess(a.index("· eval 1 · score 0.60"), a.index("· eval 2 ·"))
        idx = (out / "INDEX.md").read_text()
        self.assertLess(idx.index("| A |"), idx.index("| B/x y |"))       # hardest first

    def test_no_redact_term_anywhere_including_union_across_records(self) -> None:
        self._build()
        for f in (self.run / "case_reflections").glob("*.md"):
            self.assertNotIn(SECRET, f.read_text(), f.name)
        self._build(exposure="full")
        a = (self.run / "case_reflections" / "A.md").read_text()
        self.assertIn("FULL graded", a)
        # the SECRET is only listed on node 0's record, but node 1/2's full text is redacted too
        self.assertNotIn(SECRET, a)

    def test_exposure_off_keeps_pass_rates_only(self) -> None:
        self._build(exposure="off")
        a = (self.run / "case_reflections" / "A.md").read_text()
        self.assertIn("pass rate: 2/4", a)
        self.assertNotIn("### node", a)
        self.assertNotIn("lesson", a)

    def test_cap_drops_oldest_blocks_keeps_header(self) -> None:
        self._build(max_chars_per_case=1200)
        a = (self.run / "case_reflections" / "A.md").read_text()
        self.assertIn("pass rate: 2/4", a)
        self.assertIn("older reflection block(s) omitted", a)
        self.assertIn("· eval 2 ·", a)              # newest kept
        self.assertNotIn("check the brief", a)      # oldest dropped

    def test_fields_clipped_after_redaction(self) -> None:
        from meta_agent.reflector import render_record

        long = "x" * 590 + SECRET + " tail " * 50     # the term straddles the 600-char cut
        rec = _rec(0, None, "A", "PATCH", 1, False, 0.4, lesson=long, terms=[SECRET])
        lines = render_record(rec, "lessons_only", None, 600)
        lesson = next(x for x in lines if x.startswith("lesson: "))[len("lesson: "):]
        self.assertLessEqual(len(lesson), 600)
        self.assertTrue(lesson.endswith("\u2026"))
        self.assertNotIn(SECRET[:8], lesson)   # no half-term survives the cut
        self.assertIn("[redac", lesson)
        unclipped = render_record(rec, "lessons_only", None, None)
        self.assertIn(" tail", next(x for x in unclipped if x.startswith("lesson: ")))

    def test_defaults_keep_many_nodes_per_case(self) -> None:
        from meta_agent.case_reflections import build_case_files

        big = "word " * 2000
        n0 = self.nodes[0]
        for k in range(2, 12):   # 10 more evaluations of A on the root, each with long answers
            n0.case_results.append(CaseResult(case_id="A", passed=False, score=0.3))
            r = _rec(0, None, "A", "PATCH", k, False, 0.3, lesson=big, keep=big, conf=50, ts=10 + k,
                     probes=[("q?", big)])
            (n0.round_dir / "reflections" / f"A.PATCH.e{k}.json").write_text(json.dumps(r))
        build_case_files(self.run, self.nodes)
        a = (self.run / "case_reflections" / "A.md").read_text()
        self.assertGreaterEqual(a.count("### node "), 12)   # nothing dropped at the 60K default
        self.assertIn("pass rate: 2/14", a)

    def test_rebuild_is_pure_and_removes_stale_files(self) -> None:
        self._build()
        out = self.run / "case_reflections"
        first = {f.name: f.read_text() for f in out.glob("*.md")}
        (out / "stale.md").write_text("old")
        self._build()
        self.assertEqual({f.name: f.read_text() for f in out.glob("*.md")}, first)

    def test_excerpt_for_failure_summarizer(self) -> None:
        from meta_agent.case_reflections import excerpt_for_cases

        self._build()
        text = excerpt_for_cases(self.run, ["A", "missing"], 1500)
        self.assertIn("pass rate: 2/4", text)
        self.assertIn("### node 2", text)            # newest block
        self.assertNotIn("Evaluations by node", text)
        self.assertLessEqual(len(text), 1500)
        self.assertEqual(excerpt_for_cases(self.run, []), "")


class CasesAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = Path(tempfile.mkdtemp(prefix="casesaccess_"))
        self.addCleanup(shutil.rmtree, self.run, True)
        self.rd = self.run / "round_003"
        (self.rd / "logs").mkdir(parents=True)
        (self.rd / "eval_result.json").write_text(json.dumps(
            {"score": 0.5, "per_case": [{"case_id": "A", "passed": False, "score": 0.0},
                                        {"case_id": "Z", "passed": True, "score": 1.0}]}))
        cases = self.run / "case_reflections"
        cases.mkdir()
        (cases / "INDEX.md").write_text("| A | A.md |\n")
        (cases / "A.md").write_text("# Test case A\npass rate: 1/2\nline three\n")
        (self.run / "secret.txt").write_text("outside")

    def test_resolve_read_grep_and_escape(self) -> None:
        from meta_agent import log_access

        self.assertIn("pass rate: 1/2", log_access.read_file({}, self.rd, {"path": "cases/A.md"}))
        self.assertIn("A.md", log_access.read_file({}, self.rd, {"path": "cases/"}))   # listing
        self.assertIn("pass rate", log_access.grep({}, self.rd, {"path": "cases/A.md", "pattern": "pass"}))
        for bad in ("cases/../secret.txt", "cases/../round_003/eval_result.json"):
            self.assertIn("escapes the cases/ root", log_access.read_file({}, self.rd, {"path": bad}))
        self.assertIn("'cases/'", log_access.resolve("nope/x", sources={}, round_dir=self.rd)[1])

    def test_no_case_files_is_a_clear_error_and_no_listing(self) -> None:
        from meta_agent import log_access

        shutil.rmtree(self.run / "case_reflections")
        self.assertIn("no per-test-case reflection files",
                      log_access.read_file({}, self.rd, {"path": "cases/INDEX.md"}))
        self.assertEqual(log_access.cases_listing(self.rd), "")

    def test_listing_names_parent_case_files_only_when_present(self) -> None:
        from meta_agent import log_access

        text = log_access.cases_listing(self.rd)
        self.assertIn("cases/INDEX.md", text)
        self.assertIn("A.md", text)
        self.assertNotIn("Z.md", text)   # no file for Z

    def test_editor_reads_cases_and_refuses_writes(self) -> None:
        from meta_agent.agent_editor import AGENTIC_GREP_TOOL, AGENTIC_LOG_READ_FILE_TOOL, AgentEditor

        self.assertTrue(AgentEditor._is_log_path("cases/A.md"))
        self.assertTrue(AgentEditor._is_log_path("cases"))
        self.assertFalse(AgentEditor._is_log_path("casesX/A.md"))
        self.assertIn("cases/", AGENTIC_LOG_READ_FILE_TOOL["description"])
        self.assertIn("cases/", AGENTIC_GREP_TOOL["description"])
        self.assertIn("cases/INDEX.md", AgentEditor._cases_listing(self.rd))
        ed = AgentEditor.__new__(AgentEditor)
        out = ed._agentic_grep(self.rd / "task_agent", self.rd, {"path": "cases/A.md", "pattern": "three"}, [])
        self.assertIn("line three", out)

    def test_block_suggester_tool_description(self) -> None:
        from meta_agent.block_suggester import AGENTIC_READ_FILE_TOOL

        self.assertIn("cases/", AGENTIC_READ_FILE_TOOL["description"])


if __name__ == "__main__":
    unittest.main()

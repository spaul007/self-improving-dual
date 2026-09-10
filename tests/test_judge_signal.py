"""Tests for the judge-as-signal changes (analysis v7): standard errors on
the outcome, the effect verdict in the analysis renderer/prompt/record, the
citation forms, judge-labelled belief scoring and its report sections, the
one-line score/judge summaries, and the own-evaluation analysis gate.

    PYTHONPATH=. python3 -m unittest tests.test_judge_signal
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.belief_contract import (
    parse_citations, parse_document, record_effects, verify_citations,
)
from meta_agent.belief_scoring import (
    Scored, is_measurable, render_calibration_report, resolve,
)
from meta_agent.edit_beliefs import BeliefStore
from meta_agent.edit_memory import EditMemory, RECORD_FORMAT, render_record
from meta_agent.edit_memory_render import _load_records, build_ledger
from meta_agent.edit_outcome import compute_outcome, se_of, se_unpaired
from meta_agent.edit_usage import (
    ANALYSIS_VERSION, build_analysis_prompt, render_analysis,
)
from meta_agent.perf_text import judge_summary, perf_summary, score_context
from meta_agent.steering import compact_sibling_lines
from tests.test_belief_contract import GOOD as GOOD_DOC
from tests.test_edit_memory import _Node, _StubLLM, _Tree, _agent, _case

RECIPE = {"mode": "list", "path": "failed_checks"}


def _sub(edit, sound=True, effect="improved", evidence="strong",
         targets=("opening_hours",), reason="fixed on 6 of 8 cases where it fired"):
    return {"edit": edit, "sound": sound, "reason": "fired on case a",
            "targeted_checks": list(targets), "effect": effect,
            "evidence": evidence, "effect_reason": reason}


V7_PAYLOAD = {
    "components": [{"component": "gate/main", "role": "gate", "activated": "8x",
                    "verdict_behavior": "pass 8", "agreement": "a,b pass/pass",
                    "cause": "gate wired before emit"}],
    "targeted_constraints": [{"constraint": "opening_hours",
                              "remaining_failures": "2/8", "was": "6/8"}],
    "collateral": "none observed",
    "implementation_sound": True,
    "implementation_reason": "gate fired on 8 cases and agreed with the scorer",
    "sub_edit_verdicts": [_sub(1), _sub(2, sound=False, effect="no_effect",
                                       evidence="weak", targets=(),
                                       reason="never fired")],
    "regressions": "budget 1->3 fails (-2)",
}


# --------------------------------------------------------------------------- #
class TestStandardErrors(unittest.TestCase):
    def test_se_of_and_unpaired(self):
        self.assertIsNone(se_of([0.5]))
        self.assertEqual(se_of([1.0, 1.0, 1.0]), 0.0)
        self.assertAlmostEqual(se_of([0.0, 1.0]), 0.5)
        self.assertAlmostEqual(se_unpaired([0.0, 1.0], [0.0, 1.0]), (0.5 ** 2 * 2) ** 0.5, 4)
        self.assertIsNone(se_unpaired([0.0, 1.0], [0.3]))

    def test_outcome_carries_both_standard_errors(self):
        parent = [_case("a", 0.0), _case("b", 1.0)]
        child = [_case("c", 1.0), _case("d", 0.0)]        # disjoint
        oc = compute_outcome(parent, child, min_shared=8)
        self.assertEqual(oc.n_shared, 0)
        self.assertIsNone(oc.se_shared)
        self.assertAlmostEqual(oc.se_all, 0.7071, 3)
        self.assertIn("se_all", oc.to_dict())
        oc2 = compute_outcome(parent, [_case("a", 1.0), _case("b", 1.0)], min_shared=1)
        self.assertEqual(oc2.n_shared, 2)
        self.assertAlmostEqual(oc2.se_shared, 0.5)


# --------------------------------------------------------------------------- #
class TestAnalysisV7(unittest.TestCase):
    def test_version_bumped(self):
        self.assertGreaterEqual(ANALYSIS_VERSION, 7)

    def test_render_effect_lines_and_regressions(self):
        md = render_analysis(V7_PAYLOAD)
        self.assertIn("- **effect (edit 1)**: improved (strong; targets: opening_hours) "
                      "— fixed on 6 of 8 cases where it fired", md)
        self.assertIn("- **effect (edit 2)**: no_effect (weak) — never fired", md)
        self.assertIn("- **regressions**: budget 1->3 fails (-2)", md)
        # effect lines follow the implementation lines, in edit order
        self.assertLess(md.index("**implementation (edit 2)**"),
                        md.index("**effect (edit 1)**"))
        self.assertLess(md.index("**effect (edit 1)**"), md.index("**effect (edit 2)**"))

    def test_old_payload_renders_no_effect_line_and_bad_values_are_skipped(self):
        old = dict(V7_PAYLOAD)
        old["sub_edit_verdicts"] = [{"edit": 1, "sound": True, "reason": "r"}]
        old.pop("regressions")
        md = render_analysis(old)
        self.assertNotIn("**effect", md)
        self.assertNotIn("**regressions", md)
        bad = dict(V7_PAYLOAD)
        bad["sub_edit_verdicts"] = [_sub(1, effect="maybe", evidence="huge")]
        self.assertNotIn("**effect", render_analysis(bad))

    def test_prompt_has_unpaired_table_case_rows_and_view_header(self):
        parent = [{"case_id": "a", "score": 0.5, "passed": False,
                   "details": {"failed_checks": ["hours"]}},
                  {"case_id": "b", "score": 1.0, "passed": True,
                   "details": {"failed_checks": []}}]
        child = [{"case_id": "b", "score": 1.0, "passed": True,
                  "details": {"failed_checks": []}},
                 {"case_id": "c", "score": 0.25, "passed": False,
                  "details": {"failed_checks": ["hours", "budget"]},
                  "error": "timeout after 1800s"}]
        oc = compute_outcome(parent, child, min_shared=8,
                             recipe={"mode": "list", "path": "details.failed_checks"})
        ev = [{"label": "gate", "name": "m", "verdict": "pass", "case_id": "b"},
              {"label": "gate", "name": "m", "verdict": "fail", "case_id": "c"},
              {"label": "gate", "name": "m", "verdict": "fail", "case_id": "c"}]
        prompt = build_analysis_prompt(
            node_id=3, record_body="## Edit 1\n- **what**: w", outcome=oc,
            store={"events": ev, "surface": {}}, u_lines=[],
            parent_cases=parent, child_cases=child,
            recipe={"mode": "list", "path": "details.failed_checks"},
            code_diff="### workflow.py :: run_task (function, changed)  +1/-0",
            code_kind="view")
        self.assertIn("# Implementation vs parent (added lines per definition", prompt)
        self.assertIn("unpaired Δ (own cases, different samples): -0.1250", prompt)
        self.assertIn("paired Δ over 1 shared cases: +0.0000 — fewer than the 8-shared "
                      "minimum", prompt)
        self.assertIn("`hours` parent 1/2 -> child 1/2 fails (rate 50% -> 50%)", prompt)
        self.assertIn("`budget` parent 0/2 -> child 1/2 fails", prompt)
        # shared case first, parent row beside it; fired verdicts per case
        rows = prompt.split("# Per-case results")[1].split("# Runtime")[0]
        self.assertIn("- case b: 1.0000 PASS · failed: none · fired: gate/m=pass", rows)
        self.assertIn("- case b (parent): 1.0000 PASS · failed: none", rows)
        self.assertIn("- case c: 0.2500 FAIL · failed: budget, hours · fired: "
                      "gate/m=fail×2 · error: timeout after 1800s", rows)
        self.assertLess(rows.index("- case b:"), rows.index("- case c:"))
        self.assertIn("all 2 child cases", prompt)


# --------------------------------------------------------------------------- #
def _write_record(tmp: Path, node: int, parent: int, body: str, outcome,
                  analysis_md: str = "") -> Path:
    rd = tmp / f"round_{node:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "edit_memory.md").write_text(render_record(
        {"node": str(node), "parent": str(parent), "depth": "1",
         "lineage": f"0 > {node}"}, body, outcome, analysis_md=analysis_md),
        encoding="utf-8")
    return rd


BODY = ("## Edit 1\n- **name**: `gate`\n"
        "- **category level 1 (strategy)**: `add-verifier`\n"
        "- **category level 2 (area)**: `routing`\n"
        "- **what**: Adds a gate\n- **why**: hours were wrong\n"
        "## Edit 2\n- **name**: `log`\n"
        "- **category level 1 (strategy)**: `add-telemetry`\n"
        "- **category level 2 (area)**: `routing`\n"
        "- **what**: Logs\n- **why**: visibility")


class TestRecordAndLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        parent = [_case("a", 0.0), _case("b", 1.0), _case("e", 0.5)]
        child = [_case("c", 1.0), _case("d", 0.5), _case("e", 1.0)]
        self.oc = compute_outcome(parent, child, min_shared=8)
        _write_record(self.tmp, 1, 0, BODY, self.oc, render_analysis(V7_PAYLOAD))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_record_format_and_unpaired_line(self):
        self.assertGreaterEqual(RECORD_FORMAT, 6)
        text = (self.tmp / "round_001" / "edit_memory.md").read_text(encoding="utf-8")
        self.assertIn("- **unpaired**: child 0.8333/3 vs parent 0.5000/3 · Δ +0.3333 ± ",
                      text)
        self.assertNotIn("paired SE", text.split("- **unpaired**")[1].split("\n")[0]
                         if self.oc.n_shared == 0 else "")
        rec = _load_records(self.tmp)[1]
        self.assertAlmostEqual(rec["delta_all"], 0.3333)
        self.assertIsNotNone(rec["se_all"])
        self.assertEqual((rec["parent_n_all"], rec["parent_abs_all"]), (3, 0.5))
        self.assertEqual(rec["effect_by_edit"], {1: "improved", 2: "no_effect"})
        self.assertEqual(rec["evidence_by_edit"], {1: "strong", 2: "weak"})
        self.assertEqual(rec["targets_by_edit"][1], ["opening_hours"])
        self.assertIn("fixed on 6 of 8", rec["effect_reason_by_edit"][1])
        self.assertEqual((rec["effect"], rec["evidence"]), ("improved", "strong"))
        self.assertEqual(rec["regressions"], "budget 1->3 fails (-2)")
        self.assertEqual(rec["impl_by_edit"], {1: True, 2: False})

    def test_ledger_tallies_the_judge_per_strategy(self):
        registry = {"strategies": {
            "add-verifier": {"definition": "adds a check",
                             "edits": [{"node": 1, "edit_index": 1, "name": "gate"}]},
            "add-telemetry": {"definition": "logs",
                              "edits": [{"node": 1, "edit_index": 2, "name": "log"}]}},
            "areas": {}}
        rows = {r["id"]: r for r in build_ledger(registry, _load_records(self.tmp),
                                                 threshold=0.02, min_shared=8)}
        self.assertEqual(rows["add-verifier"]["effects"], {"improved": 1})
        self.assertEqual((rows["add-verifier"]["sound"], rows["add-verifier"]["unsound"]),
                         (1, 0))
        self.assertEqual(rows["add-telemetry"]["effects"], {"no_effect": 1})
        self.assertEqual(rows["add-telemetry"]["unsound"], 1)
        self.assertEqual(rows["add-verifier"]["targets"], {"opening_hours": 1})
        self.assertEqual(rows["add-verifier"]["n_regressed"], 1)
        from meta_agent.edit_memory_render import judge_ledger_lines
        line = next(l for l in judge_ledger_lines(list(rows.values()))
                    if l.startswith("- `add-verifier`"))
        self.assertIn("· judge: improved 1 · targets: opening_hours ×1 · regressions "
                      "on 1 node(s) · implementation: sound 1 / unsound 0 · "
                      "score (context): paired Δ median +0.5000 · unpaired Δ median "
                      "+0.3333 — adds a check", line)
        self.assertAlmostEqual(rows["add-verifier"]["unpaired_median"], 0.3333)
        # one shared case (e: 0.5 -> 1.0): the paired Δ exists but is thin
        self.assertAlmostEqual(rows["add-verifier"]["median"], 0.5)
        self.assertEqual(rows["add-verifier"]["tally"], {"inconclusive": 1})


# --------------------------------------------------------------------------- #
class TestCitations(unittest.TestCase):
    REC = {1: {"delta": 0.05, "n_shared": 8, "delta_all": 0.04, "se_all": 0.12,
               "effect_by_edit": {1: "improved", 2: "no_effect"}},
           2: {"delta": None, "n_shared": 0, "delta_all": -0.1, "se_all": 0.15,
               "effect": "regressed"},
           3: {"delta": None, "n_shared": 0, "delta_all": None}}

    def test_parse_forms(self):
        text = ("[node 1: improved] [node 1: improved; Δ+0.0500/8] "
                "[node 2: Δ-0.1000±0.15] [node 2: regressed, Δ-0.10±0.15] "
                "[node 3: unmeasured] [node 4: ]")
        cites = parse_citations(text)
        self.assertEqual(len(cites), 5)                 # the empty bracket is dropped
        self.assertEqual((cites[0]["effect"], cites[0]["delta"]), ("improved", None))
        self.assertEqual((cites[1]["effect"], cites[1]["delta"], cites[1]["n_shared"]),
                         ("improved", 0.05, 8))
        self.assertEqual((cites[2]["delta"], cites[2]["se"], cites[2]["n_shared"]),
                         (-0.1, 0.15, None))
        self.assertEqual((cites[3]["effect"], cites[3]["delta"]), ("regressed", -0.1))
        self.assertTrue(cites[4]["unmeasured"])

    def test_verify_effect_and_unpaired_forms(self):
        codes = lambda t: [(v.code, v.message) for v in verify_citations(t, self.REC)]
        self.assertEqual(codes("[node 1: improved]"), [])
        self.assertEqual(codes("[node 1: no_effect]"), [])   # any sub-edit's verdict
        bad = codes("[node 1: regressed]")
        self.assertEqual(bad[0][0], "misquote")
        self.assertIn("edit 1: improved, edit 2: no_effect", bad[0][1])
        self.assertEqual(codes("[node 2: regressed; Δ-0.1000±0.15]"), [])
        self.assertEqual(codes("[node 1: Δ+0.0400±0.12]"), [])
        wrong = codes("[node 1: Δ-0.0400±0.12]")
        self.assertIn("unpaired Δ+0.0400", wrong[0][1])
        self.assertIn("no unpaired Δ", codes("[node 3: Δ+0.0100±0.1]")[0][1])
        self.assertIn("no effect verdict", codes("[node 3: improved]")[0][1])
        stale = codes("[node 2: unmeasured]")
        self.assertEqual(stale[0][0], "misquote")
        self.assertIn("regressed", stale[0][1])
        self.assertEqual(record_effects(self.REC[2]), {0: "regressed"})


# --------------------------------------------------------------------------- #
def _rec(effect="improved", evidence="strong", impl=True, n_abs=16, delta=None,
         n_shared=0, by_edit=None, tags=None, targets=None, regressions="",
         delta_all=-0.01, se_all=0.12):
    body = ("## Edit 1\n- **name**: `gate`\n"
            "- **category level 1 (strategy)**: `add-verifier`\n"
            "- **category level 2 (area)**: `routing`\n"
            "- **what**: Adds a gate\n- **why**: hours")
    effects = by_edit if by_edit is not None else (
        {1: (effect, evidence, "gate fixed hours on 5 cases")} if effect else {})
    return {"delta": delta, "n_shared": n_shared, "n_abs": n_abs,
            "delta_all": delta_all, "se_all": se_all,
            "targets_by_edit": ({e: list(targets) for e in effects}
                                if targets else {}),
            "regressions": regressions,
            "impl_sound": impl, "impl_reason": "fired on 8 cases",
            "impl_by_edit": {e: True for e in effects} if impl is not None else {},
            "impl_reason_by_edit": {},
            "effect_by_edit": {e: v[0] for e, v in effects.items()},
            "evidence_by_edit": {e: v[1] for e, v in effects.items()},
            "effect_reason_by_edit": {e: v[2] for e, v in effects.items()},
            "effect": effects[min(effects)][0] if effects else None,
            "evidence": effects[min(effects)][1] if effects else None,
            "effect_reason": effects[min(effects)][2] if effects else "",
            "text": body, "body": body,
            "tags": tags if tags is not None else
            [{"edit": 1, "strategy": "add-verifier", "area": "routing"}]}


def _pred(strategy=("s", 0.7), implementation=("i", 0.4), matched=None):
    p = {"belief_version": 1, "instruction_version": 0, "coverable": True,
         "strategy": {"slug": strategy[0], "p": strategy[1]} if strategy else None,
         "implementation": ({"slug": implementation[0], "p": implementation[1]}
                            if implementation else None)}
    if matched is not None and p["strategy"]:
        p["strategy"]["matched_edit"] = matched
    return p


def _judge(preds, recs, **kw):
    kw.setdefault("threshold", 0.02)
    kw.setdefault("min_shared", 8)
    kw.setdefault("already", set())
    kw.setdefault("n_updates", 1)
    kw.setdefault("label_source", "judge")
    return resolve(preds, recs, **kw)


class TestResolveJudge(unittest.TestCase):
    def test_improved_scores_one_without_shared_cases(self):
        new, pending, skipped = _judge({1: _pred()}, {1: _rec()})
        self.assertEqual((pending, skipped), ([], []))
        by = {s.kind: s for s in new}
        self.assertEqual((by["strategy"].y, by["strategy"].effect,
                          by["strategy"].evidence, by["strategy"].label_source),
                         (1, "improved", "strong", "judge"))
        self.assertAlmostEqual(by["strategy"].brier, 0.09)
        self.assertEqual(by["strategy"].n_shared, 0)
        self.assertAlmostEqual(by["strategy"].delta_all, -0.01)
        self.assertEqual(by["implementation"].y, 1)
        self.assertEqual(by["strategy"].matched_edit, 1)

    def test_no_effect_and_regressed_score_zero(self):
        for eff in ("no_effect", "regressed"):
            new, _, _ = _judge({1: _pred()}, {1: _rec(effect=eff)})
            self.assertEqual({s.kind: s.y for s in new}["strategy"], 0, eff)

    def test_unclear_and_weak_stay_pending(self):
        new, pending, _ = _judge({1: _pred()}, {1: _rec(effect="unclear")})
        self.assertEqual([s.kind for s in new], ["implementation"])
        self.assertIn("effect unclear", pending[0]["reason"])
        new2, pending2, _ = _judge({1: _pred()}, {1: _rec(evidence="weak")})
        self.assertEqual([s.kind for s in new2], ["implementation"])
        self.assertIn("weak evidence below the moderate minimum", pending2[0]["reason"])
        # ...unless the run accepts weak evidence.
        new3, pending3, _ = _judge({1: _pred()}, {1: _rec(evidence="weak")},
                                   min_evidence="weak")
        self.assertEqual({s.kind for s in new3}, {"strategy", "implementation"})
        self.assertEqual(pending3, [])

    def test_no_analysis_yet_is_pending_only_with_evaluations(self):
        new, pending, _ = _judge({1: _pred()}, {1: _rec(impl=None, effect=None)})
        self.assertEqual(new, [])
        self.assertEqual(pending[0]["reason"], "awaiting the analysis verdict")
        new2, pending2, _ = _judge({1: _pred()},
                                   {1: _rec(impl=None, effect=None, n_abs=0)})
        self.assertEqual((new2, pending2), ([], []))

    def test_pre_v7_analysis_is_pending_for_strategy(self):
        rec = _rec(effect=None)              # implementation verdict, no effect line
        new, pending, _ = _judge({1: _pred()}, {1: rec})
        self.assertEqual([s.kind for s in new], ["implementation"])
        self.assertIn("analysis predates v7", pending[0]["reason"])

    def test_matched_sub_edit_effect_wins_else_first_edit(self):
        rec = _rec(by_edit={1: ("no_effect", "strong", "dead"),
                            2: ("improved", "moderate", "hours moved")})
        rec["impl_by_edit"] = {1: True, 2: True}
        new, _, _ = _judge({1: _pred(matched=2)}, {1: rec})
        s = {x.kind: x for x in new}["strategy"]
        self.assertEqual((s.y, s.effect, s.matched_edit), (1, "improved", 2))
        new2, _, _ = _judge({1: _pred()}, {1: rec})       # unmatched -> edit 1
        self.assertEqual({x.kind: x.y for x in new2}["strategy"], 0)

    def test_unsound_still_blocks_strategy(self):
        rec = _rec(impl=False)
        rec["impl_by_edit"] = {1: False}
        new, pending, _ = _judge({1: _pred()}, {1: rec})
        self.assertEqual([s.kind for s in new], ["implementation"])
        self.assertIn("unsound", pending[0]["reason"])

    def test_delta_mode_is_unchanged(self):
        # No shared cases: delta mode cannot score, judge mode can.
        new, pending, _ = resolve({1: _pred()}, {1: _rec()}, threshold=0.02,
                                  min_shared=8, already=set(), n_updates=1,
                                  label_source="delta")
        self.assertEqual((new, pending), ([], []))
        rec = _rec(effect="no_effect", delta=0.05, n_shared=8)
        new2, _, _ = resolve({1: _pred()}, {1: rec}, threshold=0.02, min_shared=8,
                             already=set(), n_updates=1, label_source="delta")
        self.assertEqual({s.kind: s.y for s in new2}["strategy"], 1)  # Δ decides
        self.assertTrue(is_measurable(rec, label_source="delta", min_shared=8))
        self.assertFalse(is_measurable(_rec(), label_source="delta", min_shared=8))
        self.assertTrue(is_measurable(_rec(), label_source="judge", min_shared=8))

    def test_scored_roundtrip_with_new_fields(self):
        s = Scored(1, "strategy", "s", 0.7, 1, 0.09, 1, 0, 2, 0, 0.0,
                   impl_reason="r", label_source="judge", effect="improved",
                   evidence="strong", effect_reason="e", matched_edit=1,
                   delta_all=-0.01, se_all=0.12)
        self.assertEqual(Scored.from_dict(s.as_dict()), s)
        legacy = Scored.from_dict({"node": 1, "kind": "strategy", "slug": "s",
                                   "p": 0.7, "y": 1, "brier": 0.09})
        self.assertEqual((legacy.label_source, legacy.effect, legacy.delta_all),
                         ("delta", "", None))          # pre-v7 rows were Δ-labelled
        fresh = Scored(1, "strategy", "s", 0.7, 1, 0.09, 1, 0, 2, 0, 0.0)
        self.assertEqual(fresh.label_source, "judge")  # the module default


class TestReportJudge(unittest.TestCase):
    def test_sections_agreement_and_flips(self):
        parsed = parse_document(GOOD_DOC)
        rec1 = _rec(delta=-0.05, n_shared=16)        # judge improved, Δ negative
        rec2 = _rec(effect="no_effect")              # flipped since scoring
        # node 3: only 3 shared cases, but the own-case Δ is far outside its
        # SE — well measured without any parent overlap; judge says improved.
        rec3 = _rec(delta=0.10, n_shared=3, delta_all=-0.30, se_all=0.10)
        rows = [Scored(1, "strategy", "add-verifier-helps", 0.65, 1, 0.1225, 1, 0, 2,
                       16, -0.05, label_source="judge", effect="improved",
                       evidence="strong", effect_reason="hours fixed on 5",
                       matched_edit=1, delta_all=-0.01, se_all=0.12),
                Scored(2, "strategy", None, 0.5, 1, 0.25, 1, 0, 2, 0, 0.0,
                       label_source="judge", effect="improved", evidence="moderate",
                       matched_edit=1),
                Scored(3, "strategy", "add-verifier-helps", 0.65, 1, 0.1225, 1, 0, 2,
                       3, 0.10, label_source="judge", effect="improved",
                       evidence="strong", matched_edit=1, delta_all=-0.30,
                       se_all=0.10)]
        report = render_calibration_report(
            parsed=parsed, scored=rows, pending=[],
            records={1: rec1, 2: rec2, 3: rec3},
            predictions={1: _pred(), 2: _pred(strategy=None), 3: _pred()},
            soft_violations=[], joins=[], version_history=[{"version": 0}],
            current_version=0, min_shared=8, label_source="judge", threshold=0.02)
        self.assertIn("labels: strategy y = the judge's effect verdict", report)
        self.assertIn("judge improved (strong) — \"hours fixed on 5\" · score hurt "
                      "Δ-0.0500/16 shared", report)
        self.assertIn("### Judge vs score — a diagnostic, not a yardstick", report)
        self.assertIn("0 agree / 2 disagree — the judge is the label", report)
        self.assertIn('node 1: judge improved (strong) — "hours fixed on 5" · score '
                      'hurt Δ-0.0500/16 shared', report)
        self.assertIn("node 3: judge improved (strong)", report)
        self.assertIn("Δ-0.3000 ± 0.100 unpaired", report)
        self.assertNotIn("Judge vs score agreement (strategy rows with ≥ 16", report)
        self.assertIn("### Verdict changes since scoring", report)
        self.assertIn("node 2: scored on `improved`; the latest analysis says "
                      "`no_effect`", report)
        self.assertIn("strategy: judge improved · Δ-0.0100 ± 0.120 unpaired", report)
        # delta-mode report keeps the old wording and has no judge sections
        old = render_calibration_report(
            parsed=parsed, scored=[], pending=[], records={}, predictions={},
            soft_violations=[], joins=[], version_history=[], current_version=0,
            min_shared=8, label_source="delta")
        self.assertIn("labels: strategy y = Δ vs parent", old)
        self.assertNotIn("Judge vs score", old)


# --------------------------------------------------------------------------- #
class TestSummaries(unittest.TestCase):
    def test_score_context_prefers_paired_then_unpaired(self):
        self.assertEqual(score_context(delta=0.05, n_shared=16), "helped Δ+0.0500/16 shared")
        self.assertEqual(score_context(delta=0.05, n_shared=3, delta_all=-0.0117,
                                       se_all=0.1461, n_child=16, n_parent=16),
                         "Δ-0.0117 ± 0.146 unpaired (16 vs 16 own cases); only 3 shared")
        self.assertEqual(score_context(delta=0.05, n_shared=3),
                         "Δ+0.0500/3 shared (below the 8-shared minimum)")
        self.assertEqual(score_context(delta=None, n_shared=0), "unmeasured")
        self.assertEqual(judge_summary({"effect": "improved", "evidence": "strong"}),
                         "improved (strong)")
        self.assertEqual(judge_summary({}), "")
        self.assertEqual(perf_summary(_rec()), "Δ-0.0100 ± 0.120 unpaired")

    def test_sibling_line_shows_judge_and_unpaired_score(self):
        rec = _rec(targets=["opening_hours"])
        rec.update({"child_abs": 0.66, "n_abs": 16, "parent_n_all": 16,
                    "usage": "", "suspect": False, "n_shared": 3, "delta": 0.05})
        lines = compact_sibling_lines({5: rec}, [(5, "Adds a gate", False)],
                                      threshold=0.02, min_shared=8)
        self.assertEqual(lines, ['- node 5: judge improved (strong; targets: '
                                 'opening_hours) · score Δ-0.0100 ± 0.120 unpaired '
                                 '(16 vs 16 own cases); only 3 shared · '
                                 'child 0.6600/16 · "Adds a gate"'])
        # Regressions the judge listed come right after the verdict.
        rec2 = _rec(effect="regressed", evidence="moderate",
                    regressions="budget 1->3 fails (-2)")
        rec2.update({"child_abs": 0.5, "n_abs": 16, "parent_n_all": 16,
                     "usage": "", "suspect": False})
        line = compact_sibling_lines({6: rec2}, [(6, "Reworks meals", False)],
                                     threshold=0.02, min_shared=8)[0]
        self.assertTrue(line.startswith("- node 6: judge regressed (moderate) · "
                                        "regressions: budget 1->3 fails (-2) · score "))
        self.assertEqual(judge_summary({"effect": "improved", "evidence": "strong",
                                        "effect_by_edit": {2: "improved"},
                                        "targets_by_edit": {2: ["a", "b"]}}),
                         "improved (strong; targets: a, b)")


# --------------------------------------------------------------------------- #
class TestJudgeGate(unittest.TestCase):
    """The analysis runs on the node's OWN evaluations in judge mode — no
    overlap with the parent — and its effect verdict lands in the record."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p, self.c = self.tmp / "round_000", self.tmp / "round_001"
        _agent(self.p, "def run_task(t):\n    return 1\n")
        _agent(self.c, "from platform_core.trace import log as trace_log\n"
                       "def run_task(t):\n"
                       "    trace_log('gate', verdict='pass', name='main')\n"
                       "    return 2\n")
        self.analysis = json.loads(json.dumps(V7_PAYLOAD))
        self.analysis["sub_edit_verdicts"] = [_sub(1)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **kw):
        llm = _StubLLM(analysis=self.analysis)
        em = EditMemory(llm, **kw)
        em.setup(self.tmp, self.p, [_case("a", 0.0, ["x"])])
        em.record_node(round_dir=self.c, parent_round_dir=self.p, node_id=1,
                       parent_id=0, ancestors=[0])
        logs = self.c / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "trace.jsonl").write_text(
            '{"kind": "mutable_log", "payload": {"label": "gate", "verdict": "pass", '
            '"name": "main", "case_id": "c"}}\n', encoding="utf-8")
        root = _Node(0, None, self.p, [_case("a", 0.0, ["x"]), _case("b", 0.5, ["x"])],
                     children=[1])
        n1 = _Node(1, 0, self.c, [_case("c", 1.0, []), _case("d", 0.5, ["x"])])
        em.refresh_outcomes(_Tree([root, n1]), 1)
        return llm, (self.c / "edit_memory.md").read_text(encoding="utf-8")

    def test_judge_mode_analyses_disjoint_cases(self):
        llm, text = self._run(strategy_label="judge", analysis_min_own_evals=2)
        self.assertIn("## Analysis", text)
        self.assertIn("- **effect (edit 1)**: improved (strong; targets: opening_hours)",
                      text)
        self.assertIn("- **unpaired**: child 0.7500/2 vs parent 0.2500/2 · Δ +0.5000 ± ",
                      text)
        self.assertIn("(no cases shared with parent yet)", text)
        prompt = [c for c in llm.calls
                  if c["tools"][0]["function"]["name"] == "submit_edit_analysis"][0]
        user = prompt["messages"][1]["content"]
        self.assertIn("# Implementation vs parent (added lines per definition", user)
        self.assertIn("run_task", user)
        self.assertIn("no shared cases yet: no paired estimate", user)
        self.assertIn("- case c: 1.0000 PASS · failed: none · fired: gate/main=pass", user)
        rec = _load_records(self.tmp)[1]
        self.assertEqual((rec["effect"], rec["evidence"]), ("improved", "strong"))
        self.assertEqual(rec["parent_n_all"], 2)

    def test_gate_waits_for_own_evaluations(self):
        _, text = self._run(strategy_label="judge", analysis_min_own_evals=3)
        self.assertNotIn("## Analysis", text)          # 2 evals < 3

    def test_delta_mode_keeps_the_shared_case_gate(self):
        _, text = self._run(strategy_label="delta", min_shared=1)
        self.assertNotIn("## Analysis", text)          # no shared cases

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            EditMemory(_StubLLM(), strategy_label="vibes")
        with self.assertRaises(ValueError):
            EditMemory(_StubLLM(), judge_min_evidence="any")
        with self.assertRaises(ValueError):
            BeliefStore(_StubLLM(), label_source="vibes")
        with self.assertRaises(ValueError):
            BeliefStore(_StubLLM(), min_evidence="lots")

    def test_beliefs_inherit_label_settings(self):
        em = EditMemory(_StubLLM(), setup_pass=False, strategy_label="delta",
                        judge_min_evidence="weak", beliefs={"enabled": True})
        self.assertEqual((em._beliefs.label_source, em._beliefs.min_evidence),
                         ("delta", "weak"))
        em2 = EditMemory(_StubLLM(), setup_pass=False,
                         beliefs={"enabled": True, "label_source": "delta"})
        self.assertEqual(em2._beliefs.label_source, "delta")   # explicit key wins


if __name__ == "__main__":
    unittest.main()

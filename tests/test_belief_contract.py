"""Unit tests for the belief document contract (belief_contract).

Pure module — no LLM, no I/O.

    PYTHONPATH=. python3 -m unittest tests.test_belief_contract
"""
from __future__ import annotations

import unittest

from meta_agent.belief_contract import (
    SUMMARY_CHAR_CAP,
    inject_track_lines,
    match_belief,
    parse_anchors,
    parse_citations,
    parse_document,
    render_violations,
    strip_track_lines,
    validate_document,
    verify_citations,
)

REGISTRY = {
    "strategies": {"add-verifier": {"definition": "adds a check"},
                   "add-cache": {"definition": "caches"}},
    "areas": {"routing": {"definition": "route problems"}},
}
RECORDS = {1: {"delta": 0.05, "n_shared": 8},
           2: {"delta": -0.01, "n_shared": 9},
           3: {"delta": None, "n_shared": 0}}

SECTION_A = """### belief:add-verifier-helps — verifiers pay off when wired
- kind: strategy
- scope: strategy=add-verifier area=routing
- predict: p=0.65
- evidence: helped once [node 1: Δ+0.0500/8]; one unmeasured attempt [node 3: unmeasured]
- next: extend the gate to intercity transfers
"""
SECTION_B = """### belief:add-verifier-impl — verifiers often ship dead
- kind: implementation
- scope: strategy=add-verifier
- predict: p=0.40
- evidence: one dead component [node 2: Δ-0.0100/9]
- next: wire the gate before adding more checks
"""
GOOD = "## Summary\nTwo strategies tried; nothing conclusive yet.\n\n" + SECTION_A + "\n" + SECTION_B


def _codes(doc):
    return sorted(v.code for v in doc.violations)


class TestParse(unittest.TestCase):
    def test_conformant_document(self):
        doc = validate_document(GOOD, registry=REGISTRY, doc_char_cap=16000,
                                records=RECORDS)
        self.assertEqual(doc.violations, [])
        self.assertEqual(doc.summary, "Two strategies tried; nothing conclusive yet.")
        self.assertEqual([b.slug for b in doc.beliefs],
                         ["add-verifier-helps", "add-verifier-impl"])
        a, b = doc.beliefs
        self.assertEqual((a.kind, a.strategy, a.area, a.p),
                         ("strategy", "add-verifier", "routing", 0.65))
        self.assertEqual((b.kind, b.strategy, b.area, b.p),
                         ("implementation", "add-verifier", None, 0.40))
        self.assertEqual(a.title, "verifiers pay off when wired")
        self.assertIn("extend the gate", a.next)
        self.assertEqual([c["node"] for c in a.citations], [1, 3])
        self.assertTrue(a.section_text.startswith("### belief:add-verifier-helps"))

    def test_continuation_lines_join_the_previous_bullet(self):
        doc = parse_document(SECTION_A.replace(
            "- next: extend the gate to intercity transfers",
            "- next: extend the gate\n  to intercity transfers"))
        self.assertEqual(doc.violations, [])
        self.assertEqual(doc.beliefs[0].next, "extend the gate to intercity transfers")

    def test_bold_keys_are_accepted(self):
        doc = parse_document(SECTION_A.replace("- kind:", "- **kind**:"))
        self.assertEqual(doc.violations, [])
        self.assertEqual(doc.beliefs[0].kind, "strategy")

    def test_snake_case_slug_parses_fully(self):
        text = "### belief:transfer_time_accuracy — t\n[node 1: Δ-0.0273/16]\n"
        self.assertEqual(parse_anchors(text), ["transfer_time_accuracy"])
        self.assertEqual(parse_citations(text)[0]["slug"], "transfer_time_accuracy")
        self.assertIsNone(parse_citations("[node 3: unmeasured]")[0]["delta"])


class TestHardViolations(unittest.TestCase):
    def _v(self, text, **kw):
        kw.setdefault("registry", REGISTRY)
        kw.setdefault("doc_char_cap", 16000)
        kw.setdefault("records", RECORDS)
        return validate_document(text, **kw)

    def test_prose_above_first_belief(self):
        doc = self._v("Some intro prose.\n\n" + SECTION_A)
        self.assertIn("prose-above-beliefs", _codes(doc))
        self.assertTrue(doc.hard)

    def test_summary_over_cap(self):
        doc = self._v("## Summary\n" + "x" * (SUMMARY_CHAR_CAP + 50) + "\n\n" + SECTION_A)
        self.assertIn("summary-cap", _codes(doc))

    def test_unknown_ids(self):
        doc = self._v(SECTION_A.replace("strategy=add-verifier area=routing",
                                        "strategy=nope area=elsewhere"))
        self.assertIn("unknown-strategy", _codes(doc))
        self.assertIn("unknown-area", _codes(doc))

    def test_registry_none_skips_id_checks(self):
        doc = self._v(SECTION_A.replace("add-verifier", "nope"), registry=None)
        self.assertNotIn("unknown-strategy", _codes(doc))

    def test_p_range_and_bad_predict(self):
        self.assertIn("p-range", _codes(self._v(SECTION_A.replace("p=0.65", "p=0.99"))))
        self.assertIn("predict", _codes(self._v(SECTION_A.replace("p=0.65", "likely"))))

    def test_missing_field(self):
        doc = self._v(SECTION_A.replace("- predict: p=0.65\n", ""))
        self.assertIn("missing-field", _codes(doc))

    def test_bad_kind(self):
        self.assertIn("kind", _codes(self._v(SECTION_A.replace("kind: strategy",
                                                                "kind: opinion"))))

    def test_duplicate_slug_and_scope(self):
        doc = self._v(SECTION_A + "\n" + SECTION_A)
        self.assertIn("duplicate-slug", _codes(doc))
        self.assertIn("duplicate-scope", _codes(doc))
        renamed = SECTION_A.replace("belief:add-verifier-helps", "belief:other")
        doc2 = self._v(SECTION_A + "\n" + renamed)
        self.assertNotIn("duplicate-slug", _codes(doc2))
        self.assertIn("duplicate-scope", _codes(doc2))

    def test_unknown_bullet_and_extra_section(self):
        doc = self._v(SECTION_A + "- stance: mixed\n")
        self.assertIn("unknown-field", _codes(doc))
        doc2 = self._v(SECTION_A + "\n## Next moves\n- do X\n")
        self.assertIn("unexpected-section", _codes(doc2))

    def test_free_prose_inside_section(self):
        doc = self._v(SECTION_A + "This strategy is great.\n")
        self.assertIn("unexpected-line", _codes(doc))

    def test_over_cap(self):
        doc = self._v(GOOD, doc_char_cap=100)
        self.assertIn("over-cap", _codes(doc))

    def test_no_beliefs(self):
        doc = self._v("## Summary\njust prose\n")
        self.assertIn("no-beliefs", _codes(doc))


class TestSoftViolations(unittest.TestCase):
    def test_misquote_is_soft(self):
        doc = validate_document(SECTION_A.replace("Δ+0.0500/8", "Δ-0.0500/8"),
                                registry=REGISTRY, doc_char_cap=16000,
                                records=RECORDS)
        self.assertEqual(doc.hard, [])
        self.assertEqual([v.code for v in doc.soft], ["misquote"])
        self.assertIn("Δ+0.0500 over 8 shared", doc.soft[0].message)
        self.assertEqual(doc.soft[0].slug, "add-verifier-helps")

    def test_verify_citations_cases(self):
        vs = verify_citations("### belief:x — t\n[node 9: Δ+0.1/3] [node 3: Δ+0.1/3] "
                              "[node 1: unmeasured]", RECORDS)
        self.assertEqual([v.code for v in vs], ["no-record", "misquote", "misquote"])
        self.assertTrue(all(not v.hard for v in vs))

    def test_render_violations(self):
        doc = parse_document("prose\n" + SECTION_A.replace("p=0.65", "p=0.99"))
        text = render_violations(doc.violations)
        self.assertIn("1. [HARD]", text)
        self.assertIn("belief:add-verifier-helps — p=0.99", text)


class TestTrackLines(unittest.TestCase):
    def test_inject_is_idempotent_and_strip_inverts(self):
        once = inject_track_lines(GOOD, {"add-verifier-helps": "n=2 · Brier 0.10"})
        twice = inject_track_lines(once, {"add-verifier-helps": "n=2 · Brier 0.10"})
        self.assertEqual(once, twice)
        self.assertIn("- track: n=2 · Brier 0.10", once)
        self.assertEqual(once.count("- track:"), 1)  # only the covered belief
        self.assertEqual(strip_track_lines(once).strip(), GOOD.strip())
        # The track line is the LAST bullet of its section.
        section = once.split("### belief:add-verifier-impl")[0]
        self.assertTrue(section.rstrip().endswith("- track: n=2 · Brier 0.10"))

    def test_echoed_track_lines_are_ignored_by_the_parser(self):
        doc = parse_document(SECTION_A + "- track: made up by the model\n")
        self.assertEqual(doc.violations, [])
        self.assertIsNone(doc.beliefs[0].track)
        self.assertNotIn("track", doc.beliefs[0].section_text)


class TestMatch(unittest.TestCase):
    def test_area_match_beats_strategy_only_across_subedit_order(self):
        beliefs = parse_document(GOOD).beliefs
        tags = [{"edit": 1, "strategy": "add-verifier", "area": None},
                {"edit": 2, "strategy": "add-verifier", "area": "routing"}]
        got = match_belief(beliefs, "strategy", tags)
        self.assertIsNotNone(got)
        self.assertEqual((got[0].slug, got[1]), ("add-verifier-helps", 2))
        got_i = match_belief(beliefs, "implementation", tags)
        self.assertEqual((got_i[0].slug, got_i[1]), ("add-verifier-impl", 1))

    def test_no_match(self):
        beliefs = parse_document(GOOD).beliefs
        self.assertIsNone(match_belief(beliefs, "strategy",
                                       [{"edit": 1, "strategy": "add-cache", "area": None}]))
        self.assertIsNone(match_belief(beliefs, "strategy", []))


if __name__ == "__main__":
    unittest.main()


class TestJudgeFirstGrammar(unittest.TestCase):
    """The grammar and the worked example the maintainer imitates lead with
    the judge's verdict; a Δ appears only as a secondary, well-measured cite."""

    def test_grammar_and_example_are_judge_first(self):
        from meta_agent.belief_contract import EXAMPLE, GRAMMAR
        self.assertIn("by the judge's verdict", GRAMMAR)
        self.assertIn("[node N: improved] / [node N: no_effect] / [node N: regressed]",
                      GRAMMAR)
        section = EXAMPLE.split("\n\n(")[0]
        doc = validate_document(
            section, registry={"strategies": {"add-constraint-enforcement": {}},
                               "areas": {}}, doc_char_cap=4000)
        self.assertEqual(doc.violations, [])
        cites = [(c["node"], c["effect"], c["delta"]) for c in parse_citations(section)]
        self.assertEqual(cites, [(13, "improved", None), (10, "improved", 0.12),
                                 (4, "no_effect", None)])
        # The verdict-only form comes first; the Δ is shown once, second.
        self.assertLess(section.index("[node 13: improved]"),
                        section.index("[node 10: improved; Δ+0.1200/25]"))

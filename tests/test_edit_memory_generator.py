"""Validators and the two single-call steps of the edit-memory layer."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent.edit_memory import generator as G
from meta_agent.edit_memory import prompts as P


def _memory(extra: str = "") -> str:
    return "\n".join(P.MEMORY_SECTIONS[i] + f"\n- retry-on-timeout — nodes {i} (E{i}.1): something\n"
                     for i in range(4)) + extra


def _section(nid, subs=P.NODE_SUBSECTIONS, *, suffix: str = "", text: str = "text") -> str:
    """A curation node section; the 'Edits' subsection carries one labelled item."""
    out = P.NODE_SECTION_HEADING.format(node_id=nid) + suffix + "\n"
    for s in subs:
        body = f"- E{nid}.1 — {text} {nid}" if s == P.EDITS_SUBSECTION else f"{text} {nid}"
        out += f"### {s}\n{body}\n"
    return out


class TestValidators(unittest.TestCase):
    def test_memory_sections_and_size(self) -> None:
        self.assertEqual(G.validate_memory(_memory(), max_chars=20000), [])
        errs = G.validate_memory("## 1. Ranked edits\nx\n", max_chars=20000)
        self.assertTrue(any("## 2. Usefulness" in e for e in errs))
        self.assertTrue(any("limit" in e for e in G.validate_memory(_memory(), max_chars=50)))
        self.assertEqual(G.validate_memory("   ", max_chars=10), ["the document is empty"])

    def test_memory_ranks_edits_not_nodes(self) -> None:
        # Node-keyed entries in section 1 are a finding; the same wording in
        # other sections, or edit-keyed entries that cite nodes, are fine.
        doc = _memory().replace("- retry-on-timeout — nodes 0 (E0.1): something",
                                "1. **Node 0** — added retry (helped)\n- node 3: budget check")
        errs = G.validate_memory(doc, max_chars=20000)
        self.assertEqual(len(errs), 2)
        self.assertIn("must rank edits, not nodes", errs[0])
        self.assertIn("1. **Node 0**", errs[0])
        ok = _memory("\n- node 9 is the parent of most of these (context)\n")   # lands in section 4
        self.assertEqual(G.validate_memory(ok, max_chars=20000), [])
        self.assertEqual(G.validate_memory(
            _memory().replace("nodes 0 (E0.1)", "nodes 0, 5, 7 (E0.1, E5.2, E7.1)"), max_chars=20000), [])
        # Task-agent utilization / capability wording is not a score prediction.
        util = _memory().replace("(E0.1): something",
                                 "(E0.1): helped; the task agent could not use the hint on 9 of 16 failing cases "
                                 "— requires a deterministic post-processing step linked to the plan output")
        self.assertEqual(G.validate_memory(util, max_chars=20000), [])

    def test_score_prediction_regex(self) -> None:
        bad = [
            "expected score 0.81 after this edit",
            "We predict node 12 will win.",
            "this will raise the composite to 0.9",
            "the repair loop will improve the score",
            "gain of +0.05 points on commonsense",
            "Forecast: the next node scores higher",
        ]
        for line in bad:
            self.assertTrue(G.score_prediction_lines(line), line)
        good = [
            "node 7 helped: diverse_meal_options failures dropped from 12 to 3 cases",
            "the behaviour was unpredictable across cases",
            "the loop will reach the 5-call limit on long itineraries",
            "expected the tool to return JSON; it returned text",
            "mean score 0.72 over 16 cases (context)",
            "node 3 scored 0.61 on its 16 cases; node 5 0.70",
        ]
        for line in good:
            self.assertEqual(G.score_prediction_lines(line), [], line)
        errs = G.validate_memory(_memory("\nexpected score 0.9 next\n"), max_chars=20000)
        self.assertTrue(any("score prediction" in e for e in errs))

    def test_addendum_validator(self) -> None:
        self.assertEqual(G.validate_addendum("- cite the failing check names\n", max_chars=4000), [])
        self.assertTrue(G.validate_addendum(P.CORE_SENTINEL + "\nx", max_chars=4000))
        self.assertTrue(G.validate_addendum("x" * 5000, max_chars=4000))
        self.assertTrue(G.validate_addendum("- predict the score of each edit", max_chars=4000))

    def test_curation_validator(self) -> None:
        section = _section
        doc = section(3) + section(5) + P.CURATION_CROSS_HEADING + "\nx\n" + \
            P.CURATION_GRADIENT_HEADING + "\ny\n"
        self.assertEqual(G.validate_curation(doc, node_ids=[3, 5]), [])
        errs = G.validate_curation(doc, node_ids=[3, 5, 8])
        self.assertEqual(errs, ["missing section '## Node 8'"])
        partial = section(3, P.NODE_SUBSECTIONS[:-1]) + section(5) + \
            P.CURATION_CROSS_HEADING + "\n" + P.CURATION_GRADIENT_HEADING + "\n"
        errs = G.validate_curation(partial, node_ids=[3, 5])
        self.assertEqual(errs, ["section '## Node 3' lacks subsection 'Usefulness verdict'"])
        errs = G.validate_curation(section(3), node_ids=[3])
        self.assertEqual(len(errs), 2)
        self.assertIn("Task-agent utilization", P.NODE_SUBSECTIONS)

    def test_curation_edits_must_be_itemised(self) -> None:
        tail = P.CURATION_CROSS_HEADING + "\nx\n" + P.CURATION_GRADIENT_HEADING + "\ny\n"
        # Prose instead of items, or items labelled for another node -> one finding for that node.
        prose = _section(3).replace("- E3.1 — text 3", "The node added a retry and a budget check.")
        errs = G.validate_curation(prose + tail, node_ids=[3])
        self.assertEqual(len(errs), 1)
        self.assertIn("'## Node 3' / 'Edits' has no itemised edit", errs[0])
        self.assertIn("E3.<k>", errs[0])
        wrong = _section(3).replace("- E3.1 — text 3", "- E5.1 — text")
        self.assertEqual(len(G.validate_curation(wrong + tail, node_ids=[3])), 1)
        # Accepted item forms: "-", "*", "1." lists, bold or code-quoted labels, several items.
        for item in ("* E3.2 — x", "1. E3.1 — x\n2. E3.2 — y", "- **E3.1** — x", "- `E3.1`: x"):
            doc = _section(3).replace("- E3.1 — text 3", item) + tail
            self.assertEqual(G.validate_curation(doc, node_ids=[3]), [], item)
        # An item further down the subsection counts; one in another subsection does not.
        later = _section(3).replace("- E3.1 — text 3", "intro line\n\n- E3.1 — x")
        self.assertEqual(G.validate_curation(later + tail, node_ids=[3]), [])
        elsewhere = _section(3).replace("- E3.1 — text 3", "none").replace("### Intent\ntext 3", "### Intent\n- E3.1 — x")
        self.assertEqual(len(G.validate_curation(elsewhere + tail, node_ids=[3])), 1)

    def test_salvage_curation_inserts_placeholders(self) -> None:
        def section(nid, subs):
            return _section(nid, subs, suffix=" (round_005)")
        doc = section(3, P.NODE_SUBSECTIONS) + section(5, P.NODE_SUBSECTIONS[:2] + P.NODE_SUBSECTIONS[3:]) + \
            P.CURATION_CROSS_HEADING + "\nx\n"
        self.assertTrue(G.validate_curation(doc, node_ids=[3, 5, 8]))
        fixed, inserted = G.salvage_curation(doc, node_ids=[3, 5, 8])
        self.assertEqual(G.validate_curation(fixed, node_ids=[3, 5, 8]), [])
        self.assertEqual(inserted, ["## Node 5 / Editor process", "## Node 8", P.CURATION_GRADIENT_HEADING])
        # The placeholder went into node 5's section, before node 8's, and the original text is intact.
        i5, i8 = fixed.index("## Node 5"), fixed.index("## Node 8")
        self.assertIn(f"### Editor process\n{G.PLACEHOLDER}", fixed[i5:i8])
        self.assertIn("text 3", fixed); self.assertIn("text 5", fixed)
        self.assertEqual(fixed.count(G.PLACEHOLDER), 1 + len(P.NODE_SUBSECTIONS) + 1)
        # The appended node 8 section carries a placeholder EDIT ITEM under 'Edits'.
        self.assertIn(f"### Edits\n- E8.1 — {G.PLACEHOLDER}", fixed[i8:])
        # A complete document is untouched; an empty one is left alone.
        ok = section(3, P.NODE_SUBSECTIONS) + P.CURATION_CROSS_HEADING + "\nx\n" + P.CURATION_GRADIENT_HEADING + "\ny\n"
        self.assertEqual(G.salvage_curation(ok, node_ids=[3]), (ok, []))
        self.assertEqual(G.salvage_curation("  ", node_ids=[3]), ("  ", []))

    def test_salvage_curation_adds_missing_edit_item(self) -> None:
        tail = P.CURATION_CROSS_HEADING + "\nx\n" + P.CURATION_GRADIENT_HEADING + "\ny\n"
        # 'Edits' present but prose only -> a placeholder item is appended at the
        # end of that subsection (before '### Intent'); nothing else changes.
        doc = _section(3).replace("- E3.1 — text 3", "prose about the diff\nmore prose") + _section(5) + tail
        fixed, inserted = G.salvage_curation(doc, node_ids=[3, 5])
        self.assertEqual(inserted, ["## Node 3 / Edits item"])
        self.assertEqual(G.validate_curation(fixed, node_ids=[3, 5]), [])
        self.assertIn(f"### Edits\nprose about the diff\nmore prose\n- E3.1 — {G.PLACEHOLDER}\n### Intent", fixed)
        self.assertEqual(fixed.count(G.PLACEHOLDER), 1)
        # Missing 'Edits' subsection -> inserted with the placeholder item.
        doc = _section(3, P.NODE_SUBSECTIONS[1:]) + tail
        fixed, inserted = G.salvage_curation(doc, node_ids=[3])
        self.assertEqual(inserted, ["## Node 3 / Edits"])
        self.assertEqual(G.validate_curation(fixed, node_ids=[3]), [])
        self.assertIn(f"### Edits\n- E3.1 — {G.PLACEHOLDER}", fixed)
        q = P.Q_SECTIONS[0] + "\nx\n"
        fixed, inserted = G.salvage_q(q)
        self.assertEqual(G.validate_q(fixed), [])
        self.assertEqual(inserted, list(P.Q_SECTIONS[1:]))

    def test_q_validator(self) -> None:
        doc = "".join(h + "\ntext\n" for h in P.Q_SECTIONS)
        self.assertEqual(G.validate_q(doc), [])
        self.assertTrue(G.validate_q(doc.replace(P.Q_SECTIONS[2], "## other")))


@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    raw: object = None


class _ScriptedLLM:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.steps.pop(0)


class TestCalls(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.spec = G.LLMSpec(model="m", reasoning_effort="low", base_url="http://x",
                              api_key_env="K", llm_timeout_s=60,
                              extra_body={"provider": {"order": ["Baidu"], "allow_fallbacks": False}})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_generate_memory_retries_once_then_keeps_the_final_draft(self) -> None:
        llm = _ScriptedLLM([_Resp("## 1. Ranked edits\nonly one section"), _Resp(_memory())])
        out, errs = G.generate_memory(llm, self.spec, previous_memory="", curation="Z", addendum="- add x",
                                      window_meta={"window_index": 1, "nodes": [{"node_id": 1, "memory_arm": "none"}]},
                                      max_chars=20000, record_path=self.tmp / "memory_call.json")
        self.assertEqual((out, errs), (_memory(), []))
        self.assertEqual(len(llm.calls), 2)
        # kwargs threaded; the rejection is fed back on the retry.
        kw = llm.calls[0]
        self.assertEqual((kw["model"], kw["reasoning_effort"], kw["base_url"], kw["api_key_env"], kw["timeout_s"]),
                         ("m", "low", "http://x", "K", 60))
        self.assertNotIn("temperature", kw)
        self.assertEqual(kw["extra_body"], {"provider": {"order": ["Baidu"], "allow_fallbacks": False}})
        self.assertIn("previous draft was rejected", llm.calls[1]["messages"][1]["content"])
        self.assertIn("## 2. Usefulness", llm.calls[1]["messages"][1]["content"])
        sys_msg = llm.calls[0]["messages"][0]["content"]
        self.assertTrue(sys_msg.startswith(P.CORE_SENTINEL))
        self.assertIn("=== ADDENDUM", sys_msg)
        self.assertIn("- add x", sys_msg)
        self.assertNotIn("0.7", llm.calls[0]["messages"][1]["content"])   # no mean scores
        rec = json.loads((self.tmp / "memory_call.json").read_text())
        self.assertTrue(rec["accepted"])
        self.assertEqual([a["attempt"] for a in rec["attempts"]], [1, 2])
        # Two failing checks → the FINAL draft is still returned, with its findings.
        llm = _ScriptedLLM([_Resp("bad"), _Resp("bad again\nexpected score 0.9")])
        out, errs = G.generate_memory(llm, self.spec, previous_memory="B", curation="Z", addendum="",
                                      window_meta={"window_index": 2, "nodes": []}, max_chars=20000,
                                      record_path=self.tmp / "memory_call.json")
        self.assertEqual(out, "bad again\nexpected score 0.9\n")
        self.assertTrue(any("## 1. Ranked edits" in e for e in errs))
        self.assertTrue(any("score prediction" in e for e in errs))
        rec = json.loads((self.tmp / "memory_call.json").read_text())
        self.assertFalse(rec["accepted"])
        self.assertEqual(len(rec["attempts"]), 2)
        # An empty final draft is the one thing that yields no document.
        llm = _ScriptedLLM([_Resp(""), _Resp("   ")])
        out, errs = G.generate_memory(llm, self.spec, previous_memory="B", curation="Z", addendum="",
                                      window_meta={"window_index": 2, "nodes": []}, max_chars=20000,
                                      record_path=self.tmp / "m2.json")
        self.assertIsNone(out)
        self.assertEqual(errs, ["the document is empty"])

    def test_update_instruction(self) -> None:
        llm = _ScriptedLLM([_Resp("- cite failing checks\n- keep entries per mechanism")])
        out, errs = G.update_instruction(llm, self.spec, addendum="", q="Q-REPORT", previous_q=["OLD-Q"],
                                         max_chars=4000, record_path=self.tmp / "update_call.json")
        self.assertEqual((out, errs), ("- cite failing checks\n- keep entries per mechanism\n", []))
        user = llm.calls[0]["messages"][1]["content"]
        self.assertIn(P.CORE_SENTINEL, user)
        self.assertIn("Q-REPORT", user)
        self.assertIn("OLD-Q", user)
        llm = _ScriptedLLM([_Resp(P.CORE_SENTINEL), _Resp("predict the score")])
        out, errs = G.update_instruction(llm, self.spec, addendum="a", q="q", previous_q=[],
                                         max_chars=4000, record_path=self.tmp / "u.json")
        self.assertEqual(out, "predict the score\n")       # final draft kept ...
        self.assertTrue(any("score prediction" in e for e in errs))   # ... findings reported
        self.assertIn("previous draft was rejected", llm.calls[1]["messages"][1]["content"])

    def test_llm_failure_is_recorded_not_raised(self) -> None:
        def boom(**kw):
            raise RuntimeError("down")
        out, errs = G.generate_memory(boom, self.spec, previous_memory="", curation="Z", addendum="",
                                      window_meta={"window_index": 1, "nodes": []}, max_chars=100,
                                      record_path=self.tmp / "m.json")
        self.assertIsNone(out)
        self.assertIn("RuntimeError", errs[0])
        rec = json.loads((self.tmp / "m.json").read_text())
        self.assertIn("RuntimeError", rec["attempts"][0]["error"])


if __name__ == "__main__":
    unittest.main()

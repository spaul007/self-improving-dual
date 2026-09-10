"""Unit tests for the belief layer (edit_beliefs): contract-bound updates,
retry, pre-registration, scoring, optimizer wiring and the full stack.

Stub LLM, no network.

    PYTHONPATH=. python3 -m unittest tests.test_edit_beliefs
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent.belief_optimizer import INSTRUCTION_ARCHIVE_DIR, INSTRUCTION_NAME, SEED_INSTRUCTION
from meta_agent.edit_beliefs import (
    BELIEF_PREDICTION_NAME,
    BELIEF_PROMPT_DIR,
    BELIEFS_ARCHIVE_DIR,
    BELIEFS_NAME,
    BELIEFS_STATE_NAME,
    PREDICTION_NAME,
    BeliefStore,
    parse_anchors,
    parse_citations,
)
from meta_agent.edit_memory import STATE_NAME, REGISTRY_NAME
from meta_agent.edit_memory_render import render_edit_memory
from tests.test_edit_archive import write_record


@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class _Call:
    name: str
    arguments: dict


GOOD_DOC = """## Summary
One strategy tried so far.

### belief:add-verifier-helps — verifiers pay off when wired
- kind: strategy
- scope: strategy=add-verifier area=routing
- predict: p=0.65
- evidence: helped once [node 1: Δ+0.0500/8]
- next: extend the gate to intercity transfers

### belief:add-verifier-impl — verifiers often ship dead
- kind: implementation
- scope: strategy=add-verifier
- predict: p=0.40
- evidence: gate wiring is fragile [node 1: Δ+0.0500/8]
- next: wire the gate before adding more checks
"""
BAD_SIGN_DOC = GOOD_DOC.replace("[node 1: Δ+0.0500/8]", "[node 1: Δ-0.0500/8]")
BAD_SCOPE_DOC = GOOD_DOC.replace("strategy=add-verifier area=routing",
                                 "strategy=no-such-strategy")
NO_ANCHOR_DOC = "## Summary\njust prose, no belief sections\n"
ECHOED_TRACK_DOC = GOOD_DOC.replace(
    "- next: extend the gate to intercity transfers",
    "- next: extend the gate to intercity transfers\n- track: n=99 · made up")


class _BeliefStub:
    """Canned responses per tool name (OpenAI-style tools). ``docs`` are
    returned in order for successive ``submit_belief_update`` calls (the
    last one repeats)."""

    def __init__(self, doc: str = GOOD_DOC, docs=None, junk: bool = False,
                 instruction: str = "Revised guidance: weigh node counts."):
        self.docs = list(docs) if docs else [doc]
        self.junk = junk
        self.instruction = instruction
        self.calls: list[tuple[str, dict]] = []
        self._i = 0

    def __call__(self, **kw):
        name = kw["tools"][0]["function"]["name"]
        self.calls.append((name, kw))
        if self.junk:
            return _Resp(content="no tool call here")
        if name == "submit_belief_update":
            doc = self.docs[min(self._i, len(self.docs) - 1)]
            self._i += 1
            payload = {"document": doc, "change_note": "n"}
        elif name == "submit_instruction_update":
            payload = {"critique": "c", "instruction": self.instruction}
        else:
            return _Resp(content="unexpected tool " + name)
        return _Resp(tool_calls=[_Call(name=name, arguments=payload)])


@dataclass
class _Node:
    node_id: int
    parent_id: object
    round_dir: Path
    mean_utility: float = 0.5
    n_evals: int = 10
    edit_failed: bool = False
    case_results: list = field(default_factory=list)
    children: list = field(default_factory=list)


class _Tree:
    def __init__(self, nodes):
        self.nodes = {n.node_id: n for n in nodes}

    def __getitem__(self, k):
        return self.nodes[k]


def _write_state(round_dir: Path, case_sig: str) -> None:
    (round_dir / STATE_NAME).write_text(
        json.dumps({"child_case_sig": case_sig, "analysis_sig": ""}),
        encoding="utf-8")


def _registry(experiment_dir: Path) -> None:
    (experiment_dir / REGISTRY_NAME).write_text(json.dumps({
        "strategies": {"add-verifier": {
            "definition": "adds a check", "first_node": 1,
            "edits": [{"node": 1, "edit_index": 1, "name": "route-check"}]}},
        "areas": {"routing": {
            "definition": "route problems", "first_node": 1,
            "edits": [{"node": 1, "edit_index": 1, "name": "route-check"}]}},
    }), encoding="utf-8")


def _body(what="Adds a route verifier", strategy="add-verifier", area="routing",
          impl=None):
    b = ("## Edit 1\n- **name**: `route-check`\n"
         f"- **category level 1 (strategy)**: `{strategy}`\n"
         f"- **category level 2 (area)**: `{area}`\n"
         f"- **what**: {what}\n- **why**: routes were wrong")
    if impl is not None:
        # v6 implementation verdict + v7 judge effect for the sub-edit.
        b += ("\n\n## Analysis\n- **implementation**: "
              + ("sound — gate fired on case a" if impl else "unsound — never fired")
              + "\n- **implementation (edit 1)**: "
              + ("sound — gate fired on case a" if impl else "unsound — never fired")
              + "\n- **effect (edit 1)**: "
              + ("improved (strong; targets: opening_hours) — opening_hours "
                 "fixed on 6 of 8 cases where the gate fired" if impl else
                 "no_effect (strong) — never fired"))
    return b


def _state(tmp):
    return json.loads((tmp / BELIEFS_STATE_NAME).read_text(encoding="utf-8"))


class BeliefBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rd1 = write_record(self.tmp, 1, 0, _body(), delta=0.05, n_shared=8)
        _write_state(self.rd1, "sig-a")
        _registry(self.tmp)
        self.tree = _Tree([
            _Node(0, None, self.tmp / "round_000", mean_utility=0.45),
            _Node(1, 0, self.rd1)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestConventions(unittest.TestCase):
    def test_parse_anchors_and_citations(self):
        self.assertEqual(parse_anchors(GOOD_DOC),
                         ["add-verifier-helps", "add-verifier-impl"])
        cites = parse_citations(GOOD_DOC)
        self.assertEqual(len(cites), 2)
        self.assertEqual(cites[0]["slug"], "add-verifier-helps")
        self.assertEqual(cites[0]["node"], 1)
        self.assertAlmostEqual(cites[0]["delta"], 0.05)
        self.assertEqual(cites[0]["n_shared"], 8)


class TestUpdate(BeliefBase):
    def test_first_update_writes_doc_state_and_track_lines(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("### belief:add-verifier-helps", text)
        self.assertEqual(text.count("- track: no scored predictions yet"), 2)
        self.assertNotIn("Machine appendix", text)
        state = _state(self.tmp)
        self.assertEqual(state["n_updates"], 1)
        self.assertEqual(state["beliefs_index"]["add-verifier-helps"]["p"], 0.65)
        sys_prompt = stub.calls[0][1]["messages"][0]["content"]
        self.assertIn("Guidance (learned", sys_prompt)
        self.assertIn(SEED_INSTRUCTION, sys_prompt)
        self.assertIn("JUDGE", sys_prompt)          # judge-labelled scoring
        self.assertIn("[node N: improved]", sys_prompt)
        self.assertIn("Brier", sys_prompt)
        user = stub.calls[0][1]["messages"][1]["content"]
        # Judge-first run context: the score is orientation, not the goal.
        self.assertIn("## Run context\nThe run optimizes what the judge finds", user)
        self.assertIn("Score context (noisy, for orientation only): seed 0.4500/10",
                      user)
        self.assertNotIn("ABSOLUTE", user)
        self.assertIn("## Registry ids you may use in scope lines", user)
        self.assertIn("`add-verifier` — 1 node(s) — adds a check", user)
        self.assertIn("## Calibration report", user)
        self.assertIn("(none yet — this is the first update", user)
        self.assertTrue((self.tmp / INSTRUCTION_NAME).exists())
        self.assertTrue((self.tmp / BELIEF_PROMPT_DIR / "update_0001.txt").exists())

    def test_sig_gating_skips_unchanged_evidence(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        self.assertFalse(store.update(self.tmp, self.tree))
        self.assertEqual(len(stub.calls), 1)  # no second LLM call

    def test_second_update_prompt_carries_track_lines(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        store.update(self.tmp, self.tree)
        _write_state(self.rd1, "sig-changed")
        self.assertTrue(store.update(self.tmp, self.tree))
        user = stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("## Current belief document (with code-generated track lines)", user)
        self.assertIn("- track: no scored predictions yet", user)

    def test_delta_evidence_only_changed_nodes_and_whole_records(self):
        stub = _BeliefStub()
        store = BeliefStore(stub, evidence_char_budget=1000)
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 0, _body("Reworks the hotel budget"),
                           delta=-0.02, n_shared=9)
        _write_state(rd2, "sig-b")
        # Node 3 alone exceeds the (floor-clamped) 1000-char evidence budget,
        # so it is shown whole and the older node 2 is dropped whole.
        rd3 = write_record(self.tmp, 3, 0, _body("Adds a cache " * 80),
                           delta=0.01, n_shared=9)
        _write_state(rd3, "sig-c")
        self.assertTrue(store.update(self.tmp, self.tree))
        user = stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("New/changed evidence", user)
        self.assertIn("### node 3", user)            # newest, verbatim
        self.assertIn("Adds a cache", user)
        self.assertIn((rd3 / "edit_memory.md").read_text(encoding="utf-8").rstrip(), user)
        self.assertNotIn("### node 2", user)         # oldest dropped WHOLE
        self.assertIn("(+1 older changed node(s) not shown", user)
        self.assertNotIn("### node 1", user)         # unchanged node not re-sent
        self.assertNotIn("chars elided", user)

    def test_no_anchor_doc_retried_then_rejected(self):
        stub = _BeliefStub(doc=NO_ANCHOR_DOC)
        store = BeliefStore(stub)
        self.assertFalse(store.update(self.tmp, self.tree))
        self.assertEqual([n for n, _ in stub.calls],
                         ["submit_belief_update", "submit_belief_update"])
        retry = stub.calls[1][1]["messages"][1]["content"]
        self.assertIn("## Your previous submission was rejected", retry)
        self.assertIn("[HARD]", retry)
        self.assertIn("no `### belief:", retry)
        self.assertFalse((self.tmp / BELIEFS_NAME).exists())
        self.assertFalse((self.tmp / BELIEFS_STATE_NAME).exists())

    def test_retry_fixes_hard_violation(self):
        stub = _BeliefStub(docs=[BAD_SCOPE_DOC, GOOD_DOC])
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        self.assertEqual(len(stub.calls), 2)
        retry = stub.calls[1][1]["messages"][1]["content"]
        self.assertIn("no-such-strategy", retry)
        self.assertIn("### belief:add-verifier-helps",
                      (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8"))

    def test_bad_citation_is_soft_accepted_and_flagged(self):
        stub = _BeliefStub(doc=BAD_SIGN_DOC)
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        self.assertEqual(len(stub.calls), 2)  # one retry names the misquote
        retry = stub.calls[1][1]["messages"][1]["content"]
        self.assertIn("[SOFT]", retry)
        self.assertIn("Δ+0.0500 over 8 shared", retry)
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("misquoted citation", text)
        self.assertIn("Δ-0.0500", text)  # the model's text is kept, flagged

    def test_over_cap_keeps_previous_document(self):
        stub = _BeliefStub()
        BeliefStore(stub).update(self.tmp, self.tree)
        before = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        big = GOOD_DOC.replace("- next: wire the gate before adding more checks",
                               "- next: " + "wire it " * 400)
        stub2 = _BeliefStub(doc=big)
        store2 = BeliefStore(stub2, doc_char_cap=1000)
        _write_state(self.rd1, "sig-changed")
        self.assertFalse(store2.update(self.tmp, self.tree))
        self.assertEqual(len(stub2.calls), 2)
        self.assertIn("over-cap", stub2.calls[1][1]["messages"][1]["content"]
                      .replace("the cap is", "over-cap"))
        self.assertEqual((self.tmp / BELIEFS_NAME).read_text(encoding="utf-8"), before)

    def test_echoed_track_lines_are_stripped(self):
        stub = _BeliefStub(doc=ECHOED_TRACK_DOC)
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        self.assertEqual(len(stub.calls), 1)
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertNotIn("made up", text)
        self.assertEqual(text.count("- track:"), 2)

    def test_junk_output_keeps_previous_state(self):
        stub = _BeliefStub(junk=True)
        store = BeliefStore(stub)
        self.assertFalse(store.update(self.tmp, self.tree))
        self.assertFalse((self.tmp / BELIEFS_NAME).exists())
        self.assertFalse((self.tmp / BELIEFS_STATE_NAME).exists())

    def test_empty_archive_no_llm_call(self):
        empty = Path(tempfile.mkdtemp())
        try:
            stub = _BeliefStub()
            self.assertFalse(BeliefStore(stub).update(empty, self.tree))
            self.assertEqual(stub.calls, [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_disabled_store_is_inert(self):
        stub = _BeliefStub()
        self.assertFalse(BeliefStore(stub, enabled=False)
                         .update(self.tmp, self.tree))
        self.assertEqual(stub.calls, [])

    def test_render_block_is_verbatim(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        store.update(self.tmp, self.tree)
        self.assertEqual(store.render_block(self.tmp),
                         (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8"))

    def test_archive_written_on_second_update(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        store.update(self.tmp, self.tree)
        _write_state(self.rd1, "sig-changed")
        store.update(self.tmp, self.tree)
        archived = list((self.tmp / BELIEFS_ARCHIVE_DIR).glob("beliefs_*.md"))
        self.assertEqual([p.name for p in archived], ["beliefs_0001.md"])
        self.assertEqual(_state(self.tmp)["versions"], ["beliefs_0001.md"])


class TestRegistrationAndScoring(BeliefBase):
    def _store(self, **kw):
        self.stub = _BeliefStub()
        return BeliefStore(self.stub, **kw)

    def test_register_matches_most_specific_belief(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body("Adds an intercity gate"),
                           delta=0.0, n_shared=0)
        got = store.register(self.tmp, 2, rd2)
        self.assertEqual(got["strategy"]["slug"], "add-verifier-helps")
        self.assertEqual(got["strategy"]["p"], 0.65)
        self.assertEqual(got["strategy"]["scope"], {"strategy": "add-verifier",
                                                    "area": "routing"})
        self.assertEqual(got["implementation"]["slug"], "add-verifier-impl")
        self.assertEqual(got["belief_version"], 1)
        self.assertEqual(got["tags"][0]["strategy"], "add-verifier")
        self.assertIs(got["coverable"], True)   # node 1 already used add-verifier
        self.assertIn("### belief:add-verifier-helps", got["strategy"]["section"])
        on_disk = json.loads((rd2 / BELIEF_PREDICTION_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk, got)
        # Idempotent: a later call returns the frozen prediction.
        (self.tmp / BELIEFS_NAME).write_text("", encoding="utf-8")
        self.assertEqual(store.register(self.tmp, 2, rd2), got)

    def test_register_without_doc_is_uncovered(self):
        store = self._store()
        rd2 = write_record(self.tmp, 2, 1, _body(), delta=0.0, n_shared=0)
        got = store.register(self.tmp, 2, rd2)
        self.assertIsNone(got["strategy"])
        self.assertIsNone(got["implementation"])
        self.assertEqual(got["belief_version"], 0)

    def test_register_first_node_of_new_strategy_is_uncoverable(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body(strategy="add-cache", area="speed"),
                           delta=0.0, n_shared=0)
        got = store.register(self.tmp, 2, rd2)
        self.assertIs(got["coverable"], False)
        self.assertIsNone(got["strategy"])
        # Once measured, it is retired without the 0.25 silence charge.
        self._measure_as(rd2, 2, 0.05, impl=True, strategy="add-cache", area="speed")
        store.update(self.tmp, self.tree)
        state = _state(self.tmp)
        self.assertEqual(state.get("scored", []), [])
        self.assertEqual({(k["node"], k["kind"]) for k in state["skipped"]},
                         {(2, "strategy"), (2, "implementation")})
        user = self.stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("2 prediction(s) skipped, not scored", user)
        # A later update does not re-report or re-score it.
        _write_state(rd2, "sig-later")
        store.update(self.tmp, self.tree)
        self.assertEqual(len(_state(self.tmp)["skipped"]), 2)

    def _measure_as(self, rd, node, delta, impl, strategy, area):
        write_record(self.tmp, node, 1,
                     _body("Adds a cache", strategy=strategy, area=area, impl=impl),
                     delta=delta, n_shared=8)
        _write_state(rd, f"sig-measured-{node}-{delta}")

    def test_register_ignores_cap_forced_tags(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        forced = _body() + ("\n- **fit**: forced by the registry cap — nearest "
                            "id by shared token, not the tagger's choice")
        rd2 = write_record(self.tmp, 2, 1, forced, delta=0.0, n_shared=0)
        got = store.register(self.tmp, 2, rd2)
        self.assertEqual(got["tags"][0]["fit"], "forced")
        self.assertIs(got["coverable"], False)      # only a forced tag: uncoverable
        self.assertIsNone(got["strategy"])          # never matched to add-verifier-helps
        self.assertIsNone(got["implementation"])

    def test_register_unrecorded_node_is_none(self):
        store = self._store()
        rd9 = self.tmp / "round_009"
        rd9.mkdir()
        self.assertIsNone(store.register(self.tmp, 9, rd9))
        self.assertFalse((rd9 / BELIEF_PREDICTION_NAME).exists())

    def _measure(self, rd, node, delta, impl):
        write_record(self.tmp, node, 1, _body("Adds an intercity gate", impl=impl),
                     delta=delta, n_shared=8)
        _write_state(rd, f"sig-measured-{node}-{delta}")

    def test_scoring_persists_and_track_line_updates(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body("Adds an intercity gate"),
                           delta=0.0, n_shared=0)
        store.register(self.tmp, 2, rd2)
        self._measure(rd2, 2, 0.05, impl=True)
        self.assertTrue(store.update(self.tmp, self.tree))
        state = _state(self.tmp)
        by_kind = {s["kind"]: s for s in state["scored"]}
        self.assertEqual(set(by_kind), {"strategy", "implementation"})
        self.assertEqual(by_kind["strategy"]["y"], 1)
        self.assertAlmostEqual(by_kind["strategy"]["brier"], (0.65 - 1) ** 2)
        self.assertEqual(by_kind["implementation"]["y"], 1)
        self.assertEqual(state["scored_since_step"], 2)
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("- track: n=1 · Brier 0.12 (0.25 = uninformative) · outcomes: 2 yes",
                      text)
        user = self.stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("2 scored prediction(s)", user)
        self.assertIn("belief:add-verifier-helps (strategy", user)
        self.assertIn("scored prediction(s) so far", store.calibration_line(self.tmp))
        # The judge ledger leads with verdicts and targets; Δ trails as context.
        self.assertIn("## Per-strategy outcomes (the judge's verdicts per sub-edit",
                      user)
        # The registry fixture lists only node 1 (unjudged) under the
        # strategy, so its row reads "not judged yet"; the judge-first row
        # shape is what matters here (targets rendering: test_judge_signal).
        self.assertIn("- `add-verifier` — 1 node(s) (1) · judge: not judged yet · "
                      "implementation: no verdict · score (context): paired Δ "
                      "median +0.0500 — adds a check", user)
        self.assertNotIn("Deterministic per-strategy ledger", user)

    def test_unsound_node_scores_implementation_only(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body(), delta=0.0, n_shared=0)
        store.register(self.tmp, 2, rd2)
        self._measure(rd2, 2, 0.05, impl=False)
        store.update(self.tmp, self.tree)
        state = _state(self.tmp)
        self.assertEqual([s["kind"] for s in state["scored"]], ["implementation"])
        self.assertEqual(state["scored"][0]["y"], 0)
        user = self.stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("implementation unsound — strategy belief not scored", user)

    def test_no_verdict_means_not_scored(self):
        store = self._store()
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body(), delta=0.0, n_shared=0)
        store.register(self.tmp, 2, rd2)
        self._measure(rd2, 2, 0.05, impl=None)
        store.update(self.tmp, self.tree)
        self.assertEqual(_state(self.tmp).get("scored", []), [])
        user = self.stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("awaiting the analysis verdict", user)

    def test_optimizer_fires_after_optimize_every(self):
        store = self._store(optimize_every=1, optimize_min_scored=1)
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 1, _body(), delta=0.0, n_shared=0)
        store.register(self.tmp, 2, rd2)
        self._measure(rd2, 2, 0.05, impl=True)
        store.update(self.tmp, self.tree)
        names = [n for n, _ in self.stub.calls]
        self.assertIn("submit_instruction_update", names)
        self.assertLess(names.index("submit_instruction_update"),
                        len(names) - 1)  # the step precedes the rewrite
        self.assertTrue((self.tmp / INSTRUCTION_ARCHIVE_DIR / "v001.md").exists())
        self.assertEqual(_state(self.tmp)["instruction"]["version"], 1)
        self.assertEqual((self.tmp / INSTRUCTION_NAME).read_text().strip(),
                         "Revised guidance: weigh node counts.")
        # The rewrite that followed used the new guidance.
        sys_prompt = self.stub.calls[-1][1]["messages"][0]["content"]
        self.assertIn("Revised guidance: weigh node counts.", sys_prompt)

    def test_proposal_predictions_reach_the_report(self):
        store = self._store()
        rd2 = write_record(self.tmp, 2, 1, _body(), delta=0.021, n_shared=14)
        _write_state(rd2, "sig-b")
        (rd2 / PREDICTION_NAME).write_text(json.dumps({
            "belief_id": "add-verifier-helps", "expected_direction": "up",
            "expected_effect": "improved", "expected_targets": ["opening_hours"]}),
            encoding="utf-8")
        self.assertTrue(store.update(self.tmp, self.tree))
        # first update: the doc did not exist, so the report has no per-belief
        # rows yet; the join is persisted for the next one.
        joins = _state(self.tmp)["prediction_joins"]
        self.assertEqual(len(joins), 1)
        self.assertEqual((joins[0]["expected_effect"], joins[0]["expected_targets"]),
                         ("improved", ["opening_hours"]))
        _write_state(rd2, "sig-c")
        store.update(self.tmp, self.tree)
        user = self.stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("cited by 1 proposal(s)", user)


class TestRenderModes(BeliefBase):
    def test_full_mode_unchanged_by_default(self):
        default = render_edit_memory(self.tmp, focus_node_id=0)
        explicit = render_edit_memory(self.tmp, focus_node_id=0, mode="full",
                                      belief_block="ignored in full mode")
        self.assertEqual(default, explicit)
        self.assertIn("### Every edit, oldest first", default)

    def test_belief_mode_is_not_rendered_here(self):
        with self.assertRaises(ValueError):
            render_edit_memory(self.tmp, mode="belief", belief_block=GOOD_DOC)


class _AllToolsStub:
    """One stub for every meta-agent tool the full stack calls (tagger,
    belief update, guidance step). Dispatches on the OpenAI-style tool name."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, **kw):
        name = kw["tools"][0]["function"]["name"]
        self.calls.append(name)
        payload = {
            "submit_node_edits": {
                "edits": [{"name": "route-check",
                           "what": "Adds a route check in workflow",
                           "why": "routes were wrong",
                           "strategy": "add-verifier", "area": "routing"}],
                "new_category_defs": {
                    "add-verifier": "adds a check",
                    "routing": "route problems"},
            },
            "submit_belief_update": {"document": GOOD_DOC, "change_note": "n"},
            "submit_instruction_update": {"critique": "c", "instruction": "g"},
        }.get(name)
        if payload is None:
            return _Resp(content="unexpected tool " + name)
        return _Resp(tool_calls=[_Call(name=name, arguments=payload)])


class _MutatingEditor:
    """Stub editor that actually changes workflow.py so record_node fires."""

    def __init__(self):
        self.calls = 0
        self.contexts: list[str] = []

    def apply(self, feedback, base_dir, out_dir, *, context=None):
        import shutil as _sh

        from meta_agent.models import EditResult, EvolutionStrategy
        self.calls += 1
        self.contexts.append(context or "")
        src, dst = Path(base_dir) / "task_agent", Path(out_dir) / "task_agent"
        if dst.exists():
            _sh.rmtree(dst)
        _sh.copytree(src, dst)
        wf = dst / "workflow.py"
        wf.write_text(wf.read_text(encoding="utf-8")
                      + f"\n\ndef extra_{self.calls}():\n"
                        f"    return {self.calls}\n", encoding="utf-8")
        return EditResult(success=True, edited_files=["workflow.py"],
                          strategy=EvolutionStrategy(
                              target_files=["workflow.py"],
                              optimization_goal=f"stub edit {self.calls}",
                              proposed_changes="stub", rationale=""))


class _ScoredEvaluator:
    def run(self, round_dir, benchmark_dir, *, case_ids=None):
        from meta_agent.models import CaseResult, EvaluationResult
        per_case = [CaseResult(case_id=cid,
                               passed=(hash(cid) % 1000) / 1000.0 >= 0.5,
                               score=(hash(cid) % 1000) / 1000.0)
                    for cid in (case_ids or [])]
        passed = sum(1 for c in per_case if c.passed)
        return EvaluationResult(
            score=sum(c.score for c in per_case) / max(len(per_case), 1),
            passed=passed, failed=len(per_case) - passed, per_case=per_case)


class TestManagerIntegration(unittest.TestCase):
    """evolve() with the real EditMemory: belief updates fire after expands
    and eval batches, predictions are pre-registered per node, code records
    are written, steering is the belief-mode block."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="beliefs_evolve_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        seed = self.tmp / "seed"
        seed.mkdir()
        (seed / "workflow.py").write_text(
            "def run_task(task):\n    return None\n", encoding="utf-8")
        self.seed = seed
        self.experiment = self.tmp / "exp"
        self.experiment.mkdir()

    def test_full_stack(self):
        from meta_agent.edit_code import CODE_NAME
        from meta_agent.edit_memory import EditMemory, RECORD_NAME
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.managers.hgm import HGMManager

        stub = _AllToolsStub()
        em = EditMemory(stub, setup_pass=False, usage_tracking=False,
                        analysis_mode="off", steering_mode="belief",
                        beliefs={"enabled": True})
        editor = _MutatingEditor()
        manager = HGMManager(eval_budget=24, init_expansions=2,
                             eval_batch_size=4, alpha=0.6, seed=7)
        manager.evolve(
            editor=editor, evaluator=_ScoredEvaluator(),
            gatherer=DefaultFeedbackGatherer(),
            seed_dir=self.seed, benchmark_dir=self.tmp / "bench",
            experiment_dir=self.experiment, max_rounds=10, score_target=None,
            train_case_ids=[f"c{i}" for i in range(12)], eval_case_ids=None,
            edit_memory=em)

        # Belief artifacts exist and advanced with the run.
        self.assertTrue((self.experiment / BELIEFS_NAME).exists())
        state = json.loads((self.experiment / BELIEFS_STATE_NAME)
                           .read_text(encoding="utf-8"))
        self.assertGreaterEqual(state["n_updates"], 1)
        self.assertGreaterEqual(stub.calls.count("submit_belief_update"), 1)
        self.assertTrue((self.experiment / INSTRUCTION_NAME).exists())

        # Every recorded round carries a pre-registered prediction.
        recorded = list(self.experiment.glob(f"round_*/{RECORD_NAME}"))
        self.assertGreater(len(recorded), 0)
        for rec in recorded:
            self.assertTrue((rec.parent / BELIEF_PREDICTION_NAME).exists(),
                            rec.parent.name)
        covered = [json.loads((r.parent / BELIEF_PREDICTION_NAME).read_text())
                   for r in recorded]
        self.assertTrue(any(c["strategy"] is not None for c in covered))

        # Code records exist for recorded (non-seed) rounds.
        self.assertGreater(len(list(self.experiment.glob(f"round_*/{CODE_NAME}"))), 0)

        # Steering is the belief-mode block, never the legacy text.
        self.assertTrue(all(c.startswith("## Objective") for c in editor.contexts))
        self.assertTrue(any("### belief:add-verifier-helps" in c
                            for c in editor.contexts))
        for c in editor.contexts:
            self.assertNotIn("aim to beat it", c)
            self.assertNotIn("### Every edit, oldest first", c)
            self.assertNotIn("chars elided", c)


if __name__ == "__main__":
    unittest.main()


class TestDeltaAblation(BeliefBase):
    """`label_source: delta` keeps the pre-judge wording verbatim, so the
    ablation isolates the label source and nothing else."""

    def test_delta_mode_keeps_the_legacy_framing(self):
        stub = _BeliefStub()
        store = BeliefStore(stub, label_source="delta")
        self.assertTrue(store.update(self.tmp, self.tree))
        sys_prompt = stub.calls[0][1]["messages"][0]["content"]
        self.assertIn("Δ vs parent ≥", sys_prompt)
        self.assertNotIn("The judge is the per-node analysis", sys_prompt)
        user = stub.calls[0][1]["messages"][1]["content"]
        self.assertIn("The goal is the highest ABSOLUTE score.", user)
        self.assertIn("## Deterministic per-strategy ledger (ground truth)", user)
        self.assertNotIn("Per-strategy outcomes (the judge's verdicts", user)

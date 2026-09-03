"""Unit tests for the belief layer (edit_beliefs) and belief-led steering.

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

from meta_agent.edit_beliefs import (
    BELIEFS_ARCHIVE_DIR,
    BELIEFS_NAME,
    BELIEFS_STATE_NAME,
    MACHINE_SECTION,
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


GOOD_DOC = """## Document structure
Organized one section per strategy; machine appendix is code-owned.

### belief:add-verifier — verification helps but gates need care
- stance: mixed — helped once [node 1: Δ+0.0500/8]
- evidence: one node only; not sufficient for a confident verdict
- next move: repair the gate check before trying again
"""

BAD_SIGN_DOC = GOOD_DOC.replace("[node 1: Δ+0.0500/8]",
                                "[node 1: Δ-0.0500/8]")
NO_ANCHOR_DOC = "## Document structure\njust prose, no belief sections\n"


class _BeliefStub:
    """Canned responses per tool name (OpenAI-style tools)."""

    def __init__(self, doc: str = GOOD_DOC, directives=None,
                 junk: bool = False):
        self.doc = doc
        self.directives = directives or ["split add-verifier by gate role"]
        self.junk = junk
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, **kw):
        name = kw["tools"][0]["function"]["name"]
        self.calls.append((name, kw))
        if self.junk:
            return _Resp(content="no tool call here")
        payload = {
            "submit_belief_update": {"document": self.doc, "change_note": "n"},
            "submit_belief_reflection": {"directives": self.directives},
        }[name]
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
        "areas": {},
    }), encoding="utf-8")


class BeliefBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rd1 = write_record(self.tmp, 1, 0,
                                "## Edit 1\n- **what**: Adds a route verifier",
                                delta=0.05, n_shared=8)
        _write_state(self.rd1, "sig-a")
        _registry(self.tmp)
        self.tree = _Tree([
            _Node(0, None, self.tmp / "round_000", mean_utility=0.45),
            _Node(1, 0, self.rd1)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestConventions(unittest.TestCase):
    def test_parse_anchors_and_citations(self):
        self.assertEqual(parse_anchors(GOOD_DOC), ["add-verifier"])
        cites = parse_citations(GOOD_DOC)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["slug"], "add-verifier")
        self.assertEqual(cites[0]["node"], 1)
        self.assertAlmostEqual(cites[0]["delta"], 0.05)
        self.assertEqual(cites[0]["n_shared"], 8)

    def test_snake_case_slug_parses_fully(self):
        # Models often write snake_case despite the kebab-case suggestion;
        # the anchor must not truncate at the first underscore (real bug,
        # caught in the 2026-09-02 smoke run).
        doc = "### belief:transfer_time_accuracy — t\n[node 1: Δ-0.0273/16]\n"
        self.assertEqual(parse_anchors(doc), ["transfer_time_accuracy"])
        self.assertEqual(parse_citations(doc)[0]["slug"],
                         "transfer_time_accuracy")

    def test_unmeasured_citation_parses(self):
        cites = parse_citations("### belief:x — t\n[node 3: unmeasured]\n")
        self.assertIsNone(cites[0]["delta"])


class TestUpdate(BeliefBase):
    def test_first_update_writes_doc_sidecar_and_appendix(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("### belief:add-verifier", text)
        self.assertIn(MACHINE_SECTION, text)
        self.assertIn("citations match the records", text)
        self.assertIn("belief:add-verifier cites 1 node(s) — 1 — covering 8 "
                      "shared case(s)", text)
        state = json.loads((self.tmp / BELIEFS_STATE_NAME)
                           .read_text(encoding="utf-8"))
        self.assertEqual(state["n_updates"], 1)
        # Noise policy travels in the system prompt.
        sys_prompt = stub.calls[0][1]["messages"][0]["content"]
        self.assertIn("sampling noise", sys_prompt)

    def test_sig_gating_skips_unchanged_evidence(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        self.assertFalse(store.update(self.tmp, self.tree))
        self.assertEqual(len(stub.calls), 1)  # no second LLM call

    def test_delta_evidence_only_changed_nodes(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        store.update(self.tmp, self.tree)
        rd2 = write_record(self.tmp, 2, 0,
                           "## Edit 1\n- **what**: Reworks the hotel budget",
                           delta=-0.02, n_shared=9)
        _write_state(rd2, "sig-b")
        self.assertTrue(store.update(self.tmp, self.tree))
        user = stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("New/changed evidence", user)
        self.assertIn("### node 2", user)
        self.assertNotIn("### node 1", user)  # unchanged node not re-sent

    def test_no_anchor_doc_rejected(self):
        stub = _BeliefStub(doc=NO_ANCHOR_DOC)
        store = BeliefStore(stub)
        self.assertFalse(store.update(self.tmp, self.tree))
        self.assertFalse((self.tmp / BELIEFS_NAME).exists())

    def test_bad_citation_flagged_not_dropped(self):
        stub = _BeliefStub(doc=BAD_SIGN_DOC)
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("misquotes the record", text)
        self.assertIn("Δ+0.0500", text)  # the corrected number is narrated

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


class TestPredictionsAndReflection(BeliefBase):
    def test_prediction_join_in_prompt_and_appendix(self):
        rd2 = write_record(self.tmp, 2, 1,
                           "## Edit 1\n- **what**: Gate repair",
                           delta=0.021, n_shared=14)
        _write_state(rd2, "sig-b")
        (rd2 / PREDICTION_NAME).write_text(json.dumps({
            "belief_id": "add-verifier", "expected_direction": "up",
            "why": "gate repair should help"}), encoding="utf-8")
        rd3 = self.tmp / "round_003"
        rd3.mkdir()
        (rd3 / PREDICTION_NAME).write_text(json.dumps({
            "belief_id": "add-verifier", "expected_direction": "up"}),
            encoding="utf-8")  # no record yet -> unmeasured
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        user = stub.calls[0][1]["messages"][1]["content"]
        self.assertIn("Predictions made by past edit proposals", user)
        self.assertIn("measured Δ+0.0210 over 14 shared", user)
        self.assertIn("not yet measured", user)
        text = (self.tmp / BELIEFS_NAME).read_text(encoding="utf-8")
        self.assertIn("Proposal outcomes", text)
        self.assertIn("belief:add-verifier justified the edit(s)", text)
        state = json.loads((self.tmp / BELIEFS_STATE_NAME)
                           .read_text(encoding="utf-8"))
        self.assertEqual(len(state["prediction_joins"]), 2)

    def test_new_measurement_retriggers_update(self):
        rd3 = self.tmp / "round_003"
        rd3.mkdir()
        (rd3 / PREDICTION_NAME).write_text(json.dumps({
            "belief_id": "add-verifier", "expected_direction": "up"}),
            encoding="utf-8")
        stub = _BeliefStub()
        store = BeliefStore(stub)
        self.assertTrue(store.update(self.tmp, self.tree))
        # The prediction's outcome lands (record appears) with no other change:
        write_record(self.tmp, 3, 1, "## Edit 1\n- **what**: x",
                     delta=-0.01, n_shared=9)
        _write_state(self.tmp / "round_003", "sig-c")
        self.assertTrue(store.update(self.tmp, self.tree))

    def test_reflection_fires_and_directives_reach_next_update(self):
        stub = _BeliefStub()
        store = BeliefStore(stub, reflect_every=1)
        self.assertTrue(store.update(self.tmp, self.tree))  # n_updates -> 1
        _write_state(self.rd1, "sig-changed")
        self.assertTrue(store.update(self.tmp, self.tree))
        names = [n for n, _kw in stub.calls]
        self.assertEqual(names, ["submit_belief_update",
                                 "submit_belief_reflection",
                                 "submit_belief_update"])
        user = stub.calls[-1][1]["messages"][1]["content"]
        self.assertIn("Representation directives", user)
        self.assertIn("split add-verifier by gate role", user)

    def test_archive_written_on_second_update(self):
        stub = _BeliefStub()
        store = BeliefStore(stub)
        store.update(self.tmp, self.tree)
        _write_state(self.rd1, "sig-changed")
        store.update(self.tmp, self.tree)
        archived = list((self.tmp / BELIEFS_ARCHIVE_DIR).glob("beliefs_*.md"))
        self.assertEqual([p.name for p in archived], ["beliefs_0001.md"])
        state = json.loads((self.tmp / BELIEFS_STATE_NAME)
                           .read_text(encoding="utf-8"))
        self.assertEqual(state["versions"], ["beliefs_0001.md"])


class TestRenderModes(BeliefBase):
    def test_belief_mode_replaces_dump(self):
        out = render_edit_memory(
            self.tmp, focus_node_id=0, mode="belief",
            belief_block=GOOD_DOC)
        self.assertIn("Belief document", out)
        self.assertIn("How to read the belief document", out)
        self.assertIn("### belief:add-verifier", out)
        self.assertIn("What has been tried, by strategy", out)
        self.assertIn("Edits already tried directly off node 0", out)
        self.assertNotIn("### Every edit, oldest first", out)

    def test_belief_mode_without_block_still_renders_ledger(self):
        out = render_edit_memory(self.tmp, mode="belief", belief_block="")
        self.assertIn("What has been tried, by strategy", out)
        self.assertNotIn("Belief document", out)

    def test_full_mode_unchanged_by_default(self):
        default = render_edit_memory(self.tmp, focus_node_id=0)
        explicit = render_edit_memory(self.tmp, focus_node_id=0, mode="full",
                                      belief_block="ignored in full mode")
        self.assertEqual(default, explicit)
        self.assertIn("### Every edit, oldest first", default)


class _AllToolsStub:
    """One stub for every meta-agent tool the full stack calls (tagger,
    belief update, reflection). Dispatches on the OpenAI-style tool name."""

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
            "submit_belief_reflection": {"directives": ["keep it simple"]},
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
    and eval batches, code records are written, steering is belief-led."""

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
        from meta_agent.edit_memory import EditMemory
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

        # Code records exist for recorded (non-seed) rounds.
        code_files = list(self.experiment.glob(f"round_*/{CODE_NAME}"))
        self.assertGreater(len(code_files), 0)

        # Steering is belief-led: after the first belief update, expand
        # contexts carry the belief document and never the record dump.
        self.assertTrue(any("Belief document" in c for c in editor.contexts))
        self.assertFalse(any("### Every edit, oldest first" in c
                             for c in editor.contexts))


if __name__ == "__main__":
    unittest.main()

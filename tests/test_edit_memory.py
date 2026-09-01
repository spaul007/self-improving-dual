"""Unit tests for the edit-memory subsystem — deterministic halves, the
markdown record contract, the candidate/registry split, and idempotency.

Uses a stub LLM (no network).

    PYTHONPATH=. python3 -m unittest tests.test_edit_memory
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from meta_agent.edit_diff import changed_mutable_files, diff_mutable_files, truncate_middle
from meta_agent.edit_memory import EditMemory, case_sig, render_record, split_record
from meta_agent.edit_memory_render import render_edit_memory
from meta_agent.edit_outcome import (
    classify, compact_check, compute_outcome, extract_checks, family_of,
    select_checks, validate_recipe,
)
from meta_agent.models import CaseResult


# --------------------------------------------------------------------------- #
@dataclass
class _Resp:
    content: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class _Call:
    name: str
    arguments: dict


class _StubLLM:
    """Returns a canned payload per tool name; records the prompts it saw."""

    def __init__(self, setup=None, node=None, analysis=None):
        self.calls: list[dict] = []
        self.setup = setup or {
            "strategies": [{"id": "add-verifier", "definition": "adds a check"},
                           {"id": "unused-proxy", "definition": "never used"}],
            "areas": [{"id": "routing", "definition": "route problems"}],
            "per_check_recipe": {"mode": "list", "path": "failed_checks"},
        }
        self.node = node or {
            "edits": [{"name": "route check", "what": "Adds a route check in workflow.py",
                       "why": "routes were wrong", "strategy": "add-verifier",
                       "area": "routing"}],
            "new_category_defs": {},
        }
        self.analysis = analysis or {
            "components": [{"component": "route_check/main",
                            "activated": "1x in observed batches",
                            "verdict_behavior": "pass 1",
                            "agreement": "case a: pass/pass",
                            "cause": "worked as intended"}],
            "targeted_constraints": [{"constraint": "x",
                                      "remaining_failures": "0/1",
                                      "was": "1/1",
                                      "evidence": "case a fixed"}],
            "collateral": "none observed",
        }

    def __call__(self, **kw):
        self.calls.append(kw)
        name = kw["tools"][0]["function"]["name"]
        payload = {"submit_setup": self.setup,
                   "submit_node_edits": self.node,
                   "submit_edit_analysis": self.analysis}.get(name, self.node)
        return _Resp(tool_calls=[_Call(name=name, arguments=payload)])


def _agent(round_dir: Path, workflow: str) -> None:
    a = round_dir / "task_agent"
    (a / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (a / "workflow.py").write_text(workflow, encoding="utf-8")
    (a / "tool_wrapper.py").write_text("def x(): return None\n", encoding="utf-8")
    (a / "tools_schema.json").write_text("[]", encoding="utf-8")
    (a / "mutable_tools" / "__init__.py").write_text("", encoding="utf-8")


def _case(cid, score, checks=None):
    return CaseResult(case_id=cid, passed=score >= 1.0, score=score,
                      details={"failed_checks": list(checks or [])})


def _dead_case(cid, score=0.0):
    """A case whose agent produced nothing gradeable: scored, but no detail
    block written. Distinct from a case that passed every check."""
    return CaseResult(case_id=cid, passed=False, score=score,
                      details={"error": "traceback ...", "hard_score": 0.0})


@dataclass
class _Node:
    node_id: int
    parent_id: object
    round_dir: Path
    case_results: list = field(default_factory=list)
    children: list = field(default_factory=list)
    edit_failed: bool = False


class _Tree:
    def __init__(self, nodes):
        self.nodes = {n.node_id: n for n in nodes}

    def __getitem__(self, k):
        return self.nodes[k]


# --------------------------------------------------------------------------- #
class TestOutcome(unittest.TestCase):
    def test_delta_uses_only_shared_cases(self):
        parent = [_case("a", 0.0), _case("b", 0.5), _case("c", 0.5)]
        child = [_case("b", 1.0), _case("c", 1.0), _case("d", 0.0)]
        oc = compute_outcome(parent, child, min_shared=2)
        self.assertEqual(oc.n_shared, 2)             # a and d excluded
        self.assertAlmostEqual(oc.delta_shared, 0.5)
        # the unshared view is confounded and must differ here
        self.assertNotAlmostEqual(oc.delta_all, oc.delta_shared)

    def test_verdict_thresholds(self):
        for delta, want in ((0.0208, "helped"), (0.0167, "neutral"),
                            (-0.0177, "neutral"), (-0.0219, "hurt")):
            self.assertEqual(classify(delta), want, delta)

    def test_thin_overlap_is_inconclusive_not_neutral(self):
        oc = compute_outcome([_case("a", 0.0)], [_case("a", 1.0)], min_shared=8)
        self.assertEqual(oc.verdict, "inconclusive")

    def test_scores_clamped_like_node_record(self):
        oc = compute_outcome([_case("a", -0.5)], [_case("a", 1.4)], min_shared=1)
        self.assertEqual(oc.parent_mean_shared, 0.0)
        self.assertEqual(oc.child_mean_shared, 1.0)

    def test_per_check_delta_sign_and_absence(self):
        parent = [_case("a", 0.0, ["x", "y"]), _case("b", 0.0, ["x"])]
        child = [_case("a", 1.0, []), _case("b", 1.0, ["x"])]
        oc = compute_outcome(parent, child, min_shared=1,
                             recipe={"mode": "list", "path": "details.failed_checks"})
        self.assertEqual(oc.per_check_delta["x"], 1)   # parent 2 fails, child 1
        self.assertEqual(oc.per_check_delta["y"], 1)
        # no recipe -> empty, never an exception
        self.assertEqual(compute_outcome(parent, child, min_shared=1).per_check_delta, {})


class TestMissingCheckData(unittest.TestCase):
    """A case that produced no check data must not be read as a case that
    passed every check.

    In a real 28-round run this inverted the meaning of the line: the three
    worst regressions in the tree were each reported as a clean sweep of
    "fixed" criteria, because the child crashed and wrote no detail block, so
    every criterion its parent failed scored +1.
    """

    RECIPE = {"mode": "list", "path": "details.failed_checks"}

    def test_absent_and_empty_are_distinguished(self):
        perfect = _case("a", 1.0, [])          # present and empty
        dead = _dead_case("b")                 # key absent entirely
        self.assertEqual(extract_checks(perfect, **self.RECIPE), set())
        self.assertIsNone(extract_checks(dead, **self.RECIPE))

    def test_wrong_shape_is_unusable_not_empty(self):
        bogus = CaseResult(case_id="a", passed=False, score=0.0,
                           details={"failed_checks": {"not": "a list"}})
        self.assertIsNone(extract_checks(bogus, **self.RECIPE))

    def test_dead_cases_are_excluded_from_the_tally(self):
        parent = [_case("a", 0.5, ["x"]), _case("b", 0.5, ["x"]),
                  _case("c", 0.5, ["x", "y"])]
        child = [_case("a", 1.0, []), _dead_case("b"), _dead_case("c")]
        oc = compute_outcome(parent, child, min_shared=1, recipe=self.RECIPE)
        self.assertEqual(oc.n_shared, 3)
        self.assertEqual(oc.n_check_cases, 1)      # only case a counts
        self.assertEqual(oc.per_check_delta, {"x": 1})
        self.assertNotIn("y", oc.per_check_delta)  # came only from a dead case

    def test_a_child_that_died_everywhere_reports_nothing(self):
        """The exact shape that produced the inverted lines: the child scores
        0.0 on every shared case, so the tally must be empty rather than a list
        of fixes."""
        parent = [_case(str(i), 0.75, ["x", "y", "z"]) for i in range(8)]
        child = [_dead_case(str(i)) for i in range(8)]
        oc = compute_outcome(parent, child, min_shared=1, recipe=self.RECIPE)
        self.assertLess(oc.delta_shared, -0.5)
        self.assertEqual(oc.n_check_cases, 0)
        self.assertEqual(oc.per_check_delta, {})

    def test_regression_shaped_like_the_run_that_surfaced_this(self):
        """16 shared cases, 3 of which the child failed to grade. The child
        genuinely broke one criterion and fixed another; the dead cases must
        not drown that out with false positives."""
        parent = ([_case(str(i), 0.7, ["broke_this"]) for i in range(6)]
                  + [_case(str(i), 0.7, ["fixed_this"]) for i in range(6, 13)]
                  + [_case(str(i), 0.8, ["dead_only"]) for i in range(13, 16)])
        child = ([_case(str(i), 0.6, ["broke_this", "also_broke"]) for i in range(6)]
                 + [_case(str(i), 0.6, []) for i in range(6, 13)]
                 + [_dead_case(str(i)) for i in range(13, 16)])
        oc = compute_outcome(parent, child, min_shared=8, recipe=self.RECIPE)
        self.assertEqual(oc.n_shared, 16)
        self.assertEqual(oc.n_check_cases, 13)
        self.assertEqual(oc.verdict, "hurt")
        self.assertEqual(oc.per_check_delta["fixed_this"], 7)
        self.assertEqual(oc.per_check_delta["also_broke"], -6)
        # The criterion seen only on the ungraded cases must not appear at all.
        self.assertNotIn("dead_only", oc.per_check_delta)


class TestCheckSelection(unittest.TestCase):
    """Every criterion that moved is reported; the cap is a safety valve, and
    when it binds it must not starve a whole family of criteria."""

    def test_nothing_is_truncated_below_the_cap(self):
        deltas = {f"c{i}": (i % 5) + 1 for i in range(25)}
        self.assertEqual(len(select_checks(deltas)), 25)

    def test_flat_ids_fall_back_to_a_global_top_k(self):
        deltas = {f"c{i}": i + 1 for i in range(10)}
        got = select_checks(deltas, top_k=3)
        self.assertEqual(list(got), ["c9", "c8", "c7"])

    def test_a_small_magnitude_family_still_gets_slots(self):
        """Criteria drawn per-case move in +-1 steps and would lose every
        magnitude comparison against criteria evaluated on every case."""
        deltas = {f"big:c{i}": 20 - i for i in range(10)}
        deltas.update({f"small:c{i}": 1 for i in range(10)})
        got = select_checks(deltas, top_k=6)
        self.assertEqual(sum(1 for k in got if k.startswith("small:")), 3)
        self.assertEqual(sum(1 for k in got if k.startswith("big:")), 3)

    def test_a_short_family_does_not_waste_slots(self):
        deltas = {f"big:c{i}": 20 - i for i in range(10)}
        deltas["small:only"] = 1
        got = select_checks(deltas, top_k=6)
        self.assertEqual(len(got), 6)
        self.assertEqual(sum(1 for k in got if k.startswith("small:")), 1)

    def test_compaction_drops_middle_segments_only(self):
        self.assertEqual(compact_check("commonsense:Activity Diversity:diverse_meals"),
                         "commonsense:diverse_meals")
        self.assertEqual(compact_check("hard:budget_constraint"),
                         "hard:budget_constraint")   # only two segments
        self.assertEqual(compact_check("flat_id"), "flat_id")
        # the original separator survives
        self.assertEqual(compact_check("a/b/c"), "a/c")

    def test_family_of(self):
        self.assertEqual(family_of("hard:x"), "hard")
        self.assertEqual(family_of("flat"), "")


class TestRecipe(unittest.TestCase):
    def test_all_modes(self):
        c = {"details": {"lst": ["a"], "keys": {"k": 1},
                         "flags": {"ok": {"passed": True}, "bad": {"passed": False}}}}
        self.assertEqual(extract_checks(c, "list", "details.lst"), {"a"})
        self.assertEqual(extract_checks(c, "dict_keys", "details.keys"), {"k"})
        self.assertEqual(extract_checks(c, "dict_flags", "details.flags"), {"bad"})
        # "none" means the project exposes no per-case failure detail, so the
        # answer is "no usable data" (None), not "nothing failed" (set()).
        self.assertIsNone(extract_checks(c, "none", "details.lst"))

    def test_validate_recovers_missing_container_prefix(self):
        sample = [{"details": {"failed_checks": ["x"]}}]
        # model drops the "details." prefix - the common failure
        got = validate_recipe({"mode": "list", "path": "failed_checks"}, sample)
        self.assertEqual(got, {"mode": "list", "path": "details.failed_checks"})

    def test_validate_falls_back_to_none(self):
        self.assertEqual(validate_recipe({"mode": "list", "path": "nope"}, [{"a": 1}]),
                         {"mode": "none", "path": ""})

    def test_empty_proposal_does_not_grab_the_whole_details_object(self):
        sample = [{"details": {"failed_checks": ["x"], "score": 1, "query": "q"}}]
        for proposed in (None, {"mode": "list", "path": ""}):
            got = validate_recipe(proposed, sample)
            self.assertEqual(got, {"mode": "list", "path": "details.failed_checks"},
                             f"proposed={proposed}")

    def test_shape_beats_a_wrong_proposed_mode(self):
        # {name: {passed: bool}} must resolve to dict_flags. dict_keys would
        # return every criterion, so parent and child sets match and every
        # delta cancels to zero -- silently empty rather than an error.
        sample = [{"details": {"hard_constraints": {
            "a": {"passed": True}, "b": {"passed": False}}}}]
        got = validate_recipe({"mode": "dict_keys", "path": "details.hard_constraints"},
                              sample)
        self.assertEqual(got, {"mode": "dict_flags", "path": "details.hard_constraints"})

    def test_wrong_mode_would_have_cancelled_to_zero(self):
        parent = [CaseResult(case_id="a", passed=False, score=0.0, details={
            "hard_constraints": {"x": {"passed": False}, "y": {"passed": True}}})]
        child = [CaseResult(case_id="a", passed=True, score=1.0, details={
            "hard_constraints": {"x": {"passed": True}, "y": {"passed": True}}})]
        flags = compute_outcome(parent, child, min_shared=1, recipe={
            "mode": "dict_flags", "path": "details.hard_constraints"})
        keys = compute_outcome(parent, child, min_shared=1, recipe={
            "mode": "dict_keys", "path": "details.hard_constraints"})
        self.assertEqual(flags.per_check_delta, {"x": 1})
        self.assertEqual(keys.per_check_delta, {})   # the failure being guarded


class TestRecordFormat(unittest.TestCase):
    def test_split_and_refresh_preserves_body_bytes(self):
        body = "# Node 5\n\n## Edit 1\n- **name**: `x`\n- **what**: y"
        from meta_agent.edit_outcome import EditOutcome
        text = render_record({"node": 5, "parent": 1}, body, EditOutcome())
        fm, got_body = split_record(text)
        self.assertEqual(fm["node"], "5")
        self.assertEqual(got_body, body)
        # re-rendering with a new outcome must not disturb the body
        oc = compute_outcome([_case("a", 0.0)], [_case("a", 1.0)], min_shared=1)
        again = render_record(fm, got_body, oc)
        self.assertEqual(split_record(again)[1], body)
        self.assertIn("performance", again)

    def test_case_sig_is_order_independent(self):
        a = [_case("x", 0.5), _case("y", 1.0)]
        self.assertEqual(case_sig(a), case_sig(list(reversed(a))))
        self.assertNotEqual(case_sig(a), case_sig([_case("x", 0.6), _case("y", 1.0)]))


class TestEditMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p, self.c = self.tmp / "round_000", self.tmp / "round_001"
        _agent(self.p, "def run_task(t):\n    return 1\n")
        _agent(self.c, "def run_task(t):\n    return 2\n\ndef check():\n    pass\n")
        self.llm = _StubLLM()
        self.em = EditMemory(self.llm, min_shared=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _setup(self):
        self.em.setup(self.tmp, self.p, [_case("a", 0.0, ["x"])])

    def test_candidates_never_enter_registry_unused(self):
        self._setup()
        cand = json.loads((self.tmp / "edit_memory_candidates.json").read_text())
        reg = json.loads((self.tmp / "edit_memory_registry.json").read_text())
        self.assertIn("unused-proxy", cand["strategies"])
        self.assertEqual(reg["strategies"], {})       # nothing used yet
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        reg = json.loads((self.tmp / "edit_memory_registry.json").read_text())
        # promoted at first use, definition carried over from the candidate pool
        self.assertIn("add-verifier", reg["strategies"])
        self.assertEqual(reg["strategies"]["add-verifier"]["definition"], "adds a check")
        # the unused proxy is absent from the registry at every point
        self.assertNotIn("unused-proxy", reg["strategies"])
        self.assertFalse([k for k, v in reg["strategies"].items() if v["n_nodes"] == 0])

    def test_registry_lists_edits_with_names_matching_the_record(self):
        self._setup()
        path = self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                                   node_id=1, parent_id=0, ancestors=[0])
        text = path.read_text()
        self.assertIn("- **name**: `route-check`", text)
        reg = json.loads((self.tmp / "edit_memory_registry.json").read_text())
        rows = reg["strategies"]["add-verifier"]["edits"]
        self.assertEqual(rows, [{"node": 1, "edit_index": 1, "name": "route-check"}])

    def test_record_has_five_keys_in_order_and_no_diff_hash(self):
        self._setup()
        text = self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                                   node_id=1, parent_id=0, ancestors=[0]).read_text()
        keys = [l.split("**")[1] for l in text.splitlines()
                if l.startswith("- **") and "Edit" not in l]
        self.assertEqual(keys[:5], ["name", "category level 1 (strategy)",
                                    "category level 2 (area)", "what", "why"])
        self.assertNotIn("diff_hash", text)
        self.assertNotIn("verdict", text)   # derived at render, never stored

    def test_paths_stripped_but_word_lists_survive(self):
        self.llm.node = {"edits": [{
            "name": "n", "strategy": "add-verifier", "area": "routing",
            "what": "Edits workflow.py to add a check",
            "why": "constraints on transport/hotel/restaurant choices"}]}
        self._setup()
        text = self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                                   node_id=1, parent_id=0, ancestors=[0]).read_text()
        self.assertNotIn("workflow.py", text)
        self.assertIn("transport/hotel/restaurant", text)   # bare slashes preserved

    def test_second_pass_makes_no_llm_call(self):
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        n = len(self.llm.calls)
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        self.assertEqual(len(self.llm.calls), n)    # presence-based freeze

    def test_refresh_is_radius_one_and_skips_when_unchanged(self):
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        gc = self.tmp / "round_002"
        _agent(gc, "def run_task(t):\n    return 3\n")
        self.em.record_node(round_dir=gc, parent_round_dir=self.c,
                            node_id=2, parent_id=1, ancestors=[0, 1])
        root = _Node(0, None, self.p, [_case("a", 0.0)], children=[1])
        n1 = _Node(1, 0, self.c, [_case("a", 1.0)], children=[2])
        n2 = _Node(2, 1, gc, [_case("a", 1.0)])
        tree = _Tree([root, n1, n2])
        self.assertEqual(self.em.refresh_outcomes(tree, 0), 1)   # only child 1
        self.assertEqual(self.em.refresh_outcomes(tree, 0), 0)   # skip guard holds
        self.assertIn("performance", (self.c / "edit_memory.md").read_text())

    def test_llm_failure_writes_nothing_and_does_not_raise(self):
        class _Dead:
            def __call__(self, **kw):
                raise RuntimeError("boom")
        em = EditMemory(_Dead())
        em.setup(self.tmp, self.p, [_case("a", 0.0)])
        self.assertIsNone(em.record_node(round_dir=self.c, parent_round_dir=self.p,
                                         node_id=1, parent_id=0, ancestors=[0]))

    def test_checks_moved_no_longer_rendered(self):
        # per_check_delta is still computed (it grounds the analysis prompt)
        # but the "checks moved" line was replaced by the analysis section's
        # constraint-effect line — it must appear nowhere.
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        root = _Node(0, None, self.p, [_case("a", 0.0, ["x"])], children=[1])
        n1 = _Node(1, 0, self.c, [_case("a", 1.0, [])])
        self.em.refresh_outcomes(_Tree([root, n1]), 0)
        self.assertNotIn("checks moved", (self.c / "edit_memory.md").read_text())
        block = render_edit_memory(self.tmp, token_budget=48000, min_shared=1)
        self.assertNotIn("checks moved", block)

    def test_usage_and_analysis_flow_end_to_end(self):
        """record -> sidecar; eval batch trace -> usage lines with an explicit
        0-calls, and an LLM analysis section; both reach the steering block;
        a second refresh makes no further write and no further LLM call."""
        # child adds an instrumented log point AND a mutable tool that the
        # trace never shows being called
        _agent(self.c, "from platform_core.trace import log as trace_log\n"
                       "def run_task(t):\n"
                       "    trace_log('route_check', verdict='pass', name='main')\n"
                       "    return 2\n")
        (self.c / "task_agent" / "mutable_tools" / "new_tool.py").write_text(
            "def run():\n    return 1\n", encoding="utf-8")
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        self.assertTrue((self.c / "edit_usage.json").exists())

        logs = self.c / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "trace.jsonl").write_text(
            '{"kind": "mutable_log", "payload": {"label": "route_check", '
            '"verdict": "pass", "name": "main", "case_id": "a"}}\n'
            '{"kind": "tool_call", "payload": {"name": "other_tool", '
            '"case_id": "a"}}\n', encoding="utf-8")
        root = _Node(0, None, self.p, [_case("a", 0.0, ["x"])], children=[1])
        n1 = _Node(1, 0, self.c, [_case("a", 1.0, [])])
        tree = _Tree([root, n1])
        self.assertEqual(self.em.refresh_outcomes(tree, 1), 1)

        text = (self.c / "edit_memory.md").read_text()
        self.assertIn("new log point `route_check/main`**: fired 1x", text)
        self.assertIn("`new_tool` **0 calls**", text)
        self.assertIn("## Analysis", text)
        self.assertIn("likely cause", text)
        self.assertIn("**target `x`** — remaining 0/1 (was 1/1, +1)", text)
        self.assertIn("**collateral**", text)
        block = render_edit_memory(self.tmp, token_budget=48000, min_shared=1)
        self.assertIn("new log point", block)
        self.assertIn("**0 calls**", block)
        self.assertIn("**collateral**", block)

        n_calls = len(self.llm.calls)
        self.assertEqual(self.em.refresh_outcomes(tree, 1), 0)  # skip guard
        self.assertEqual(len(self.llm.calls), n_calls)          # no re-analysis

    def test_no_trace_means_no_usage_lines_not_zero(self):
        # A store with no consumed batches must render nothing — "0 calls"
        # is a statement about an observed batch, not about missing data.
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        root = _Node(0, None, self.p, [_case("a", 0.0)], children=[1])
        n1 = _Node(1, 0, self.c, [_case("a", 1.0)])
        self.em.refresh_outcomes(_Tree([root, n1]), 1)
        text = (self.c / "edit_memory.md").read_text()
        self.assertNotIn("0 calls", text)
        self.assertNotIn("new tools", text)
        self.assertNotIn("## Analysis", text)   # no evidence -> no LLM call

    def test_steering_block_excludes_candidates_and_respects_budget(self):
        self._setup()
        self.em.record_node(round_dir=self.c, parent_round_dir=self.p,
                            node_id=1, parent_id=0, ancestors=[0])
        block = render_edit_memory(self.tmp, token_budget=12000, min_shared=1)
        self.assertIn("add-verifier", block)
        self.assertNotIn("unused-proxy", block)     # proxy never reaches the editor
        self.assertEqual(render_edit_memory(self.tmp, token_budget=0), "")


class TestCategoryDefinitions(unittest.TestCase):
    """A registry entry with no definition is nearly useless: rendering the
    registry *with* definitions is what keeps categorisation stable across
    nodes, and a bare id teaches neither the tagger nor the editor anything.
    A real 28-round run ended up with 9 of 14 strategies undefined."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p, self.c = self.tmp / "round_000", self.tmp / "round_001"
        _agent(self.p, "def run_task(t):\n    return 1\n")
        _agent(self.c, "def run_task(t):\n    return 2\n\ndef check():\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, node_payload):
        llm = _StubLLM(node=node_payload)
        em = EditMemory(llm, min_shared=1)
        em.setup(self.tmp, self.p, [_case("a", 0.0, ["x"])])
        em.record_node(round_dir=self.c, parent_round_dir=self.p,
                       node_id=1, parent_id=0, ancestors=[0])
        return json.loads((self.tmp / "edit_memory_registry.json").read_text())

    def test_undefined_id_falls_back_instead_of_being_admitted_bare(self):
        # The candidate pool defines add-verifier; 'add-verifier-retry' is
        # invented by the model with no definition, so it must map onto the
        # defined neighbour rather than enter the registry undefined.
        reg = self._run({
            "edits": [{"name": "retry", "what": "Adds a retry", "why": "flaky",
                       "strategy": "add-verifier-retry", "area": "routing"}],
            "new_category_defs": {},
        })
        self.assertNotIn("add-verifier-retry", reg["strategies"])
        self.assertIn("add-verifier", reg["strategies"])
        self.assertEqual(reg["strategies"]["add-verifier"]["edits"][0]["name"],
                         "retry")

    def test_no_registry_entry_is_ever_left_undefined(self):
        reg = self._run({
            "edits": [{"name": "a", "what": "w", "why": "y",
                       "strategy": "brand-new-thing", "area": "brand-new-area"}],
            "new_category_defs": {},
        })
        for axis in ("strategies", "areas"):
            for cid, entry in reg[axis].items():
                self.assertNotIn("no definition supplied", entry["definition"],
                                 f"{axis}/{cid} admitted without a definition")

    def test_a_generic_shared_word_is_not_a_fit(self):
        """Taken from a live run: three undefined area ids each shared exactly
        the token 'validation' with the candidate 'transfer-time-validation'
        and were all folded into it, filing a system-prompt edit, a logging
        edit and a parsing edit under commute-time buffers. Similarity, not
        bare overlap, is what keeps them apart."""
        reg = self._run({
            "edits": [{"name": "a", "what": "Adds a pre-plan check", "why": "y",
                       "strategy": "add-verifier", "area": "pre-plan-validation"}],
            "new_category_defs": {},
        })
        self.assertIn("pre-plan-validation", reg["areas"])
        self.assertNotIn("pre-plan-validation",
                         str(reg["areas"].get("routing", {})))
        # kept, and defined from the edit rather than left bare
        self.assertTrue(reg["areas"]["pre-plan-validation"]["definition"])
        self.assertNotIn("no definition supplied",
                         reg["areas"]["pre-plan-validation"]["definition"])

    def test_a_close_variant_still_folds_in(self):
        """The fix must not stop real variants from merging: 'add-verifier-retry'
        against 'add-verifier' is 0.67 similar and should still fold."""
        reg = self._run({
            "edits": [{"name": "retry", "what": "Adds a retry", "why": "flaky",
                       "strategy": "add-verifier-retry", "area": "routing"}],
            "new_category_defs": {},
        })
        self.assertNotIn("add-verifier-retry", reg["strategies"])
        self.assertIn("add-verifier", reg["strategies"])

    def test_a_supplied_definition_is_accepted(self):
        reg = self._run({
            "edits": [{"name": "a", "what": "w", "why": "y",
                       "strategy": "prune-context", "area": "routing"}],
            "new_category_defs": {"prune-context": "trims prompt context"},
        })
        self.assertEqual(reg["strategies"]["prune-context"]["definition"],
                         "trims prompt context")


class TestDiffHelpers(unittest.TestCase):
    def test_truncate_middle_keeps_both_ends(self):
        got = truncate_middle("A" * 500 + "B" * 500, 400)
        self.assertTrue(got.startswith("A"))
        self.assertTrue(got.endswith("B"))
        self.assertIn("chars elided", got)

    def test_changed_and_diff(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p, c = tmp / "a", tmp / "b"
            _agent(p, "x = 1\n")
            _agent(c, "x = 2\n")
            self.assertEqual(changed_mutable_files(p, c), ["workflow.py"])
            self.assertIn("-x = 1", diff_mutable_files(p, c))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPerformanceFirst(unittest.TestCase):
    """Round-2 revisions: absolute per-check tallies, run context, the fmt-2
    record line, and backward compatibility with pre-fmt records."""

    RECIPE = {"mode": "list", "path": "details.failed_checks"}

    def test_per_check_absolute_tallies_keep_unmoved_checks(self):
        parent = [_case("a", 0.0, ["x", "y"]), _case("b", 0.0, ["y"])]
        child = [_case("a", 1.0, ["y"]), _case("b", 1.0, ["y"])]
        oc = compute_outcome(parent, child, min_shared=1, recipe=self.RECIPE)
        self.assertEqual(oc.per_check["x"], (1, 0))
        self.assertEqual(oc.per_check["y"], (2, 2))  # unmoved but still failing
        self.assertNotIn("y", oc.per_check_delta)    # the delta view drops it

    def test_select_check_tallies_orders_by_child_fails(self):
        from meta_agent.edit_outcome import select_check_tallies
        got = select_check_tallies(
            {"a:one": (0, 3), "a:two": (5, 1), "a:three": (2, 3)})
        self.assertEqual(list(got), ["a:three", "a:one", "a:two"])

    def test_run_context_reads_seed_and_best(self):
        from types import SimpleNamespace
        from meta_agent.edit_outcome import run_context
        mk = lambda i, p, m, n, ef=False: SimpleNamespace(  # noqa: E731
            node_id=i, parent_id=p, mean_utility=m, n_evals=n, edit_failed=ef)
        tree = SimpleNamespace(nodes={
            0: mk(0, None, 0.6, 60),
            1: mk(1, 0, 0.75, 32),
            2: mk(2, 0, 0.9, 0),           # unevaluated: ineligible for best
            3: mk(3, 0, 0.95, 16, True)})  # edit-failed: ineligible
        ctx = run_context(tree)
        self.assertEqual(ctx["seed_mean"], 0.6)
        self.assertEqual(ctx["best_node"], 1)
        self.assertEqual(ctx["best_n"], 32)
        self.assertEqual(run_context(SimpleNamespace(nodes={})), {})

    def test_old_format_record_feeds_absolute_ledger(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "round_001"
            d.mkdir(parents=True)
            (d / "edit_memory.md").write_text(
                "---\nnode: 1\nparent: 0\nlineage: 0 > 1\n---\n\n"
                "## Edit 1\n- **name**: `x`\n"
                "- **category level 1 (strategy)**: `add-verifier`\n"
                "- **category level 2 (area)**: `routing`\n"
                "- **what**: w\n- **why**: y\n\n## Outcome\n"
                "- **delta shared**: +0.0938 over 60 shared cases "
                "(parent 0.6229 -> child 0.7167)\n", encoding="utf-8")
            (tmp / "edit_memory_registry.json").write_text(json.dumps(
                {"strategies": {"add-verifier": {"definition": "d",
                 "edits": [{"node": 1, "edit_index": 1, "name": "x"}]}},
                 "areas": {}}), encoding="utf-8")
            from meta_agent.edit_memory_render import _load_records
            rec = _load_records(tmp)[1]
            # absolutes recovered from the OLD line itself — no rewrite needed
            self.assertEqual(rec["child_abs"], 0.7167)
            self.assertEqual(rec["parent_abs"], 0.6229)
            self.assertEqual(rec["delta"], 0.0938)
            block = render_edit_memory(
                tmp, min_shared=8,
                run_context={"seed_mean": 0.62, "seed_n": 60,
                             "best_mean": 0.76, "best_n": 60, "best_node": 8})
            self.assertIn("child median 0.717", block)
            self.assertIn("best so far 0.7600/60 (node 8)", block)
            self.assertIn("child 0.7167/60 · Δ +0.0938 vs parent on 60 shared, helped", block)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_machine_state_lives_in_sidecar_not_frontmatter(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p, c = tmp / "round_000", tmp / "round_001"
            _agent(p, "def run_task(t):\n    return 1\n")
            _agent(c, "def run_task(t):\n    return 2\n")
            em = EditMemory(_StubLLM(), min_shared=1)
            em.setup(tmp, p, [_case("a", 0.0, ["x"])])
            em.record_node(round_dir=c, parent_round_dir=p,
                           node_id=1, parent_id=0, ancestors=[0])
            root = _Node(0, None, p, [_case("a", 0.0, ["x"])], children=[1])
            n1 = _Node(1, 0, c, [_case("a", 1.0, [])])
            tree = _Tree([root, n1])
            self.assertEqual(em.refresh_outcomes(tree, 0), 1)
            path = c / "edit_memory.md"
            text = path.read_text()
            self.assertIn("performance", text)
            # the record carries human keys ONLY
            for noise in ("child_case_sig", "parent_case_sig", "threshold",
                          "min_shared", "fmt", "analysis_sig"):
                self.assertNotIn(noise, text)
            from meta_agent.edit_memory import STATE_NAME, RECORD_FORMAT
            state = json.loads((c / STATE_NAME).read_text())
            self.assertEqual(state["fmt"], RECORD_FORMAT)
            self.assertTrue(state["child_case_sig"])
            self.assertEqual(em.refresh_outcomes(tree, 0), 0)  # guard holds
            # a record with no sidecar (legacy / resumed run): exactly one
            # migration rewrite, then quiet again
            (c / STATE_NAME).unlink()
            self.assertEqual(em.refresh_outcomes(tree, 0), 1)
            self.assertEqual(em.refresh_outcomes(tree, 0), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_legacy_frontmatter_keys_migrate_to_sidecar_once(self):
        # A pre-v3 record whose machine keys sit in the frontmatter must be
        # rewritten exactly once (keys stripped, sidecar created) even when
        # its case sigs are current.
        tmp = Path(tempfile.mkdtemp())
        try:
            p, c = tmp / "round_000", tmp / "round_001"
            _agent(p, "def run_task(t):\n    return 1\n")
            _agent(c, "def run_task(t):\n    return 2\n")
            em = EditMemory(_StubLLM(), min_shared=1)
            em.setup(tmp, p, [_case("a", 0.0, ["x"])])
            em.record_node(round_dir=c, parent_round_dir=p,
                           node_id=1, parent_id=0, ancestors=[0])
            root = _Node(0, None, p, [_case("a", 0.0, ["x"])], children=[1])
            n1 = _Node(1, 0, c, [_case("a", 1.0, [])])
            tree = _Tree([root, n1])
            em.refresh_outcomes(tree, 0)
            path = c / "edit_memory.md"
            from meta_agent.edit_memory import STATE_NAME
            state = json.loads((c / STATE_NAME).read_text())
            (c / STATE_NAME).unlink()
            # forge the legacy layout: same sigs, but keys in the frontmatter
            text = path.read_text()
            legacy = text.replace(
                "---\n\n",
                "child_case_sig: %s\nparent_case_sig: %s\nthreshold: 0.02\n"
                "min_shared: 1\nfmt: 2\n---\n\n"
                % (state["child_case_sig"], state["parent_case_sig"]), 1)
            path.write_text(legacy, encoding="utf-8")
            self.assertEqual(em.refresh_outcomes(tree, 0), 1)   # migrates
            self.assertNotIn("child_case_sig", path.read_text())
            self.assertTrue((c / STATE_NAME).exists())
            self.assertEqual(em.refresh_outcomes(tree, 0), 0)   # then quiet
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSeenSplit(unittest.TestCase):
    def test_legs_and_thin_parent_guard(self):
        from meta_agent.edit_outcome import seen_split
        parent = [_case(str(i), 0.5) for i in range(6)]
        child = ([_case(str(i), 0.9) for i in range(6)]      # seen, covered
                 + [_case(str(i), 0.3) for i in range(6, 10)])  # unseen
        gs = seen_split(parent, child, [str(i) for i in range(6)])
        self.assertEqual(gs["seen"], {"mean": 0.9, "n": 6, "delta": 0.4})
        # parent covers 0 of the 4 unseen cases -> delta None
        self.assertEqual(gs["unseen"], {"mean": 0.3, "n": 4, "delta": None})
        # no unseen cases at all -> leg is None
        gs2 = seen_split(parent, child[:6], [str(i) for i in range(6)])
        self.assertIsNone(gs2["unseen"])

    def test_render_generalization_line(self):
        from meta_agent.edit_memory import render_generalization
        line = render_generalization({
            "seen": {"mean": 0.71, "n": 16, "delta": 0.09},
            "unseen": {"mean": 0.62, "n": 16, "delta": None}})
        self.assertEqual(
            line, "- **generalization**: seen 0.7100/16 (Δ +0.0900) · "
                  "unseen 0.6200/16 (Δ unmeasured) · gap -0.0900")
        self.assertEqual(render_generalization(None), "")

    def test_record_gains_generalization_and_cross_tab(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p, c = tmp / "round_000", tmp / "round_001"
            _agent(p, "def run_task(t):\n    return 1\n")
            _agent(c, "from platform_core.trace import log as trace_log\n"
                      "def run_task(t):\n"
                      "    trace_log('gate', verdict='pass', name='main')\n"
                      "    return 2\n")
            # parent evaluated on cases a,b,c BEFORE the child exists
            (p / "eval_result.json").write_text(json.dumps(
                {"per_case": [{"case_id": "a"}, {"case_id": "b"},
                              {"case_id": "c"}]}))
            em = EditMemory(_StubLLM(), min_shared=1)
            em.setup(tmp, p, [_case("a", 0.0, ["x"])])
            em.record_node(round_dir=c, parent_round_dir=p,
                           node_id=1, parent_id=0, ancestors=[0])
            store = json.loads((c / "edit_usage.json").read_text())
            self.assertEqual(store["seen_case_ids"], ["a", "b", "c"])
            logs = c / "logs"
            logs.mkdir(exist_ok=True)
            (logs / "trace.jsonl").write_text(
                json.dumps({"kind": "mutable_log", "payload": {
                    "label": "gate", "verdict": "pass", "name": "main",
                    "case_id": "d"}}) + "\n")
            root = _Node(0, None, p, [_case("a", 0.25, ["x"]),
                                     _case("b", 0.25, ["x"]),
                                     _case("c", 0.25, ["x"])], children=[1])
            n1 = _Node(1, 0, c, [_case("a", 1.0, []), _case("b", 1.0, []),
                                 _case("c", 1.0, []), _case("d", 0.2, ["x"])])
            self.assertEqual(em.refresh_outcomes(_Tree([root, n1]), 1), 1)
            text = (c / "edit_memory.md").read_text()
            self.assertIn("- **generalization**: seen 1.0000/3 (Δ +0.7500) · "
                          "unseen 0.2000/1 (Δ unmeasured) · gap -0.8000", text)
            # gate passed only on case d, which the scorer failed -> suspect
            self.assertIn("-> scorer on those cases: 0 pass / 1 fail", text)
            self.assertIn("SUSPECT VERIFIER", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_suspect_flag_reaches_ledger(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "round_001"
            d.mkdir(parents=True)
            (d / "edit_memory.md").write_text(
                "---\nnode: 1\nparent: 0\nlineage: 0 > 1\n---\n\n"
                "## Edit 1\n- **name**: `x`\n"
                "- **category level 1 (strategy)**: `add-verifier`\n"
                "- **category level 2 (area)**: `routing`\n"
                "- **what**: w\n- **why**: y\n\n## Outcome\n"
                "- **performance**: child 0.7000 over 16 evaluated cases "
                "(vs parent on 16 shared: child 0.7000, parent 0.6000, "
                "Δ +0.1000)\n"
                "- **new log point `g`**: fired 4x (pass 4) (1 batches, 4 "
                "cases) -> scorer on those cases: 1 pass / 3 fail · "
                "SUSPECT VERIFIER\n", encoding="utf-8")
            (tmp / "edit_memory_registry.json").write_text(json.dumps(
                {"strategies": {"add-verifier": {"definition": "d",
                 "edits": [{"node": 1, "edit_index": 1, "name": "x"}]}},
                 "areas": {}}), encoding="utf-8")
            block = render_edit_memory(tmp, min_shared=8)
            self.assertIn("suspect-verifier in 1 node(s)", block)
            self.assertIn("REPAIR a promising category", block)
            self.assertIn("You can potentially utilize", block)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

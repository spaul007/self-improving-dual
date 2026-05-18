"""Smoke tests that don't require an API key.

Covers:
  * Validators reject malformed agents and accept the seed.
  * The subprocess evaluator can run a stub task agent and produce results.
  * The config loader instantiates real components from configs/default.yaml.
  * tools_schema_consistency catches name collisions and dangling names.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_smoke
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _BuildEnv(unittest.TestCase):
    """Each test runs in its own temp dir copied from the seed."""

    def setUp(self) -> None:
        # Make sure the math project's tools are registered so the schema
        # consistency validator finds `calculate`.
        import projects.math.tools  # noqa: F401

        self.tmp = Path(tempfile.mkdtemp(prefix="metaagent_test_"))
        self.round_dir = self.tmp / "round_000"
        self.round_dir.mkdir()
        seed = REPO_ROOT / "projects" / "math" / "seed"
        shutil.copytree(seed, self.round_dir / "task_agent")
        (self.round_dir / "logs").mkdir()
        # base_dir for the immutable_files validator: same content, so no diffs.
        self.base_dir = self.tmp / "base"
        shutil.copytree(self.round_dir, self.base_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class ValidatorTests(_BuildEnv):
    def test_seed_passes_all_validators(self) -> None:
        from meta_agent.editor_validators import (
            ImmutableFilesValidator,
            ImportValidator,
            LoadTestValidator,
            MutableToolImportValidator,
            SchemaWrapperConsistencyValidator,
            SignatureValidator,
            SyntaxValidator,
        )

        for v in (
            SyntaxValidator(),
            SignatureValidator(),
            ImportValidator(),
            SchemaWrapperConsistencyValidator(),
            MutableToolImportValidator(),
            ImmutableFilesValidator(),
            LoadTestValidator(),
        ):
            errors = v.validate(self.round_dir, self.base_dir)
            self.assertEqual(errors, [], f"{type(v).__name__} unexpectedly failed: {errors}")

    def test_load_test_validator_catches_name_error(self) -> None:
        """``LoadTestValidator`` spawns a subprocess that imports the
        agent's mutable modules. A workflow.py that references an
        undefined name fails at module load time — none of the static
        validators catch it (it's syntactically valid). This validator
        must surface the NameError so the editor knows what to fix."""
        from meta_agent.editor_validators import LoadTestValidator

        wf = self.round_dir / "task_agent" / "workflow.py"
        wf.write_text(
            "from platform_core.runner import AgentOutput\n"
            "UNDEFINED_THING.foo  # NameError at module load\n"
            "def run_task(task):\n    return AgentOutput(result='')\n",
            encoding="utf-8",
        )
        errors = LoadTestValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(errors, "expected LoadTestValidator to report errors")
        self.assertTrue(
            any("NameError" in e for e in errors),
            f"expected NameError in errors, got {errors!r}",
        )

    def test_signature_validator_rejects_wrong_signature(self) -> None:
        from meta_agent.editor_validators import SignatureValidator

        wf = self.round_dir / "task_agent" / "workflow.py"
        wf.write_text("def run_task(x):\n    return ''\n", encoding="utf-8")
        errors = SignatureValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(errors)
        self.assertIn("run_task", errors[0])

    def test_imports_validator_rejects_forbidden_platform_import(self) -> None:
        from meta_agent.editor_validators import ImportValidator

        wf = self.round_dir / "task_agent" / "workflow.py"
        wf.write_text(
            "from platform_core.tools import call_tool\n"
            "def run_task(task):\n    return ''\n",
            encoding="utf-8",
        )
        errors = ImportValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(any("platform_core.tools" in e for e in errors))

    def test_mutable_tool_imports_rejects_llm_wrapper(self) -> None:
        from meta_agent.editor_validators import MutableToolImportValidator

        bad = self.round_dir / "task_agent" / "mutable_tools" / "rogue.py"
        bad.write_text(
            "from platform_core.llm_wrapper import call_llm\n"
            "NAME='rogue'\nSCHEMA={}\n"
            "def run(**kw):\n    return ''\n",
            encoding="utf-8",
        )
        errors = MutableToolImportValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(errors, "expected forbidden-import to be rejected")

    def test_schema_consistency_rejects_dangling_name(self) -> None:
        from meta_agent.editor_validators import SchemaWrapperConsistencyValidator

        schema_path = self.round_dir / "task_agent" / "tools_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema.append(
            {"name": "ghost_tool", "description": "x", "input_schema": {"type": "object"}}
        )
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        errors = SchemaWrapperConsistencyValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(any("ghost_tool" in e for e in errors))

    def test_immutable_files_rejects_new_file_outside_mutable(self) -> None:
        from meta_agent.editor_validators import ImmutableFilesValidator

        rogue = self.round_dir / "task_agent" / "secret.py"
        rogue.write_text("# rogue", encoding="utf-8")
        errors = ImmutableFilesValidator().validate(self.round_dir, self.base_dir)
        self.assertTrue(any("secret.py" in e for e in errors))


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        # Tell the subprocess child which project's tools to load via
        # the same env var the parent normally exports.
        import os
        self._prev_project = os.environ.get("META_AGENT_PROJECT")
        os.environ["META_AGENT_PROJECT"] = "math"

        self.tmp = Path(tempfile.mkdtemp(prefix="metaagent_eval_"))
        self.round_dir = self.tmp / "round_000"
        agent = self.round_dir / "task_agent"
        agent.mkdir(parents=True)
        # Stub workflow that calls the immutable calculator directly,
        # so the evaluator can run without an LLM key.
        (agent / "workflow.py").write_text(
            "from platform_core.runner import AgentOutput\n"
            "from platform_core.tools import call_tool\n"
            "def run_task(task) -> AgentOutput:\n"
            "    expr = task.description.split(':',1)[-1].strip()\n"
            "    return AgentOutput(result=call_tool('calculate', expression=expr))\n",
            encoding="utf-8",
        )
        # Minimal benchmark.
        self.bench = self.tmp / "bench"
        self.bench.mkdir()
        (self.bench / "cases.jsonl").write_text(
            json.dumps({"id": "a", "input": "compute: 2+3", "expected": "5"}) + "\n"
            + json.dumps({"id": "b", "input": "compute: 4*4", "expected": "16"}) + "\n",
            encoding="utf-8",
        )
        (self.bench / "scorer.py").write_text(
            "def score(case, output):\n"
            "    raw = str(getattr(output, 'result', output) or '').strip()\n"
            "    p = raw == str(case['expected']).strip()\n"
            "    return {'score': 1.0 if p else 0.0, 'passed': p, 'details': {}}\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        import os
        if self._prev_project is None:
            os.environ.pop("META_AGENT_PROJECT", None)
        else:
            os.environ["META_AGENT_PROJECT"] = self._prev_project
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_subprocess_evaluator_runs_stub_agent(self) -> None:
        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator(wall_time_s_per_case=15, memory_mb=512)
        result = ev.run(self.round_dir, self.bench)
        self.assertEqual(result.passed, 2)
        self.assertEqual(result.failed, 0)
        self.assertFalse(result.crashed)
        self.assertAlmostEqual(result.score, 1.0)

    def test_subprocess_evaluator_handles_timeout(self) -> None:
        from meta_agent.evaluator import SubprocessEvaluator

        agent = self.round_dir / "task_agent"
        (agent / "workflow.py").write_text(
            "def run_task(task):\n"
            "    while True: pass\n",
            encoding="utf-8",
        )
        ev = SubprocessEvaluator(wall_time_s_per_case=2, memory_mb=512)
        result = ev.run(self.round_dir, self.bench)
        self.assertTrue(result.crashed)
        self.assertEqual(result.passed, 0)

    def test_subprocess_evaluator_filters_by_case_ids(self) -> None:
        """When ``case_ids`` is given, the evaluator runs exactly those
        cases (in order) and skips the rest."""
        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator(wall_time_s_per_case=15, memory_mb=512)
        result = ev.run(self.round_dir, self.bench, case_ids=["b"])
        self.assertEqual(len(result.per_case), 1)
        self.assertEqual(result.per_case[0].case_id, "b")
        self.assertTrue(result.per_case[0].passed)

    def test_subprocess_evaluator_unknown_case_id_raises(self) -> None:
        from meta_agent.evaluator import SubprocessEvaluator

        ev = SubprocessEvaluator(wall_time_s_per_case=15, memory_mb=512)
        with self.assertRaises(KeyError):
            ev.run(self.round_dir, self.bench, case_ids=["does_not_exist"])


class ConfigLoaderTests(unittest.TestCase):
    def test_default_config_assembles(self) -> None:
        from meta_agent.config import build_components, load

        cfg = load(REPO_ROOT / "configs" / "default.yaml")
        fw = build_components(cfg)
        self.assertEqual(type(fw.manager).__name__, "HillClimbingManager")
        self.assertEqual(type(fw.evaluator).__name__, "SubprocessEvaluator")
        self.assertEqual(type(fw.gatherer).__name__, "DefaultFeedbackGatherer")
        self.assertEqual(type(fw.editor).__name__, "AgentEditor")
        self.assertEqual(len(fw.validators), 7)
        self.assertTrue(fw.seed_dir.exists())
        self.assertTrue(fw.benchmark_dir.exists())

    def test_editor_is_pluggable_via_registry(self) -> None:
        """Editor is a ComponentSpec like every other pluggable component.
        The YAML declares ``editor.type: "default"`` explicitly; the
        framework looks it up in the registry."""
        from meta_agent import registry
        from meta_agent.config import build_components, load

        cfg = load(REPO_ROOT / "configs" / "default.yaml")
        self.assertEqual(cfg.editor.type, "default")
        self.assertEqual(cfg.editor.config.get("max_attempts"), 2)

        fw = build_components(cfg)
        self.assertEqual(type(fw.editor).__name__, "AgentEditor")
        self.assertIn("default", registry.available("editor"))

    def test_project_field_parses(self) -> None:
        from meta_agent.config import load

        default_cfg = load(REPO_ROOT / "configs" / "default.yaml")
        self.assertEqual(default_cfg.project, "math")

        travel_cfg = load(REPO_ROOT / "configs" / "travel.yaml")
        self.assertEqual(travel_cfg.project, "travel")

    def test_load_project_filters_immutable_tools(self) -> None:
        """``load_project`` populates only the named project's tools. Run
        in a clean subprocess so the singleton ``_LOADED`` flag and
        registry state are pristine for each scenario."""
        cmd = (
            "from platform_core.tools import load_project, is_immutable; "
            "load_project('math'); "
            "print('cal=' + str(is_immutable('calculate')) + "
            "',flight=' + str(is_immutable('query_flight_info')))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**__import__('os').environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("cal=True,flight=False", proc.stdout)

        cmd2 = (
            "from platform_core.tools import load_project, is_immutable; "
            "load_project('travel'); "
            "print('cal=' + str(is_immutable('calculate')) + "
            "',flight=' + str(is_immutable('query_flight_info')))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", cmd2],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**__import__('os').environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("cal=False,flight=True", proc.stdout)

    def test_partial_yaml_raises_field_required(self) -> None:
        """Component types must be declared in the YAML — defaults are
        not embedded in framework code. A minimal config that omits the
        component blocks should fail Pydantic validation with a clear
        ``field required`` message rather than silently picking
        defaults."""
        from pydantic import ValidationError

        from meta_agent.config import FrameworkConfig

        with self.assertRaises(ValidationError) as ctx:
            FrameworkConfig.model_validate(
                {
                    "experiment_name": "x",
                    "project": "math",
                    "loop": {"max_rounds": 1},
                }
            )
        msg = str(ctx.exception)
        for required in ("manager", "evaluator", "editor", "gatherer", "validators"):
            self.assertIn(required, msg)

    def test_travel_yaml_uses_project_specific_scorer(self) -> None:
        """``configs/travel.yaml`` names the project-specific scorer
        (``travel_default``); the framework gatherer is the generic
        ``"default"`` one. Both end up holding the same scorer instance
        so the gatherer can call ``scorer.aggregate(...)`` to populate
        ``AgentFeedback.project_metrics``."""
        from meta_agent.config import build_components, load

        cfg = load(REPO_ROOT / "configs" / "travel.yaml")
        self.assertEqual(cfg.gatherer.type, "default")
        self.assertEqual(cfg.evaluator.config.get("scorer"), "travel_default")

        fw = build_components(cfg)
        self.assertEqual(type(fw.gatherer).__name__, "DefaultFeedbackGatherer")
        self.assertEqual(type(fw.evaluator.scorer).__name__, "TravelCompositeScorer")
        # Same scorer instance is shared between evaluator and gatherer.
        self.assertIs(fw.gatherer.scorer, fw.evaluator.scorer)
        # Aggregate method is the gatherer's source for project_metrics.
        self.assertTrue(callable(getattr(fw.gatherer.scorer, "aggregate", None)))

    def test_unknown_manager_gives_clear_error(self) -> None:
        from meta_agent.config import FrameworkConfig, build_components

        cfg = FrameworkConfig.model_validate(
            {
                "experiment_name": "x",
                "project": "math",
                "loop": {"max_rounds": 1},
                "manager": {"type": "no_such_manager"},
                "evaluator": {"type": "subprocess"},
                "gatherer": {"type": "default"},
                "editor": {"type": "default"},
                "validators": [],
            }
        )
        with self.assertRaises(KeyError) as cm:
            build_components(cfg)
        self.assertIn("Available", str(cm.exception))


class CaseSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        """Same seed → same split. Train and eval are disjoint and
        their union covers every case in cases.jsonl."""
        from meta_agent.config import compute_split

        bench = REPO_ROOT / "projects" / "travel" / "benchmark"
        if not (bench / "cases.jsonl").exists():
            self.skipTest("travel benchmark not present")

        train_a, eval_a = compute_split(bench, seed=42, train_size=60)
        train_b, eval_b = compute_split(bench, seed=42, train_size=60)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(len(train_a), 60)
        self.assertEqual(len(eval_a), 60)
        self.assertTrue(set(train_a).isdisjoint(eval_a))

        # Coverage: every id in cases.jsonl is in exactly one split.
        all_ids = set(train_a) | set(eval_a)
        from meta_agent.evaluator import load_cases
        loaded = load_cases(bench)
        self.assertEqual(len(loaded), len(all_ids))

    def test_different_seed_changes_split(self) -> None:
        from meta_agent.config import compute_split

        bench = REPO_ROOT / "projects" / "travel" / "benchmark"
        if not (bench / "cases.jsonl").exists():
            self.skipTest("travel benchmark not present")

        train_a, _ = compute_split(bench, seed=42, train_size=60)
        train_b, _ = compute_split(bench, seed=43, train_size=60)
        self.assertNotEqual(train_a, train_b)

    def test_travel_yaml_exposes_split(self) -> None:
        from meta_agent.config import load

        cfg = load(REPO_ROOT / "configs" / "travel.yaml")
        self.assertIsNotNone(cfg.split)
        self.assertEqual(cfg.split.seed, 42)
        self.assertEqual(cfg.split.train_size, 60)

    def test_default_yaml_has_no_split(self) -> None:
        """Math benchmark must keep working without any split block."""
        from meta_agent.config import load

        cfg = load(REPO_ROOT / "configs" / "default.yaml")
        self.assertIsNone(cfg.split)


class FeedbackGathererTests(unittest.TestCase):
    """The framework gatherer is project-agnostic: it produces trace
    aggregates (tool_usage / tool_error_rate / llm_calls / runtime
    exceptions / log_excerpt) plus an empty ``project_metrics``.
    Project-specific roll-ups (no_plan_rate, etc.) move to a
    project-specific gatherer subclass; see TravelGathererTests."""

    def _build_round_dir(self, trace_lines: list[dict]) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="metaagent_gatherer_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        logs = tmp / "logs"
        logs.mkdir()
        with (logs / "trace.jsonl").open("w", encoding="utf-8") as fh:
            for ev in trace_lines:
                fh.write(json.dumps(ev) + "\n")
        return tmp

    def _build_eval_result(self, per_case_dicts: list[dict]):
        from meta_agent.models import CaseResult, EvaluationResult

        cases = [CaseResult(**c) for c in per_case_dicts]
        passed = sum(1 for c in cases if c.passed)
        return EvaluationResult(
            score=sum(c.score for c in cases) / max(len(cases), 1),
            passed=passed,
            failed=len(cases) - passed,
            per_case=cases,
        )

    def _gather(self, eval_result, trace_events):
        from meta_agent.feedback_gatherer import DefaultFeedbackGatherer
        from meta_agent.models import EvolutionStrategy

        round_dir = self._build_round_dir(trace_events)
        strategy = EvolutionStrategy(
            target_files=["workflow.py"],
            optimization_goal="x",
            proposed_changes="y",
        )
        return DefaultFeedbackGatherer().compile(
            round_number=1,
            base_round=0,
            strategy=strategy,
            eval_result=eval_result,
            round_dir=round_dir,
        )

    def test_default_gatherer_leaves_project_metrics_empty(self) -> None:
        """The framework's default gatherer has no project-specific
        knowledge. Even when per-case ``details`` carry data, it should
        leave ``project_metrics`` empty — only a project-specific
        gatherer (e.g. ``TravelGatherer``) reads those fields."""
        eval_result = self._build_eval_result([
            {
                "case_id": "0", "passed": False, "score": 0.0,
                "details": {"error": "plan conversion failed: empty"},
            },
            {
                "case_id": "1", "passed": False, "score": 0.4,
                "details": {
                    "failed_checks": ["foo:bar"],
                    "dimension_scores": {"x": 0.5},
                },
            },
        ])
        fb = self._gather(eval_result, trace_events=[])
        self.assertEqual(fb.project_metrics, {})

    def test_tool_error_rate_pairs_call_with_result(self) -> None:
        """A tool_result whose result_preview begins with "Error" or
        contains an "error" JSON key counts as a failed call for the
        matching tool_call (paired by payload.id)."""
        trace = [
            {"timestamp": 1.0, "kind": "tool_call",
             "payload": {"id": "a", "name": "search_location"}},
            {"timestamp": 1.1, "kind": "tool_result",
             "payload": {"id": "a", "name": "search_location",
                         "result_preview": "Error: missing place_name"}},
            {"timestamp": 1.2, "kind": "tool_call",
             "payload": {"id": "b", "name": "search_location"}},
            {"timestamp": 1.3, "kind": "tool_result",
             "payload": {"id": "b", "name": "search_location",
                         "result_preview": '{"place_name": "X", "latitude": 30}'}},
            {"timestamp": 1.4, "kind": "tool_call",
             "payload": {"id": "c", "name": "query_flight_info"}},
            {"timestamp": 1.5, "kind": "tool_result",
             "payload": {"id": "c",
                         "result_preview": '{"error": "no flights found"}'}},
        ]
        empty_eval = self._build_eval_result([])
        fb = self._gather(empty_eval, trace_events=trace)
        # search_location: 2 calls, 1 error → 0.5
        # query_flight_info: 1 call, 1 error → 1.0
        self.assertAlmostEqual(fb.tool_error_rate["search_location"], 0.5)
        self.assertAlmostEqual(fb.tool_error_rate["query_flight_info"], 1.0)
        self.assertEqual(fb.tool_usage["search_location"], 2)
        self.assertEqual(fb.tool_usage["query_flight_info"], 1)

    def test_empty_per_case_yields_safe_defaults(self) -> None:
        """Manager-synthesized failed-edit feedback has no per_case and
        no trace events; the gatherer must not crash and must produce
        zero/empty roll-ups."""
        empty_eval = self._build_eval_result([])
        fb = self._gather(empty_eval, trace_events=[])
        self.assertEqual(fb.tool_error_rate, {})
        self.assertEqual(fb.tool_usage, {})
        self.assertEqual(fb.llm_calls, 0)
        self.assertEqual(fb.project_metrics, {})


class RunnerTests(unittest.TestCase):
    """The platform_core.runner module is the cross-process contract; it
    must round-trip a Task → AgentOutput in both stdin and standalone modes
    without an API key."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="metaagent_runner_"))
        self.agent = self.tmp / "agent"
        self.agent.mkdir()
        # Stub workflow that doesn't need an LLM.
        (self.agent / "workflow.py").write_text(
            "from platform_core.runner import AgentOutput\n"
            "def run_task(task):\n"
            "    return AgentOutput(result=task.description.upper(),\n"
            "                       metadata={'case_id': task.case_id})\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _env(self) -> dict[str, str]:
        return {**__import__('os').environ, "PYTHONPATH": str(REPO_ROOT)}

    def test_runner_stdin_mode_round_trips_task(self) -> None:
        task_payload = json.dumps(
            {"description": "hello", "case_id": "x", "context": {}}
        )
        proc = subprocess.run(
            [sys.executable, "-m", "platform_core.runner"],
            input=task_payload,
            capture_output=True,
            text=True,
            cwd=str(self.agent),
            env=self._env(),
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        envelope = json.loads(proc.stdout)
        self.assertTrue(envelope["ok"], msg=envelope)
        self.assertEqual(envelope["output"]["result"], "HELLO")
        self.assertEqual(envelope["output"]["metadata"]["case_id"], "x")

    def test_runner_standalone_mode_with_task_file(self) -> None:
        task_file = self.tmp / "task.json"
        task_file.write_text(
            json.dumps({"description": "hi", "case_id": "1", "context": {}}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable, "-m", "platform_core.runner",
                "--agent-dir", str(self.agent),
                "--task-file", str(task_file),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=self._env(),
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"], "HI")

    def test_runner_stdin_mode_reports_workflow_error(self) -> None:
        """If the workflow raises, the envelope reports ok=False with the
        traceback. The child must still exit 0 so the evaluator can read
        the envelope."""
        (self.agent / "workflow.py").write_text(
            "def run_task(task):\n    raise RuntimeError('boom')\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-m", "platform_core.runner"],
            input='{"description": "x", "case_id": "0", "context": {}}',
            capture_output=True,
            text=True,
            cwd=str(self.agent),
            env=self._env(),
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        envelope = json.loads(proc.stdout)
        self.assertFalse(envelope["ok"])
        self.assertIn("RuntimeError", envelope["error"])
        self.assertIn("boom", envelope["error"])


class FailedEditFeedbackTests(unittest.TestCase):
    """When the editor's mutation fails validation, the round must (a)
    skip the held-out eval split (the workspace is identical to the
    base round; running eval is wasted compute) and (b) surface the
    ``edit_errors`` list into both the strategy prompt and the editor
    prompt so the next round's optimizer can pivot."""

    def _make_failed_edit_feedback(self, errors: list[str]):
        from meta_agent.models import (
            AgentFeedback,
            EvaluationResult,
            EvolutionStrategy,
        )

        return AgentFeedback(
            round_number=1,
            base_round=0,
            strategy=EvolutionStrategy(
                target_files=["workflow.py"],
                optimization_goal="proposed something",
                proposed_changes="x",
            ),
            eval_result=EvaluationResult(score=0.0),
            edit_errors=errors,
        )

    def test_run_eval_split_skips_when_edit_errors(self) -> None:
        """Eval split must be skipped (and the evaluator never called)
        when ``feedback.edit_errors`` is non-empty."""
        from meta_agent.managers.hill_climbing import HillClimbingManager

        sentinel = []

        class _ShouldNotRunEvaluator:
            def run(self, *args, **kwargs):
                sentinel.append(("called", args, kwargs))
                from meta_agent.models import EvaluationResult
                return EvaluationResult(score=0.0)

        m = HillClimbingManager()
        m._eval_case_ids = ["x"]  # split is configured

        tmp = Path(tempfile.mkdtemp(prefix="failed_edit_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "task_agent").mkdir()  # exists, mimicking editor's reset

        fb = self._make_failed_edit_feedback(["signature broken"])
        m._run_eval_split(
            1, tmp, _ShouldNotRunEvaluator(), Path("."), fb,
        )
        self.assertEqual(sentinel, [], "evaluator must not have been called")

        # Verify the skipped sidecar is written with the edit_errors.
        score_path = tmp / "eval_score.json"
        self.assertTrue(score_path.exists(), "expected eval_score.json")
        side = json.loads(score_path.read_text(encoding="utf-8"))
        self.assertTrue(side.get("skipped"))
        self.assertEqual(side.get("edit_errors"), ["signature broken"])

    def test_strategy_prompt_renders_edit_errors(self) -> None:
        """The manager's steering context must surface ``edit_errors`` in
        its recent-rounds section so the editor avoids the same mistake."""
        from meta_agent.managers.hill_climbing import HillClimbingManager

        m = HillClimbingManager()
        fb = self._make_failed_edit_feedback(
            ["run_task signature must be run_task(task) -> AgentOutput"]
        )
        text = m._render_change_context([fb], best=fb)
        self.assertIn("edit_errors", text)
        self.assertIn("run_task signature must be", text)

    def test_editor_format_feedback_renders_edit_errors(self) -> None:
        """The editor's ``_format_feedback`` (cross-round view of the
        previous round's outcome) must include ``edit_errors`` so the
        editor LLM avoids re-producing the broken structure."""
        from meta_agent.agent_editor import AgentEditor

        editor = AgentEditor(llm_caller=lambda **kw: None, validators=[])
        fb = self._make_failed_edit_feedback(
            ["forbidden edit path: 'platform_core/foo.py'"]
        )
        text = editor._format_feedback(fb)
        self.assertIn("edit_errors", text)
        self.assertIn("forbidden edit path", text)


class RuntimeEnvTests(unittest.TestCase):
    """The framework forwards arbitrary env-var overrides via the YAML's
    ``env:`` block. There is no project-specific helper; project tools
    compute their own defaults."""

    def test_apply_user_env_exports_each_pair(self) -> None:
        import os

        from meta_agent.runtime_env import apply_user_env

        prev_a = os.environ.pop("META_TEST_KEY_A", None)
        prev_b = os.environ.pop("META_TEST_KEY_B", None)
        try:
            apply_user_env({"META_TEST_KEY_A": "alpha", "META_TEST_KEY_B": "42"})
            self.assertEqual(os.environ.get("META_TEST_KEY_A"), "alpha")
            self.assertEqual(os.environ.get("META_TEST_KEY_B"), "42")
        finally:
            for k, prev in (("META_TEST_KEY_A", prev_a), ("META_TEST_KEY_B", prev_b)):
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev

    def test_travel_yaml_env_block_default_empty(self) -> None:
        """The shipped travel YAML doesn't set any env overrides; the
        project's tools default to ``projects/travel/data/database_en``
        on their own."""
        from meta_agent.config import load

        cfg = load(REPO_ROOT / "configs" / "travel.yaml")
        self.assertEqual(cfg.env, {})

    def test_csv_database_root_falls_back_to_project_default(self) -> None:
        """When ``TRAVEL_DATABASE_ROOT`` is unset, the travel project's
        ``database_root()`` returns the project-relative default if the
        directory exists. This is what makes a typical run zero-config."""
        import os

        prev = os.environ.pop("TRAVEL_DATABASE_ROOT", None)
        try:
            from projects.travel.tools._csv import database_root

            root = database_root()
            expected = REPO_ROOT / "projects" / "travel" / "data" / "database_en"
            if expected.exists():
                self.assertEqual(root, expected)
            else:
                self.assertIsNone(root)
        finally:
            if prev is not None:
                os.environ["TRAVEL_DATABASE_ROOT"] = prev


class StrategyCoercionTests(unittest.TestCase):
    """Defensive coercion for malformed self-improvement tool-call args.

    Local vLLM-hosted open-weights models don't always honor the tool's
    declared JSON schema. The 2026-05-12 gpt-oss bake-off crashed
    mid-run on a Pydantic ValidationError because the model returned
    `target_files: "workflow.py"` (string) instead of
    `target_files: ["workflow.py"]` (list). The coercion helpers now live
    in ``meta_agent.agent_editor`` (the single self-improvement step).
    """

    def setUp(self) -> None:
        from meta_agent.agent_editor import (
            _coerce_target_files, _coerce_str,
        )
        self._coerce_target_files = _coerce_target_files
        self._coerce_str = _coerce_str

    # ---- target_files coercion ----

    def test_target_files_string_becomes_list(self) -> None:
        # The actual bug we found in the live eval.
        self.assertEqual(
            self._coerce_target_files("workflow.py"),
            ["workflow.py"],
        )

    def test_target_files_proper_list_passes_through(self) -> None:
        self.assertEqual(
            self._coerce_target_files(["workflow.py", "tool_wrapper.py"]),
            ["workflow.py", "tool_wrapper.py"],
        )

    def test_target_files_none_falls_back(self) -> None:
        self.assertEqual(self._coerce_target_files(None), ["workflow.py"])

    def test_target_files_empty_string_falls_back(self) -> None:
        self.assertEqual(self._coerce_target_files(""), ["workflow.py"])

    def test_target_files_unknown_values_dropped(self) -> None:
        # The schema enum only allows the three mutable files.
        self.assertEqual(
            self._coerce_target_files(["workflow.py", "../platform_core/x.py"]),
            ["workflow.py"],
        )

    def test_target_files_all_unknown_falls_back(self) -> None:
        self.assertEqual(
            self._coerce_target_files(["../platform_core/x.py"]),
            ["workflow.py"],
        )

    def test_target_files_non_iterable_falls_back(self) -> None:
        # e.g. model returns a dict or an int.
        self.assertEqual(self._coerce_target_files(42), ["workflow.py"])
        self.assertEqual(self._coerce_target_files({"k": "v"}), ["workflow.py"])

    # ---- _coerce_str ----

    def test_coerce_str_passes_through_strings(self) -> None:
        self.assertEqual(self._coerce_str("hello"), "hello")
        self.assertEqual(self._coerce_str(""), "")

    def test_coerce_str_none_becomes_empty(self) -> None:
        self.assertEqual(self._coerce_str(None), "")

    def test_coerce_str_stringifies_non_strings(self) -> None:
        self.assertEqual(self._coerce_str(42), "42")
        self.assertEqual(self._coerce_str(True), "True")


class SelfImprovementParsingTests(unittest.TestCase):
    """The editor's `submit_self_improvement` parsing must coerce malformed
    tool-call args into a valid EvolutionStrategy instead of crashing on
    Pydantic validation. Reproduces the gpt-oss-120b crash from job 140902
    (now guarded inside ``AgentEditor._parse_self_improvement``)."""

    def test_malformed_tool_args_do_not_crash(self) -> None:
        from meta_agent.agent_editor import AgentEditor
        from meta_agent.models import EvolutionStrategy

        # Non-string scalars for text fields — the shape open-weights
        # models produce when they ignore the declared schema.
        strategy, files = AgentEditor._parse_self_improvement(
            {
                "optimization_goal": 42,
                "proposed_changes": None,
                "rationale": True,
                "files": [{"path": "workflow.py", "content": "x"}],
            }
        )
        self.assertIsInstance(strategy, EvolutionStrategy)
        self.assertEqual(strategy.optimization_goal, "42")
        self.assertEqual(strategy.proposed_changes, "")
        self.assertEqual(strategy.target_files, ["workflow.py"])
        self.assertEqual(len(files), 1)

    def test_target_files_derived_from_emitted_files(self) -> None:
        """`target_files` is derived from the emitted file paths; a
        mutable_tools/* edit is applied but not an enum value, so it is
        dropped from the summary's `target_files`."""
        from meta_agent.agent_editor import AgentEditor

        strategy, files = AgentEditor._parse_self_improvement(
            {
                "optimization_goal": "g",
                "proposed_changes": "c",
                "files": [
                    {"path": "tool_wrapper.py", "content": "x"},
                    {"path": "mutable_tools/foo.py", "content": "y"},
                ],
            }
        )
        self.assertEqual(strategy.target_files, ["tool_wrapper.py"])
        self.assertEqual(len(files), 2)


class EvaluateScriptTests(unittest.TestCase):
    def test_evaluate_help_runs(self) -> None:
        """The standalone evaluate.py CLI parses arguments without
        importing anything that needs an OpenAI key."""
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evaluate.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**__import__('os').environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--config", proc.stdout)
        self.assertIn("--agent", proc.stdout)


if __name__ == "__main__":
    unittest.main()

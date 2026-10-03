"""projects/deepswe_seedling: trial reading, infra-vs-model boundary, scorer trust,
hidden-test isolation of exported artifacts, transcript bounds, validator logic.
Self-contained (synthetic trial dirs); the Pier-interpreter pieces (invariant self-test,
dry run, mutation suite) are exercised by adapter/selftest/mutation_suite.py.

    PYTHONPATH=. python3 -m pytest -q tests/test_deepswe_seedling_project.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1] / "projects" / "deepswe_seedling"
sys.path.insert(0, str(PROJECT))

import benchmark.scorer as S  # noqa: E402
from adapter import pier_case  # noqa: E402
from adapter.categorizer import categorize_errors  # noqa: E402
from adapter.render import MAX_LINE, render_trial, run_report  # noqa: E402
from adapter.trial import dispatch_outputs, failure_classes, load_outcome  # noqa: E402
from adapter.validators import check_dry_run, check_settings, scan_source  # noqa: E402
from meta_agent.models import CaseResult  # noqa: E402
from platform_core.runner import AgentOutput  # noqa: E402


def make_trial(root: Path, *, reward=0, f2p=0.5, p2p=1.0, exc=None, verdicts=("pass",),
               patch=b"diff --git a/x b/x\n") -> Path:
    t = root / "task-x__abc1234"
    (t / "agent" / "conv").mkdir(parents=True)
    (t / "artifacts").mkdir()
    (t / "verifier").mkdir()
    rewards = None if reward is None else {
        "reward": reward, "f2p": f2p, "p2p": p2p, "partial": 0.9, "f2p_total": 10,
        "f2p_passed": int(10 * f2p), "p2p_total": 20, "p2p_passed": int(20 * p2p)}
    (t / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": rewards} if rewards else None,
        "exception_info": {"exception_type": exc} if exc else None,
        "task_id": {"path": "/x/tasks/task-x"}}))
    rs = [{"role": "baseline", "attempt": 0, "stop_reason": "end_turn", "edits": 0}]
    for i, v in enumerate(verdicts, 1):
        rs += [{"role": "patch", "attempt": i, "stop_reason": "end_turn", "edits": 3, "llm_calls": 2},
               {"role": "verify", "attempt": i, "stop_reason": "end_turn", "verdict": v, "llm_calls": 1}]
    (t / "agent" / "run_summary.json").write_text(json.dumps(
        {"outcome": "completed", "role_stats": rs, "signals": {}, "llm": {"n_calls": 5}}))
    (t / "agent" / "conv" / "patch.1.json").write_text(json.dumps({
        "role": "patch", "attempt": 1, "n_messages": 3, "stop_reason": "end_turn",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                     {"role": "assistant", "content": "x" * 5000, "reasoning": "y" * 9000,
                      "tool_calls": [{"function": {"name": "Bash",
                                                   "arguments": json.dumps({"command": "z" * 3000})}}]}]}))
    (t / "agent" / "trajectory.json").write_text(json.dumps({"steps": [
        {"extra": {"role": "verify"}, "observation": {"results": [
            {"content": json.dumps({"verdict": verdicts[-1], "issues": ["a"]})}]}}]}))
    (t / "artifacts" / "model.patch").write_bytes(patch)
    (t / "verifier" / "test-stdout.txt").write_text("HIDDEN_TEST_OUTPUT test_secret_behaviour FAILED")
    return t


class TrialTests(unittest.TestCase):
    def test_reward_and_classes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            o = load_outcome(make_trial(Path(d)))
        self.assertEqual(o["reward"], 0)
        self.assertIsNone(o["infra_class"])
        self.assertEqual(o["verdicts"], ["pass"])
        self.assertIn("near_miss", failure_classes(o))
        self.assertIn("verify_false_pass", failure_classes(o))

    def test_build_break_and_empty_patch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            o = load_outcome(make_trial(Path(d), f2p=0.0, p2p=0.0, patch=b""))
        self.assertTrue({"build_break", "empty_patch"} <= set(failure_classes(o)))

    def test_verifier_timeout_is_infra(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            o = load_outcome(make_trial(Path(d), reward=None, exc="VerifierTimeoutError"))
        self.assertEqual(o["infra_class"], "verifier_timeout")

    def test_solved_has_no_failure_classes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            o = load_outcome(make_trial(Path(d), reward=1, f2p=1.0))
        self.assertEqual(failure_classes(o), [])


class ScorerTests(unittest.TestCase):
    def test_trusted_path_scored_untrusted_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            t = make_trial(Path(d), reward=1, f2p=1.0)
            sc = S.DeepSWESeedlingScorer()
            os.environ["SID_PIER_JOBS_ROOT"] = d
            ok = sc.score({"meta_info": {"language": "go"}}, AgentOutput(result="", metadata={"trial_dir": str(t)}))
            os.environ["SID_PIER_JOBS_ROOT"] = str(Path(d) / "elsewhere")
            bad = sc.score({}, AgentOutput(result="", metadata={"trial_dir": str(t)}))
        self.assertEqual((ok["score"], ok["passed"], ok["details"]["excluded"]), (1.0, True, False))
        self.assertTrue(bad["details"]["excluded"])
        self.assertEqual(bad["details"]["infra_class"], "untrusted_trial_dir")

    def test_aggregate_divides_by_scored(self) -> None:
        cases = [CaseResult(case_id="a", score=1.0, passed=True, details={"reward": 1, "verdicts": ["pass"], "language": "go", "patch_bytes": 5}),
                 CaseResult(case_id="b", score=0.0, passed=False, details={"reward": 0, "verdicts": ["pass"], "language": "go", "failure_classes": ["near_miss"], "patch_bytes": 5}),
                 CaseResult(case_id="c", score=0.0, passed=False, details={"excluded": True, "infra_class": "env_start"})]
        agg = S.DeepSWESeedlingScorer().aggregate(cases, [])
        self.assertEqual((agg["n_scored"], agg["n_excluded_infra"], agg["resolve_rate_of_scored"]), (2, 1, 0.5))
        self.assertEqual(agg["verify_false_pass_rate"], 0.5)
        self.assertEqual(agg["harness_checks"], [("env_start", 1)])
        cats = {c["category_id"] for c in categorize_errors(cases)}
        self.assertEqual(cats, {"seedling__near_miss", "infra__env_start"})


class IsolationTests(unittest.TestCase):
    def test_export_never_contains_verifier_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            t = make_trial(Path(d))
            o = load_outcome(t)
            dest = Path(d) / "scratch"
            pier_case.export_artifacts(t, dest, o, run_report(o, dispatch_outputs(t)))
            blob = "".join(p.read_text(errors="ignore") for p in dest.rglob("*") if p.is_file())
            longest = max(len(l) for p in dest.rglob("*.txt") for l in p.read_text().splitlines())
        self.assertNotIn("HIDDEN_TEST_OUTPUT", blob)
        self.assertNotIn("test_secret_behaviour", blob)
        self.assertLessEqual(longest, MAX_LINE)

    def test_host_isolation_scan(self) -> None:
        self.assertEqual(scan_source("p = PROMPTS / 'x.md'\ns = p.read_text()\n"), [])
        bad = [m for _, m in scan_source(
            "import subprocess\nopen('/etc/passwd')\nfrom pathlib import Path\nPath('x').read_text()\n"
            "import os\nos.listdir('.')\nq = '../../groups'\n")]
        self.assertTrue(any("subprocess" in m for m in bad))
        self.assertTrue(any("open()" in m for m in bad))
        self.assertTrue(any("read_text" in m for m in bad))
        self.assertTrue(any("os.listdir" in m for m in bad))
        self.assertTrue(any("/groups" in m for m in bad))


class ValidatorLogicTests(unittest.TestCase):
    def test_role_mandate_and_a0(self) -> None:
        good = {"outcome": "completed", "role_stats": [
            {"role": "patch", "missing_keys": [], "report_ok": True},
            {"role": "verify", "missing_keys": [], "report_ok": True}],
            "role_prompt_sha": {"patch": "p", "verify": "v"}, "sys_sha_seen": ["p", "v"]}
        self.assertEqual(check_dry_run(good), [])
        solo = dict(good, role_stats=[good["role_stats"][0]])
        self.assertTrue(any("ROLE MANDATE" in e for e in check_dry_run(solo)))
        a0 = dict(good, sys_sha_seen=["p"])
        self.assertTrue(any("A0" in e for e in check_dry_run(a0)))

    def test_settings_guard(self) -> None:
        ok = {"REASONING_EFFORT": "medium", "MAX_TOKENS": 32768, "MAX_PATCH_ATTEMPTS": 8,
              "wall_frac": {"baseline": 0.06, "patch": 0.22, "verify": 0.15, "solo": 0.92}}
        self.assertEqual(check_settings(ok), [])
        self.assertTrue(check_settings(dict(ok, REASONING_EFFORT="high")))
        self.assertTrue(check_settings(dict(ok, wall_frac={"baseline": 0.5, "patch": 0.5, "verify": 0.5})))


class CasesTests(unittest.TestCase):
    def test_cases_file(self) -> None:
        rows = [json.loads(l) for l in (PROJECT / "benchmark" / "cases.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 113)
        self.assertEqual(len({r["id"] for r in rows}), 113)
        self.assertTrue(all("task_dir" in r["context"] and r["meta_info"]["language"] for r in rows))


if __name__ == "__main__":
    unittest.main()

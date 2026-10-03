"""Run the curated SAFETY-INVARIANT subset of seedling's own tests against a candidate tree.

    $PIERPY run_invariants.py <candidate task_agent dir>      (exit 0 = all pass)

Runs under PIER's interpreter (py3.13 + litellm + pier), because seedling imports them.
The vendored test files (``vendored/``, verbatim copies of seedling_v8_3/tests at the
seed's tree_sha a6a2f291) locate the package as ``Path(__file__).parents[1]/"seedling"``,
so they are staged into a temp dir next to a symlink to the candidate's package.

Only INVARIANTS gate an edit -- properties any evolved harness must keep (methods it calls
exist, per-run state, truncation != end_turn, VERIFY write confinement, compaction never
corrupts/erases the conversation, run() never escapes). Tests that pin v8.3 DESIGN choices
(schema-only pipeline, BASELINE sharing VERIFY's conversation, transient messages,
v8.3-specific constants/strings) are deliberately NOT run: they would veto legitimate
evolution of exactly the seam the meta-agent is meant to change. The role mandate and the
A0 prompt check are enforced behaviourally by ``dry_run.py`` instead.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVARIANTS = {
    "test_wiring.py": [
        "test_every_llm_method_called_actually_exists",
        "test_every_git_method_called_actually_exists",
        "test_compact_uses_an_async_llm_method",
        "test_token_totals_reach_final_metrics_and_are_idempotent",
        "test_cached_tokens_are_actually_incremented_and_reported_per_call",
        "test_arg_path_recognises_every_real_tool_schema",
        "test_truncated_generation_is_not_treated_as_end_turn",
        "test_max_tokens_exceeds_an_observed_thinking_block",
        "test_two_runstates_are_independent",
    ],
    "test_compact.py": [
        "test_stub_matches_the_real_chat_signature",
        "test_below_threshold_is_a_noop",
        "test_above_threshold_compacts_and_preserves_system_and_tail",
        "test_no_orphaned_tool_result_at_the_seam",
        "test_service_failure_leaves_the_conversation_intact",
        "test_empty_summary_is_treated_as_failure",
        "test_wiring_error_is_raised_not_swallowed",
        "test_stub_summary_is_rejected_and_conversation_kept",
        "test_continuation_is_detected_and_rejected",
    ],
}


def main(candidate: Path) -> int:
    pkg = candidate / "seedling"
    if not (pkg / "__init__.py").is_file():
        print(f"FAIL: no seedling package under {candidate}")
        return 1
    fails = 0
    with tempfile.TemporaryDirectory(prefix="sid_selftest_") as tmp:
        root = Path(tmp)
        (root / "seedling").symlink_to(pkg.resolve(), target_is_directory=True)
        (root / "tests").mkdir()
        for f in ("test_wiring.py", "test_compact.py", "test_containment.py"):
            shutil.copyfile(HERE / "vendored" / f, root / "tests" / f)
        sys.path.insert(0, str(root))
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        for fname, names in INVARIANTS.items():
            spec = importlib.util.spec_from_file_location(fname[:-3], root / "tests" / fname)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {fname}: import error {type(exc).__name__}: {exc}")
                fails += 1
                continue
            for name in names:
                fn = getattr(mod, name, None)
                if fn is None:
                    print(f"FAIL {fname}::{name}: missing from the vendored file")
                    fails += 1
                    continue
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    fails += 1
                    print(f"FAIL {fname}::{name}: {type(exc).__name__}: {str(exc)[:300]}")
        # test_containment.py is a script (fault injection: run() must never let a
        # non-CancelledError escape and must always leave trajectory + run_summary).
        env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run([sys.executable, str(root / "tests" / "test_containment.py")],
                           capture_output=True, text=True, timeout=120, env=env, cwd=str(root))
        if r.returncode != 0:
            fails += 1
            print("FAIL test_containment.py:\n" + (r.stdout + r.stderr)[-1500:])
    print("ALL INVARIANTS PASS" if not fails else f"{fails} INVARIANT FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))

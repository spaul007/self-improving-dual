"""Tests for the per-agent output_failure/budget_exhausted metadata and
trace.jsonl 'error' event surfacing in
projects/travel_mas_refactored/seed/mas_workflow.py.

Real gap this closes: the old design only ever modeled failure for the
sightseeing stage specifically (`sightseeing_failed`/`task_failure`), and
`budget_exhausted` was a single flag OR'd across all three tool-calling
stages -- losing which stage actually hit its cap. Confirmed live this
session: `task_failure` and `sightseeing_failed` were IDENTICAL sets on
every real case in the current seed (task_failure = sightseeing_failed
AND NOT budget_exhausted, and budget_exhausted never fires in practice),
so tracking both as separate curriculum goals was pure redundancy.

The new design is fully generic over all four stages (flight/train/
sightseeing/accounting): two INDEPENDENT per-agent flags,
`{agent}_output_failure` and `{agent}_budget_exhausted` -- whether they
co-occur for the same agent is itself the signal for whether running out
of iterations was the likely cause, without a third derived flag to
encode that. Each failing/budget-exhausted stage also gets a structured
`trace.emit("error", {...})` event, so an agentic reader (block_suggester/
editor) can find "which agent failed and why" directly from
logs/trace.jsonl instead of only from the case's final AgentOutput.metadata.

    PYTHONPATH=. python3 -m unittest tests.test_travel_mas_refactored_mas_workflow_metadata
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEED_DIR = REPO_ROOT / "projects" / "travel_mas_refactored" / "seed"

# Real modules under the seed's own `agents` package/`tool_wrapper` sibling
# (used by mas_workflow.py's own imports) that must be evicted from
# sys.modules after each test -- several projects in this repo reuse the
# generic names `agents`/`tool_wrapper`, so leaving a stale entry behind
# would silently break an unrelated test run later in the same session.
_SEED_MODULE_NAMES = [
    "agents", "agents.immutable", "agents.immutable.message",
    "agents.flight", "agents.train", "agents.sightseeing", "agents.accounting",
    "tool_wrapper",
]


def _fake_msg(sender, **kwargs):
    from agents.immutable.message import AgentMessage
    return AgentMessage(sender=sender, content="", **kwargs)


class MasWorkflowMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in _SEED_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SEED_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SEED_DIR)))
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in _SEED_MODULE_NAMES])

        spec = importlib.util.spec_from_file_location(
            "travel_mas_refactored_seed_mas_workflow", SEED_DIR / "mas_workflow.py"
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

        self.trace_events: list[tuple[str, dict]] = []
        fake_trace = SimpleNamespace(emit=lambda kind, payload: self.trace_events.append((kind, payload)))
        self.mod.trace = fake_trace

    def _run(self, *, flight=None, train=None, sightseeing=None, accounting=None):
        flight = flight or _fake_msg("flight", ok=True, iterations=3)
        train = train or _fake_msg("train", ok=True, iterations=2)
        sightseeing = sightseeing or _fake_msg("sightseeing", ok=True, iterations=10)
        accounting = accounting or _fake_msg("accounting", ok=True, iterations=1)

        self.mod.run_flight_stage = MagicMock(return_value=flight)
        self.mod.run_train_stage = MagicMock(return_value=train)
        self.mod.run_sightseeing_stage = MagicMock(return_value=sightseeing)
        self.mod.run_accounting_stage = MagicMock(return_value=accounting)

        task = SimpleNamespace(description="x", case_id="1", context={})
        return self.mod.run_task(task)

    def test_all_stages_ok_produces_no_failure_flags(self) -> None:
        out = self._run()
        self.assertEqual(
            out.metadata["stage_iterations"],
            {"flight": 3, "train": 2, "sightseeing": 10},
        )
        self.assertFalse(any(k.endswith("_output_failure") for k in out.metadata))
        self.assertFalse(any(k.endswith("_budget_exhausted") for k in out.metadata))
        self.assertEqual(self.trace_events, [])
        self.mod.run_accounting_stage.assert_called_once()

    def test_sightseeing_output_failure_sets_per_agent_flag_and_reason(self) -> None:
        out = self._run(
            sightseeing=_fake_msg(
                "sightseeing", ok=False, iterations=8, error="no <itinerary> tag",
            )
        )
        self.assertTrue(out.metadata["sightseeing_output_failure"])
        self.assertEqual(out.metadata["sightseeing_output_failure_reason"], "no <itinerary> tag")
        self.assertNotIn("sightseeing_budget_exhausted", out.metadata)
        self.assertEqual(out.result, "")
        # Accounting must never run on a sightseeing failure.
        self.mod.run_accounting_stage.assert_not_called()

    def test_sightseeing_failure_with_budget_exhausted_sets_both_independent_flags(self) -> None:
        out = self._run(
            sightseeing=_fake_msg(
                "sightseeing", ok=False, iterations=80, budget_exhausted=True,
                error="hit iteration cap",
            )
        )
        self.assertTrue(out.metadata["sightseeing_output_failure"])
        self.assertTrue(out.metadata["sightseeing_budget_exhausted"])

    def test_output_truncated_is_its_own_flag(self) -> None:
        out = self._run(
            sightseeing=_fake_msg(
                "sightseeing", ok=False, iterations=8, output_truncated=True,
                error="cut off",
            )
        )
        self.assertTrue(out.metadata["sightseeing_output_truncated"])

    def test_flight_budget_exhausted_fires_independently_even_when_ok(self) -> None:
        # A stage can hit its own iteration cap and STILL recover a usable
        # answer on its own (ok=True) -- budget_exhausted is not
        # conditioned on ok being False.
        out = self._run(flight=_fake_msg("flight", ok=True, iterations=80, budget_exhausted=True))
        self.assertTrue(out.metadata["flight_budget_exhausted"])
        self.assertNotIn("flight_output_failure", out.metadata)

    def test_trace_emits_one_error_event_per_failing_or_exhausted_stage(self) -> None:
        self._run(
            flight=_fake_msg("flight", ok=True, iterations=80, budget_exhausted=True),
            sightseeing=_fake_msg("sightseeing", ok=False, iterations=8, error="boom"),
        )
        kinds_agents = [(kind, payload["agent"]) for kind, payload in self.trace_events]
        self.assertIn(("error", "flight"), kinds_agents)
        self.assertIn(("error", "sightseeing"), kinds_agents)
        sightseeing_payload = next(p for k, p in self.trace_events if p["agent"] == "sightseeing")
        self.assertEqual(sightseeing_payload["output_failure"], True)
        self.assertEqual(sightseeing_payload["budget_exhausted"], False)
        self.assertEqual(sightseeing_payload["message"], "boom")

    def test_accounting_failure_is_tracked_too_after_sightseeing_succeeds(self) -> None:
        out = self._run(accounting=_fake_msg("accounting", ok=False, error="bad tally"))
        self.assertTrue(out.metadata["accounting_output_failure"])
        self.assertEqual(out.metadata["accounting_output_failure_reason"], "bad tally")


if __name__ == "__main__":
    unittest.main()

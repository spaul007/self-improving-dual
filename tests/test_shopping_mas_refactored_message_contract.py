"""Tests for the frozen AgentMessage/inbox inter-agent contract introduced
in projects/shopping_mas_refactored/shopping_mas/ -- the one structural
idea adopted from travel_mas_refactored's own refactor (see that
project's agents/immutable/message.py), replacing the shared mutable
`State` object the un-forked projects/shopping_mas/ still uses.

Real gap this closes: a pure internal-representation refactor (shared
mutable State -> frozen per-stage AgentMessage passed via inbox) has no
behavior change as its explicit goal, but that's exactly the kind of
change where a migration bug (a stray `state.*` reference, the
product_scout fan-out losing an item, the repair loop picking up the
wrong iteration's plan) is easy to introduce silently. These tests pin
the contract itself (frozen, from_sender's lookup-by-name semantics) and
the two trickiest migration points call out in that project's own
mas_workflow.py docstring: the product_scout fan-out collapsing into one
aggregate message, and the cart_optimizer<->cart_executor repair loop
only exposing its FINAL iteration through the frozen contract while still
recording every iteration into agent_log.

Does NOT attempt the full-120-case behavioral-parity check against the
un-forked projects/shopping_mas/ baseline (0.8829) -- that needs a live
LLM endpoint and is a separate, manual verification step (see this
project's own entry in the implementation plan), not a unit test.

    PYTHONPATH=. python3 -m unittest tests.test_shopping_mas_refactored_message_contract
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEED_DIR = REPO_ROOT / "projects" / "shopping_mas_refactored" / "shopping_mas"

# Bare top-level names shopping_mas_refactored's own seed code imports
# (agents/config/llm_client/tools) -- several projects in this repo reuse
# these same generic names, so stale sys.modules entries left behind by one
# test would silently corrupt an unrelated test run later in the same
# process. Evicted before AND after every test.
_SEED_MODULE_NAMES = [
    "agents", "agents.base", "agents.immutable", "agents.immutable.message",
    "agents.requirement_parser", "agents.requirement_parser.workflow",
    "agents.product_scout", "agents.product_scout.workflow",
    "agents.cart_optimizer", "agents.cart_optimizer.workflow",
    "agents.cart_executor", "agents.cart_executor.workflow",
    "config", "llm_client", "mas_prompt_cfg", "tools", "tools.loader",
]


def _line_item(item_id, description="a shirt"):
    from agents.base import LineItem
    return LineItem(item_id=item_id, constraints={"description": description})


class _MasWorkflowTestCase(unittest.TestCase):
    """Loads shopping_mas_refactored's real mas_workflow.py by file path
    (never through the package's own __init__ chain, so this stays
    independent of however the rest of the test suite imports
    projects.shopping_mas_refactored) and stubs out the four stage
    modules' own `run` functions directly on the loaded module object --
    same pattern tests/test_travel_mas_refactored_mas_workflow_metadata.py
    already uses for travel_mas_refactored's own mas_workflow.py."""

    def setUp(self) -> None:
        for name in _SEED_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SEED_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SEED_DIR)))
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in _SEED_MODULE_NAMES])

        spec = importlib.util.spec_from_file_location(
            "shopping_mas_refactored_seed_mas_workflow", SEED_DIR / "mas_workflow.py"
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

        self.tmpdir = tempfile.mkdtemp(prefix="shopping_mas_refactored_test_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir, ignore_errors=True))

        # Never touch the real benchmark tools/database in these tests --
        # only the orchestration layer (mas_workflow.py) is under test.
        self.mod._fetch_user_info = MagicMock(return_value={"demographics": {}})
        self.mod._fetch_products = MagicMock(return_value={})
        self.mod.loader = SimpleNamespace(make_toolset=lambda *_a, **_k: {})

        self.workflow = self.mod.MASWorkflow.__new__(self.mod.MASWorkflow)
        self.workflow.cfg = SimpleNamespace(
            level=1, max_llm_calls=240, max_repair_iterations=2, scout_workers=4,
        )
        self.workflow.llm = MagicMock()  # never actually called; every stage is stubbed


class AgentMessageContractTests(_MasWorkflowTestCase):
    def test_agent_message_is_frozen(self) -> None:
        from agents.immutable.message import AgentMessage

        msg = AgentMessage(sender="x", content={"a": 1})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            msg.sender = "y"  # type: ignore[misc]

    def test_from_sender_raises_cleanly_on_missing_sender(self) -> None:
        from agents.immutable.message import AgentMessage, from_sender

        inbox = [AgentMessage(sender="requirement_parser", content={})]
        with self.assertRaises(KeyError) as ctx:
            from_sender(inbox, "product_scout")
        self.assertIn("requirement_parser", str(ctx.exception))

    def test_from_sender_returns_the_matching_message(self) -> None:
        from agents.immutable.message import AgentMessage, from_sender

        target = AgentMessage(sender="cart_optimizer", content={"plan": True})
        inbox = [AgentMessage(sender="requirement_parser", content={}), target]
        self.assertIs(from_sender(inbox, "cart_optimizer"), target)


class ProductScoutAggregateMessageTests(_MasWorkflowTestCase):
    """The product_scout fan-out (one scout per line item, run concurrently)
    must collapse into exactly ONE "product_scout" AgentMessage, keyed by
    item_id -- not one message per scout. Verified by capturing the inbox
    cart_optimizer.run actually receives."""

    def test_two_line_items_produce_one_aggregate_message_keyed_by_item_id(self) -> None:
        from agents.immutable.message import from_sender

        items = [_line_item(1, "shirt"), _line_item(2, "shoes")]
        self.mod.parser.run = MagicMock(return_value=(items, None, {"raw": True}))

        def fake_scout_run(llm, cfg, item, user_info, toolset, products_map,
                           trace=None, counter=None):
            return [{"product_id": f"p{item.item_id}"}], f"note-{item.item_id}"

        self.mod.scout.run = MagicMock(side_effect=fake_scout_run)

        captured = {}

        def fake_optimizer_run(llm, cfg, items, query, user_info, inbox, products_map,
                               repair_notes=None, trace=None, counter=None):
            captured["inbox"] = inbox
            return None  # end the case here; we only care about the inbox shape

        self.mod.optimizer.run = MagicMock(side_effect=fake_optimizer_run)
        self.mod.executor.run = MagicMock()

        self.workflow.run_task("case-1", "buy a shirt and shoes", self.tmpdir)

        scout_msg = from_sender(captured["inbox"], "product_scout")
        self.assertEqual(
            scout_msg.content,
            {
                1: {"candidates": [{"product_id": "p1"}], "note": "note-1"},
                2: {"candidates": [{"product_id": "p2"}], "note": "note-2"},
            },
        )
        self.mod.executor.run.assert_not_called()


class RepairLoopFinalMessageTests(_MasWorkflowTestCase):
    """cart_optimizer<->cart_executor can repeat up to
    cfg.max_repair_iterations times; the frozen contract must only ever
    reflect the LAST iteration's outcome, while agent_log keeps every
    iteration (no information lost)."""

    def test_fails_once_then_succeeds_records_both_but_keeps_last_plan(self) -> None:
        items = [_line_item(1)]
        self.mod.parser.run = MagicMock(return_value=(items, None, {"raw": True}))
        self.mod.scout.run = MagicMock(return_value=([{"product_id": "p1"}], "note"))

        plan_attempt_1 = {
            "assignments": {1: {"product_id": "p1"}}, "coupons": {},
            "skipped_items": [], "base_total": 10.0, "discount": 0.0,
            "final_price": 10.0, "source": "optimize",
        }
        plan_attempt_2 = {
            "assignments": {1: {"product_id": "p2"}}, "coupons": {},
            "skipped_items": [], "base_total": 12.0, "discount": 0.0,
            "final_price": 12.0, "source": "optimize",
        }
        self.mod.optimizer.run = MagicMock(side_effect=[
            dict(plan_attempt_1), dict(plan_attempt_2),
        ])
        self.mod.executor.run = MagicMock(side_effect=[
            ("failed", [{"kind": "product", "id": "p1"}], {"raw": 1}),
            ("ok", [], {"raw": 2}),
        ])

        result = self.workflow.run_task("case-1", "buy a shirt", self.tmpdir)

        # Both iterations recorded -- no information lost.
        optimizer_records = [r for r in result["agent_log"] if r["agent"] == "cart_optimizer"]
        executor_records = [r for r in result["agent_log"] if r["agent"] == "cart_executor"]
        self.assertEqual(len(optimizer_records), 2)
        self.assertEqual(len(executor_records), 2)

        # But the final, externally-visible result reflects ONLY the last
        # iteration's plan (p2, not p1) and its "ok" status.
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plan"]["assignments"], {1: "p2"})
        self.assertEqual(self.mod.optimizer.run.call_count, 2)
        self.assertEqual(self.mod.executor.run.call_count, 2)

    def test_optimizer_producing_no_plan_records_issue_and_stops(self) -> None:
        items = [_line_item(1)]
        self.mod.parser.run = MagicMock(return_value=(items, None, {"raw": True}))
        self.mod.scout.run = MagicMock(return_value=([], "no candidates"))
        self.mod.optimizer.run = MagicMock(return_value=None)
        self.mod.executor.run = MagicMock()

        result = self.workflow.run_task("case-1", "buy a shirt", self.tmpdir)

        self.assertIsNone(result["plan"])
        self.assertIn("optimizer: no usable plan", result["issues"])
        self.mod.executor.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

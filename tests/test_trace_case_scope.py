"""Tests for ``platform_core.trace.case_scope`` — the ambient case_id
context that the runner wraps around each ``workflow.run_task`` call so
every emitted event (``tool_call``, ``tool_result``, ``llm_call``,
``mutable_log``, ``error``) automatically carries the originating case_id.

    PYTHONPATH=. python3 -m unittest tests.test_trace_case_scope
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from platform_core import trace


class CaseScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="trace_test_")
        os.close(fd)
        self.trace_path = Path(path)
        self.addCleanup(self.trace_path.unlink, missing_ok=True)
        self._prev_env = os.environ.get(trace.TRACE_PATH_ENV)
        os.environ[trace.TRACE_PATH_ENV] = str(self.trace_path)
        def _restore() -> None:
            if self._prev_env is None:
                os.environ.pop(trace.TRACE_PATH_ENV, None)
            else:
                os.environ[trace.TRACE_PATH_ENV] = self._prev_env
        self.addCleanup(_restore)

    def _read_events(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # ---- No ambient scope → events carry no case_id ----
    def test_emit_outside_scope_has_no_case_id(self) -> None:
        trace.emit("tool_call", {"name": "x", "id": "1"})
        evs = self._read_events()
        self.assertEqual(len(evs), 1)
        self.assertNotIn("case_id", evs[0]["payload"])

    # ---- Inside scope → ambient case_id is merged into the payload ----
    def test_emit_inside_scope_inherits_case_id(self) -> None:
        with trace.case_scope("c42"):
            trace.emit("tool_call", {"name": "x", "id": "1"})
            trace.log("verifier_fired", verdict="pass", name="budget")
        evs = self._read_events()
        self.assertEqual(len(evs), 2)
        for ev in evs:
            self.assertEqual(ev["payload"].get("case_id"), "c42")
        # Mutable_log specifics survive.
        log_ev = next(e for e in evs if e["kind"] == "mutable_log")
        self.assertEqual(log_ev["payload"]["label"], "verifier_fired")
        self.assertEqual(log_ev["payload"]["verdict"], "pass")

    # ---- Explicit case_id in payload always wins over ambient ----
    def test_explicit_case_id_in_payload_wins(self) -> None:
        with trace.case_scope("ambient"):
            trace.emit("tool_call", {"name": "x", "case_id": "explicit"})
        evs = self._read_events()
        self.assertEqual(evs[0]["payload"]["case_id"], "explicit")

    # ---- Falsy case_id (empty string) means "no ambient scope" ----
    def test_empty_case_id_scope_does_not_tag(self) -> None:
        with trace.case_scope(""):
            trace.emit("tool_call", {"name": "x"})
        evs = self._read_events()
        self.assertNotIn("case_id", evs[0]["payload"])

    # ---- Scope is properly reset on exit ----
    def test_scope_reset_after_exit(self) -> None:
        with trace.case_scope("c1"):
            trace.emit("tool_call", {"name": "in"})
        trace.emit("tool_call", {"name": "after"})
        evs = self._read_events()
        self.assertEqual(evs[0]["payload"].get("case_id"), "c1")
        self.assertNotIn("case_id", evs[1]["payload"])

    # ---- Nested scopes — inner wins, outer restored on exit ----
    def test_nested_scopes(self) -> None:
        with trace.case_scope("outer"):
            trace.emit("tool_call", {"name": "o1"})
            with trace.case_scope("inner"):
                trace.emit("tool_call", {"name": "i"})
            trace.emit("tool_call", {"name": "o2"})
        evs = self._read_events()
        self.assertEqual(evs[0]["payload"]["case_id"], "outer")
        self.assertEqual(evs[1]["payload"]["case_id"], "inner")
        self.assertEqual(evs[2]["payload"]["case_id"], "outer")

    # ---- Scope is per-thread/per-async-task (ContextVar guarantee) ----
    def test_scope_is_per_thread(self) -> None:
        results: dict[str, str] = {}
        # Block thread A inside its scope until thread B has emitted, so we
        # know thread B did NOT see thread A's ambient case_id.
        ready_a = threading.Event()
        proceed_a = threading.Event()

        def thread_a() -> None:
            with trace.case_scope("A"):
                ready_a.set()
                proceed_a.wait()
                trace.emit("tool_call", {"name": "from_a"})

        def thread_b() -> None:
            ready_a.wait()  # ensure A is inside its scope
            trace.emit("tool_call", {"name": "from_b"})
            proceed_a.set()  # release A to finish

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start(); tb.start()
        ta.join(); tb.join()

        evs = self._read_events()
        by_name = {ev["payload"]["name"]: ev for ev in evs}
        self.assertEqual(by_name["from_a"]["payload"].get("case_id"), "A")
        self.assertNotIn("case_id", by_name["from_b"]["payload"])

    # ---- No env var → no file written (even inside scope) ----
    def test_no_env_var_no_op(self) -> None:
        del os.environ[trace.TRACE_PATH_ENV]
        with trace.case_scope("c1"):
            trace.emit("tool_call", {"name": "x"})
            trace.log("verifier_fired", verdict="pass")
        # The trace_path file we created during setUp should still be empty.
        self.assertEqual(self.trace_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()

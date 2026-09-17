"""Tests for meta_agent.run_inspect_agentic: transcript parsing and the
edit_memory/ loader."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meta_agent import run_inspect_agentic as ra


def _events():
    return [
        {"t": 10.0, "kind": "llm_call", "i": 0, "n_messages": 2, "tools": ["bash", "editor"]},
        {"t": 12.0, "kind": "llm_response", "i": 0, "elapsed_s": 2.0, "content": "look",
         "tool_calls": [{"id": "c1", "name": "bash"}], "usage": {"input_tokens": 100, "output_tokens": 10, "reasoning_tokens": 1},
         "stop_reason": "completed"},
        {"t": 12.1, "kind": "tool_call", "i": 0, "call_id": "c1", "name": "bash", "input": {"command": "ls"},
         "result": "a\nb", "result_chars": 3, "elapsed_s": 0.1},
        {"t": 12.2, "kind": "budget_reminder", "used": 1, "total": 40},
        {"t": 13.0, "kind": "llm_call", "i": 1, "n_messages": 4, "tools": ["bash", "editor"]},
        {"t": 15.0, "kind": "llm_response", "i": 1, "elapsed_s": 2.0, "content": "edit",
         "tool_calls": [{"id": "c2", "name": "editor"}, {"id": "c3", "name": "validate"}],
         "usage": {"input_tokens": 200, "output_tokens": 20, "reasoning_tokens": 2}, "stop_reason": "completed"},
        {"t": 15.1, "kind": "tool_call", "i": 1, "call_id": "c2", "name": "editor",
         "input": {"command": "str_replace", "path": "$NODE_DIR/task_agent/workflow.py", "old_str": "a\nb", "new_str": "a\nc"},
         "result": "edited", "result_chars": 6, "elapsed_s": 0.01},
        {"t": 15.2, "kind": "tool_call", "i": 1, "call_id": "c3", "name": "validate", "input": {},
         "result": "All validators passed.", "result_chars": 22, "elapsed_s": 0.2},
        {"t": 15.3, "kind": "validation", "round": 1, "changed_files": ["workflow.py"], "errors": []},
        {"t": 16.0, "kind": "llm_call", "i": 2, "n_messages": 6, "tools": []},
        {"t": 16.5, "kind": "llm_error", "i": 2, "error": "boom"},
        {"t": 17.0, "kind": "end", "reason": "llm_error", "success": False, "n_llm_calls": 3, "validation_rounds": 1},
    ]


class TranscriptTests(unittest.TestCase):
    def test_groups_by_llm_index(self):
        tr = ra.parse_transcript_lines(json.dumps(e) for e in _events())
        self.assertEqual([s.i for s in tr.steps], [0, 1, 2])
        s0, s1, s2 = tr.steps
        self.assertEqual(s0.content, "look")
        self.assertEqual([tc.name for tc in s0.tool_calls], ["bash"])
        self.assertEqual(s0.notes[0]["kind"], "budget_reminder")
        self.assertEqual([tc.name for tc in s1.tool_calls], ["editor", "validate"])
        self.assertEqual(s1.notes[0]["kind"], "validation")
        self.assertEqual(s2.error, "boom")
        self.assertEqual(tr.end["reason"], "llm_error")
        self.assertEqual(len(tr.validations), 1)
        self.assertEqual((tr.t_start, tr.t_end), (10.0, 17.0))
        self.assertFalse(tr.truncated_tail)
        self.assertEqual(ra.tool_call_counts(tr), {"bash": 1, "editor": 1, "validate": 1})

    def test_token_curve_cumulative(self):
        tr = ra.parse_transcript_lines(json.dumps(e) for e in _events())
        curve = ra.token_curve(tr)
        self.assertEqual([c["cum_input"] for c in curve], [100, 300, 300])
        self.assertEqual([c["cum_output"] for c in curve], [10, 30, 30])
        self.assertEqual(curve[1]["reasoning"], 2)

    def test_partial_last_line(self):
        lines = [json.dumps(e) for e in _events()[:3]] + ['{"t": 12.2, "kind": "tool_ca']
        tr = ra.parse_transcript_lines(lines)
        self.assertTrue(tr.truncated_tail)
        self.assertIsNone(tr.end)
        self.assertEqual(len(tr.steps), 1)

    def test_editor_call_helpers(self):
        sr = {"command": "str_replace", "path": "$NODE_DIR/task_agent/workflow.py", "old_str": "a\nb", "new_str": "a\nc"}
        self.assertEqual(ra.editor_call_summary(sr), "str_replace workflow.py")
        d = ra.editor_call_as_diff(sr)
        self.assertIn("-b", d)
        self.assertIn("+c", d)
        self.assertIn("b/workflow.py", d)
        ins = {"command": "insert", "path": "x/task_agent/mutable_tools/t.py", "insert_line": 3, "new_str": "z"}
        self.assertEqual(ra.editor_call_summary(ins), "insert mutable_tools/t.py after line 3")
        self.assertIn("+z", ra.editor_call_as_diff(ins))
        cr = {"command": "create", "path": "/abs/task_agent/mutable_tools/new.py", "file_text": "\n".join(str(i) for i in range(300))}
        dc = ra.editor_call_as_diff(cr)
        self.assertIn("+199", dc)
        self.assertNotIn("+200\n", dc)
        self.assertIn("100 more lines", dc)
        view = {"command": "view", "path": "$NODE_DIR/task_agent/workflow.py", "view_range": [10, 40]}
        self.assertEqual(ra.editor_call_summary(view), "view workflow.py[10:40]")
        self.assertIsNone(ra.editor_call_as_diff(view))

    def test_load_transcript_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ra.load_transcript(Path(d) / "nope.jsonl"))
            self.assertIsNone(ra.load_session(Path(d) / "nope.json"))


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_edit_memory(exp: Path) -> Path:
    em = exp / "edit_memory"
    state = {
        "memory_version": 2, "instruction_version": 1, "window_index": 2, "window": [9],
        "window_failed": [], "node_arms": {"1": ["none", None], "5": ["with", 1], "7": ["without", 1]},
        "pulls": {"with": 2, "without": 1},
        "config": {"window_size": 4},
        "events": [
            {"t": 1789639664.226, "event": "setup"},
            {"t": 1789646218.716, "event": "window_add", "node_id": 1},
            {"t": 1789650000.0, "event": "memory_written", "window": 1, "memory_version": 1},
            {"t": 1789650001.0, "event": "instruction_failed", "update": 1, "error": "E" * 500},
        ],
    }
    _w(em / "state.json", json.dumps(state))
    _w(em / "edit_memory_v001.md", "## 1. Ranked edits\nold\n")
    _w(em / "edit_memory_v002.md", "## 1. Ranked edits\nnew\n")
    _w(em / "edit_memory.md", "## 1. Ranked edits\nnew\n")
    _w(em / "instruction_v000.md", "")
    _w(em / "instruction_v001.md", "- do x\n")
    _w(em / "instruction.md", "- do x\n")
    w1 = em / "window_001"
    _w(w1 / "window.json", json.dumps({"window_index": 1, "memory_version_before": 0, "instruction_version": 0, "t": 1.0,
                                      "nodes": [{"node_id": 1, "parent_id": 0, "memory_arm": "none", "n_evals": 16, "mean_utility": 0.2}]}))
    _w(w1 / "curation.md", "## Node 1\n")
    _w(w1 / "memory_call.json", json.dumps({"kind": "memory_call", "accepted": True,
                                           "attempts": [{"attempt": 1, "elapsed_s": 5.0, "input_tokens": 1, "output_tokens": 2, "chars": 3, "errors": []}]}))
    _w(w1 / "agentic" / "transcript.jsonl", '{"t": 1.0, "kind": "end", "reason": "submitted", "success": true}\n')
    _w(w1 / "agentic" / "session.json", json.dumps({"success": True, "summary": "ok"}))
    # window_002 exists but is still empty (curator running)
    (em / "window_002").mkdir()
    u1 = em / "instruction_update_001"
    _w(u1 / "nodes.json", json.dumps([{"node_id": 5}, {"node_id": 6}]))
    _w(u1 / "q.md", "## Edit memory usage by editors\n")
    _w(u1 / "update_call.json", json.dumps({"kind": "update_call", "accepted": False,
                                           "attempts": [{"attempt": 1, "errors": ["missing section"]}]}))
    return em


class EditMemoryTests(unittest.TestCase):
    def test_load_layout(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            make_edit_memory(exp)
            em = ra.load_edit_memory(exp)
            self.assertIsNotNone(em)
            self.assertEqual([v for v, _ in em.memory_versions], [1, 2])
            self.assertEqual([v for v, _ in em.instruction_versions], [0, 1])
            self.assertEqual([w.index for w in em.windows], [1, 2])
            w1, w2 = em.windows
            self.assertEqual(w1.window["nodes"][0]["node_id"], 1)
            self.assertTrue(w1.has_agentic)
            self.assertEqual(ra.memory_call_rows(w1.memory_call)[0]["elapsed_s"], 5.0)
            self.assertEqual(w2.window, {})
            self.assertIsNone(w2.curation_md)
            self.assertFalse(w2.has_agentic)
            self.assertEqual(len(em.updates), 1)
            u = em.updates[0]
            self.assertEqual([n["node_id"] for n in u.nodes], [5, 6])
            self.assertEqual(ra.memory_call_rows(u.update_call)[0]["errors"], "missing section")
            self.assertEqual(em.current_memory_path.read_text(), "## 1. Ranked edits\nnew\n")
            self.assertEqual(ra.session_path(w1.dir / "agentic").name, "session.json")
            self.assertIsInstance(ra.edit_memory_signature(em.dir), tuple)

    def test_state_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            make_edit_memory(exp)
            em = ra.load_edit_memory(exp)
            rows = ra.events_rows(em.state)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["event"], "setup")
            self.assertTrue(rows[0]["time"].startswith("2026-09-17"))
            self.assertIn("PDT", rows[0]["time"])
            self.assertEqual(rows[2]["memory_version"], 1)
            self.assertEqual(len(rows[3]["error"]), 200)
            arms = ra.node_arm_rows(em.state)
            self.assertEqual(arms, [
                {"node_id": 1, "arm": "none", "memory_version": None},
                {"node_id": 5, "arm": "with", "memory_version": 1},
                {"node_id": 7, "arm": "without", "memory_version": 1},
            ])
            diff = ra.diff_markdown("a\nold\n", "a\nnew\n", old_label="v1", new_label="v2")
            self.assertIn("-old", diff)
            self.assertIn("+new", diff)

    def test_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ra.load_edit_memory(Path(d)))
            self.assertIsNone(ra.edit_memory_dir(Path(d)))


if __name__ == "__main__":
    unittest.main()

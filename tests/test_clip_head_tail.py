"""clip_head_tail (meta_agent/evaluator.py): a traceback clipped for the
editor must keep its opening AND its final exception line.

    PYTHONPATH=. python3 -m unittest tests.test_clip_head_tail
"""
from __future__ import annotations

import unittest

from meta_agent.evaluator import _CLIP_MARKER, clip_head_tail


class ClipHeadTailTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(clip_head_tail("x" * 2000), "x" * 2000)

    def test_keeps_first_300_and_last_1700(self):
        text = "H" * 300 + "M" * 5000 + "T" * 1700
        out = clip_head_tail(text)
        self.assertTrue(out.startswith("H" * 300 + _CLIP_MARKER))
        self.assertTrue(out.endswith("T" * 1700))
        self.assertNotIn("M", out.replace("[... middle of traceback omitted ...]", ""))

    def test_exception_line_survives(self):
        text = "Traceback (most recent call last):\n" + "  File x\n" * 500 + "ImportError: no module named foo"
        self.assertTrue(clip_head_tail(text).endswith("ImportError: no module named foo"))

    def test_idempotent(self):
        once = clip_head_tail("a" * 9000)
        self.assertEqual(clip_head_tail(once), once)


if __name__ == "__main__":
    unittest.main()

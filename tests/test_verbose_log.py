"""Tests for the verbose-logging helper module.

The helpers must:
  * No-op every writer when ``META_AGENT_VERBOSE`` is unset.
  * Write the file under ``<round_dir>/verbose/`` when set to ``"1"``.
  * Round-trip JSON cleanly with non-ASCII content.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_verbose_log
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from meta_agent import verbose_log  # noqa: E402


class VerboseLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="verbose_log_test_"))
        # Snapshot the env var so tests don't leak state to each other.
        self._orig_env = os.environ.get("META_AGENT_VERBOSE")
        os.environ.pop("META_AGENT_VERBOSE", None)

    def tearDown(self) -> None:
        if self._orig_env is None:
            os.environ.pop("META_AGENT_VERBOSE", None)
        else:
            os.environ["META_AGENT_VERBOSE"] = self._orig_env
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_disabled_by_default(self) -> None:
        self.assertFalse(verbose_log.is_enabled())

    def test_writers_noop_when_disabled(self) -> None:
        verbose_log.write_text(self.tmp, "foo.txt", "hello")
        verbose_log.write_json(self.tmp, "foo.json", {"k": 1})
        # No verbose dir, no files
        self.assertFalse((self.tmp / "verbose").exists())

    def test_writers_emit_when_enabled(self) -> None:
        os.environ["META_AGENT_VERBOSE"] = "1"
        self.assertTrue(verbose_log.is_enabled())

        verbose_log.write_text(self.tmp, "foo.txt", "hello world")
        verbose_log.write_json(self.tmp, "foo.json", {"k": 1, "s": "héllo"})

        verbose = self.tmp / "verbose"
        self.assertTrue(verbose.is_dir())
        self.assertEqual((verbose / "foo.txt").read_text(encoding="utf-8"), "hello world")
        loaded = json.loads((verbose / "foo.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded, {"k": 1, "s": "héllo"})

    def test_arbitrary_value_is_stringified_for_json(self) -> None:
        os.environ["META_AGENT_VERBOSE"] = "1"
        # Path is non-trivially JSON-serialisable; the writer's default=str
        # should turn it into a string instead of crashing.
        verbose_log.write_json(self.tmp, "p.json", {"path": Path("/tmp/x")})
        body = (self.tmp / "verbose" / "p.json").read_text(encoding="utf-8")
        self.assertIn("/tmp/x", body)


if __name__ == "__main__":
    unittest.main()

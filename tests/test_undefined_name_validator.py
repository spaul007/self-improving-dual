"""Tests for UndefinedNameValidator (meta_agent/editor_validators.py) --
catches a name referenced inside a function body but never bound in any
enclosing scope, which ast.parse()/compile() can't (Python only resolves
names inside a `def` at call time, not at parse/compile time).

    PYTHONPATH=. python3 -m unittest tests.test_undefined_name_validator
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from meta_agent.editor_validators import UndefinedNameValidator


class UndefinedNameValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="undefined_name_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out_dir = self.tmp / "out"
        self.agent_dir = self.out_dir / "task_agent"
        self.agent_dir.mkdir(parents=True)
        self.validator = UndefinedNameValidator()

    def _write(self, rel_path: str, source: str) -> None:
        path = self.agent_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def test_flags_undefined_name_used_inside_a_function_body(self) -> None:
        # The real bug this validator exists for: a helper that constructs
        # AgentMessage(...) without ever importing AgentMessage. Invisible
        # to ast.parse()/compile() -- Python only resolves the name when
        # the function actually runs.
        self._write(
            "mas_workflow.py",
            "def merge(a, b):\n"
            "    return AgentMessage(sender='x', content=a + b)\n",
        )
        errors = self.validator.validate(self.out_dir, self.tmp)
        self.assertEqual(len(errors), 1)
        self.assertIn("AgentMessage", errors[0])
        self.assertIn("mas_workflow.py", errors[0])

    def test_clean_when_name_is_properly_imported(self) -> None:
        self._write(
            "mas_workflow.py",
            "from agents.immutable.message import AgentMessage\n\n"
            "def merge(a, b):\n"
            "    return AgentMessage(sender='x', content=a + b)\n",
        )
        # The import target doesn't need to exist on disk for pyflakes'
        # static check -- it only needs the name to be bound in scope.
        self.assertEqual(self.validator.validate(self.out_dir, self.tmp), [])

    def test_does_not_flag_unused_import(self) -> None:
        # Deliberately narrow: style-only pyflakes findings (unused
        # imports/variables, redefinitions) are not correctness bugs and
        # must not gate an edit.
        self._write(
            "mas_workflow.py",
            "import json\n\n"
            "def foo():\n"
            "    return 1\n",
        )
        self.assertEqual(self.validator.validate(self.out_dir, self.tmp), [])

    def test_syntax_error_is_skipped_not_crashed_on(self) -> None:
        # SyntaxValidator's job to report -- this validator must not raise
        # or double-report when ast.parse itself fails.
        self._write("mas_workflow.py", "def broken(:\n    pass\n")
        errors = self.validator.validate(self.out_dir, self.tmp)
        self.assertEqual(errors, [])

    def test_only_the_buggy_file_is_flagged_among_several(self) -> None:
        self._write("clean.py", "def foo():\n    return 1\n")
        self._write(
            "buggy.py",
            "def bar():\n    return SomethingUndefined()\n",
        )
        errors = self.validator.validate(self.out_dir, self.tmp)
        self.assertEqual(len(errors), 1)
        self.assertIn("buggy.py", errors[0])
        self.assertIn("SomethingUndefined", errors[0])

    def test_missing_task_agent_dir_returns_no_errors(self) -> None:
        empty_out = self.tmp / "empty_out"
        empty_out.mkdir()
        self.assertEqual(self.validator.validate(empty_out, self.tmp), [])


if __name__ == "__main__":
    unittest.main()

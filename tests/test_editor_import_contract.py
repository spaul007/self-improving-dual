"""The editor prompt and the validators must agree about imports.

A live run lost 7 of its first 14 expansions to failed edits; 82% of the
validator errors were the single mistake ``from platform_core import X``. The
prompt was *stricter* than the code it was judged by — rules 2 and 4 omitted
``platform_core.trace`` while rule 7 mandated calling it — so obeying the rules
made the mandate unsatisfiable, and the rejection message named the module
rather than the statement form that caused it.

These tests pin the three properties whose absence allowed that: the prompt and
the validators list the same platform paths, the four import spellings validate
exactly as documented, and a rejection of the bare package carries the fix (the
retry replays that string, so it is the only channel that can make attempt 2
differ from attempt 1).
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from meta_agent.editor_validators import (
    ImportValidator,
    MutableToolImportValidator,
    SchemaWrapperConsistencyValidator,
)


class _Stop(Exception):
    pass


def _editor_system_prompt() -> str:
    """The system prompt the editor is actually sent, captured at the LLM call
    rather than duplicated here — a copy would drift out of date silently."""
    from meta_agent.agent_editor import AgentEditor

    captured: dict = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    editor = AgentEditor(llm_caller=fake_llm, validators=[])
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "round_001"
        (out_dir / "task_agent" / "mutable_tools").mkdir(parents=True)
        (out_dir / "task_agent" / "workflow.py").write_text(
            "def run_task(task):\n    return None\n"
        )
        (out_dir / "task_agent" / "tool_wrapper.py").write_text("")
        (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
        try:
            editor._self_improve(
                out_dir=out_dir, feedback=None, context=None,
                prior_errors=[], attempt=1,
            )
        except _Stop:
            pass
    for msg in captured["messages"]:
        if msg["role"] == "system":
            return msg["content"]
    raise AssertionError("editor sent no system message")


def _rule(prompt: str, number: int) -> str:
    """The text of hard rule ``number``, up to the next numbered rule."""
    m = re.search(rf"^  {number}\. (.*?)(?=^  {number + 1}\. |\n\n)",
                  prompt, re.S | re.M)
    assert m, f"rule {number} not found in the editor prompt"
    return m.group(1)


def _paths_named(text: str) -> set[str]:
    return set(re.findall(r"platform_core\.[a-z_]+", text))


def _write_agent(root: Path, *, workflow: str = "def run_task(task):\n    return None\n",
                 wrapper: str = "", tool: str | None = None) -> Path:
    out_dir = root / "round_001"
    agent = out_dir / "task_agent" / "mutable_tools"
    agent.mkdir(parents=True)
    (out_dir / "task_agent" / "workflow.py").write_text(workflow)
    (out_dir / "task_agent" / "tool_wrapper.py").write_text(wrapper)
    (out_dir / "task_agent" / "tools_schema.json").write_text("[]")
    if tool is not None:
        (agent / "helper.py").write_text(tool)
    return out_dir


class TestPromptMatchesValidators(unittest.TestCase):
    """Every platform path the prompt calls allowed must actually be allowed,
    and vice versa. This is the check whose absence let the contradiction sit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.prompt = _editor_system_prompt()

    def test_workflow_rule_matches_import_validator(self) -> None:
        named = _paths_named(_rule(self.prompt, 2))
        allowed = {p for p in ImportValidator.WORKFLOW_ALLOWED
                   if p.startswith("platform_core.")}
        # The rule names platform_core.tools only to forbid it there.
        self.assertNotIn("platform_core.tools", allowed)
        self.assertEqual(named - {"platform_core.tools"}, allowed)

    def test_wrapper_rule_matches_import_validator(self) -> None:
        named = _paths_named(_rule(self.prompt, 3))
        allowed = {p for p in ImportValidator.WRAPPER_ALLOWED
                   if p.startswith("platform_core.")}
        self.assertEqual(named, allowed)

    def test_mutable_tool_rule_matches_its_validator(self) -> None:
        named = _paths_named(_rule(self.prompt, 4))
        allowed = {p for p in MutableToolImportValidator.ALLOWED_PREFIXES
                   if p.startswith("platform_core.")}
        self.assertEqual(named, allowed)

    def test_trace_is_importable_wherever_it_is_mandated(self) -> None:
        """Rule 7 requires platform_core.trace.log in workflow.py and in
        mutable tools. Both must be able to import it."""
        self.assertIn("platform_core.trace", _rule(self.prompt, 7))
        self.assertIn("platform_core.trace", ImportValidator.WORKFLOW_ALLOWED)
        self.assertIn("platform_core.trace",
                      MutableToolImportValidator.ALLOWED_PREFIXES)

    def test_prompt_shows_the_exact_legal_and_illegal_forms(self) -> None:
        self.assertIn("from platform_core.trace import log", self.prompt)
        self.assertIn("import platform_core.trace", self.prompt)
        self.assertIn("from platform_core import trace", self.prompt)

    def test_prompt_flags_the_wrapper_only_spelling(self) -> None:
        """tool_wrapper.py's `from platform_core import tools` is the only
        in-context example, and copying it into workflow.py is exactly what
        failed. The prompt must say it is wrapper-only."""
        self.assertIn("from platform_core import tools", self.prompt)
        self.assertRegex(self.prompt, r"ONLY in tool_wrapper\.py")


class TestImportForms(unittest.TestCase):
    """The four spellings, validated as documented."""

    def _errors(self, workflow_src: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_agent(Path(tmp), workflow=workflow_src)
            return ImportValidator().validate(out_dir, out_dir)

    def test_dotted_from_import_is_legal(self) -> None:
        self.assertEqual(
            self._errors("from platform_core.trace import log\n"
                         "def run_task(task):\n    return None\n"), [])

    def test_dotted_plain_import_is_legal(self) -> None:
        self.assertEqual(
            self._errors("import platform_core.trace\n"
                         "def run_task(task):\n    return None\n"), [])

    def test_from_bare_package_is_rejected(self) -> None:
        errors = self._errors("from platform_core import trace\n"
                              "def run_task(task):\n    return None\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("'platform_core'", errors[0])

    def test_bare_package_import_is_rejected(self) -> None:
        errors = self._errors("import platform_core\n"
                              "def run_task(task):\n    return None\n")
        self.assertEqual(len(errors), 1)

    def test_workflow_may_not_import_tools_directly(self) -> None:
        errors = self._errors("from platform_core.tools import call_tool\n"
                              "def run_task(task):\n    return None\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("platform_core.tools", errors[0])


class TestErrorsCarryTheFix(unittest.TestCase):
    """max_attempts=2 replays the error string against the same prompt, so a
    message that does not contain the fix guarantees the retry repeats."""

    def test_bare_package_error_gives_the_corrective_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_agent(
                Path(tmp),
                workflow="from platform_core import trace\n"
                         "def run_task(task):\n    return None\n")
            errors = ImportValidator().validate(out_dir, out_dir)
        self.assertIn("from platform_core.X import name", errors[0])
        self.assertIn("import platform_core.X", errors[0])
        self.assertIn("trace", errors[0])

    def test_mutable_tool_error_lists_trace_as_allowed(self) -> None:
        """The old message said mutable tools may import only
        platform_core.tools — contradicting its own ALLOWED_PREFIXES and
        telling the editor to strip the instrumentation rule 7 demands."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_agent(Path(tmp),
                                   tool="from platform_core import trace\n")
            errors = MutableToolImportValidator().validate(out_dir, out_dir)
        self.assertEqual(len(errors), 1)
        self.assertIn("platform_core.trace", errors[0])
        self.assertIn("from platform_core.X import name", errors[0])

    def test_mutable_tool_may_import_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_agent(Path(tmp),
                                   tool="from platform_core.trace import log\n")
            self.assertEqual(
                MutableToolImportValidator().validate(out_dir, out_dir), [])

    def test_unbacked_schema_entry_names_both_remedies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = _write_agent(Path(tmp))
            (out_dir / "task_agent" / "tools_schema.json").write_text(
                '[{"type": "function", "function": '
                '{"name": "validate_itinerary", "parameters": {}}}]')
            errors = SchemaWrapperConsistencyValidator().validate(out_dir, out_dir)
        self.assertEqual(len(errors), 1)
        self.assertIn("mutable_tools/validate_itinerary.py", errors[0])
        self.assertIn("remove", errors[0])


if __name__ == "__main__":
    unittest.main()

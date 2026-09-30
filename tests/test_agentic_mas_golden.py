"""Golden lock on the agentic editor's multi-agent (exclude-list) prompts.

``tests/test_agentic_golden.py`` pins sep18's single-agent texts; this pins
everything the block-HGM port adds: the multi-agent system prompt (both read
scopes, with and without the skipped-validator and hard-coded-answers
wording), the instruction with and without an assignment, on both memory
arms, with a suggestion, and with the implementation-strategy and curriculum
axes on, plus the exclude-mode tool texts and every block's scope. Property
checks: the without-memory arm renders exactly like a run with no memory
layer, and no absolute path leaks into any text.

    PYTHONPATH=. python3 -m unittest tests.test_agentic_mas_golden
    # deliberate prompt changes only:
    PYTHONPATH=. python3 -m tests.test_agentic_mas_golden --regen
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from meta_agent.agent_editor_agentic import AgenticEditor
from meta_agent.agentic.policy import MEMORY_DIR_NAME, REPO_ROOT, RUN_ROOT_MARKER
from meta_agent.assignment import ExpandAssignment
from meta_agent.block_suggester import _BLOCK_BODIES, block_scope
from meta_agent.editor_validators import SmokeTestValidator, SyntaxValidator
from meta_agent.implementation_strategy import _IMPLEMENTATION_STRATEGY_BODIES

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "agentic_golden_mas.json"
PROJECT = REPO_ROOT / "projects" / "travel_mas_refactored"
EXCLUDE = ["agents/immutable/", "benchmark/", "workflow.py"]
OVERVIEW = "projects/travel_mas_refactored/adapter/agentic_editor_overview.md"


class _HardcodeCheck:
    """Stands in for the hardcoded_answers validator (only the flag matters)."""
    rejects_hardcoded_answers = True

    def validate(self, out_dir, base_dir):
        return []


class _Stop(Exception):
    pass


def _mas_agent(root: Path) -> None:
    ta = root / "task_agent"
    (ta / "agents" / "immutable").mkdir(parents=True)
    (ta / "mutable_tools").mkdir()
    for rel, text in {
        "workflow.py": "import mas_workflow\ndef run_task(task):\n    return mas_workflow.run_task(task)\n",
        "mas_workflow.py": "def run_task(task):\n    return None\n",
        "agents/flight.py": "PROMPT = 'fly'\n",
        "agents/common.py": "X = 1\n",
        "agents/immutable/message.py": "class AgentMessage: ...\n",
        "mas_llm_backbone.yaml": "default: {}\n",
        "tool_wrapper.py": "from platform_core import tools\n",
        "tools_schema.json": "[]\n",
        "mutable_tools/__init__.py": "",
    }.items():
        (ta / rel).write_text(text, encoding="utf-8")


def _render(*, scope: str = "run", assignment: Optional[ExpandAssignment] = None,
            memory: bool = False, memory_dir: bool = False,
            skipped: bool = False, hardcode: bool = False) -> dict:
    """Render one session's system prompt, instruction and tools through
    the real ``apply`` (the fake LLM captures the first call and stops)."""
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "runs" / "exp"
        run.mkdir(parents=True)
        (run / RUN_ROOT_MARKER).write_text("")
        parent = run / "round_001"
        _mas_agent(parent)
        (parent / "logs").mkdir()
        for name in ("hgm_node.json", "strategy.json", "feedback.json", "eval_result.json",
                     "failure_summary.md", "behavior_memory.md", "assignment.json"):
            (parent / name).write_text("{}")
        (parent / "logs" / "case_1.json").write_text("{}")
        memory_path = None
        if memory or memory_dir:
            (run / MEMORY_DIR_NAME).mkdir()
        if memory:
            memory_path = run / MEMORY_DIR_NAME / "edit_memory_v001.md"
            memory_path.write_text("memory")
        captured: dict = {}

        def fake_llm(**kw):
            captured["messages"] = kw["messages"]
            captured["tools"] = kw["tools"]
            raise _Stop()

        validators = [SyntaxValidator()]
        if skipped:
            validators.append(SmokeTestValidator())
        if hardcode:
            validators.append(_HardcodeCheck())
        editor = AgenticEditor(
            fake_llm, validators, mutable_exclude=EXCLUDE, project_root=PROJECT,
            sandbox="none", read_scope=scope, max_llm_calls=150, timeout_s=5400,
            max_attempts=3, strategies_path="strategies.md",
            project_overview_path=OVERVIEW,
        )
        editor.apply(None, parent, run / "round_002", memory_path=memory_path,
                     assignment=assignment)
        texts = {
            "system": captured["messages"][0]["content"],
            "instruction": captured["messages"][1]["content"],
            "tools": {t["name"]: t.get("description", "") for t in captured["tools"]},
        }
        blob = json.dumps(texts)
        if d in blob:
            raise AssertionError(f"temp path leaked into the prompts: {d}")
        return texts


def _assignment(block: str = "verifiers", **kw) -> ExpandAssignment:
    return ExpandAssignment(block=block, block_scope=block_scope(block), **kw)


def render_all() -> dict[str, str]:
    out: dict[str, str] = {}
    base = _render()
    out["system:run"] = base["system"]
    out["system:parent"] = _render(scope="parent")["system"]
    both = _render(skipped=True, hardcode=True)
    out["system:run:smoke_skipped+hardcode"] = both["system"]
    out["tool:bash"] = base["tools"]["bash"]
    out["tool:editor"] = base["tools"]["editor"]
    out["tool:validate:smoke_skipped"] = both["tools"]["validate"]
    out["instruction:no_assignment"] = base["instruction"]
    out["instruction:parent_scope:assignment"] = _render(
        scope="parent", assignment=_assignment())["instruction"]
    out["instruction:assignment"] = _render(assignment=_assignment())["instruction"]
    out["instruction:assignment:with_memory"] = _render(
        assignment=_assignment(), memory=True)["instruction"]
    out["instruction:assignment:mixed"] = _render(assignment=_assignment("mixed"))["instruction"]
    suggestion = "Target: accounting\nDiagnosis: totals drift\nProposed change: re-sum costs"
    out["instruction:assignment:suggestion"] = _render(
        assignment=_assignment(suggestion=suggestion))["instruction"]
    out["instruction:assignment:suggestion:with_memory"] = _render(
        assignment=_assignment(suggestion=suggestion), memory=True)["instruction"]
    out["instruction:assignment:impl+curriculum"] = _render(assignment=_assignment(
        implementation_strategy="harness_heavy",
        implementation_strategy_body=_IMPLEMENTATION_STRATEGY_BODIES["harness_heavy"],
        curriculum_directive="Target the check `transfer_gap` (failing in 12/40 cases)."))["instruction"]
    for block in sorted(_BLOCK_BODIES):
        out[f"block_scope:{block}"] = block_scope(block)
    return out


class AgenticMasGoldenTests(unittest.TestCase):
    def test_texts_match_fixture(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = render_all()
        self.assertEqual(sorted(actual), sorted(expected))
        for key in expected:
            with self.subTest(text=key):
                self.assertEqual(actual[key], expected[key])

    def test_without_memory_arm_equals_no_memory_run(self) -> None:
        """The without arm (edit_memory/ exists and is masked, no memory
        file) must read exactly like a run with no memory layer at all."""
        for a in (None, _assignment()):
            with self.subTest(assignment=a is not None):
                self.assertEqual(_render(assignment=a, memory_dir=True),
                                 _render(assignment=a))

    def test_memory_is_mentioned_only_on_the_with_arm(self) -> None:
        without = _render(assignment=_assignment(), memory_dir=True)
        self.assertNotIn("memory", (without["system"] + without["instruction"]).replace(
            "behavior_memory.md", "").lower())
        with_arm = _render(assignment=_assignment(), memory=True)
        self.assertIn("$EDIT_MEMORY_FILE", with_arm["instruction"])
        self.assertIn("stay within the selected block", with_arm["instruction"])
        self.assertEqual(with_arm["system"], without["system"])

    def test_default_assignment_shows_only_the_block(self) -> None:
        text = _render(assignment=_assignment())["instruction"]
        self.assertIn("## Selected block for this EXPAND: verifiers\nScope: ", text)
        for absent in ("## Current curriculum focus", "## Implementation strategy",
                       "## Advisory suggestion"):
            self.assertNotIn(absent, text)


if __name__ == "__main__":
    if "--regen" in sys.argv:
        FIXTURE.write_text(json.dumps(render_all(), indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()

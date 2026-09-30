"""Golden lock on the DEFAULT editor's prompts (``editor.type: default``).

Pins, byte for byte, every message ``AgentEditor`` sends in its single-shot
mode and in its ``agentic_editing`` tool-loop mode, for both mutable
surfaces (include-list and ``mutable_exclude``), with and without a
block suggestion, including the validation-retry turn. The fixture was
generated from unmodified vivek_mas @ 1bd6884, before the agentic-editor /
edit-memory port factored the shared exclude-mode rules into
``exclude_surface_and_core_rules`` -- this test is what proves that refactor
(and everything else in the port) left the default editor untouched.

    PYTHONPATH=. python3 -m unittest tests.test_default_editor_prompt_golden
    # deliberate default-editor prompt changes only:
    PYTHONPATH=. python3 -m tests.test_default_editor_prompt_golden --regen
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "default_editor_prompts.json"
EXCLUDE = ["agents/immutable/", "benchmark/", "workflow.py"]


def _feedback():
    from meta_agent.models import AgentFeedback, CaseResult, EvaluationResult, EvolutionStrategy

    return AgentFeedback(
        round_number=3,
        base_round=1,
        strategy=EvolutionStrategy(
            target_files=["workflow.py"], optimization_goal="prior goal",
            proposed_changes="prior change", rationale="prior why",
        ),
        eval_result=EvaluationResult(
            score=0.425, passed=1, failed=3,
            per_case=[
                CaseResult(case_id=f"c{i}", passed=i == 0, score=0.25 * i) for i in range(4)
            ],
        ),
        tool_usage={"search_trains": 7, "search_hotels": 3},
        llm_calls=11,
        tool_error_rate={"search_trains": 0.25, "search_hotels": 0.0},
        runtime_exceptions=["KeyError: 'price'"],
        edit_errors=[],
        trace_n_cases=2,
        project_metrics={
            "composite_mean": 0.425,
            "top_failed_checks": [["transfer_gap", 3], ["budget", 2]],
            "dimension_scores": {"cost": 0.4, "time": 0.7},
        },
    )


def _make_workspace(root: Path, *, exclude_mode: bool) -> Path:
    base = root / "base"
    agent = base / "task_agent"
    (agent / "mutable_tools").mkdir(parents=True)
    (agent / "workflow.py").write_text("def run_task(task):\n    return None\n", encoding="utf-8")
    (agent / "tool_wrapper.py").write_text("from platform_core import tools\n", encoding="utf-8")
    (agent / "tools_schema.json").write_text("[]\n", encoding="utf-8")
    (agent / "mutable_tools" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    if exclude_mode:
        (agent / "agents" / "immutable").mkdir(parents=True)
        (agent / "mas_workflow.py").write_text("def run_task(task):\n    return 1\n", encoding="utf-8")
        (agent / "agents" / "flight.py").write_text("PROMPT = 'fly'\n", encoding="utf-8")
        (agent / "agents" / "immutable" / "message.py").write_text("class M: ...\n", encoding="utf-8")
        (agent / "mas_llm_backbone.yaml").write_text("default: {}\n", encoding="utf-8")
    return base


def _capture(*, exclude_mode: bool, agentic: bool, has_suggestion: bool) -> list[dict]:
    """Run ``apply`` with a fake LLM that never edits (2 attempts, so the
    retry turn is captured too) and return every call's kwargs, with the temp
    dir replaced by ``<TMP>``."""
    from meta_agent.agent_editor import AgentEditor

    tmp = Path(tempfile.mkdtemp(prefix="default_editor_golden_"))
    try:
        base = _make_workspace(tmp, exclude_mode=exclude_mode)
        calls: list[dict] = []

        def fake_llm(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(content="no edits this time", tool_calls=[])

        editor = AgentEditor(
            llm_caller=fake_llm, validators=[], max_attempts=2,
            model="meta-model", base_url="http://meta/v1",
            tools_source="# tools source\ndef search_trains(): ...\n",
            db_schema="trains(id, no, dep, arr)",
            mutable_exclude=list(EXCLUDE) if exclude_mode else None,
            agentic_editing=agentic, agentic_max_turns=3,
        )
        editor.apply(
            _feedback(), base, tmp / "out",
            context="## Selected block for this EXPAND: verifiers\nSTEERING TEXT",
            has_suggestion=has_suggestion,
        )
        text = json.dumps(calls, sort_keys=True, default=str).replace(str(tmp), "<TMP>")
        return json.loads(text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render_all() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for exclude_mode in (False, True):
        for agentic in (False, True):
            for has_suggestion in (False, True):
                key = (
                    f"{'exclude' if exclude_mode else 'include'}:"
                    f"{'agentic' if agentic else 'single'}:"
                    f"{'suggestion' if has_suggestion else 'plain'}"
                )
                out[key] = _capture(
                    exclude_mode=exclude_mode, agentic=agentic, has_suggestion=has_suggestion
                )
    return out


class DefaultEditorPromptGoldenTests(unittest.TestCase):
    def test_prompts_match_fixture(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = render_all()
        self.assertEqual(sorted(actual), sorted(expected))
        for key in expected:
            with self.subTest(scenario=key):
                self.assertEqual(actual[key], expected[key])

    def test_fixture_covers_the_retry_turn(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for key, calls in expected.items():
            with self.subTest(scenario=key):
                self.assertEqual(len(calls), 2)  # attempt 1 + the retry
                self.assertIn("Previous attempt failed validation", calls[1]["messages"][1]["content"])


if __name__ == "__main__":
    if "--regen" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(render_all(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()

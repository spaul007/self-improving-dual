"""Golden test: the agentic editor's prompts and tool descriptions are pinned
byte for byte.

Everything the model sees in a no-memory session — system prompt (both read
scopes), instruction message on a fixed layout, the bash/editor/validate/
submit tool descriptions and the four reminder texts — is compared against
``tests/fixtures/agentic_golden.json``. Any refactor of the session machinery
(the SubmitSpec seam, the edit-memory delivery) must leave these identical:
the finished 2026-09-13 no-editmem run was produced with exactly these texts.

Regenerate the fixture ONLY for a deliberate prompt change:
    PYTHONPATH=. python -m tests.test_agentic_golden --regen
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "agentic_golden.json"


def _agent(round_dir: Path, workflow: str) -> None:
    a = round_dir / "task_agent"
    (a / "mutable_tools").mkdir(parents=True, exist_ok=True)
    (a / "workflow.py").write_text(workflow, encoding="utf-8")
    (a / "tool_wrapper.py").write_text("def x(): return None\n", encoding="utf-8")
    (a / "tools_schema.json").write_text("[]", encoding="utf-8")
    (a / "mutable_tools" / "__init__.py").write_text("", encoding="utf-8")


def render_all() -> dict[str, str]:
    """Every text the editor sends, rendered on a fixed temp layout with the
    absolute temp path normalised out (the instruction never contains it,
    which is itself one of the pinned properties)."""
    from meta_agent.agentic import session as S
    from meta_agent.agentic.policy import RUN_ROOT_MARKER, build_policy
    from meta_agent.agentic.tools import (
        SUBMIT_TOOL, VALIDATE_TOOL, bash_tool_info, editor_tool_info,
    )

    out: dict[str, str] = {}
    for scope in ("run", "parent"):
        out[f"system:{scope}"] = S.agentic_system_prompt(scope)
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        (run / RUN_ROOT_MARKER).write_text("")
        base, node = run / "round_001", run / "round_002"
        _agent(base, "def run_task(task):\n    return None\n")
        (base / "feedback.json").write_text("{}")
        (base / "strategy.json").write_text("{}")
        (base / "hgm_node.json").write_text("{}")
        (base / "eval_result.json").write_text("{}")
        (base / "logs").mkdir()
        (base / "logs" / "case_1.json").write_text("{}")
        (base / "logs" / "trace.jsonl").write_text("")
        _agent(node, "def run_task(task):\n    return None\n")
        (node / "agentic" / "scratch").mkdir(parents=True)
        for scope in ("run", "parent"):
            pol = build_policy(out_dir=node, base_dir=base, repo_root=Path(tmp) / "repo",
                               project_root=None, read_scope=scope)
            text = S.render_instruction(pol, max_llm_calls=150, timeout_s=5400, max_attempts=3)
            assert tmp not in text
            out[f"instruction:{scope}"] = text
            out[f"bash_tool:{scope}"] = json.dumps(
                bash_tool_info(bash_timeout_s=120, max_output_chars=20000,
                               root_vars=pol.root_vars()), sort_keys=True)
    out["editor_tool"] = json.dumps(editor_tool_info(max_view_chars=40000), sort_keys=True)
    out["validate_tool"] = json.dumps(VALIDATE_TOOL, sort_keys=True)
    out["submit_tool"] = json.dumps(SUBMIT_TOOL, sort_keys=True)
    out["msg:nudge"] = S.NUDGE_MESSAGE
    out["msg:halfway"] = S.HALFWAY_MESSAGE
    out["msg:final_stretch"] = S.FINAL_STRETCH_MESSAGE
    out["msg:wrap_up"] = S.WRAP_UP_MESSAGE
    return out


class TestAgenticGolden(unittest.TestCase):
    def test_matches_fixture(self) -> None:
        self.assertTrue(FIXTURE.exists(), f"missing fixture {FIXTURE}; run --regen once")
        want = json.loads(FIXTURE.read_text(encoding="utf-8"))
        got = render_all()
        self.assertEqual(sorted(got), sorted(want))
        for key in want:
            self.assertEqual(got[key], want[key], f"{key} drifted from the golden fixture")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(render_all(), indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()

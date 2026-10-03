"""logs_guide: a project-supplied description of logs/ for the agentic editor and block
suggester. Unset must leave both prompts byte-identical."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from meta_agent import block_suggester as bs
from meta_agent.agent_editor import AgentEditor
from meta_agent.block_suggester import BlockSuggester
from meta_agent.log_access import load_logs_guide


class LogsGuideTests(unittest.TestCase):
    def test_default_closing_is_unchanged_text(self) -> None:
        self.assertIn("'converted_plan' field", bs._SYSTEM_CLOSING_AGENTIC)
        self.assertNotIn(bs._LOGS_DESCRIPTION_MARK, bs._SYSTEM_CLOSING_AGENTIC)
        sug = BlockSuggester(llm_caller=lambda **k: None, agentic_access=True)
        self.assertEqual(sug._agentic_closing(), bs._SYSTEM_CLOSING_AGENTIC)

    def test_guide_replaces_travel_description(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            g = Path(d) / "logs_guide.md"
            g.write_text("Start with logs/DOSSIERS.md; transcripts under logs/scratch/<case>/<run>/transcripts/.")
            sug = BlockSuggester(llm_caller=lambda **k: None, agentic_access=True, logs_guide=str(g))
            text = sug._agentic_closing()
            self.assertNotIn("converted_plan", text)
            self.assertIn("LOGS LAYOUT FOR THIS PROJECT", text)
            self.assertIn("DOSSIERS.md", text)
            ed = AgentEditor(llm_caller=lambda **k: None, validators=[], agentic_editing=True,
                             agentic_log_access=True, logs_guide=str(g))
            self.assertIn("DOSSIERS.md", ed.logs_guide)

    def test_missing_guide_falls_back(self) -> None:
        self.assertEqual(load_logs_guide("/nonexistent/guide.md"), "")
        self.assertEqual(load_logs_guide(None), "")


if __name__ == "__main__":
    unittest.main()

"""What the manager hands an assignment-steered editor for one EXPAND.

Block HGM steers each expansion with a selected block (and, when those
optional axes are on, an implementation strategy and a curriculum focus). The
default editor receives that steering as prose inside its ``context`` string
(``HGMManager._render_expand_context``). The agentic editor
(``editor.type: agentic``, ``steering: assignment``) instead gets this
structured record and renders its own ``## Selected block for this EXPAND``
section (``meta_agent/agentic/session.py::render_assignment``); it reads the
rest of the evidence -- lineage, siblings, summaries -- from the run directory
itself. The manager also writes it to ``round_NNN/assignment.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

ASSIGNMENT_FILE = "assignment.json"


@dataclass(frozen=True)
class ExpandAssignment:
    block: str
    # The block's scope text (block_suggester.block_scope): what part of the
    # agent this expansion should change.
    block_scope: str
    implementation_strategy: Optional[str] = None
    implementation_strategy_body: Optional[str] = None
    curriculum_directive: Optional[str] = None
    # A configured block suggester's proposal; advisory for this editor.
    suggestion: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

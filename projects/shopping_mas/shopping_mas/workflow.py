"""Framework-mandated entry point: `platform_core.runner._invoke_workflow`
does a bare `import workflow`, relying on this exact module name/location.
Bridges this vendor MAS's own sync, file-side-effect-based
`mas_workflow.MASWorkflow.run_task` to the framework's `Task -> AgentOutput`
contract. No asyncio needed anywhere -- `_invoke_workflow` is confirmed
purely synchronous, and the vendor's own `run_task` is already sync (unlike
math_mas, whose vendor code was async and needed an `asyncio.run` wrapper).

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput

This file is excluded from the editor's mutable surface -- it's fixed
framework-integration glue plus the scratch-dir/ground-truth-safety logic
below, not the MAS's own behavior.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from config import MASConfig
from mas_workflow import MASWorkflow
from platform_core.runner import AgentOutput, Task


def _resolve_case() -> tuple[str, str]:
    """(level, sample_id) strings, sourced from the same per-case env vars
    `benchmark/generate_cases.py`'s `env` block populates and
    `adapter/scorer_impl.py` (a copy of the sibling `projects/shopping`
    project's scorer) already relies on -- so workflow and scorer always
    agree on which case is running, without a second identification
    channel to keep in sync."""
    level = os.environ.get("SHOPPING_LEVEL")
    sample_id = os.environ.get("SHOPPING_SAMPLE_ID")
    if not level or not sample_id:
        raise RuntimeError(
            "SHOPPING_LEVEL / SHOPPING_SAMPLE_ID must be set (see "
            "benchmark/generate_cases.py's per-case env block)"
        )
    return level, sample_id


def _prepare_scratch_dir(task: Task, level: str, sample_id: str) -> Path:
    """Copy the read-only source case directory into a fresh, writable
    per-case scratch directory.

    Only `products.jsonl`/`user_info.json` are copied -- `validation_cases.json`
    (ground truth) is deliberately never copied, so it is physically absent
    from anything the MAS's agents/tools can reach, the same non-leak
    guarantee every other project's `_to_*_item` enforces structurally
    rather than by convention. `cart.json` is intentionally NOT copied
    either, even though it exists in the source tree: the benchmark tools
    self-initialize an empty cart when `cart.json` is missing (confirmed:
    every cart tool's `FileNotFoundError` handling in `tools/immutable/`),
    which is a cleaner way to guarantee a fresh cart per run than resetting
    a copied file, and avoids ever propagating stale cart state left over
    from a previous run against the shared, read-only source data.
    """
    source_root = Path(os.environ["SHOPPING_DATABASE_ROOT"])
    source_case_dir = source_root / f"database_level{level}" / f"case_{sample_id}"

    scratch_base = Path(os.environ.get("META_AGENT_SCRATCH_DIR") or tempfile.gettempdir())
    scratch_case_dir = scratch_base / f"shopping_mas_L{level}_S{sample_id}_{task.case_id}"
    if scratch_case_dir.exists():
        shutil.rmtree(scratch_case_dir)
    scratch_case_dir.mkdir(parents=True)

    for fname in ("products.jsonl", "user_info.json"):
        shutil.copy2(source_case_dir / fname, scratch_case_dir / fname)

    return scratch_case_dir


def run_task(task: Task) -> AgentOutput:
    level, sample_id = _resolve_case()
    scratch_case_dir = _prepare_scratch_dir(task, level, sample_id)
    try:
        cfg = MASConfig()
        # MASConfig's own `level` field defaults from MAS_LEVEL at *import*
        # time (a plain dataclass default expression, not a factory) --
        # explicitly override with the real per-case level resolved above
        # so it's always correct regardless of whether MAS_LEVEL happens to
        # be set in the environment.
        cfg.level = int(level)

        result_dict = MASWorkflow(cfg).run_task(
            case_id=task.case_id,
            query=task.description,
            database_path=scratch_case_dir,
        )

        cart_path = scratch_case_dir / "cart.json"
        final_cart: dict[str, Any] = (
            json.loads(cart_path.read_text(encoding="utf-8"))
            if cart_path.exists()
            else {"items": [], "used_coupons": [], "summary": {}}
        )

        return AgentOutput(result=final_cart, metadata=result_dict)
    finally:
        if os.environ.get("SHOPPING_MAS_KEEP_SCRATCH") != "1":
            shutil.rmtree(scratch_case_dir, ignore_errors=True)

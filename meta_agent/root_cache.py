"""Cache of the ROOT (round_000) evaluation across experiments.

A root evaluation re-scores the unmodified seed on the full train set; every run of the
same seed repeats it (for DeepSWE ~6 h at parallelism 16). With ``root_cache_dir`` set on
the HGM manager, a finished root evaluation is stored under a key that pins everything the
result depends on, and a later run with an identical key replays it instead of
re-evaluating:

* every file under ``round_000/task_agent`` (relative path + bytes; ``__pycache__`` skipped),
* the sorted train case ids,
* a config fingerprint set by ``meta_agent.config`` (project, scorer, task-agent LLM spec,
  evaluator config minus parallelism, ``env``, and the benchmark's cases file digest).

Stored per key: ``eval_result.json`` (the evaluator's raw ``EvaluationResult``),
``logs/`` (the root's per-case logs + trace.jsonl, so downstream artifacts -- failure
health, dossiers, the editor's log access -- are identical) and ``manifest.json``.

A replayed root is ONE sample reused -- the same as today's single root evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .models import EvaluationResult

_SKIP_DIRS = {"__pycache__"}


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or set(p.relative_to(root).parts) & _SKIP_DIRS:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def cache_key(agent_dir: Path, case_ids: Optional[list[str]], fingerprint: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    parts = {
        "tree": tree_digest(agent_dir),
        "case_ids": sorted(str(c) for c in (case_ids or [])),
        "fingerprint": fingerprint,
    }
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest(), parts


def load(cache_dir: Path, key: str) -> Optional[tuple[EvaluationResult, Path, dict[str, Any]]]:
    entry = Path(cache_dir) / key
    try:
        result = EvaluationResult.model_validate_json((entry / "eval_result.json").read_text())
        manifest = json.loads((entry / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    if not (entry / "logs").is_dir():
        return None
    return result, entry / "logs", manifest


def store(cache_dir: Path, key: str, parts: dict[str, Any], result: EvaluationResult,
          logs_dir: Path, source: str) -> Optional[Path]:
    """Write atomically (temp dir + rename); never overwrite an existing entry."""
    cache_dir = Path(cache_dir)
    entry = cache_dir / key
    if entry.exists():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".tmp-{key[:12]}-{uuid.uuid4().hex[:8]}"
    try:
        tmp.mkdir()
        shutil.copytree(logs_dir, tmp / "logs", symlinks=True)
        (tmp / "eval_result.json").write_text(result.model_dump_json())
        (tmp / "manifest.json").write_text(json.dumps({
            "key": key,
            "source": source,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_cases": len(result.per_case),
            "score": result.score,
            "tree": parts["tree"],
            "case_ids": parts["case_ids"],
            "fingerprint": parts["fingerprint"],
        }, indent=1, default=str))
        os.rename(tmp, entry)
        return entry
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        return None

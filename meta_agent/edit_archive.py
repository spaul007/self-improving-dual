"""Deterministic retrieval over the run's edit archive.

Resolves a stage-1 memory query — explicit node ids, category ids, keywords —
into rendered slices of the selected nodes' ``edit_memory.md`` (prose record,
outcome, analysis) and ``edit_code.md`` (implementation), under a character
budget. Pure reads plus one audit-manifest write; no LLM anywhere, so the same
query against the same archive always yields byte-identical output.

Selection order is part of the contract (explicit > category > keyword):
the proposer's own citations outrank fuzzy matches, and within a source the
order is stable so retries retrieve the same context.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .edit_code import CODE_NAME
from .edit_diff import truncate_middle
from .edit_memory import REGISTRY_NAME
from .edit_memory_render import _load_records

RETRIEVAL_MANIFEST = "retrieval_manifest.json"
DEFAULT_CHAR_BUDGET = 60000
DEFAULT_MAX_NODES = 8

_DIFF_HEADER = "## Diff vs parent"


@dataclass
class RetrievalResult:
    blocks: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_registry(experiment_dir: Path) -> dict[str, Any]:
    path = Path(experiment_dir) / REGISTRY_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _category_nodes(registry: Mapping[str, Any], axis: str, cid: str) -> list[int]:
    entry = (registry.get(axis) or {}).get(cid) or {}
    seen: list[int] = []
    for r in entry.get("edits", []):
        n = r.get("node")
        if isinstance(n, int) and n not in seen:
            seen.append(n)
    return seen


def _code_slice(round_dir: Path) -> str:
    """``edit_code.md`` with the diff section moved last (the sub-edit map and
    final-state defs are the denser signal, so they survive truncation)."""
    path = Path(round_dir) / CODE_NAME
    try:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    except (OSError, UnicodeDecodeError):
        return ""
    if not text:
        return ""
    idx = text.find(_DIFF_HEADER)
    if idx == -1:
        return text
    end = text.find("\n## ", idx + 1)
    if end == -1:
        return text
    diff_section = text[idx:end]
    return text[:idx] + text[end + 1:] + "\n" + diff_section


def resolve_query(
    experiment_dir: Path,
    query: Mapping[str, Any],
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    max_nodes: int = DEFAULT_MAX_NODES,
    include_code: bool = True,
) -> RetrievalResult:
    """Resolve a memory query into per-node text slices + an audit manifest.

    Never raises: an unreadable archive yields empty blocks with the manifest
    explaining what was asked for.
    """
    experiment_dir = Path(experiment_dir)
    query = dict(query or {})
    include_code = bool(query.get("include_code", include_code))
    records = _load_records(experiment_dir)
    registry = _load_registry(experiment_dir)

    selected: list[tuple[int, str]] = []  # (node, why), dedup on node
    picked: set[int] = set()
    dropped: list[dict[str, Any]] = []

    def _take(node: Any, why: str) -> None:
        if not isinstance(node, int) or node in picked:
            return
        if node not in records:
            dropped.append({"node": node, "why": why, "reason": "no record"})
            return
        picked.add(node)
        selected.append((node, why))

    for n in query.get("nodes") or []:
        try:
            _take(int(n), "explicit")
        except (TypeError, ValueError):
            dropped.append({"node": n, "why": "explicit", "reason": "bad id"})
    for axis, key in (("strategies", "strategy"), ("areas", "area")):
        for cid in query.get(axis) or []:
            for n in _category_nodes(registry, axis, str(cid)):
                _take(n, f"{key}:{cid}")
    keywords = [str(k).lower() for k in (query.get("keywords") or []) if k]
    if keywords:
        scored: list[tuple[int, int]] = []
        for n, rec in records.items():
            if n in picked:
                continue
            body = (rec.get("text") or rec.get("body") or "").lower()
            hits = sum(body.count(kw) for kw in keywords)
            if hits:
                scored.append((hits, n))
        scored.sort(key=lambda t: (-t[0], -t[1]))  # matches desc, newest first
        for hits, n in scored:
            _take(n, f"keyword({hits} hits)")

    overflow = selected[max_nodes:]
    selected = selected[:max_nodes]
    for n, why in overflow:
        dropped.append({"node": n, "why": why, "reason": "over max_nodes"})

    per_node = max(1, char_budget // max(1, len(selected))) if selected else 0
    blocks: list[str] = []
    rows: list[dict[str, Any]] = []
    total = 0
    for n, why in selected:
        rec = records[n]
        piece = rec.get("text") or rec.get("body") or ""
        if include_code:
            code = _code_slice(experiment_dir / f"round_{n:03d}")
            if code:
                piece += "\n\n" + code
        truncated = len(piece) > per_node
        piece = truncate_middle(piece, per_node)
        blocks.append(f"### Retrieved node {n} ({why})\n{piece.rstrip()}")
        rows.append({"node": n, "why": why, "chars": len(piece),
                     "truncated": truncated})
        total += len(piece)

    manifest = {
        "version": 1,
        "query": {k: query.get(k) for k in
                  ("nodes", "strategies", "areas", "keywords", "include_code")},
        "selected": rows,
        "char_budget": char_budget,
        "total_chars": total,
        "dropped": dropped,
    }
    return RetrievalResult(blocks=blocks, manifest=manifest)


def render_retrieved(result: RetrievalResult) -> str:
    if not result.blocks:
        return "(nothing retrieved — the query matched no recorded node)"
    return "\n\n".join(result.blocks)


def write_manifest(round_dir: Path, result: RetrievalResult) -> None:
    """Best-effort audit write; a failure never blocks the edit."""
    try:
        _atomic_write(Path(round_dir) / RETRIEVAL_MANIFEST,
                      json.dumps(result.manifest, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[edit_archive] manifest write failed: {exc!r}", flush=True)

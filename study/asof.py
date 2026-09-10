"""Rebuild the run directory as it stood the moment BEFORE one expand.

Every study that replays an expand must not let the editor (or the retriever)
see nodes that did not exist yet. `snapshots/tree_snapshots.jsonl` records the
tree at each search event, so the entry immediately preceding the child's first
appearance is exactly the world the expand was issued against.

The result is a directory of SYMLINKS to the real round dirs plus two rewritten
files, so it costs nothing and the production code paths
(`edit_memory_render._load_records`, `edit_archive.resolve_query`) run unmodified
against it.

What is faithful:
  * node membership — only rounds present pre-expand
  * the strategy/area registry — node lists filtered to those rounds
  * the belief document — restored from `edit_memory_beliefs_archive/` at the
    version recorded in the child's `belief_prediction.json`

What is NOT faithful (state the limitation wherever this is used): a node's
`edit_memory.md` carries the Outcome/Analysis text as most recently refreshed,
not as it read at this moment. Records are refreshed in place and no history is
kept, so an as-of record is slightly better informed than the real one was.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

SNAP = "snapshots/tree_snapshots.jsonl"
REGISTRY = "edit_memory_registry.json"
BELIEF_DOC = "edit_memory_beliefs.md"
BELIEF_ARCHIVE = "edit_memory_beliefs_archive"


def nodes_before(run: Path, child: int) -> Optional[set[int]]:
    """Node ids present in the snapshot just before `child` first appears."""
    path = run / SNAP
    if not path.exists():
        return None
    prev: Optional[set[int]] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            snap = json.loads(line)
        except json.JSONDecodeError:
            continue
        ids = {n["node_id"] for n in snap.get("nodes") or []}
        if child in ids:
            return prev
        prev = ids
    return prev


def belief_version_for(run: Path, child: int) -> Optional[int]:
    p = run / f"round_{child:03d}" / "belief_prediction.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("belief_version")
    except (OSError, json.JSONDecodeError):
        return None


def belief_doc_at(run: Path, version: Optional[int]) -> Optional[Path]:
    """The archived belief document at `version`.

    The archive is written on every update, so file NNNN is the document that
    was current when the store's version was NNNN.
    """
    if version is None:
        return None
    cand = run / BELIEF_ARCHIVE / f"beliefs_{int(version):04d}.md"
    return cand if cand.exists() else None


def _filter_registry(registry: dict[str, Any], keep: set[int]) -> dict[str, Any]:
    """Drop nodes that did not exist yet from every category's node list."""
    out = json.loads(json.dumps(registry))  # deep copy
    for axis in ("strategies", "areas"):
        cats = out.get(axis) or {}
        for cid in list(cats):
            entry = cats[cid] or {}
            edits = [e for e in (entry.get("edits") or [])
                     if e.get("node") in keep]
            if not edits:
                cats.pop(cid)               # category nothing had used yet
                continue
            entry["edits"] = edits
            entry["n_nodes"] = len({e["node"] for e in edits})
            firsts = sorted({e["node"] for e in edits})
            entry["first_node"] = firsts[0]
            cats[cid] = entry
    return out


def build(run: Path, child: int, dest: Path, *,
          overwrite: bool = True) -> dict[str, Any]:
    """Materialise the as-of view for the expand that produced `child`."""
    run = Path(run).resolve()
    dest = Path(dest)
    keep = nodes_before(run, child)
    if keep is None:
        raise SystemExit(f"no pre-expand snapshot for child {child} in {run}")
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    linked = []
    for n in sorted(keep):
        src = run / f"round_{n:03d}"
        if not src.is_dir():
            continue
        (dest / f"round_{n:03d}").symlink_to(src, target_is_directory=True)
        linked.append(n)

    reg_src = run / REGISTRY
    if reg_src.exists():
        reg = json.loads(reg_src.read_text())
        (dest / REGISTRY).write_text(
            json.dumps(_filter_registry(reg, set(linked)), indent=2) + "\n",
            encoding="utf-8")

    bver = belief_version_for(run, child)
    doc = belief_doc_at(run, bver)
    if doc is not None:
        shutil.copyfile(doc, dest / BELIEF_DOC)

    # anything else the production readers may touch, linked as-is
    for name in ("config.snapshot.yaml", "edit_memory_candidates.json",
                 "belief_instruction.md"):
        src = run / name
        if src.exists():
            (dest / name).symlink_to(src)

    meta = {"source_run": str(run), "child": child,
            "nodes_present": linked, "belief_version": bver,
            "belief_doc": str(doc) if doc else None}
    (dest / "_ASOF.json").write_text(json.dumps(meta, indent=2) + "\n",
                                     encoding="utf-8")
    return meta


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--child", required=True, type=int)
    ap.add_argument("--dest", required=True, type=Path)
    args = ap.parse_args()
    meta = build(args.run, args.child, args.dest)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

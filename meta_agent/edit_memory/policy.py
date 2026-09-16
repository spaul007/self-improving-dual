"""Read/write surface of the edit-memory curators.

A curator is an agentic session like the editor's, but its job is to read a
set of finished nodes and write one document. Its ``PathPolicy`` therefore
has:

* ``named_roots`` — ``$WORK_DIR`` (its own output dir, the only writable
  path), ``$MEMORY_DIR`` (the run's ``edit_memory/``: previous memory,
  instruction, earlier curations — read-only), ``$NODE_1..$NODE_m`` and
  ``$PARENT_1..$PARENT_m`` (the nodes under review and their parents, so a
  diff is one ``diff -u`` away), ``$REPO_DIR``;
* the same deny list as the editor (benchmark, data, categorizer,
  meta_agent, tests): the curator can no more see ground truth than the
  editor can.

The sandbox binds exactly ``read_roots`` and ``write_dirs``, so bash is
confined identically to the ``editor`` tool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..agentic.policy import PathPolicy, _real

WORK_ROOT_VAR = "WORK_DIR"
MEMORY_ROOT_VAR = "MEMORY_DIR"
REPO_ROOT_VAR = "REPO_DIR"


def build_curator_policy(
    *,
    workspace: Path,
    memory_dir: Path,
    node_dirs: Sequence[Path],
    parent_dirs: Sequence[Optional[Path]],
    repo_root: Path,
    project_root: Optional[Path] = None,
    project_name: str = "",
) -> PathPolicy:
    """``node_dirs[i]`` becomes ``$NODE_{i+1}`` and ``parent_dirs[i]``
    ``$PARENT_{i+1}`` (a ``None`` parent is tolerated and omitted). The
    workspace is created; nothing else is touched."""
    workspace = _real(Path(workspace))
    memory_dir = _real(Path(memory_dir))
    repo_root = _real(Path(repo_root))
    project_root = _real(Path(project_root)) if project_root else None
    if project_root is not None and not project_name:
        project_name = project_root.name
    workspace.mkdir(parents=True, exist_ok=True)

    named: list[tuple[str, Path]] = [(WORK_ROOT_VAR, workspace), (MEMORY_ROOT_VAR, memory_dir)]
    read: list[Path] = [workspace, memory_dir]
    # Distinct parents only once as a root, but every $PARENT_i var is defined.
    for i, (nd, pd) in enumerate(zip(node_dirs, parent_dirs), start=1):
        nd = _real(Path(nd))
        named.append((f"NODE_{i}", nd))
        read.append(nd)
        if pd is not None:
            pd = _real(Path(pd))
            named.append((f"PARENT_{i}", pd))
            read.append(pd)
    named.append((REPO_ROOT_VAR, repo_root))
    read.append(repo_root / "platform_core")
    read.append(repo_root / "projects" / "__init__.py")
    if project_root is not None:
        read += [project_root / "__init__.py", project_root / "tools",
                 project_root / "db_schema.md"]
    read_roots = tuple(dict.fromkeys(p for p in read if p.exists()))

    deny: list[Path] = [repo_root / "meta_agent", repo_root / "tests"]
    if project_root is not None:
        deny += [project_root / "benchmark", project_root / "data"]
        deny += sorted(project_root.glob("*_error_categorizer.py"))
    return PathPolicy(
        out_dir=workspace,
        base_dir=workspace,
        task_agent=workspace,          # bash cwd + relative-path base
        scratch=workspace,             # everything under it is writable
        write_files=(),
        write_dirs=(workspace,),
        read_roots=read_roots,
        deny_roots=tuple(deny),
        run_root=memory_dir.parent,
        repo_root=repo_root,
        project_root=project_root,
        project_name=project_name,
        named_roots=tuple(named),
    )

"""What the agentic editor's coding agent may read and write.

One absolute-path allow-list, used two ways: the ``editor`` tool checks every
path against it in-process, and the sandbox turns it into bubblewrap binds
so ``bash`` sees exactly the same surface. Both layers see the same
``PathPolicy``, so they can never disagree.

Writable: the task agent's mutable surface in the round being produced
(``MUTABLE_FILES`` + ``mutable_tools/*.py``, the same contract the validators
enforce) plus a scratch directory for throwaway scripts.

Readable: the whole run directory (every node's code, evaluation evidence,
edit memory record, diff, and the run-level registry / belief files — all of
which the framework writes dynamically, so the agent reads them straight from
disk), ``platform_core/`` and the project's tool implementations + database
schema (needed to ``import workflow`` and to understand the tools). The
project's ``benchmark/`` (scorer, cases) and ``data/`` are never readable —
they are not roots, and a deny-list guards against a misconfigured root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..editor_validators import MUTABLE_DIRS, MUTABLE_FILES

# The file the framework writes at the run root (see main_loop.py); walking
# up from any round / variant dir to it locates the run.
RUN_ROOT_MARKER = "config.snapshot.yaml"
# Repo root: meta_agent/agentic/policy.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_SUBDIR = ("agentic", "scratch")
LIST_SKIP_NAMES = {"__pycache__"}
# Run-root files the instruction prompt explains to the agent (only the ones
# present are listed). Name -> one-line meaning.
RUN_ROOT_LEGEND = (
    ("edit_memory_registry.json", "category registry across all edits (deterministic)"),
    ("edit_memory_candidates.json", "proxy categories from the setup pass (tagger-only)"),
    ("edit_memory_beliefs.md", "current belief document about which edit strategies work"),
    ("edit_memory_beliefs_state.json", "machine bookkeeping for the beliefs"),
    ("edit_memory_beliefs_archive", "earlier belief documents"),
    ("beliefs_archive", "earlier belief documents"),
    ("tree_snapshots.jsonl", "search-tree history, one snapshot per line"),
)
ROUND_LEGEND = (
    ("hgm_node.json", "tree stats: parent_id, mean_utility, n_evals, cmp"),
    ("strategy.json", "the edit summary that produced this node"),
    ("feedback.json", "evaluation digest incl. the failure report"),
    ("eval_result.json", "per-case scores and details"),
    ("logs/case_<id>.json", "one CaseResult per evaluated case"),
    ("logs/trace.jsonl", "llm/tool trace of the evaluation"),
    ("edit_memory.md", "memory record of this node's edit"),
    ("edit_code.md", "the diff vs its parent + changed defs"),
    ("edit_prediction.json", "the belief prediction made for this edit"),
    ("task_agent/", "this node's code"),
)


@dataclass(frozen=True)
class PathPolicy:
    out_dir: Path
    base_dir: Path
    task_agent: Path
    scratch: Path
    write_files: tuple[Path, ...]
    write_dirs: tuple[Path, ...]
    read_roots: tuple[Path, ...]
    deny_roots: tuple[Path, ...]
    run_root: Optional[Path]
    repo_root: Path
    project_root: Optional[Path]
    project_name: str

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def can_write(self, path: Path) -> bool:
        """``path`` must already be realpath'd (see :func:`resolve`) — so a
        symlink planted inside a writable dir that points elsewhere has
        already been followed and is judged by where it lands. For a file
        that does not exist yet, realpath resolves the existing prefix, so
        the same check covers ``create``."""
        if path in self.write_files:
            return True
        return self._under_write_dir(path)

    def can_read(self, path: Path) -> bool:
        if _is_denied(path, self.deny_roots):
            return False
        if self.can_write(path):
            return True
        return any(_within(path, root) for root in self.read_roots)

    def _under_write_dir(self, path: Path) -> bool:
        for d in self.write_dirs:
            if not _within(path, d) or path == d:
                continue
            if d == self.scratch or _within(d, self.scratch):
                return True
            # Mutable tool dir: direct child, *.py only (the same rule as
            # AgentEditor._is_path_allowed and the validators).
            if path.parent == d and path.suffix == ".py":
                return True
        return False

    # ------------------------------------------------------------------ #
    # Rendering for the instruction prompt
    # ------------------------------------------------------------------ #

    def describe(self, *, listing_depth: int = 2) -> str:
        lines = [
            "## Workspace (absolute paths; nothing outside these exists for "
            "bash or the editor tool)",
            f"Agent under edit — bash cwd: {self.task_agent}",
            "  WRITABLE (edit with the editor tool):",
        ]
        for f in self.write_files:
            lines.append(f"    {f}")
        for d in self.write_dirs:
            if d == self.scratch:
                continue
            lines.append(f"    {d}/      (new *.py files allowed)")
        lines.append(f"  Scratch (writable, ignored by validators): {self.scratch}/")
        lines.append("READ-ONLY reference:")
        lines.append(
            f"  parent node (the agent you are improving + its evidence): {self.base_dir}/"
        )
        present = [(n, m) for n, m in ROUND_LEGEND
                   if _round_entry_exists(self.base_dir, n)]
        for name, meaning in present:
            lines.append(f"     {name:<22} {meaning}")
        if self.run_root is not None:
            lines.append(
                f"  run directory (every node round_NNN/ has the same layout as "
                f"the parent above; 'node N' in any memory/belief citation is "
                f"round_NNN/ zero-padded, e.g. node 17 -> round_017/): {self.run_root}/"
            )
            for name, meaning in RUN_ROOT_LEGEND:
                if (self.run_root / name).exists():
                    lines.append(f"     {name:<32} {meaning}")
        lines.append(
            f"  platform (call_llm, runner, trace, tools registry): "
            f"{self.repo_root / 'platform_core'}/"
        )
        if self.project_root is not None:
            tools_dir = self.project_root / "tools"
            if tools_dir.exists():
                lines.append(
                    f"  immutable tools reached via call_tool: {tools_dir}/"
                )
            schema = self.project_root / "db_schema.md"
            if schema.exists():
                lines.append(f"  database schema the tools query against: {schema}")
        lines.append(
            "  NOT available anywhere: the benchmark's cases, scoring code and "
            "the database itself; the model API; the network."
        )
        lines.append("")
        lines.append(f"Listing of {self.task_agent} ({listing_depth} levels):")
        lines.append(self.list_dir(self.task_agent, depth=listing_depth, indent="  "))
        return "\n".join(lines) + "\n"

    def list_dir(self, directory: Path, *, depth: int = 2, indent: str = "") -> str:
        """Policy-filtered listing, ``depth`` levels deep, hidden and
        ``__pycache__`` entries excluded."""
        out: list[str] = []

        def walk(d: Path, level: int, pad: str) -> None:
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except OSError as exc:
                out.append(f"{pad}(error listing {d}: {exc})")
                return
            for p in entries:
                if p.name.startswith(".") or p.name in LIST_SKIP_NAMES:
                    continue
                rp = _real(p)
                if not self.can_read(rp) and not (p.is_dir() and self._has_readable_below(rp)):
                    continue
                if p.is_dir():
                    out.append(f"{pad}{p.name}/")
                    if level < depth:
                        walk(p, level + 1, pad + "  ")
                else:
                    out.append(f"{pad}{p.name}")

        walk(directory, 1, indent)
        return "\n".join(out) if out else f"{indent}(empty)"

    def _has_readable_below(self, directory: Path) -> bool:
        return any(_within(root, directory) for root in self.read_roots + self.write_dirs)


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #

def build_policy(
    *,
    out_dir: Path,
    base_dir: Path,
    repo_root: Path = REPO_ROOT,
    project_root: Optional[Path] = None,
    project_name: str = "",
) -> PathPolicy:
    out_dir = _real(Path(out_dir))
    base_dir = _real(Path(base_dir))
    repo_root = _real(Path(repo_root))
    project_root = _real(Path(project_root)) if project_root else None
    if project_root is not None and not project_name:
        project_name = project_root.name

    task_agent = out_dir / "task_agent"
    scratch = out_dir.joinpath(*SCRATCH_SUBDIR)
    write_files = tuple(task_agent / f for f in sorted(MUTABLE_FILES))
    write_dirs = tuple(task_agent / d for d in sorted(MUTABLE_DIRS)) + (scratch,)

    run_root = find_run_root(out_dir)
    read_candidates: list[Path] = [task_agent]
    if run_root is not None:
        read_candidates.append(run_root)
    else:
        # No run marker (tests, ad-hoc dirs): expose the two round dirs only.
        read_candidates += [base_dir, out_dir]
    read_candidates.append(repo_root / "platform_core")
    read_candidates.append(repo_root / "projects" / "__init__.py")
    if project_root is not None:
        read_candidates += [
            project_root / "__init__.py",
            project_root / "tools",
            project_root / "db_schema.md",
        ]
    read_roots = tuple(dict.fromkeys(p for p in read_candidates if p.exists()))

    deny: list[Path] = [repo_root / "meta_agent", repo_root / "tests"]
    if project_root is not None:
        deny += [project_root / "benchmark", project_root / "data"]
        deny += sorted(project_root.glob("*_error_categorizer.py"))
    return PathPolicy(
        out_dir=out_dir,
        base_dir=base_dir,
        task_agent=task_agent,
        scratch=scratch,
        write_files=write_files,
        write_dirs=write_dirs,
        read_roots=read_roots,
        deny_roots=tuple(deny),
        run_root=run_root,
        repo_root=repo_root,
        project_root=project_root,
        project_name=project_name,
    )


def find_run_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` to the directory holding ``config.snapshot.yaml``
    (the run root). ``None`` when there is none (e.g. unit-test temp dirs)."""
    p = _real(Path(start))
    for candidate in (p, *p.parents):
        if (candidate / RUN_ROOT_MARKER).exists():
            return candidate
    return None


def resolve(path_str: str) -> Path:
    """Turn a tool-supplied path into the realpath the policy is keyed on.
    Raises ``ValueError`` with the message the model should see."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Error: path is required")
    p = Path(path_str.strip())
    if not p.is_absolute():
        raise ValueError(f"Error: path must be absolute (got {path_str!r})")
    return _real(p)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _real(p: Path) -> Path:
    return Path(os.path.realpath(p))


def _within(path: Path, root: Path) -> bool:
    """``path`` is ``root`` itself or somewhere beneath it."""
    return path == root or root in path.parents


def _is_denied(path: Path, deny_roots: tuple[Path, ...]) -> bool:
    if path.name == "cases.jsonl":
        return True
    return any(_within(path, d) for d in deny_roots)


def _round_entry_exists(round_dir: Path, name: str) -> bool:
    if "<" in name:  # glob-ish legend entries like logs/case_<id>.json
        prefix = name.split("<", 1)[0]
        parent = round_dir / Path(prefix).parent
        return parent.exists() and any(
            child.name.startswith(Path(prefix).name) for child in parent.iterdir()
        )
    return (round_dir / name.rstrip("/")).exists()

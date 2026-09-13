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
import re
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
# Root variables the instruction prompt and the sandbox share: the prompt
# names paths as ``$VAR/...``, every bash call has them exported, and the
# editor tool expands them. Keeps the prompt free of experiment paths.
ROOT_VARS = ("RUN_DIR", "NODE_DIR", "PARENT_DIR", "REPO_DIR")
# Run-level edit-memory files the prompt explains (only listed when present).
MEMORY_LEGEND = (
    ("edit_memory_beliefs.md",
     "belief document — which edit strategies have worked, which have not and why, with node citations"),
    ("edit_memory_registry.json", "category registry of every edit made so far"),
)
MEMORY_MARKERS = ("edit_memory_registry.json", "edit_memory_candidates.json",
                  "edit_memory_beliefs.md")
BELIEFS_FILE = "edit_memory_beliefs.md"
ROUND_LEGEND = (
    ("hgm_node.json", "tree stats: parent_id, mean_utility, n_evals"),
    ("strategy.json", "the edit summary that produced this node"),
    ("feedback.json", "evaluation digest incl. the failure report"),
    ("eval_result.json", "per-case scores and details"),
    ("logs/case_<id>.json", "one result per evaluated case"),
    ("logs/trace.jsonl", "llm/tool trace of the evaluation"),
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
    # Roots
    # ------------------------------------------------------------------ #

    def roots(self) -> dict[str, Path]:
        """``RUN_DIR`` / ``NODE_DIR`` / ``PARENT_DIR`` / ``REPO_DIR`` — the
        only absolute paths the agent ever needs; exported as env vars in
        every bash call and expanded by the editor tool."""
        return {
            "RUN_DIR": self.run_root if self.run_root is not None else self.out_dir.parent,
            "NODE_DIR": self.out_dir,
            "PARENT_DIR": self.base_dir,
            "REPO_DIR": self.repo_root,
        }

    def var_path(self, path: Path) -> str:
        """Render ``path`` as ``$VAR/...`` using the most specific root
        (NODE_DIR before RUN_DIR, since it lies inside it)."""
        path = Path(path)
        for var in ("NODE_DIR", "PARENT_DIR", "RUN_DIR", "REPO_DIR"):
            root = self.roots()[var]
            if path == root:
                return f"${var}"
            if root in path.parents:
                return f"${var}/{path.relative_to(root)}"
        return str(path)

    def resolve(self, path_str: str) -> Path:
        """Turn a tool-supplied path into the realpath the policy is keyed
        on: ``$VAR/...`` and ``${VAR}/...`` expand from :meth:`roots`,
        absolute paths pass through, relative paths resolve against the
        task_agent directory (the bash cwd). Raises ``ValueError`` with the
        message the model should see."""
        if not isinstance(path_str, str) or not path_str.strip():
            raise ValueError("Error: path is required")
        text = path_str.strip()
        m = re.match(r"^\$\{?([A-Z_]+)\}?(?:/|$)(.*)$", text)
        if m:
            var, rest = m.group(1), m.group(2)
            if var not in self.roots():
                raise ValueError(
                    f"Error: unknown root ${var}; known roots: "
                    + ", ".join(f"${v}" for v in ROOT_VARS)
                )
            p = self.roots()[var] / rest if rest else self.roots()[var]
        else:
            p = Path(text)
            if not p.is_absolute():
                p = self.task_agent / p
        return _real(p)

    def memory_enabled(self) -> bool:
        """Edit memory is on for this run iff the framework has written any
        of its run-level files (the setup pass writes the registry and
        candidates before the first expansion)."""
        run = self.roots()["RUN_DIR"]
        return any((run / name).exists() for name in MEMORY_MARKERS)

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

    def describe(self, *, listing_depth: int = 2, memory: Optional[bool] = None) -> str:
        """The workspace map for the instruction prompt: roots as ``$VAR``,
        writable files, the parent's evidence legend, the run-level memory
        files (only when ``memory`` — default: :meth:`memory_enabled`), the
        platform/project reference, and a listing of the agent."""
        if memory is None:
            memory = self.memory_enabled()
        run = self.roots()["RUN_DIR"]

        def under_run(p: Path) -> str:
            # The roots table defines NODE_DIR / PARENT_DIR in terms of RUN_DIR.
            return (f"$RUN_DIR/{p.relative_to(run)}" if run in p.parents
                    else str(p))

        node_rel = under_run(self.out_dir)
        parent_rel = under_run(self.base_dir)
        lines = [
            "## Roots (environment variables in every bash call; the editor "
            "tool accepts the same $VAR form)",
            "  RUN_DIR     the run directory (runs/<experiment>/) — one "
            "round_NNN/ per node; 'node N' means round_NNN/ (zero-padded)",
            f"  NODE_DIR    {node_rel}/   this node",
            f"  PARENT_DIR  {parent_rel}/   its parent",
            "  REPO_DIR    the repository root",
            "",
            "## Workspace (nothing outside these exists for bash or the "
            "editor tool)",
            "Agent under edit — bash cwd: $NODE_DIR/task_agent/",
            "  WRITABLE (edit with the editor tool):",
        ]
        order = ("workflow.py", "tool_wrapper.py", "tools_schema.json")
        for name in order:
            f = self.task_agent / name
            if f in self.write_files:
                lines.append(f"    {self.var_path(f)}")
        for f in self.write_files:
            if f.name not in order:
                lines.append(f"    {self.var_path(f)}")
        for d in self.write_dirs:
            if d == self.scratch:
                continue
            lines.append(f"    {self.var_path(d)}/      (new *.py files allowed)")
        lines.append(
            "  Scratch for your own throwaway scripts (not part of the agent, "
            f"not validated): {self.var_path(self.scratch)}/"
        )
        lines.append("READ-ONLY reference:")
        lines.append(
            "  parent node $PARENT_DIR/ — the agent you are improving and its "
            "evaluation evidence:"
        )
        for name, meaning in ROUND_LEGEND:
            if _round_entry_exists(self.base_dir, name):
                lines.append(f"     {name:<22} {meaning}")
        lines.append("  every other node $RUN_DIR/round_NNN/ has the same layout.")
        if memory:
            lines.append("  accumulated understanding of previous edits (run-level):")
            for name, meaning in MEMORY_LEGEND:
                if (run / name).exists():
                    lines.append(f"     $RUN_DIR/{name:<26} {meaning}")
            lines.append(
                "     $RUN_DIR/round_NNN/edit_memory.md    memory record of "
                "that node's edit; edit_code.md next to it is its diff"
            )
        lines.append(
            "  platform (call_llm, runner, trace, tools registry): "
            "$REPO_DIR/platform_core/"
        )
        if self.project_root is not None:
            tools_dir = self.project_root / "tools"
            if tools_dir.exists():
                lines.append(
                    f"  immutable tools reached via call_tool: {self.var_path(tools_dir)}/"
                )
            schema = self.project_root / "db_schema.md"
            if schema.exists():
                lines.append(
                    f"  database schema the tools query against: {self.var_path(schema)}"
                )
        lines.append(
            "  NOT available anywhere: the benchmark's cases, scoring code and "
            "the database itself; the model API; the network."
        )
        lines.append("")
        lines.append(f"Listing of $NODE_DIR/task_agent ({listing_depth} levels):")
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
    """Absolute-path realpath (no roots, no relative resolution). Kept for
    callers that have no policy at hand; tools use ``PathPolicy.resolve``."""
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

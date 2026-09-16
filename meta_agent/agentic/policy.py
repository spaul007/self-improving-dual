"""What the agentic editor's coding agent may read and write.

One absolute-path allow-list, used two ways: the ``editor`` tool checks every
path against it in-process, and the sandbox turns it into bubblewrap binds
so ``bash`` sees exactly the same surface. Both layers see the same
``PathPolicy``, so they can never disagree.

Writable: the task agent's mutable surface in the round being produced
(``MUTABLE_FILES`` + ``mutable_tools/*.py``, the same contract the validators
enforce) plus a scratch directory for throwaway scripts.

Readable — decided by ``read_scope``:
  * ``"run"``: the whole run directory (every node's code and evaluation
    evidence, which the framework writes dynamically, so the agent reads
    them straight from disk);
  * ``"parent"``: only the parent node's round dir and this node's own
    round dir — no ``$RUN_DIR`` root exists, so sibling / ancestor nodes are
    neither named in the prompt nor bound into the sandbox.
plus, in both scopes, ``platform_core/`` and the project's tool
implementations + database schema (needed to ``import workflow`` and to
understand the tools). The project's ``benchmark/`` (scorer, cases) and
``data/`` are never readable — they are not roots, and a deny-list guards
against a misconfigured root.

Edit memory: the run-level ``edit_memory/`` directory is always denied to
the editor (both arms of the with/without-memory bandit must not browse it);
an expansion on the with-memory arm gets exactly one file back through
``read_files`` — checked before the deny list — exposed as
``$EDIT_MEMORY_FILE``. The curators of the edit-memory layer use the same
``PathPolicy`` with ``named_roots`` (their own ``$WORK_DIR`` / ``$NODE_i``
roots) instead of the editor's four.

The bash side of the policy is enforced by the bubblewrap binds built from
``read_roots`` (deny roots inside a read root are masked with a tmpfs, then
``read_files`` are bound back); when the sandbox falls back to unconfined
mode only the ``editor`` tool enforces it.
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
# ``RUN_DIR`` exists only under read_scope "run".
ROOT_VARS = ("RUN_DIR", "NODE_DIR", "PARENT_DIR", "REPO_DIR")
# What the meta-agent may read beyond its own node (see module docstring).
READ_SCOPE_RUN = "run"
READ_SCOPE_PARENT = "parent"
READ_SCOPES = (READ_SCOPE_RUN, READ_SCOPE_PARENT)
# The edit-memory layer's directory under the run root (see
# meta_agent/edit_memory); always denied to the editor.
MEMORY_DIR_NAME = "edit_memory"
# Root var + prompt row for the edit-memory file on the with-memory arm.
MEMORY_ROOT_VAR = "EDIT_MEMORY_FILE"
MEMORY_LEGEND_LINE = (
    "  accumulated edit memory of previous edits in this run (ranked edits, "
    "what worked, shortcomings): $EDIT_MEMORY_FILE"
)
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
    read_scope: str = READ_SCOPE_RUN
    # Single files readable even inside a denied directory (checked before
    # ``deny_roots``); the sandbox binds them back after masking the deny.
    read_files: tuple[Path, ...] = ()
    # The with-memory arm's memory file; rendered as $EDIT_MEMORY_FILE.
    memory_file: Optional[Path] = None
    # Curator policies: when non-empty these ARE the roots (name, path), in
    # this order, and the editor's four are not used.
    named_roots: tuple[tuple[str, Path], ...] = ()

    # ------------------------------------------------------------------ #
    # Roots
    # ------------------------------------------------------------------ #

    def roots(self) -> dict[str, Path]:
        """``RUN_DIR`` / ``NODE_DIR`` / ``PARENT_DIR`` / ``REPO_DIR`` — the
        only absolute paths the agent ever needs; exported as env vars in
        every bash call and expanded by the editor tool. ``RUN_DIR`` is
        absent under read_scope "parent"; ``EDIT_MEMORY_FILE`` present only
        with a ``memory_file``. A curator policy returns its ``named_roots``
        instead."""
        if self.named_roots:
            return {name: path for name, path in self.named_roots}
        out: dict[str, Path] = {}
        if self.read_scope == READ_SCOPE_RUN:
            out["RUN_DIR"] = (self.run_root if self.run_root is not None
                              else self.out_dir.parent)
        out["NODE_DIR"] = self.out_dir
        out["PARENT_DIR"] = self.base_dir
        out["REPO_DIR"] = self.repo_root
        if self.memory_file is not None:
            out[MEMORY_ROOT_VAR] = self.memory_file
        return out

    def root_vars(self) -> tuple[str, ...]:
        """The ``$VAR`` names this policy defines, in prompt order."""
        return tuple(self.roots())

    def var_path(self, path: Path) -> str:
        """Render ``path`` as ``$VAR/...`` using the most specific root
        (NODE_DIR before RUN_DIR, since it lies inside it)."""
        path = Path(path)
        roots = self.roots()
        # Most specific first for the editor's roots; curator roots in order.
        order = [v for v in ("NODE_DIR", "PARENT_DIR", "RUN_DIR", "REPO_DIR") if v in roots]
        order += [v for v in roots if v not in order]
        for var in order:                       # an exact root (e.g. a file root) wins
            if path == roots[var]:
                return f"${var}"
        for var in order:
            root = roots[var]
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
        m = re.match(r"^\$\{?([A-Z_][A-Z0-9_]*)\}?(?:/|$)(.*)$", text)
        if m:
            var, rest = m.group(1), m.group(2)
            if var not in self.roots():
                raise ValueError(
                    f"Error: unknown root ${var}; known roots: "
                    + ", ".join(f"${v}" for v in self.root_vars())
                )
            p = self.roots()[var] / rest if rest else self.roots()[var]
        else:
            p = Path(text)
            if not p.is_absolute():
                p = self.task_agent / p
        return _real(p)

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
        if path in self.read_files:
            return True
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
        """The workspace map for the instruction prompt: roots as ``$VAR``,
        writable files, the parent's evidence legend, the platform/project
        reference, and a listing of the agent. Under read_scope "parent"
        there is no ``RUN_DIR`` row and no mention of other nodes. With a
        ``memory_file`` the roots table and the reference gain one row each.
        Editor policies only (curators render their own map)."""
        if self.named_roots:
            raise ValueError("describe() renders the editor's map; curator "
                             "policies use their own renderer")
        run_scope = self.read_scope == READ_SCOPE_RUN
        lines = [
            "## Roots (environment variables in every bash call; the editor "
            "tool accepts the same $VAR form)",
        ]
        if run_scope:
            run = self.roots()["RUN_DIR"]

            def under_run(p: Path) -> str:
                # The roots table defines NODE_DIR / PARENT_DIR in terms of RUN_DIR.
                return (f"$RUN_DIR/{p.relative_to(run)}" if run in p.parents
                        else str(p))

            lines += [
                "  RUN_DIR     the run directory (runs/<experiment>/) — one "
                "round_NNN/ per node; 'node N' means round_NNN/ (zero-padded)",
                f"  NODE_DIR    {under_run(self.out_dir)}/   this node",
                f"  PARENT_DIR  {under_run(self.base_dir)}/   its parent",
            ]
        else:
            lines += [
                "  NODE_DIR    this node (the agent you are producing)",
                "  PARENT_DIR  its parent (the agent you are improving)",
            ]
        lines += [
            "  REPO_DIR    the repository root",
        ]
        if self.memory_file is not None:
            lines.append("  EDIT_MEMORY_FILE  the edit memory of this run (see below)")
        lines += [
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
        if run_scope:
            lines.append("  every other node $RUN_DIR/round_NNN/ has the same layout.")
        if self.memory_file is not None:
            lines.append(MEMORY_LEGEND_LINE)
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
    read_scope: str = READ_SCOPE_RUN,
    memory_dir: Optional[Path] = None,
    memory_file: Optional[Path] = None,
) -> PathPolicy:
    """The editor's policy. ``memory_dir`` (the run's ``edit_memory/``) is
    always denied; ``memory_file`` (a file inside it) is readable on the
    with-memory arm only, as ``$EDIT_MEMORY_FILE``."""
    if read_scope not in READ_SCOPES:
        raise ValueError(f"read_scope must be one of {sorted(READ_SCOPES)}, "
                         f"got {read_scope!r}")
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
    if read_scope == READ_SCOPE_RUN and run_root is not None:
        read_candidates.append(run_root)
    else:
        # Parent scope, or no run marker (tests, ad-hoc dirs): expose the two
        # round dirs only. The sandbox binds exactly these, so sibling nodes
        # are invisible to bash as well as to the editor tool.
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
    memory_dir_r = _real(Path(memory_dir)) if memory_dir else None
    if memory_dir_r is not None:
        deny.append(memory_dir_r)
    memory_file_r = _real(Path(memory_file)) if memory_file else None
    if memory_file_r is not None and memory_dir_r is None:
        raise ValueError("memory_file requires memory_dir")
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
        read_scope=read_scope,
        read_files=(memory_file_r,) if memory_file_r is not None else (),
        memory_file=memory_file_r,
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

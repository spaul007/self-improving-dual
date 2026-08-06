"""Config loader + component factory.

Loads a YAML file, imports built-ins (so their @register decorators run) plus
any third-party plugin modules listed in ``plugins:``, then asks the registry
for each named component class and instantiates it.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from . import registry
from .evaluator import load_cases


REPO_ROOT = Path(__file__).resolve().parent.parent


class ComponentSpec(BaseModel):
    """One pluggable component. The YAML must declare both fields
    explicitly so the run's full configuration is visible in the config,
    not embedded in framework defaults."""

    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class LoopSpec(BaseModel):
    max_rounds: int
    score_target: Optional[float] = None


class LLMSpec(BaseModel):
    """Model + reasoning settings shared by a component.

    ``base_url`` lets a component target an OpenAI-compatible endpoint
    other than OpenAI's own (e.g. a locally-hosted vLLM Responses-API
    server). Leave as ``None`` to use the SDK default. Each component
    type can carry its own base_url so meta-agent and task-agent calls
    can be routed independently.
    """
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    base_url: Optional[str] = None
    temperature: Optional[float] = None  # None -> call_llm's own default (1.0)
    max_output_tokens: Optional[int] = None  # None -> call_llm sends no cap at all


class TaskAgentSpec(LLMSpec):
    """Settings the task-agent subprocess inherits via env vars."""


class SplitSpec(BaseModel):
    """Train/eval split applied to the benchmark's case list.

    When set, optimization sees only the train half (``train_size`` cases
    selected by a seeded shuffle); the remaining cases form the held-out
    eval set. The eval split is a sidecar metric — it does not feed the
    strategy or drive "best round" selection.

    ``stratify_by`` is an optional dotted path into each case dict (e.g.
    ``"context.level"``). When set, the split is balanced across the distinct
    values of that field: every stratum contributes the same fraction
    (``train_size / total``) of its cases to train, so both halves keep the
    same level mix and random skew is removed. When ``None`` (the default),
    the split is a plain seeded shuffle — identical to the legacy behavior, so
    projects that don't opt in are unaffected.
    """

    seed: int = 42
    # Number of train cases for the seeded shuffle/stratified split. Optional:
    # omit when supplying an explicit ``train_ids`` / ``train_ids_path`` instead.
    train_size: Optional[int] = None
    stratify_by: Optional[str] = None
    # Optional cap on the held-out eval set size. ``None`` (default) keeps all
    # remaining cases as held-out (legacy behavior). ``0`` disables the held-out
    # eval entirely (no ``_run_eval_split``) — handy for fast debug runs that
    # want a small train set and no extra evaluation. Any N keeps the first N
    # remaining cases. The leftover cases are simply unused.
    eval_size: Optional[int] = None
    # Predetermined optimization (train) case ids. When either is set, the
    # seeded shuffle is bypassed and EXACTLY these ids are optimized (validated
    # against the benchmark); the held-out eval set is empty unless ``eval_size``
    # selects from the remaining cases. ``train_ids`` is an inline list;
    # ``train_ids_path`` points to a JSON file containing a list of ids (or an
    # object with a top-level ``train_ids`` list), resolved relative to the repo
    # root if not absolute. Use this to fix the same set across runs (e.g. to
    # compare hgm vs hgm_dual on identical cases).
    train_ids: Optional[list[str]] = None
    train_ids_path: Optional[str] = None


class FrameworkConfig(BaseModel):
    """A run is defined by a ``project`` plus the framework-component blocks.

    ``project`` resolves these filesystem things by convention:
      - seed dir       → ``projects/<project>/<seed_dir_name>/`` (default ``seed/``)
      - benchmark dir  → ``projects/<project>/benchmark/``
      - tools package  → ``projects.<project>.tools`` (default; see
        ``tool_source_dirs`` to scan other folders instead), auto-imported

    Every pluggable component (manager / editor / evaluator / gatherer /
    validators) must be declared in the YAML — defaults are not
    embedded in framework code. The YAML is the single source of truth
    for what a run is doing.

    ``env`` is a free-form dict of environment variables to export into
    the evaluator's child subprocesses (any project-specific config —
    e.g. per-sample database paths, API keys for project tools — goes
    here). Project tools that need a default value when the env var
    isn't set should compute the default themselves.
    """

    experiment_name: str
    project: str
    loop: LoopSpec

    manager: ComponentSpec
    evaluator: ComponentSpec
    editor: ComponentSpec
    gatherer: ComponentSpec
    validators: list[ComponentSpec]
    # Optional. When set, an LLM-summarized "behavior_memory.md" is written
    # after every (non-seed) round and injected into descendants' steering
    # contexts. Omit (or null) to disable; existing configs keep current
    # behavior without changes.
    summarizer: Optional[ComponentSpec] = None
    # Optional. When set, an LLM-synthesized "failure_summary.md" (main
    # failure patterns + hardest cases, across ALL failing cases -- not just
    # the small char-capped sample failure_report.py renders for direct
    # display) is written after every evaluation batch, including the
    # root/seed's, and injected into the editor's steering context for
    # descendants expanding from that node. Omit (or null) to disable;
    # existing configs keep current behavior without changes. See
    # meta_agent/failure_summarizer.py.
    failure_summarizer: Optional[ComponentSpec] = None
    plugins: list[str] = Field(default_factory=list)

    task_agent: TaskAgentSpec = Field(default_factory=TaskAgentSpec)
    env: dict[str, str] = Field(default_factory=dict)
    split: Optional[SplitSpec] = None

    # Evaluation visibility for the meta-agent editor. "blackbox" (default)
    # preserves current behavior: the editor sees the agent's own code, tool
    # implementations and DB schema, plus behavioral feedback (scores +
    # query/plan/what-failed examples) — but NOT the scoring code. "whitebox"
    # additionally injects the project's scorer source (benchmark/scorer.py +
    # benchmark/_eval/*.py) so the editor can read exactly how output is
    # graded. Ground-truth data (data/, cases.jsonl, validation files) is never
    # exposed in either mode.
    eval_visibility: Literal["blackbox", "whitebox"] = "blackbox"

    # Name of the project subfolder treated as the task-agent's seed source
    # (copied into round_000/task_agent, then round-over-round by the
    # editor). Defaults to "seed" — every existing project relies on this
    # default and is unaffected. Set this when a project's own real
    # implementation folder should double as the seed dir directly, instead
    # of requiring a separate wrapper folder just to satisfy this convention.
    seed_dir_name: str = "seed"

    # Files/directories (relative to the seed dir) the editor may NOT modify.
    # `None` (default) preserves the current behavior everywhere: the editor
    # may ONLY touch the hardcoded include-list (`workflow.py`,
    # `tool_wrapper.py`, `tools_schema.json`, `mutable_tools/*.py` — see
    # `editor_validators.MUTABLE_FILES`/`MUTABLE_DIRS`). Setting this list
    # flips the editor (and the `immutable_files` validator) into
    # exclude-list mode instead: everything under the seed dir is editable
    # EXCEPT these paths. Injected into both the editor and every validator
    # via `_build_with_injection` so one list can't drift out of sync
    # between what the editor is told and what the validator enforces.
    mutable_exclude: Optional[list[str]] = None

    # Directories (relative to the seed dir) to scan for immutable-tool
    # registrations. `None` (default) preserves today's hardcoded
    # convention — importing `projects.<project>.tools` as a single package
    # — every existing project relies on this default. Set this instead for
    # a project whose tool-defining code is spread across multiple folders
    # inside its own vendored implementation (e.g. a shared `common_tools/`
    # folder plus per-agent folders like `agents/coordinator/`) rather than
    # living in one dedicated top-level `tools/` package: every `*.py` file
    # under each listed directory (recursively) is loaded directly by path,
    # so whichever ones call `platform_core.tools.register_tool` at import
    # time register into the shared registry — files that don't (not every
    # project's tool code follows this convention) load harmlessly as a
    # no-op. A project with neither pattern is valid too.
    tool_source_dirs: Optional[list[str]] = None

    verbose: bool = False

    # Where per-experiment run folders are written. Defaults to repo-local
    # `runs/`. Redirect it to a larger filesystem when the local disk is
    # small — two ways: an explicit `runs_root:` in the YAML, or the
    # META_AGENT_RUNS_ROOT env var (handy for deploy scripts — see
    # slurm/run_hgm.sh). An explicit YAML value wins over the env var.
    runs_root: str = Field(
        default_factory=lambda: os.environ.get("META_AGENT_RUNS_ROOT", "runs")
    )


@dataclass
class AssembledFramework:
    config: FrameworkConfig
    config_path: Path
    manager: Any
    evaluator: Any
    gatherer: Any
    editor: Any
    validators: list[Any]
    seed_dir: Path
    benchmark_dir: Path
    runs_root: Path
    summarizer: Any = None
    failure_summarizer: Any = None
    train_case_ids: Optional[list[str]] = None
    eval_case_ids: Optional[list[str]] = None


def _ensure_builtins_loaded() -> None:
    """Import every built-in component module exactly once so their @register
    decorators populate the registry."""
    importlib.import_module("meta_agent.editor_validators")
    importlib.import_module("meta_agent.evaluator")
    importlib.import_module("meta_agent.feedback_gatherer")
    importlib.import_module("meta_agent.agent_editor")
    importlib.import_module("meta_agent.behavior_summarizer")
    importlib.import_module("meta_agent.failure_summarizer")
    importlib.import_module("meta_agent.managers")  # imports submodules


def _load_project_components(project_name: str) -> None:
    """Import a project's ``benchmark/scorer.py`` so its ``@register``
    decorators run before ``build_components`` looks the scorer up by
    name. (Per-project gatherer modules used to live at
    ``projects/<name>/gatherer.py``; that abstraction was merged into
    the scorer — see the project scorer's ``aggregate`` method.)

    Missing modules are fine — the scorer's module-level ``score()``
    fallback covers projects without a registered class.
    """
    scorer_path = REPO_ROOT / "projects" / project_name / "benchmark" / "scorer.py"
    if scorer_path.exists():
        spec = importlib.util.spec_from_file_location(
            f"_project_scorer_{project_name}", scorer_path
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                # The evaluator will surface a real error later; here we
                # just want side-effect-free best-effort import for the
                # @register decorators.
                pass


def load(path: Path) -> FrameworkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return FrameworkConfig.model_validate(raw)


def resolve_seed_dir(cfg: FrameworkConfig) -> Path:
    """The one place `project` + `seed_dir_name` become a filesystem path.
    Shared by `build_components` and `runtime_env.apply_all` (the latter
    needs it to resolve `tool_source_dirs` before `build_components` runs)."""
    return REPO_ROOT / "projects" / cfg.project / cfg.seed_dir_name


def build_components(cfg: FrameworkConfig) -> AssembledFramework:
    _ensure_builtins_loaded()
    _load_project_components(cfg.project)
    for plugin in cfg.plugins:
        importlib.import_module(plugin)

    # Construct the scorer once and inject the same instance into both
    # the evaluator (per-case ``score()``) and the gatherer (round-level
    # ``aggregate()``). This keeps the per-case shape and the round-level
    # aggregation co-located on one project class.
    scorer_name = cfg.evaluator.config.get("scorer")
    scorer_obj: Any = (
        registry.get("scorer", scorer_name)() if scorer_name else None
    )

    # The evaluator config has the scorer name as a string; we strip it
    # here and inject the resolved instance via signature injection.
    evaluator_spec = ComponentSpec(
        type=cfg.evaluator.type,
        config={k: v for k, v in cfg.evaluator.config.items() if k != "scorer"},
    )
    evaluator_obj = _build_with_injection(
        evaluator_spec, "evaluator", {"scorer": scorer_obj}
    )

    gatherer_obj = _build_with_injection(
        cfg.gatherer, "gatherer", {"scorer": scorer_obj}
    )
    validators_obj = [
        _build_with_injection(v, "validator", {"mutable_exclude": cfg.mutable_exclude})
        for v in cfg.validators
    ]

    # Lazy import: keeps the YAML loader free of an OpenAI import for
    # tests that don't build components.
    from platform_core.llm_wrapper import call_llm

    # Resolve project paths up front — the editor's static project context
    # (tool implementations, DB schema, optional scorer code) is read from here.
    runs_root = _resolve_root(cfg.runs_root)
    project_root = REPO_ROOT / "projects" / cfg.project
    seed_dir = resolve_seed_dir(cfg)
    benchmark_dir = project_root / "benchmark"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"seed directory not found: {seed_dir} "
            f"(check `project: \"{cfg.project}\"` and `seed_dir_name` in the YAML)"
        )
    if not benchmark_dir.exists():
        raise FileNotFoundError(
            f"benchmark directory not found: {benchmark_dir} "
            f"(check `project: \"{cfg.project}\"` in the YAML)"
        )

    # Static project context for the editor. tools_source + db_schema are
    # shown in both modes; scorer_source only in whitebox. None of these
    # read ground-truth data (data/, cases.jsonl, validation files).
    editor_injections: dict[str, Any] = {
        "llm_caller": call_llm,
        "validators": validators_obj,
        "tools_source": _read_tools_source(project_root, seed_dir),
        "db_schema": _read_db_schema(project_root),
        "mutable_exclude": cfg.mutable_exclude,
    }
    if cfg.eval_visibility == "whitebox":
        editor_injections["scorer_source"] = _read_scorer_source(benchmark_dir)
    editor_obj = _build_with_injection(cfg.editor, "editor", editor_injections)

    summarizer_obj: Any = None
    if cfg.summarizer is not None:
        summarizer_obj = _build_with_injection(
            cfg.summarizer,
            "summarizer",
            {"llm_caller": call_llm, "mutable_exclude": cfg.mutable_exclude},
        )

    failure_summarizer_obj: Any = None
    if cfg.failure_summarizer is not None:
        failure_summarizer_obj = _build_with_injection(
            cfg.failure_summarizer,
            "failure_summarizer",
            {"llm_caller": call_llm},
        )

    manager_obj = registry.get("manager", cfg.manager.type)(**cfg.manager.config)

    train_ids: Optional[list[str]] = None
    eval_ids: Optional[list[str]] = None
    if cfg.split is not None:
        if cfg.split.train_ids or cfg.split.train_ids_path:
            # Predetermined optimization set — bypass the seeded shuffle.
            train_ids, eval_ids = _resolve_explicit_train_ids(cfg.split, benchmark_dir)
        elif cfg.split.train_size is not None:
            train_ids, eval_ids = compute_split(
                benchmark_dir,
                seed=cfg.split.seed,
                train_size=cfg.split.train_size,
                stratify_by=cfg.split.stratify_by,
                eval_size=cfg.split.eval_size,
            )
        else:
            raise ValueError(
                "split: needs either train_size (seeded split) or "
                "train_ids / train_ids_path (predetermined set)"
            )

    return AssembledFramework(
        config=cfg,
        config_path=Path(),  # filled in by caller if it cares
        manager=manager_obj,
        evaluator=evaluator_obj,
        gatherer=gatherer_obj,
        editor=editor_obj,
        validators=validators_obj,
        seed_dir=seed_dir,
        benchmark_dir=benchmark_dir,
        runs_root=runs_root,
        summarizer=summarizer_obj,
        failure_summarizer=failure_summarizer_obj,
        train_case_ids=train_ids,
        eval_case_ids=eval_ids,
    )


def _build_with_injection(
    spec: ComponentSpec,
    kind: str,
    injections: dict[str, Any],
) -> Any:
    """Instantiate a component via the registry. Each entry in
    ``injections`` is a kwarg name → value; the framework only injects
    keys the constructor's signature actually accepts, so simpler
    implementations don't have to declare parameters they don't need.
    YAML ``config`` values still take precedence (we use
    ``setdefault``)."""
    cls = registry.get(kind, spec.type)
    kwargs = dict(spec.config)
    sig = inspect.signature(cls).parameters
    for name, value in injections.items():
        if name in sig:
            kwargs.setdefault(name, value)
    return cls(**kwargs)


def _resolve_explicit_train_ids(
    split: "SplitSpec", benchmark_dir: Path
) -> tuple[list[str], list[str]]:
    """Resolve a predetermined optimization set from ``split.train_ids`` or
    ``split.train_ids_path`` (JSON list, or an object with a ``train_ids`` key;
    path resolved relative to the repo root if not absolute). Validates every id
    exists in the benchmark and preserves order. Held-out eval is empty unless
    ``eval_size`` selects from the remaining cases."""
    if split.train_ids is not None:
        ids = [str(x) for x in split.train_ids]
    else:
        p = Path(split.train_ids_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"split.train_ids_path not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("train_ids")
        if not isinstance(data, list):
            raise ValueError(
                f"{p}: expected a JSON list of ids (or an object with a "
                "'train_ids' list)"
            )
        ids = [str(x) for x in data]

    all_ids = {
        str(c.get("id") or c.get("case_id")) for c in load_cases(benchmark_dir)
    }
    # De-dup while preserving order.
    seen: set[str] = set()
    train: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            train.append(cid)
    unknown = [cid for cid in train if cid not in all_ids]
    if unknown:
        raise ValueError(
            f"split.train_ids contains ids not in the benchmark: {unknown[:10]}"
            f"{' …' if len(unknown) > 10 else ''}"
        )
    if not train:
        raise ValueError("split.train_ids resolved to an empty set")

    eval_ids: list[str] = []
    if split.eval_size:  # optional held-out from the remaining cases
        remaining = [cid for cid in sorted(all_ids) if cid not in seen]
        eval_ids = remaining[: max(0, split.eval_size)]
    return train, eval_ids


# ---------------------------------------------------------------------- #
# Static project context for the editor (read by convention from the
# project folder). SAFETY: these read ONLY curated source files — never
# data/, cases.jsonl, or validation files — so no ground truth is exposed.
# ---------------------------------------------------------------------- #

# Total size cap (chars) per injected source bundle, to bound prompt growth.
_SOURCE_BUNDLE_CAP = 24000


def _read_py_bundle(py_files: list[Path], *, cap: int = _SOURCE_BUNDLE_CAP) -> Optional[str]:
    """Concatenate the given .py source files with per-file headers, capped."""
    chunks: list[str] = []
    total = 0
    truncated = False
    for path in sorted(py_files):
        if path.name == "__init__.py":
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        chunk = f"### {path.name}\n{body}\n"
        if total + len(chunk) > cap:
            chunks.append(f"### {path.name}\n(omitted — source bundle size cap reached)\n")
            truncated = True
            continue
        chunks.append(chunk)
        total += len(chunk)
    if not chunks:
        return None
    out = "\n".join(chunks)
    if truncated:
        out += "\n(note: some files omitted to bound prompt size)\n"
    return out


def _read_tools_source(
    project_root: Path, seed_dir: Optional[Path] = None
) -> Optional[str]:
    """Tool-implementation reference shown to the editor. Two sources, both
    optional, concatenated:

    - ``projects/<p>/tools/*.py`` — the original convention (travel/shopping/
      math's project-level immutable tool package).
    - When ``seed_dir`` is given: ``common_tools/**/*.py`` inside it
      (interfaces every agent in the workspace shares, e.g. db_mas's
      query_db/report_findings — worth surfacing as reference regardless of
      whether a given one is also separately editable) plus any ``tools.py``
      file nested inside a per-agent subfolder (e.g.
      ``agents/coordinator/tools.py``).

    Either source can be empty/absent; this generalizes cleanly to projects
    with neither pattern (returns ``None``, same as before)."""
    files: set[Path] = set()
    tools_dir = project_root / "tools"
    if tools_dir.is_dir():
        files.update(tools_dir.glob("*.py"))
    if seed_dir is not None and seed_dir.is_dir():
        common_tools_dir = seed_dir / "common_tools"
        if common_tools_dir.is_dir():
            files.update(common_tools_dir.rglob("*.py"))
        files.update(seed_dir.glob("**/tools.py"))
    if not files:
        return None
    return _read_py_bundle(sorted(files))


def _read_db_schema(project_root: Path) -> Optional[str]:
    """The project's hand-authored ``db_schema.md`` (if present)."""
    schema = project_root / "db_schema.md"
    if not schema.is_file():
        return None
    try:
        text = schema.read_text(encoding="utf-8")
    except OSError:
        return None
    return text[:_SOURCE_BUNDLE_CAP] or None


def _read_scorer_source(benchmark_dir: Path) -> Optional[str]:
    """Scoring code: ``benchmark/scorer.py`` + ``benchmark/_eval/*.py``.
    Whitebox only. Reads source TEXT only — never executes, never reads data."""
    files: list[Path] = []
    scorer = benchmark_dir / "scorer.py"
    if scorer.is_file():
        files.append(scorer)
    eval_dir = benchmark_dir / "_eval"
    if eval_dir.is_dir():
        files.extend(eval_dir.glob("*.py"))
    if not files:
        return None
    return _read_py_bundle(files)


def _dig(case: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted path (e.g. ``"context.level"``) inside a case dict.
    Returns ``None`` if any segment is missing or a non-dict is encountered —
    generic, with no project-specific field names baked in."""
    cur: Any = case
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def compute_split(
    benchmark_dir: Path,
    *,
    seed: int,
    train_size: int,
    stratify_by: Optional[str] = None,
    eval_size: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Deterministic train/eval split. Reproducible from ``seed`` alone, so
    a run can be re-derived from the YAML without a secondary state file.
    Returned in shuffled order; their union covers every case in
    ``benchmark_dir/cases.jsonl``.

    When ``stratify_by`` is given (a dotted path into each case dict, e.g.
    ``"context.level"``), the split is balanced across the distinct values of
    that field: each stratum contributes the same fraction
    (``train_size / total``) of its cases to train (largest-remainder rounding
    to land on ``train_size`` exactly), so both halves keep the same value mix.
    When ``None``, the split is a plain seeded shuffle — identical to the
    legacy behavior."""
    cases = load_cases(benchmark_dir)
    ids = [str(c.get("id") or c.get("case_id")) for c in cases]
    if train_size < 0 or train_size > len(ids):
        raise ValueError(f"train_size={train_size} out of range for {len(ids)} cases")
    rng = random.Random(seed)

    def _cap_eval(ev: list[str]) -> list[str]:
        return ev if eval_size is None else ev[: max(0, eval_size)]

    if stratify_by is None:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        return shuffled[:train_size], _cap_eval(shuffled[train_size:])

    # Stratified proportional split. Group ids by the stratify key (preserving
    # file order within each stratum), shuffle within each stratum, then
    # allocate train_size across strata proportionally with largest-remainder.
    total = len(ids)
    strata: dict[Any, list[str]] = {}
    for case, cid in zip(cases, ids):
        key = _dig(case, stratify_by)
        # Keys may be unhashable/None — normalize to a stable string bucket so
        # grouping never crashes and nothing is silently dropped.
        bucket = key if isinstance(key, (str, int, float, bool, type(None))) else str(key)
        strata.setdefault(bucket, []).append(cid)

    # Deterministic stratum order (by string form of the key) so allocation and
    # tie-breaks are reproducible regardless of dict insertion order.
    ordered_keys = sorted(strata, key=lambda k: (k is None, str(k)))
    for k in ordered_keys:
        rng.shuffle(strata[k])

    # base = floor(train_size * |stratum| / total); hand out the leftover to the
    # strata with the largest fractional remainders (stable tie-break by key).
    alloc: dict[Any, int] = {}
    remainders: list[tuple[float, str, Any]] = []
    for k in ordered_keys:
        exact = train_size * len(strata[k]) / total
        base = int(exact)  # floor for non-negative
        alloc[k] = min(base, len(strata[k]))
        remainders.append((exact - base, str(k), k))
    leftover = train_size - sum(alloc.values())
    # Largest remainder first; ties broken by stratum key for determinism.
    remainders.sort(key=lambda t: (-t[0], t[1]))
    i = 0
    while leftover > 0 and remainders:
        _, _, k = remainders[i % len(remainders)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            leftover -= 1
        i += 1
        # Safety: if every stratum is capped, stop (can't happen when
        # train_size <= total, which the range check above guarantees).
        if i > len(remainders) * (total + 1):
            break

    train: list[str] = []
    eval_: list[str] = []
    for k in ordered_keys:
        n = alloc[k]
        train.extend(strata[k][:n])
        eval_.extend(strata[k][n:])
    # Final shuffle so callers don't see cases grouped by stratum.
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, _cap_eval(eval_)


def _resolve_root(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (REPO_ROOT / p)


def init_experiment_dir(
    cfg: FrameworkConfig, config_path: Path, runs_root: Path
) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = runs_root / f"{stamp}_{cfg.experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    snapshot = experiment_dir / "config.snapshot.yaml"
    snapshot.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    return experiment_dir

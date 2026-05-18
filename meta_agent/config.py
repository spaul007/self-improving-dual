"""Config loader + component factory.

Loads a YAML file, imports built-ins (so their @register decorators run) plus
any third-party plugin modules listed in ``plugins:``, then asks the registry
for each named component class and instantiates it.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


class TaskAgentSpec(LLMSpec):
    """Settings the task-agent subprocess inherits via env vars."""


class SplitSpec(BaseModel):
    """Train/eval split applied to the benchmark's case list.

    When set, optimization sees only the train half (``train_size`` cases
    selected by a seeded shuffle); the remaining cases form the held-out
    eval set. The eval split is a sidecar metric — it does not feed the
    strategy or drive "best round" selection.
    """

    seed: int
    train_size: int


class FrameworkConfig(BaseModel):
    """A run is defined by a ``project`` plus the framework-component blocks.

    ``project`` resolves three filesystem things by convention:
      - seed dir       → ``projects/<project>/seed/``
      - benchmark dir  → ``projects/<project>/benchmark/``
      - tools package  → ``projects.<project>.tools`` (auto-imported)

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
    plugins: list[str] = Field(default_factory=list)

    task_agent: TaskAgentSpec = Field(default_factory=TaskAgentSpec)
    env: dict[str, str] = Field(default_factory=dict)
    split: Optional[SplitSpec] = None

    verbose: bool = False

    # Where per-experiment run folders are written. Defaults to repo-local
    # `runs/`. Set this in the YAML to redirect output to a larger
    # filesystem when the local disk is small — e.g.
    # `runs_root: /groups/AIC-MV/n.tzou/meta-agent/runs`.
    runs_root: str = "runs"


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
    train_case_ids: Optional[list[str]] = None
    eval_case_ids: Optional[list[str]] = None


def _ensure_builtins_loaded() -> None:
    """Import every built-in component module exactly once so their @register
    decorators populate the registry."""
    importlib.import_module("meta_agent.editor_validators")
    importlib.import_module("meta_agent.evaluator")
    importlib.import_module("meta_agent.feedback_gatherer")
    importlib.import_module("meta_agent.agent_editor")
    importlib.import_module("meta_agent.managers")  # imports submodules


def _load_project_components(project_name: str) -> None:
    """Import a project's ``benchmark/scorer.py`` so its ``@register``
    decorators run before ``build_components`` looks the scorer up by
    name. (Per-project gatherer modules used to live at
    ``projects/<name>/gatherer.py``; that abstraction was merged into
    the scorer — see ``TravelCompositeScorer.aggregate``.)

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
        registry.get("validator", v.type)(**v.config) for v in cfg.validators
    ]

    # Lazy import: keeps the YAML loader free of an OpenAI import for
    # tests that don't build components.
    from platform_core.llm_wrapper import call_llm

    editor_obj = _build_with_injection(
        cfg.editor,
        "editor",
        {"llm_caller": call_llm, "validators": validators_obj},
    )

    manager_obj = registry.get("manager", cfg.manager.type)(**cfg.manager.config)

    runs_root = _resolve_root(cfg.runs_root)
    project_root = REPO_ROOT / "projects" / cfg.project
    seed_dir = project_root / "seed"
    benchmark_dir = project_root / "benchmark"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"seed directory not found: {seed_dir} "
            f"(check `project: \"{cfg.project}\"` in the YAML)"
        )
    if not benchmark_dir.exists():
        raise FileNotFoundError(
            f"benchmark directory not found: {benchmark_dir} "
            f"(check `project: \"{cfg.project}\"` in the YAML)"
        )

    train_ids: Optional[list[str]] = None
    eval_ids: Optional[list[str]] = None
    if cfg.split is not None:
        train_ids, eval_ids = compute_split(
            benchmark_dir,
            seed=cfg.split.seed,
            train_size=cfg.split.train_size,
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


def compute_split(
    benchmark_dir: Path,
    *,
    seed: int,
    train_size: int,
) -> tuple[list[str], list[str]]:
    """Deterministic train/eval split. Reproducible from ``seed`` alone, so
    a run can be re-derived from the YAML without a secondary state file.
    Returned in shuffled order; their union covers every case in
    ``benchmark_dir/cases.jsonl``."""
    cases = load_cases(benchmark_dir)
    ids = [str(c.get("id") or c.get("case_id")) for c in cases]
    if train_size < 0 or train_size > len(ids):
        raise ValueError(f"train_size={train_size} out of range for {len(ids)} cases")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size:]


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

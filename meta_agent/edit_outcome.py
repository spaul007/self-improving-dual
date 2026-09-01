"""What an edit did to the benchmark — the half of an edit memory that cannot
be read off the diff.

Pure and deterministic: no LLM, no I/O. Everything here is derived from
``CaseResult.score``, a required field of the framework's own result model, so
this module makes no project-specific assumption.

The one project-shaped thing — *which individual criteria* a case failed — is
handled by an extraction ``recipe`` chosen once per run (by an LLM, elsewhere)
and applied here deterministically. Projects expose that information under
different keys and shapes, so a hardcoded key name does not generalise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Mapping, Optional, Protocol, Sequence

# Calibrated against a 75-node reference run: this threshold reproduces that
# run's published helped/hurt/neutral tally exactly, with an admissible window
# of [0.018, 0.0208]. It assumes scores on [0, 1] where higher is better, which
# is what HGMNode.record enforces. Re-check it for a project scoring otherwise.
NEUTRAL_BAND = 0.02
# Every criterion that moved is reported. This is a safety cap for a project
# with hundreds of criteria, not a display budget — truncating by magnitude
# silently starves whichever family of criteria happens to move in small
# increments (see FAMILY_SEPARATORS).
TOP_K_CHECKS = 40
MIN_SHARED_FOR_VERDICT = 8

# Namespace separators used to split a check id into <family>:<...>:<leaf>.
# Only the *presence* of a separator is interpreted, never what the segments
# mean, so this carries no project-specific knowledge.
FAMILY_SEPARATORS = (":", "/", ".", "|")

# Recipe modes for locating per-case failed criteria.
RECIPE_MODES = ("list", "dict_keys", "dict_flags", "none")


class NodeLike(Protocol):
    """Structural type. Satisfied by ``HGMNode`` and by rehydrated sidecar
    data alike, so this module never imports the manager package."""
    case_results: Sequence[Any]


def classify(delta: float, *, threshold: float = NEUTRAL_BAND) -> str:
    d = round(float(delta), 4)
    if d >= threshold - 1e-9:
        return "helped"
    if d <= -threshold + 1e-9:
        return "hurt"
    return "neutral"


def extract_checks(case: Any, mode: Optional[str],
                   path: Optional[str]) -> Optional[set[str]]:
    """Failing criteria for one case, or ``None`` when the case produced no
    usable check data. Never raises.

    ``None`` and ``set()`` mean opposite things and must not be conflated. A
    case that scored perfectly has the container present and empty (``set()``);
    a case whose agent crashed has no container at all (``None``). Reading the
    second as the first is what made a child that failed every case appear to
    have *fixed* every criterion its parent failed.
    """
    if not mode or mode == "none":
        return None
    d: Any = case if isinstance(case, dict) else getattr(case, "__dict__", {})
    if not isinstance(case, dict):
        # CaseResult-like: walk attributes first, then dict keys.
        d = {"case_id": getattr(case, "case_id", None),
             "score": getattr(case, "score", None),
             "passed": getattr(case, "passed", None),
             "error": getattr(case, "error", None),
             "details": getattr(case, "details", None)}
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    if d is None:
        return None
    if mode == "list":
        return ({t for t in d if isinstance(t, str)}
                if isinstance(d, (list, tuple)) else None)
    if mode == "dict_keys":
        return {str(k) for k in d} if isinstance(d, dict) else None
    if mode == "dict_flags":
        if not isinstance(d, dict):
            return None
        return {str(k) for k, v in d.items()
                if isinstance(v, dict) and v.get("passed") is False}
    return None


def _split_id(check: str) -> tuple[str, list[str]]:
    """``(separator, segments)`` for a namespaced check id, else ``("", [id])``."""
    for sep in FAMILY_SEPARATORS:
        if sep in check:
            return sep, check.split(sep)
    return "", [check]


def family_of(check: str) -> str:
    """First namespace segment, or ``""`` for a flat id."""
    sep, parts = _split_id(check)
    return parts[0] if sep else ""


def compact_check(check: str) -> str:
    """Drop middle segments, keep family and leaf: ``a:b:c`` -> ``a:c``.

    The middle segments are a grouping the leaf already implies, and they are
    the bulk of the characters. Ids with fewer than three segments are returned
    unchanged, and the original separator is preserved.
    """
    sep, parts = _split_id(check)
    if not sep or len(parts) < 3:
        return check
    return f"{parts[0]}{sep}{parts[-1]}"


def _family_fair(ordered: list[tuple[str, Any]], top_k: int,
                 rank: Any) -> list[tuple[str, Any]]:
    """Overflow policy shared by :func:`select_checks` and
    :func:`select_check_tallies`: past ``top_k`` — a safety valve for projects
    with hundreds of criteria, not a display budget — slots are filled
    round-robin across families instead of by raw rank. Families whose
    criteria are drawn per-case move in ±1 increments and would otherwise be
    shut out entirely by families evaluated on every case, which move in bulk.
    With a single family this is exactly a global top-k. ``ordered`` must
    already be sorted by ``rank``."""
    if len(ordered) <= top_k:
        return ordered
    buckets: dict[str, list[tuple[str, Any]]] = {}
    for item in ordered:                      # already sorted within each family
        buckets.setdefault(family_of(item[0]), []).append(item)
    if len(buckets) < 2:
        return ordered[:top_k]
    # Strongest first, so the dominant family still leads the line.
    order = sorted(buckets, key=lambda f: (rank(buckets[f][0]), f))
    picked: list[tuple[str, Any]] = []
    depth = 0
    while len(picked) < top_k and any(len(buckets[f]) > depth for f in order):
        for f in order:
            if len(picked) >= top_k:
                break
            if len(buckets[f]) > depth:
                picked.append(buckets[f][depth])
        depth += 1
    return sorted(picked, key=rank)


def select_checks(deltas: Mapping[str, int], *,
                  top_k: int = TOP_K_CHECKS) -> dict[str, int]:
    """Every criterion that moved, largest |Δ| first (family-fair past top_k)."""
    rank = lambda kv: (-abs(kv[1]), kv[0])  # noqa: E731
    ordered = sorted(deltas.items(), key=rank)
    return dict(_family_fair(ordered, top_k, rank))


def select_check_tallies(tallies: Mapping[str, tuple[int, int]], *,
                         top_k: int = TOP_K_CHECKS) -> dict[str, tuple[int, int]]:
    """Absolute per-check tallies ``{check: (parent_fails, child_fails)}``,
    worst-after-the-edit first — a check the child still fails many times
    matters even when its delta is zero, which is exactly what a delta-only
    selection drops."""
    rank = lambda kv: (-kv[1][1], -kv[1][0], kv[0])  # noqa: E731
    ordered = sorted(tallies.items(), key=rank)
    return dict(_family_fair(ordered, top_k, rank))


def _resolve(case: Any, path: str) -> Any:
    d: Any = case if isinstance(case, dict) else {
        "details": getattr(case, "details", None)}
    for part in (path or "").split("."):
        if not part:
            continue
        d = d.get(part) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def mode_for_shape(value: Any) -> Optional[str]:
    """The only correct mode for a value's actual shape, or ``None``.

    The shape is authoritative and the model's proposed ``mode`` is only a
    hint, because the modes are not interchangeable: on a
    ``{name: {"passed": bool}}`` map, ``dict_keys`` returns *every* criterion
    rather than the failing ones, so parent and child produce identical sets
    and every delta cancels to zero — a silently empty result rather than an
    error.
    """
    if isinstance(value, (list, tuple)):
        return "list" if any(isinstance(t, str) for t in value) else None
    if isinstance(value, dict):
        if not value:
            return None
        vals = list(value.values())
        if any(isinstance(v, dict) and "passed" in v for v in vals):
            return "dict_flags"
        return "dict_keys"
    return None


def validate_recipe(recipe: Optional[Mapping[str, str]], sample: Sequence[Any],
                    *, min_hits: int = 1) -> dict[str, str]:
    """Pick an extraction recipe that actually yields *failing* criteria.

    Candidate paths come from the model's proposal (models reliably name the
    right key but often drop its container prefix) plus a few conventional
    ones. For each path the mode is decided by :func:`mode_for_shape` rather
    than by the proposal. Falls back to ``none``, a valid state for a project
    with no per-case failure detail.
    """
    base = ((recipe or {}).get("path") or "").strip()
    leaf = base.split(".")[-1].strip()
    # An empty leaf would build "details." — which resolves to the whole
    # details object, whose keys are field names, not failing criteria.
    derived = [f"details.{leaf}", leaf] if leaf else []
    paths = [p for p in ([base] + derived +
                         ["details.failed_checks", "details.hard_constraints",
                          "details.failure_causes"]) if p]

    best, hits, seen = {"mode": "none", "path": ""}, 0, set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        modes = {mode_for_shape(_resolve(c, p)) for c in sample}
        modes.discard(None)
        if len(modes) != 1:
            continue  # absent, or inconsistent across cases — untrustworthy
        m = modes.pop()
        # Recipe selection only asks "does this path flag anything?", so an
        # unusable case (None) and a clean one (empty set) count the same here.
        n = sum(1 for c in sample if extract_checks(c, m, p))  # None is falsy
        # Prefer the path that flags the most cases; ties go to the earlier
        # (model-proposed, then conventional) path.
        if n > hits:
            best, hits = {"mode": m, "path": p}, n
    return best if hits >= min_hits else {"mode": "none", "path": ""}


@dataclass
class EditOutcome:
    n_shared: int = 0
    parent_mean_shared: float = 0.0
    child_mean_shared: float = 0.0
    delta_shared: float = 0.0
    verdict: str = "unmeasured"
    per_check_delta: dict[str, int] = field(default_factory=dict)
    # Absolute tallies {check: (parent_fails, child_fails)} over the
    # n_check_cases shared cases with check data. Unlike per_check_delta this
    # keeps checks that did not move but still fail — "still fails 12×, delta
    # 0" is signal, not noise, when the question is absolute performance.
    per_check: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Shared cases where BOTH sides produced check data. Always <= n_shared, and
    # rendered alongside per_check_delta: a check tally resting on a fraction of
    # the shared set says something much weaker than one resting on all of it.
    n_check_cases: int = 0
    # Cumulative means over ALL evaluated cases. child_mean_all is the node's
    # headline absolute performance (a 3-shared-case mean can sit far from the
    # 16-case truth); the delta verdict still comes from the shared set only.
    parent_n_all: int = 0
    child_n_all: int = 0
    parent_mean_all: float = 0.0
    child_mean_all: float = 0.0
    delta_all: float = 0.0
    threshold: float = NEUTRAL_BAND
    min_shared: int = MIN_SHARED_FOR_VERDICT

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shared": self.n_shared,
            "parent_mean_shared": self.parent_mean_shared,
            "child_mean_shared": self.child_mean_shared,
            "delta_shared": self.delta_shared,
            "verdict": self.verdict,
            "per_check_delta": self.per_check_delta,
            "per_check": self.per_check,
            "n_check_cases": self.n_check_cases,
            "parent_mean_all": self.parent_mean_all,
            "child_mean_all": self.child_mean_all,
            "delta_all": self.delta_all,
            "threshold": self.threshold,
            "min_shared": self.min_shared,
        }


def _by_id(cases: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cases or ():
        cid = c["case_id"] if isinstance(c, dict) else getattr(c, "case_id", None)
        if cid is not None:
            out[str(cid)] = c
    return out


def _score(c: Any) -> float:
    raw = c["score"] if isinstance(c, dict) else getattr(c, "score", 0.0)
    # Mirrors HGMNode.record's clamp exactly, so means agree with node tallies.
    return min(1.0, max(0.0, float(raw or 0.0)))


def compute_outcome(
    parent_cases: Sequence[Any],
    child_cases: Sequence[Any],
    *,
    threshold: float = NEUTRAL_BAND,
    top_k_checks: int = TOP_K_CHECKS,
    min_shared: int = MIN_SHARED_FOR_VERDICT,
    recipe: Optional[Mapping[str, str]] = None,
) -> EditOutcome:
    """Delta measured **only on cases the parent and child both ran**.

    Restricting to the shared set is what keeps the number from being an
    artifact of which cases each node happened to draw.
    """
    p, c = _by_id(parent_cases), _by_id(child_cases)
    out = EditOutcome(threshold=threshold, min_shared=min_shared)
    out.parent_n_all, out.child_n_all = len(p), len(c)
    pm = round(mean([_score(v) for v in p.values()]), 4) if p else 0.0
    cm = round(mean([_score(v) for v in c.values()]), 4) if c else 0.0
    out.parent_mean_all, out.child_mean_all = pm, cm
    out.delta_all = round(cm - pm, 4)

    shared = sorted(set(p) & set(c))
    if not shared:
        return out

    out.n_shared = len(shared)
    out.parent_mean_shared = round(mean([_score(p[i]) for i in shared]), 4)
    out.child_mean_shared = round(mean([_score(c[i]) for i in shared]), 4)
    out.delta_shared = round(out.child_mean_shared - out.parent_mean_shared, 4)
    # Too thin an overlap to call: the standard error dwarfs the band.
    out.verdict = ("inconclusive" if out.n_shared < min_shared
                   else classify(out.delta_shared, threshold=threshold))

    if recipe and recipe.get("mode") not in (None, "none"):
        mode, path = recipe.get("mode"), recipe.get("path")
        pf: dict[str, int] = {}
        cf: dict[str, int] = {}
        for i in shared:
            ps = extract_checks(p[i], mode, path)
            cs = extract_checks(c[i], mode, path)
            # A case only counts when BOTH sides reported. Otherwise the side
            # that produced nothing looks like it passed everything, which turns
            # a total failure into a clean sweep of "fixed" criteria.
            if ps is None or cs is None:
                continue
            out.n_check_cases += 1
            for t in ps:
                pf[t] = pf.get(t, 0) + 1
            for t in cs:
                cf[t] = cf.get(t, 0) + 1
        # Positive = the child fixes that criterion on that many more cases.
        d = {t: pf.get(t, 0) - cf.get(t, 0) for t in set(pf) | set(cf)}
        out.per_check_delta = select_checks(
            {k: v for k, v in d.items() if v}, top_k=top_k_checks)
        out.per_check = select_check_tallies(
            {t: (pf.get(t, 0), cf.get(t, 0)) for t in set(pf) | set(cf)},
            top_k=top_k_checks)
    return out


def outcome_from_nodes(parent: NodeLike, child: NodeLike, **kw: Any) -> EditOutcome:
    return compute_outcome(list(parent.case_results), list(child.case_results), **kw)


def run_context(tree: Any) -> dict[str, Any]:
    """Run-global reference points for interpreting a node's absolute score:
    the seed baseline and the best evaluated node so far.

    Injected at RENDER/ANALYSIS time only, never stored in records — refresh
    invalidation radius is 1, so a stored "best so far" would go stale in
    every untouched record the moment a better node lands.

    Cumulative means are case-sample-confounded across nodes (different nodes
    may have run different case subsets), which is why every number carries
    its ``n``. Returns ``{}`` on any miss; never raises.
    """
    try:
        nodes = list(getattr(tree, "nodes", {}).values())
        seed = next((n for n in nodes if getattr(n, "parent_id", 0) is None), None)
        live = [n for n in nodes
                if not getattr(n, "edit_failed", False)
                and getattr(n, "n_evals", 0) > 0]
        if seed is None or not live:
            return {}
        best = max(live, key=lambda n: n.mean_utility)
        return {
            "seed_mean": round(float(seed.mean_utility), 4),
            "seed_n": int(seed.n_evals),
            "best_node": int(best.node_id),
            "best_mean": round(float(best.mean_utility), 4),
            "best_n": int(best.n_evals),
        }
    except Exception:  # noqa: BLE001
        return {}


def seen_split(parent_cases: Sequence[Any], child_cases: Sequence[Any],
               seen_ids: Sequence[str]) -> dict[str, Any]:
    """Child performance split by whether a case was in the parent's evaluated
    set at edit time ("seen" — the cases the editor's feedback was computed
    from) vs not ("unseen").

    Returns {"seen": leg, "unseen": leg} where each leg is
    ``{"mean": float, "n": int, "delta": float|None}`` or ``None`` when the
    child ran no cases on that side. ``delta`` is vs the parent on the same
    subset and is None when the parent covers fewer than max(3, n/2) of the
    leg's cases — too thin to attribute. Scores are clamped like everywhere
    else. Pure; never raises on malformed cases (they are skipped)."""
    seen = {str(i) for i in seen_ids or ()}
    p = {}
    for c in parent_cases or ():
        cid = c["case_id"] if isinstance(c, dict) else getattr(c, "case_id", None)
        if cid is not None:
            p[str(cid)] = _score(c)
    ch = {}
    for c in child_cases or ():
        cid = c["case_id"] if isinstance(c, dict) else getattr(c, "case_id", None)
        if cid is not None:
            ch[str(cid)] = _score(c)
    out: dict[str, Any] = {}
    for leg, ids in (("seen", set(ch) & seen), ("unseen", set(ch) - seen)):
        if not ids:
            out[leg] = None
            continue
        cm = round(mean(ch[i] for i in ids), 4)
        pids = ids & set(p)
        d = (round(cm - mean(p[i] for i in pids), 4)
             if len(pids) >= max(3, len(ids) // 2) else None)
        out[leg] = {"mean": cm, "n": len(ids), "delta": d}
    return out

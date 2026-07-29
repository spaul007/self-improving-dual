"""db_mas benchmark scorer: wraps db-mas's own score.py::deterministic_metrics
unchanged (set-based precision/recall/F1/exact_match over predicted vs. gold
root causes). No LLM-judge call here -- that scoring dimension exists in
db-mas's own score.py for its own batch-run CLI and is orthogonal to what the
framework needs from a per-case scorer.

Ground truth (`root_causes`) lives in the case's `meta_info`, populated by
generate_cases.py, never in `context` -- the agent (workflow.py) never sees it.

`score` is F1 rather than exact_match: this benchmark structurally requires
predicting `number_of_labels_pred > len(root_causes)` labels (see db-mas's
design_doc.md, section on precision/exact_match), so `exact_match` (predicted
set == gold set) can never be true for any case -- using it as the composite
score would make every case score 0. `passed` uses recall == 1.0 (all true
root causes were found) as the meaningful boolean success criterion instead.

Imports the vendored db-mas copy via `dbmas_path.py`; `DBMAS_ROOT` overrides it.

Registered as a class (`DBMASScorer`, scorer name `"db_mas_default"`) rather
than a bare function so its `aggregate()` method gets picked up by the
gatherer (see `meta_agent/feedback_gatherer.py`::`_project_metrics` --
project_metrics dispatch requires a registered scorer *instance*, not just a
module-level `score()`). `benchmark/scorer.py` is a thin shim that imports
this module (plus `gatherer_impl.py`/`summarizer_impl.py`, for their
registration side effects) -- it's the only per-project file
`meta_agent.config._load_project_components` auto-imports.
"""
from __future__ import annotations

from typing import Any

from meta_agent.registry import register

from . import dbmas_path

dbmas_path.ensure_on_path()

import score as dbmas_score  # noqa: E402 -- db-mas's own module, unmodified


@register("scorer", "db_mas_default")
class DBMASScorer:
    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        result = getattr(agent_output, "result", agent_output) or {}
        predicted = result.get("predicted_root_causes") if isinstance(result, dict) else None
        gold = (case.get("meta_info") or {}).get("root_causes") or []

        metrics = dbmas_score.deterministic_metrics(predicted or [], gold)
        return {
            "score": metrics["f1"],
            "passed": metrics["recall"] == 1.0,
            "details": {
                **metrics,
                "predicted_root_causes": predicted,
                "gold_root_causes": gold,
            },
        }

    def aggregate(
        self, per_case: list[Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Round-level rollups computed purely from each case's `details`
        (no tracing dependency, no LLM calls). `top_false_positive_labels` /
        `top_false_negative_labels` surface which wrong label the agent most
        often over-predicts and which true root cause it most often misses --
        both `(label, count)` lists, matching `render_metrics`'s supported
        shape.

        Also rolls up two runtime signals from each case's
        `details["agent_metadata"]` (populated by the evaluator's generic
        agent-artifact injection, same source `evaluate_task_agent.py`'s
        timing breakdown already uses): `forced_fallback_rate` (the
        coordinator exhausted its turn budget without voluntarily calling
        submit_verdict, see agents/coordinator/workflow.py) and mean
        per-phase wall time. `mean_coordinator_s` in particular is worth
        watching against `mean_specialists_s`: the 5 specialists run in
        parallel (ThreadPoolExecutor) so their cost is bounded by the
        slowest one, but the coordinator's up-to-~8 turns are strictly
        sequential -- it's the dominant driver of this project's latency
        tail, and a large forced_fallback_rate alongside a high
        mean_coordinator_s is the signature of that: turns being burned on
        slow LLM round-trips rather than genuine deliberation."""
        cases = list(per_case or [])
        if not cases:
            return {}
        n = len(cases)
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        exact_matches = 0
        fp_counts: dict[str, int] = {}
        fn_counts: dict[str, int] = {}
        error_count = 0
        forced_fallback_count = 0
        total_s_vals: list[float] = []
        coordinator_s_vals: list[float] = []
        specialists_s_vals: list[float] = []

        for case in cases:
            details = getattr(case, "details", None) or {}
            precisions.append(float(details.get("precision", 0.0)))
            recalls.append(float(details.get("recall", 0.0)))
            f1s.append(float(details.get("f1", 0.0)))
            if details.get("exact_match"):
                exact_matches += 1
            for label in details.get("false_positives") or []:
                fp_counts[label] = fp_counts.get(label, 0) + 1
            for label in details.get("false_negatives") or []:
                fn_counts[label] = fn_counts.get(label, 0) + 1
            if getattr(case, "error", None):
                error_count += 1

            meta = details.get("agent_metadata") or {}
            if meta.get("forced_fallback"):
                forced_fallback_count += 1
            timing = meta.get("timing") or {}
            if timing.get("total_s") is not None:
                total_s_vals.append(float(timing["total_s"]))
            if timing.get("coordinator_s") is not None:
                coordinator_s_vals.append(float(timing["coordinator_s"]))
            if timing.get("specialists_s") is not None:
                specialists_s_vals.append(float(timing["specialists_s"]))

        result: dict[str, Any] = {
            "mean_precision": sum(precisions) / n,
            "mean_recall": sum(recalls) / n,
            "mean_f1": sum(f1s) / n,
            "exact_match_rate": exact_matches / n,
            "error_rate": error_count / n,
            "forced_fallback_rate": forced_fallback_count / n,
            "top_false_positive_labels": sorted(
                fp_counts.items(), key=lambda kv: (-kv[1], kv[0])
            ),
            "top_false_negative_labels": sorted(
                fn_counts.items(), key=lambda kv: (-kv[1], kv[0])
            ),
        }
        if total_s_vals:
            result["mean_total_s"] = sum(total_s_vals) / len(total_s_vals)
        if coordinator_s_vals:
            result["mean_coordinator_s"] = sum(coordinator_s_vals) / len(coordinator_s_vals)
        if specialists_s_vals:
            result["mean_specialists_s"] = sum(specialists_s_vals) / len(specialists_s_vals)
        return result


_DEFAULT_SCORER = DBMASScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level fallback used by `SubprocessEvaluator` when no scorer name
    is configured in YAML (`evaluator.config.scorer` unset)."""
    return _DEFAULT_SCORER.score(case, agent_output)

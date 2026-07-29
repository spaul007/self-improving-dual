"""db_mas_snapshot benchmark scorer: wraps this project's own
`eval/metrics.py::score_record`/`summarize` UNCHANGED -- no scoring math is
reimplemented here, only reconstructed into the exact input shape those
functions already expect (the same shape `mas_workflow.run_task`'s own
return value has).

Ground truth (`root_causes`) lives in the case's `meta_info`, populated by
generate_cases.py, never in `context` -- the agent (workflow.py) never sees
it (see `workflow.py::_to_db_item`'s docstring).

`score` is `recall` here, NOT F1 like the other, already-integrated
`db_mas` project -- confirmed via a real sample record that this dataset's
`number_of_labels_pred == len(root_causes)` exactly for every task (unlike
the other db_mas project, where it's always `len(root_causes)+1`), so
recall genuinely can reach 1.0, and the vendor's own `eval/metrics.py`
docstring states "recall==1 is also an exact match" for this dataset.
`passed` is `record["correct"]` (the vendor's own `exact_match == 1.0` flag).

Imports the vendored db_mas copy via `dbmassnapshot_path.py`;
`DBMASSNAPSHOT_ROOT` overrides it. Registered as a class (`DBMasSnapshotScorer`,
scorer name "db_mas_snapshot_default") so its `aggregate()` method gets
picked up by the gatherer (see `meta_agent/feedback_gatherer.py`::
`_project_metrics`). `benchmark/scorer.py` is a thin shim that imports this
module (plus `gatherer_impl.py`/`summarizer_impl.py`, for their registration
side effects) -- it's the only per-project file
`meta_agent.config._load_project_components` auto-imports.
"""
from __future__ import annotations

from typing import Any

from meta_agent.registry import register

from . import dbmassnapshot_path

dbmassnapshot_path.ensure_on_path()

from eval.metrics import score_record, summarize  # noqa: E402 -- db_mas's own module, unmodified


def _to_score_record_input(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Reconstruct the exact dict shape `eval.metrics.score_record` expects
    -- the same shape `mas_workflow.run_task`'s own return value has.
    Includes `elapsed_s`/`trajectory`/`unique_id` (not read by `score_record`
    itself, but passed through into its output via `{**record, ...}`) so
    `summarize()`'s timing/`per_label_recall`/`tool_usage` rollups keep
    working unmodified in `aggregate()` below."""
    result = getattr(agent_output, "result", agent_output) or {}
    metadata = getattr(agent_output, "metadata", None) or {}
    meta_info = case.get("meta_info") or {}
    context = case.get("context") or {}
    return {
        "unique_id": metadata.get("unique_id"),
        "root_causes": meta_info.get("root_causes", []),
        "labels": context.get("labels", []),
        "prediction": result.get("prediction", "") if isinstance(result, dict) else "",
        "error": metadata.get("error"),
        "elapsed_s": metadata.get("elapsed_s"),
        "trajectory": metadata.get("trajectory", []),
        "snapshot_found": metadata.get("snapshot_found"),
    }


@register("scorer", "db_mas_snapshot_default")
class DBMasSnapshotScorer:
    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        record = score_record(_to_score_record_input(case, agent_output))
        return {
            "score": record["recall"],
            "passed": record["correct"],
            "details": record,
        }

    def aggregate(
        self, per_case: list[Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Round-level rollups. Reconstructs the list of `score_record`-shaped
        dicts from each case's `details` (already exactly that shape, since
        `.score()` stores `record` verbatim) and calls `summarize()`
        UNCHANGED for task_score/recall/precision/f1/exact_match_rate/
        extraction_failed/over_named/errors/avg_elapsed_s/per_label/
        tool_usage -- no aggregation math reimplemented.

        Adds one small, genuinely new rollup this project has that neither
        math_mas/wikihop_mas/the other db_mas project do: `snapshot_found_rate`
        -- the mean of `details["snapshot_found"]` across cases. A rate below
        1.0 on a PRE-EDIT seed round is a wiring bug (case id doesn't match
        any db_cache/*.json snapshot file), not an agent-quality signal, and
        should be root-caused before trusting any score from that round.
        """
        cases = list(per_case or [])
        if not cases:
            return {}
        scored = [
            details
            for c in cases
            if (details := getattr(c, "details", None) or {})
        ]
        summary = summarize(scored) if scored else {}

        found = [d.get("snapshot_found") for d in scored if d.get("snapshot_found") is not None]
        summary["snapshot_found_rate"] = (
            sum(1 for f in found if f) / len(found) if found else None
        )
        return summary


_DEFAULT_SCORER = DBMasSnapshotScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level fallback used by `SubprocessEvaluator` when no scorer name
    is configured in YAML (`evaluator.config.scorer` unset)."""
    return _DEFAULT_SCORER.score(case, agent_output)

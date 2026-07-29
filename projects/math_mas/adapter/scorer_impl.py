"""math_mas benchmark scorer: wraps math_mas's own
`eval/metrics.py::normalize_answer`/`is_correct` unchanged (byte-compatible
with MASPO's MATH normalization + strict-equality rule). No LLM-judge call --
math_mas has none; scoring is deterministic exact-match only.

Ground truth (`answer`) lives in the case's `meta_info`, populated by
generate_cases.py, never in `context` -- the agent (workflow.py) never sees
it (see `workflow.py::_to_math_item`'s docstring).

`score` is strictly binary (0.0/1.0) here, unlike db_mas's continuous F1 --
`is_correct` is exact string equality after normalization, so `passed` is
just `score == 1.0`.

Imports the vendored math_mas copy via `mathmas_path.py`; `MATHMAS_ROOT`
overrides it. Registered as a class (`MathMASScorer`, scorer name
"math_mas_default") so its `aggregate()` method gets picked up by the
gatherer (see `meta_agent/feedback_gatherer.py`::`_project_metrics`).
`benchmark/scorer.py` is a thin shim that imports this module (plus
`gatherer_impl.py`/`summarizer_impl.py`, for their registration side
effects) -- it's the only per-project file
`meta_agent.config._load_project_components` auto-imports.
"""
from __future__ import annotations

from typing import Any

from meta_agent.registry import register

from . import mathmas_path

mathmas_path.ensure_on_path()

from eval.metrics import is_correct, normalize_answer  # noqa: E402 -- math_mas's own module, unmodified


@register("scorer", "math_mas_default")
class MathMASScorer:
    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        result = getattr(agent_output, "result", agent_output) or {}
        metadata = getattr(agent_output, "metadata", None) or {}
        prediction = result.get("prediction") if isinstance(result, dict) else None
        gold = (case.get("meta_info") or {}).get("answer", "")

        predictor_answer = metadata.get("predictor_answer") if isinstance(metadata, dict) else None
        error = metadata.get("error") if isinstance(metadata, dict) else None

        correct = error is None and is_correct(prediction or "", gold)
        predictor_correct = (
            error is None and predictor_answer is not None and is_correct(predictor_answer, gold)
        )

        return {
            "score": 1.0 if correct else 0.0,
            "passed": correct,
            "details": {
                "normalized_prediction": normalize_answer(prediction or ""),
                "normalized_gold": normalize_answer(gold),
                "prediction": prediction,
                "gold_answer": gold,
                "predictor_answer": predictor_answer,
                "predictor_correct": predictor_correct,
                "final_correct": correct,
                "error": error,
            },
        }

    def aggregate(
        self, per_case: list[Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Round-level rollups computed purely from each case's `details` --
        mirrors `eval/metrics.py`'s own `summarize`/`reflector_delta`, just
        re-derived from `CaseResult.details` instead of a raw-result list.
        `fixed_by_reflector`/`broken_by_reflector` is the single most useful
        diagnostic for this project (see design doc): the one existing
        full-benchmark run showed the reflector hurting far more than it
        helped (predictor_accuracy=0.906 vs. final accuracy=0.73).

        Also rolls up per-case wall-clock time (`elapsed_s`, timed inside
        math_mas's own `mas_workflow.run_task` -- predictor + optional
        compress + reflector -- carried through automatically via the
        evaluator's generic `agent_metadata` injection, see
        `meta_agent/evaluator.py:311-323`; nothing math_mas-specific had to
        be added to capture it). Surfaced per-round so a run's timing can be
        tracked case by case, not just guessed from wall-clock start/end."""
        cases = list(per_case or [])
        if not cases:
            return {}
        n = len(cases)
        correct = 0
        predictor_correct_n = 0
        fixed = broken = unchanged = 0
        error_count = 0
        evaluated = 0
        elapsed_times: list[float] = []

        for case in cases:
            details = getattr(case, "details", None) or {}
            agent_meta = details.get("agent_metadata") or {}
            elapsed = agent_meta.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                elapsed_times.append(float(elapsed))
            if getattr(case, "error", None) or details.get("error"):
                error_count += 1
                continue
            f_ok = bool(details.get("final_correct"))
            p_ok = bool(details.get("predictor_correct"))
            correct += int(f_ok)
            predictor_correct_n += int(p_ok)
            evaluated += 1
            if not p_ok and f_ok:
                fixed += 1
            elif p_ok and not f_ok:
                broken += 1
            else:
                unchanged += 1

        timing: dict[str, Any] = {}
        if elapsed_times:
            timing = {
                "mean_elapsed_s": round(sum(elapsed_times) / len(elapsed_times), 3),
                "min_elapsed_s": round(min(elapsed_times), 3),
                "max_elapsed_s": round(max(elapsed_times), 3),
                "total_elapsed_s": round(sum(elapsed_times), 3),
            }

        return {
            **timing,
            "accuracy": correct / n,
            "predictor_accuracy": predictor_correct_n / evaluated if evaluated else 0.0,
            "fixed_by_reflector": fixed,
            "broken_by_reflector": broken,
            "unchanged_by_reflector": unchanged,
            "error_rate": error_count / n,
        }


_DEFAULT_SCORER = MathMASScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level fallback used by `SubprocessEvaluator` when no scorer name
    is configured in YAML (`evaluator.config.scorer` unset)."""
    return _DEFAULT_SCORER.score(case, agent_output)

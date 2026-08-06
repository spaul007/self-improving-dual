"""math_mas_pathology benchmark scorer: wraps this project's own
`eval/metrics.py::normalize_answer`/`is_correct` unchanged (byte-compatible
with math_mas's, itself byte-compatible with MASPO's `utils.py`). No
LLM-judge call -- scoring is deterministic exact-match only, same as
math_mas.

Ground truth (`answer`) lives in the case's `meta_info`, populated by
generate_cases.py, never in `context` -- the agent (workflow.py) never sees
it (see `workflow.py::_to_math_item`'s docstring).

Beyond math_mas's `predictor_correct`/`final_correct`, this scorer also
surfaces `verifier_correct` and pathology-specific diagnostics (see
README.md "Communication Pathologies") derived from the trajectory
`workflow.py` attaches to `AgentOutput.metadata`:
  trajectory[0] -- predictor's AgentOutput.to_dict()
  trajectory[1] -- verifier's VerifierResult.to_dict() (all N turns + final)
  trajectory[2] -- reflector's AgentOutput.to_dict() (.meta has deafness info)
That fixed 3-element shape comes directly from
`mas_workflow.py::run_task`'s `"trajectory"` field -- indexed rather than
duck-typed since the order is a stable contract of this project's own code,
not external input.

Imports the vendored math_mas_pathology copy via `mathmaspathology_path.py`;
`MATHMASPATHOLOGY_ROOT` overrides it. Registered as a class
(`MathMASPathologyScorer`, scorer name "math_mas_pathology_default") so its
`aggregate()` method gets picked up by the gatherer (see
`meta_agent/feedback_gatherer.py`::`_project_metrics`). `benchmark/scorer.py`
is a thin shim that imports this module (plus `gatherer_impl.py`/
`summarizer_impl.py`, for their registration side effects) -- it's the only
per-project file `meta_agent.config._load_project_components` auto-imports.
"""
from __future__ import annotations

from typing import Any

from meta_agent.registry import register

from . import mathmaspathology_path

mathmaspathology_path.ensure_on_path()

from eval.metrics import is_correct, normalize_answer  # noqa: E402 -- this project's own module, unmodified


def _trajectory_parts(metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (verifier_turns, reflector_meta) from the fixed-shape trajectory."""
    trajectory = metadata.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 3:
        return [], None
    verifier_stage = trajectory[1] or {}
    reflector_stage = trajectory[2] or {}
    verifier_turns = verifier_stage.get("turns") or []
    reflector_meta = reflector_stage.get("meta")
    return verifier_turns, reflector_meta


@register("scorer", "math_mas_pathology_default")
class MathMASPathologyScorer:
    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        result = getattr(agent_output, "result", agent_output) or {}
        metadata = getattr(agent_output, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        prediction = result.get("prediction") if isinstance(result, dict) else None
        gold = (case.get("meta_info") or {}).get("answer", "")

        predictor_answer = metadata.get("predictor_answer")
        verifier_answer = metadata.get("verifier_answer")
        error = metadata.get("error")

        correct = error is None and is_correct(prediction or "", gold)
        predictor_correct = (
            error is None and predictor_answer is not None and is_correct(predictor_answer, gold)
        )
        verifier_correct = (
            error is None and verifier_answer is not None and is_correct(verifier_answer, gold)
        )

        verifier_turns, reflector_meta = _trajectory_parts(metadata)
        turn_answers = [t.get("answer") for t in verifier_turns if isinstance(t, dict)]
        verifier_turn_drift = bool(turn_answers) and turn_answers[0] != turn_answers[-1]

        first_draft = metadata.get("first_draft")
        verifier_final_context = metadata.get("verifier_final_context")
        context_divergent = bool(
            first_draft is not None
            and verifier_final_context is not None
            and first_draft != verifier_final_context
        )
        reflector_meta = reflector_meta or {}

        return {
            "score": 1.0 if correct else 0.0,
            "passed": correct,
            "details": {
                "normalized_prediction": normalize_answer(prediction or ""),
                "normalized_gold": normalize_answer(gold),
                "prediction": prediction,
                "gold_answer": gold,
                "predictor_answer": predictor_answer,
                "verifier_answer": verifier_answer,
                "predictor_correct": predictor_correct,
                "verifier_correct": verifier_correct,
                "final_correct": correct,
                "error": error,
                "verifier_turn_drift": verifier_turn_drift,
                "verifier_unique_answer_count": len(set(turn_answers)) if turn_answers else 0,
                "verifier_rounds_run": len(turn_answers),
                "context_divergent": context_divergent,
                "context_used_by_reflector": metadata.get("context_used_by_reflector"),
                "pathology_flags": metadata.get("pathology_flags"),
                "deafness_active": reflector_meta.get("deafness_active"),
                "deafness_sentences_dropped": reflector_meta.get("sentences_dropped"),
                "deafness_chars_dropped": reflector_meta.get("chars_dropped"),
            },
        }

    def aggregate(
        self, per_case: list[Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Round-level rollups computed purely from each case's `details`.

        Beyond math_mas's `accuracy`/`predictor_accuracy`/
        `fixed_by_reflector`/`broken_by_reflector`, this adds:
        - `verifier_accuracy`/`fixed_by_verifier`/`broken_by_verifier` --
          predictor -> verifier transition, showing whether the (possibly
          discarded) verifier stage would have helped in principle.
        - `verifier_turn_drift_rate` -- fraction of cases where the
          verifier's first and last (identical-input) turns disagree --
          proxy for whether the N-1 discarded repetition-pathology calls
          carried any real signal at all.
        - `context_divergence_rate` -- fraction of cases where the stale
          first-draft context differs from what the verifier actually
          concluded -- how much pathology 2 could matter, independent of
          whether the toggle is on.
        - `selective_deafness_*_dropped_avg` -- averaged from the
          reflector's own diagnostics, only over cases where deafness was
          active.
        - `avg_verify_rounds_run` -- direct cost-visibility diagnostic.
        """
        cases = list(per_case or [])
        if not cases:
            return {}
        n = len(cases)
        correct = 0
        predictor_correct_n = 0
        verifier_correct_n = 0
        fixed = broken = unchanged = 0
        v_fixed = v_broken = v_unchanged = 0
        error_count = 0
        evaluated = 0
        elapsed_times: list[float] = []
        turn_drift_count = 0
        context_divergent_count = 0
        verify_rounds_total = 0
        deafness_chars: list[float] = []
        deafness_sentences: list[float] = []

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
            v_ok = bool(details.get("verifier_correct"))
            correct += int(f_ok)
            predictor_correct_n += int(p_ok)
            verifier_correct_n += int(v_ok)
            evaluated += 1

            if not p_ok and f_ok:
                fixed += 1
            elif p_ok and not f_ok:
                broken += 1
            else:
                unchanged += 1

            if not p_ok and v_ok:
                v_fixed += 1
            elif p_ok and not v_ok:
                v_broken += 1
            else:
                v_unchanged += 1

            if details.get("verifier_turn_drift"):
                turn_drift_count += 1
            if details.get("context_divergent"):
                context_divergent_count += 1
            rounds_run = details.get("verifier_rounds_run")
            if isinstance(rounds_run, (int, float)):
                verify_rounds_total += rounds_run
            if details.get("deafness_active"):
                cd = details.get("deafness_chars_dropped")
                sd = details.get("deafness_sentences_dropped")
                if isinstance(cd, (int, float)):
                    deafness_chars.append(cd)
                if isinstance(sd, (int, float)):
                    deafness_sentences.append(sd)

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
            "verifier_accuracy": verifier_correct_n / evaluated if evaluated else 0.0,
            "fixed_by_reflector": fixed,
            "broken_by_reflector": broken,
            "unchanged_by_reflector": unchanged,
            "fixed_by_verifier": v_fixed,
            "broken_by_verifier": v_broken,
            "unchanged_by_verifier": v_unchanged,
            "verifier_turn_drift_rate": turn_drift_count / evaluated if evaluated else 0.0,
            "context_divergence_rate": context_divergent_count / evaluated if evaluated else 0.0,
            "selective_deafness_chars_dropped_avg": (
                sum(deafness_chars) / len(deafness_chars) if deafness_chars else 0.0
            ),
            "selective_deafness_sentences_dropped_avg": (
                sum(deafness_sentences) / len(deafness_sentences) if deafness_sentences else 0.0
            ),
            "avg_verify_rounds_run": verify_rounds_total / evaluated if evaluated else 0.0,
            "error_rate": error_count / n,
        }


_DEFAULT_SCORER = MathMASPathologyScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level fallback used by `SubprocessEvaluator` when no scorer name
    is configured in YAML (`evaluator.config.scorer` unset)."""
    return _DEFAULT_SCORER.score(case, agent_output)

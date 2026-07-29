"""wikihop_mas benchmark scorer: wraps wikihop_mas's own `eval/metrics.py`
(normalize_answer/exact_match/f1_prec_recall/sp_prf/joint_f1) unchanged --
standard HotpotQA/2WikiMultihopQA short-answer + supporting-fact scoring.
No LLM-judge call.

Ground truth (`answer`/`type`/`supporting_facts`) lives in the case's
`meta_info`, populated by generate_cases.py, never in `context` --
`workflow.py` never sees it (see workflow.py's `_to_wikihop_item`
docstring).

`score` = answer_f1 (continuous token-overlap F1 on the final MAS answer),
NOT answer_em. Deliberate: `meta_agent/managers/hgm.py`'s
`HGMNode.record(case)` accumulates `mean_utility` directly from
`CaseResult.score`, and `_expandable()` only allows expansion from a node
with `mean_utility > 0`. A continuous partial-credit signal is strictly
more informative than binary 0/1 for a small train pool, and it further
shrinks the (already small) risk of an all-zero-scoring seed: a
token-overlapping-but-not-exact answer still contributes positive utility
under F1, but exactly zero under EM -- the same failure mode that forced
math_mas's train_size bump from 2 to 6.

`passed` is still the strict `answer_em` boolean (kept for diagnostics/
n_hard_cases sampling). NOTE: `CaseResult.passed` (from this `passed` key)
and `failure_report.py`'s own "failing" bucket (score < gatherer's
pass_threshold, default 1.0) are two independent mechanisms and can, in a
rare edge case, disagree: a multi-word answer with F1=1.0 (bag-of-words
match) but a token-order mismatch (EM=False) would have `passed=False` yet
NOT be flagged as "failing" by the score-based bucket. Expected to be rare
for wikihop_mas's typically short (entity/date) answers; not worth a
workaround.

Imports the vendored wikihop_mas copy via `wikihopmas_path.py`;
`WIKIHOPMAS_ROOT` overrides it. Registered as a class (`WikihopMASScorer`,
scorer name "wikihop_mas_default") so its `aggregate()` method gets picked
up by the gatherer (see `meta_agent/feedback_gatherer.py::_project_metrics`).
`benchmark/scorer.py` is a thin shim that imports this module (plus
`gatherer_impl.py`/`summarizer_impl.py`, for their registration side
effects) -- it's the only per-project file
`meta_agent.config._load_project_components` auto-imports.
"""
from __future__ import annotations

from typing import Any

from meta_agent.registry import register

from . import wikihopmas_path

wikihopmas_path.ensure_on_path()

from eval.metrics import (  # noqa: E402 -- wikihop_mas's own module, unmodified
    exact_match,
    f1_prec_recall,
    joint_f1,
    normalize_answer,
    sp_prf,
)


@register("scorer", "wikihop_mas_default")
class WikihopMASScorer:
    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        result = getattr(agent_output, "result", agent_output) or {}
        metadata = getattr(agent_output, "metadata", None) or {}
        meta_info = case.get("meta_info") or {}

        prediction = result.get("prediction", "") if isinstance(result, dict) else ""
        gold_answer = meta_info.get("answer", "")
        gold_type = meta_info.get("type", "")
        gold_supporting_facts = meta_info.get("supporting_facts", []) or []

        error = metadata.get("error") if isinstance(metadata, dict) else None
        predicted_type = metadata.get("predicted_type") if isinstance(metadata, dict) else None
        predicted_supporting_facts = (
            metadata.get("predicted_supporting_facts") if isinstance(metadata, dict) else None
        ) or []
        concluder_rounds = metadata.get("concluder_rounds") if isinstance(metadata, dict) else None
        final_answer_pre_retry = (
            metadata.get("final_answer_pre_retry") if isinstance(metadata, dict) else None
        )

        ok = error is None
        ans_p, ans_r, ans_f1 = f1_prec_recall(prediction, gold_answer)
        answer_em = ok and exact_match(prediction, gold_answer)

        pred_sf = {tuple(x) for x in predicted_supporting_facts}
        gold_sf = {tuple(x) for x in gold_supporting_facts}
        sp_p, sp_r, sp_f1 = sp_prf(pred_sf, gold_sf)
        sp_em = pred_sf == gold_sf

        type_correct = bool(predicted_type) and predicted_type == gold_type

        return {
            "score": round(ans_f1, 4) if ok else 0.0,
            "passed": bool(answer_em),
            "details": {
                "prediction": prediction,
                "gold_answer": gold_answer,
                "normalized_prediction": normalize_answer(prediction),
                "normalized_gold": normalize_answer(gold_answer),
                "answer_em": answer_em,
                "answer_f1": round(ans_f1, 4),
                "sp_em": sp_em,
                "sp_f1": round(sp_f1, 4),
                "joint_em": bool(answer_em and sp_em),
                "joint_f1": round(joint_f1(ans_p, ans_r, sp_p, sp_r), 4),
                "predicted_type": predicted_type,
                "gold_type": gold_type,
                "type_correct": type_correct,
                "concluder_rounds": concluder_rounds,
                "final_answer_pre_retry": final_answer_pre_retry,
                "error": error,
            },
        }

    def aggregate(
        self, per_case: list[Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Round-level rollups computed purely from each case's `details` --
        mirrors `eval/metrics.py`'s own `summarize`/`retry_delta`, re-derived
        from `CaseResult.details` instead of a raw-result list (never call
        `eval.metrics.summarize`/`retry_delta` directly here -- they operate
        on a differently-shaped raw-record list, not CaseResult.details).

        `fixed_by_grounding_retry`/`broken_by_grounding_retry` is this
        project's single most useful diagnostic (mirrors math_mas's
        fixed/broken_by_reflector): among cases where the bounded grounding
        retry actually fired (concluder_rounds == 2), did it rescue a wrong
        pre-retry answer or break a correct one?

        `per_type` fills a real, previously-confirmed gap: `gold_type`
        (compositional/comparison/bridge_comparison/inference) has a huge
        spread in this benchmark (measured directly on the pristine seed:
        comparison ~0.96 mean score vs. compositional ~0.46, and
        compositional is ~45% of the dataset) but nothing in this scorer
        used to break performance down by it -- `decomposer_type_accuracy`
        looks like it would help but is a DIFFERENT thing entirely (whether
        the Decomposer's own type classification matched gold, not accuracy
        BY type). Without this, 9 straight rounds of editor edits oscillated
        on one global grounding-strictness knob (several pairs of edits
        pulling in literally opposite directions) because the real,
        concentrated failure mode (compositional multi-hop questions) was
        invisible in both project_metrics and the failure_summary's
        qualitative read -- confirmed directly, neither ever named it.

        Per-case elapsed_s is carried through automatically via the
        evaluator's generic `agent_metadata` injection -- nothing
        wikihop_mas-specific had to be added to capture it.
        """
        cases = list(per_case or [])
        if not cases:
            return {}
        n = len(cases)
        error_count = 0
        evaluated = 0
        answer_em_n = 0
        type_correct_n = 0
        retried_n = 0
        fixed = broken = unchanged = 0
        elapsed_times: list[float] = []
        answer_f1s: list[float] = []
        sp_f1s: list[float] = []
        joint_f1s: list[float] = []
        by_type: dict[str, dict[str, list[float]]] = {}

        for case in cases:
            details = getattr(case, "details", None) or {}
            agent_meta = details.get("agent_metadata") or {}
            elapsed = agent_meta.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                elapsed_times.append(float(elapsed))
            if getattr(case, "error", None) or details.get("error"):
                error_count += 1
                continue

            evaluated += 1
            answer_em_n += int(bool(details.get("answer_em")))
            type_correct_n += int(bool(details.get("type_correct")))
            answer_f1s.append(float(details.get("answer_f1", 0.0)))
            sp_f1s.append(float(details.get("sp_f1", 0.0)))
            joint_f1s.append(float(details.get("joint_f1", 0.0)))

            gold_type = details.get("gold_type") or "unknown"
            bucket = by_type.setdefault(gold_type, {"f1": [], "em": []})
            bucket["f1"].append(float(details.get("answer_f1", 0.0)))
            bucket["em"].append(1.0 if details.get("answer_em") else 0.0)

            if details.get("concluder_rounds") == 2:
                retried_n += 1
                pre_ok = exact_match(
                    str(details.get("final_answer_pre_retry") or ""),
                    str(details.get("gold_answer") or ""),
                )
                post_ok = bool(details.get("answer_em"))
                if not pre_ok and post_ok:
                    fixed += 1
                elif pre_ok and not post_ok:
                    broken += 1
                else:
                    unchanged += 1

        def _mean(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        timing: dict[str, Any] = {}
        if elapsed_times:
            timing = {
                "mean_elapsed_s": round(sum(elapsed_times) / len(elapsed_times), 3),
                "min_elapsed_s": round(min(elapsed_times), 3),
                "max_elapsed_s": round(max(elapsed_times), 3),
                "total_elapsed_s": round(sum(elapsed_times), 3),
            }

        # dict of name -> number: `render_metrics` renders this as a
        # "weakest N" summary, sorted low-to-high -- exactly what's wanted
        # here (surface the worst-performing question type first). A
        # separate `per_type_n` dict (plain scalars, not a "name -> number"
        # metric to rank) carries how many cases back each mean, so the
        # editor can tell "compositional is bad AND is half the dataset"
        # apart from "some rare type had 2 unlucky cases."
        per_type_f1 = {type_name: _mean(b["f1"]) for type_name, b in by_type.items()}
        per_type_n = {type_name: len(b["f1"]) for type_name, b in by_type.items()}

        return {
            **timing,
            "answer_em": round(answer_em_n / evaluated, 4) if evaluated else 0.0,
            "answer_f1": _mean(answer_f1s),
            "sp_f1": _mean(sp_f1s),
            "joint_f1": _mean(joint_f1s),
            "answer_f1_by_type": per_type_f1,
            "n_cases_by_type": per_type_n,
            "decomposer_type_accuracy": round(type_correct_n / evaluated, 4) if evaluated else 0.0,
            "pct_cases_with_retry": round(retried_n / evaluated, 4) if evaluated else 0.0,
            "fixed_by_grounding_retry": fixed,
            "broken_by_grounding_retry": broken,
            "unchanged_by_grounding_retry": unchanged,
            "error_rate": round(error_count / n, 4),
        }


_DEFAULT_SCORER = WikihopMASScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level fallback used by `SubprocessEvaluator` when no scorer
    name is configured in YAML (`evaluator.config.scorer` unset)."""
    return _DEFAULT_SCORER.score(case, agent_output)

"""Evaluation metrics for the database RCA MAS.

All scoring logic lives here; `evaluate.py` is only a CLI around it.

Scoring matches MASPO_v2's DatabaseTaskJudge (judges.py): labels are read out
of the lead DBA's diagnosis deterministically (tools/immutable/
label_extraction.py), truncated to k = |gold set|, then set metrics are
computed case-insensitively. The headline task score is ROOT-CAUSE RECALL —
the fraction of gold root causes recovered — reported per task and averaged
over the run (a crashed task counts as 0: with ground truth, a run that
crashed IS a wrong answer). Since the task text asks for exactly |gold| labels
(see snapshot/prepare_dataset.py), recall == 1 is also an exact match.
"""

from typing import Any

import config
from tools.immutable.label_extraction import extract_labels

# --------------------------------------------------------------------------
# Per-sample metrics
# --------------------------------------------------------------------------


def score_predictions(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    """Set metrics for one task, case-folded on both sides.

    precision/recall/f1 coincide whenever the team names the requested number
    of labels, and diverge only when it under-names — which is exactly the
    failure precision is here to expose.
    """
    gold_set = {g.strip().lower() for g in gold if g and g.strip()}
    pred_set = {p.strip().lower() for p in predicted if p and p.strip()}
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "exact_match": 1.0 if pred_set == gold_set and gold_set else 0.0,
    }


_ZERO_SCORES = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "exact_match": 0.0}


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach label extraction + set scores to one raw inference record."""
    gold = record.get("root_causes", []) or []
    labels = record.get("labels") or config.LABELS
    # Single source of truth for the count: the gold set itself (the task text
    # was rewritten to ask for the same count).
    k = len({g.strip().lower() for g in gold if g and g.strip()})

    if record.get("error") is not None or not (record.get("prediction") or "").strip():
        return {
            **record,
            **_ZERO_SCORES,
            "predicted": [],
            "extraction": "unscorable",
            "n_named": 0,
            "n_requested": k,
            "correct": False,
        }

    # Parse untruncated first, so naming more than k labels is recorded rather
    # than silently cut. `n_named` > k means the team ignored the requested
    # count — a prompt-compliance problem, not a diagnostic one.
    named = extract_labels(record["prediction"], labels)
    predicted = named[:k]
    scores = score_predictions(predicted, gold)
    return {
        **record,
        **scores,
        "predicted": predicted,
        "extraction": "deterministic" if predicted else "none",
        "n_named": len(named),
        "n_requested": k,
        "correct": scores["exact_match"] == 1.0,
    }


# --------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def per_label_recall(scored: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each canonical label: how often it was recovered when it was gold."""
    out: dict[str, dict[str, Any]] = {}
    for label in config.LABELS:
        lnorm = label.lower()
        gold_tasks = [
            r for r in scored
            if lnorm in {g.strip().lower() for g in (r.get("root_causes") or [])}
        ]
        hits = sum(
            1 for r in gold_tasks
            if lnorm in {p.strip().lower() for p in (r.get("predicted") or [])}
        )
        out[label] = {
            "gold_tasks": len(gold_tasks),
            "recovered": hits,
            "recall": round(hits / len(gold_tasks), 3) if gold_tasks else None,
        }
    return out


def tool_usage(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate query_db usage out of the saved trajectories.

    Derivable from any raw file (even re-scored old ones), unlike the live
    counters in tools/immutable/query_db.py which exist only during inference.
    """
    calls = 0
    by_replay: dict[str, int] = {}
    for r in scored:
        for agent in r.get("trajectory") or []:
            for tc in agent.get("tool_calls") or []:
                calls += 1
                kind = tc.get("replay") or "unknown"
                by_replay[kind] = by_replay.get(kind, 0) + 1
    n = len(scored) or 1
    return {
        "query_db_calls": calls,
        "avg_calls_per_task": round(calls / n, 2),
        "by_replay": dict(sorted(by_replay.items())),
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics for a scored run. Averages include errored tasks as 0."""
    errors = [r for r in scored if r.get("error") is not None]
    times = [r["elapsed_s"] for r in scored if r.get("elapsed_s") is not None]

    return {
        "total": len(scored),
        # Headline task score: mean root-cause recall (MASPO surfaces the same
        # number under its historical name `accuracy`).
        "task_score": round(_mean([r["recall"] for r in scored]), 4),
        "recall": round(_mean([r["recall"] for r in scored]), 4),
        "precision": round(_mean([r["precision"] for r in scored]), 4),
        "f1": round(_mean([r["f1"] for r in scored]), 4),
        "exact_match": sum(1 for r in scored if r["correct"]),
        "exact_match_rate": round(_mean([1.0 if r["correct"] else 0.0 for r in scored]), 4),
        "extraction_failed": sum(1 for r in scored if r.get("extraction") in ("none", "unscorable")),
        "over_named": sum(1 for r in scored if r.get("n_named", 0) > r.get("n_requested", 0)),
        "errors": len(errors),
        "avg_elapsed_s": round(sum(times) / len(times), 3) if times else 0.0,
        "per_label": per_label_recall(scored),
        "tool_usage": tool_usage(scored),
    }

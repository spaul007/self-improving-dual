"""Evaluation metrics for the pathological math MAS.

All scoring logic lives here; `evaluate.py` is only a CLI around it.

`normalize_answer`/`is_correct` are copied verbatim from math_mas (kept
byte-compatible with MASPO's `utils.normalize_answer` and its non-judge MATH
correctness rule) -- unchanged, since this is the scoring contract, not MAS
behavior.

`stage_delta` generalizes math_mas's `reflector_delta` into a reusable
before/after correctness-transition counter between any two raw-answer
fields on a scored record -- this pipeline now has three stages
(predictor/verifier/reflector) instead of two, so both `reflector_delta`
(predictor -> final) and `verifier_delta` (predictor -> verifier) reuse it.
"""

import re
from typing import Any

# --------------------------------------------------------------------------
# Answer normalization
# --------------------------------------------------------------------------


def normalize_answer(answer: str) -> str:
    """Canonicalize a LaTeX-ish math answer for string comparison."""
    if not answer:
        return ""

    answer = answer.strip()
    answer = re.sub(r"\$(.*?)\$", r"\1", answer)
    answer = re.sub(r"\\\[(.*?)\\\]", r"\1", answer, flags=re.S)
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\s*{((?:[^{}]|{[^}]*})*?)}", r"\1", answer)
    answer = re.sub(r"\\\((.*?)\\\)", r"\1", answer)
    answer = re.sub(r"\\?°", "", answer)
    answer = re.sub(r"\^?\\?circ", "", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = re.sub(r"\\sqrt\s*{([^}]*)}", r"sqrt(\1)", answer)
    answer = re.sub(r"√(\d+)", r"sqrt(\1)", answer)
    answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\pi", "π", answer)
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = answer.replace("[", "(").replace("]", ")")

    # A bare list of integers is order-insensitive.
    if re.fullmatch(r"[,\s\-0-9]+", answer):
        nums = [int(x) for x in re.findall(r"-?\d+", answer)]
        return ",".join(map(str, sorted(nums)))

    return answer.lower()


# --------------------------------------------------------------------------
# Per-sample metrics
# --------------------------------------------------------------------------


def is_correct(prediction: str, gold: str) -> bool:
    """Exact match after normalization (MASPO's MATH rule)."""
    return normalize_answer(prediction) == normalize_answer(gold)


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach correctness to one raw inference record."""
    correct = record.get("error") is None and is_correct(
        record.get("prediction", ""), record.get("gold_answer", "")
    )
    return {
        **record,
        "normalized_prediction": normalize_answer(record.get("prediction", "")),
        "normalized_gold": normalize_answer(record.get("gold_answer", "")),
        "correct": correct,
    }


# --------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------


def accuracy(scored: list[dict[str, Any]]) -> float:
    if not scored:
        return 0.0
    return sum(1 for r in scored if r["correct"]) / len(scored)


def stage_delta(scored: list[dict[str, Any]], before_key: str, after_key: str) -> dict[str, Any]:
    """Correctness transition between two raw-answer fields on each record.

    `before_key`/`after_key` name raw-answer fields present on each scored
    record (e.g. "predictor_answer" -> "verifier_answer", or
    "predictor_answer" -> "prediction" for the pipeline's final answer);
    both are compared against `gold_answer` independently via `is_correct`.
    """
    fixed = broken = unchanged = 0
    before_correct_n = 0
    evaluated = 0

    for r in scored:
        if r.get("error") is not None:
            continue
        gold = r.get("gold_answer", "")
        before_ok = is_correct(r.get(before_key, ""), gold)
        after_ok = is_correct(r.get(after_key, ""), gold)
        before_correct_n += int(before_ok)
        evaluated += 1
        if not before_ok and after_ok:
            fixed += 1
        elif before_ok and not after_ok:
            broken += 1
        else:
            unchanged += 1

    return {
        "before_accuracy": before_correct_n / evaluated if evaluated else 0.0,
        "fixed": fixed,
        "broken": broken,
        "unchanged": unchanged,
        "evaluated": evaluated,
    }


def reflector_delta(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Predictor -> final(reflector) transition -- isolates the combined
    verifier+reflector stages' net contribution, same headline metric name
    math_mas uses."""
    d = stage_delta(scored, "predictor_answer", "prediction")
    return {
        "predictor_accuracy": d["before_accuracy"],
        "fixed_by_reflector": d["fixed"],
        "broken_by_reflector": d["broken"],
        "unchanged": d["unchanged"],
    }


def verifier_delta(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Predictor -> verifier transition -- shows whether the verifier stage
    changes anything *in principle*, before pathology 2 (stale context
    injection) potentially discards its conclusion entirely on the way to
    the reflector."""
    d = stage_delta(scored, "predictor_answer", "verifier_answer")
    return {
        "predictor_accuracy": d["before_accuracy"],
        "fixed_by_verifier": d["fixed"],
        "broken_by_verifier": d["broken"],
        "unchanged_by_verifier": d["unchanged"],
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics for a scored run."""
    errors = [r for r in scored if r.get("error") is not None]
    times = [r["elapsed_s"] for r in scored if r.get("elapsed_s") is not None]

    return {
        "total": len(scored),
        "correct": sum(1 for r in scored if r["correct"]),
        "accuracy": round(accuracy(scored), 4),
        "errors": len(errors),
        "avg_elapsed_s": round(sum(times) / len(times), 3) if times else 0.0,
        **reflector_delta(scored),
        **verifier_delta(scored),
    }

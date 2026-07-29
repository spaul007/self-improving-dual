"""Evaluation metrics for the math MAS.

All scoring logic lives here; `evaluate.py` is only a CLI around it.

`normalize_answer` is kept byte-compatible with MASPO's `utils.normalize_answer`
and the correctness rule matches MASPO's non-judge MATH path (strict equality of
normalized strings), so accuracy numbers are directly comparable.
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


def reflector_delta(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """How the reflector changed the predictor's answer.

    Isolates the second stage's contribution: how often it rescued a wrong
    predictor answer vs. broke a correct one.
    """
    fixed = broken = unchanged = 0
    pred_correct = 0

    for r in scored:
        if r.get("error") is not None:
            continue
        p_ok = is_correct(r.get("predictor_answer", ""), r.get("gold_answer", ""))
        f_ok = r["correct"]
        pred_correct += int(p_ok)
        if not p_ok and f_ok:
            fixed += 1
        elif p_ok and not f_ok:
            broken += 1
        else:
            unchanged += 1

    evaluated = fixed + broken + unchanged
    return {
        "predictor_accuracy": pred_correct / evaluated if evaluated else 0.0,
        "fixed_by_reflector": fixed,
        "broken_by_reflector": broken,
        "unchanged": unchanged,
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
    }

"""Evaluation metrics for the wikihop MAS.

All scoring logic lives here; `evaluate.py` is only a CLI around it.

Completely different normalizer from math_mas's LaTeX-aware one -- this is the
standard HotpotQA/2WikiMultihopQA short-answer normalization (lowercase, strip
articles/punctuation, whitespace-fix), feeding Answer EM/F1, Supporting-Fact
EM/F1, and a Joint EM/F1 that combines them (2WikiMultihopQA's standard
metric trio).
"""

import re
import string
from collections import Counter
from typing import Any

# --------------------------------------------------------------------------
# Answer normalization
# --------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Canonicalize a short answer for string/token comparison."""
    if not s:
        return ""

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        return "".join(c for c in t if c not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_prec_recall(prediction: str, gold: str) -> tuple[float, float, float]:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return (0.0, 0.0, float(pred_toks == gold_toks))
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return (0.0, 0.0, 0.0)
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# --------------------------------------------------------------------------
# Supporting-fact + joint metrics
# --------------------------------------------------------------------------


def sp_prf(pred_facts: set[tuple], gold_facts: set[tuple]) -> tuple[float, float, float]:
    tp = len(pred_facts & gold_facts)
    precision = tp / len(pred_facts) if pred_facts else 0.0
    recall = tp / len(gold_facts) if gold_facts else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def joint_f1(ans_p: float, ans_r: float, sp_p: float, sp_r: float) -> float:
    jp, jr = ans_p * sp_p, ans_r * sp_r
    return 0.0 if jp + jr == 0 else 2 * jp * jr / (jp + jr)


# --------------------------------------------------------------------------
# Per-sample scoring
# --------------------------------------------------------------------------


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach answer/supporting-fact/joint metrics to one raw inference record."""
    prediction = record.get("prediction", "")
    gold = record.get("gold_answer", "")
    ans_p, ans_r, ans_f1 = f1_prec_recall(prediction, gold)
    ans_em = record.get("error") is None and exact_match(prediction, gold)

    pred_sf = {tuple(x) for x in record.get("predicted_supporting_facts", [])}
    gold_sf = {tuple(x) for x in record.get("gold_supporting_facts", [])}
    sp_p, sp_r, sp_f1 = sp_prf(pred_sf, gold_sf)
    sp_em = pred_sf == gold_sf

    return {
        **record,
        "normalized_prediction": normalize_answer(prediction),
        "normalized_gold": normalize_answer(gold),
        "answer_em": ans_em,
        "answer_f1": round(ans_f1, 4),
        "sp_em": sp_em,
        "sp_f1": round(sp_f1, 4),
        "joint_em": bool(ans_em and sp_em),
        "joint_f1": round(joint_f1(ans_p, ans_r, sp_p, sp_r), 4),
        "type_correct": record.get("predicted_type") == record.get("gold_type"),
    }


# --------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def retry_delta(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """How the bounded grounding-retry loop changed the pre-retry answer.

    Isolates the retry loop's marginal contribution (analogous to math_mas's
    reflector_delta): among records where a hop retry actually fired
    (concluder_rounds == 2), how often did it rescue a wrong answer vs. break
    a correct one?
    """
    fixed = broken = unchanged = 0
    for r in scored:
        if r.get("error") is not None or r.get("concluder_rounds", 1) < 2:
            continue
        pre_ok = exact_match(r.get("final_answer_pre_retry", ""), r.get("gold_answer", ""))
        post_ok = r["answer_em"]
        if not pre_ok and post_ok:
            fixed += 1
        elif pre_ok and not post_ok:
            broken += 1
        else:
            unchanged += 1
    return {
        "fixed_by_grounding_retry": fixed,
        "broken_by_grounding_retry": broken,
        "unchanged_by_grounding_retry": unchanged,
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics for a scored run."""
    errors = [r for r in scored if r.get("error") is not None]
    times = [r["elapsed_s"] for r in scored if r.get("elapsed_s") is not None]
    retriever_rounds = [
        hop.get("retriever_rounds_used", 0)
        for r in scored
        for hop in r.get("trajectory", {}).get("hops", {}).values()
    ]
    hops_total = sum(len(r.get("trajectory", {}).get("hops", {})) for r in scored)
    hops_retried = sum(
        1
        for r in scored
        for hop in r.get("trajectory", {}).get("hops", {}).values()
        if hop.get("retry_count", 0) > 0
    )

    def _rate(key: str) -> float:
        return round(sum(1 for r in scored if r[key]) / len(scored), 4) if scored else 0.0

    return {
        "total": len(scored),
        "answer_em": _rate("answer_em"),
        "answer_f1": _mean([r["answer_f1"] for r in scored]),
        "sp_em": _rate("sp_em"),
        "sp_f1": _mean([r["sp_f1"] for r in scored]),
        "joint_em": _rate("joint_em"),
        "joint_f1": _mean([r["joint_f1"] for r in scored]),
        "decomposer_type_accuracy": _rate("type_correct"),
        "avg_retriever_rounds": _mean(retriever_rounds),
        "pct_hops_retried": round(hops_retried / hops_total, 4) if hops_total else 0.0,
        "errors": len(errors),
        "avg_elapsed_s": round(sum(times) / len(times), 3) if times else 0.0,
        **retry_delta(scored),
    }

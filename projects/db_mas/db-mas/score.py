"""Scores TaskResult JSON records: deterministic set metrics (no LLM) + one
LLM-judge call for qualitative feedback. Produces per-task and aggregate JSON."""
import argparse
import glob
import json
import os
import statistics
from typing import Any, Dict, List

import benchmark
import config
from llm_client import call_llm

JUDGE_CRITERIA = [
    "evidence_grounding",
    "reasoning_soundness",
    "investigation_thoroughness",
    "label_justification",
]

SUBMIT_JUDGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_judgment",
        "description": "Submit your qualitative evaluation of this diagnosis session.",
        "parameters": {
            "type": "object",
            "properties": {
                "evidence_grounding": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning_soundness": {"type": "integer", "minimum": 1, "maximum": 5},
                "investigation_thoroughness": {"type": "integer", "minimum": 1, "maximum": 5},
                "label_justification": {"type": "integer", "minimum": 1, "maximum": 5},
                "feedback": {"type": "string"},
            },
            "required": [*JUDGE_CRITERIA, "feedback"],
            "additionalProperties": False,
        },
    },
}

JUDGE_PROMPT_TEMPLATE = """You are evaluating a multi-agent database root-cause diagnosis session.

Task description:
{task_content}

Ground-truth root cause(s): {gold_root_causes}
Agent's predicted root cause(s): {predicted_root_causes}
Agent's stated reasoning: {reasoning}

Specialist investigation findings (evidence each specialist gathered):
{formatted_specialist_evidence}

Rate the following on a 1-5 integer scale:
1. evidence_grounding: Did specialists cite concrete, real pg_stat_*/pg_locks/pg_indexes query results (numbers, table/index names) rather than vague or fabricated claims?
2. reasoning_soundness: Does the coordinator's final reasoning logically follow from the specialist evidence presented (not from outside knowledge or guessing)?
3. investigation_thoroughness: Did specialists actually run enough distinct queries to support their conclusion, versus a single shallow query?
4. label_justification: For each predicted label, is there a specific piece of cited evidence supporting it (vs. an unsupported label included only to fill number_of_labels_pred)?

Call `submit_judgment` with your scores and a short (2-3 sentence) natural-language feedback string.
"""


def deterministic_metrics(predicted: List[str], gold: List[str]) -> Dict[str, Any]:
    pred_set, gold_set = set(predicted), set(gold)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": pred_set == gold_set,
        "true_positives": sorted(pred_set & gold_set),
        "false_positives": sorted(pred_set - gold_set),
        "false_negatives": sorted(gold_set - pred_set),
    }


def _format_specialist_evidence(findings: List[Dict[str, Any]]) -> str:
    lines = []
    for f in findings:
        lines.append(
            f"- {f.get('label')}: supports_label={f.get('supports_label')}, "
            f"confidence={f.get('confidence')}\n  Evidence: {f.get('evidence')}"
        )
    return "\n".join(lines)


COORDINATION_JUDGE_CRITERIA = ["communication_score", "coordination_score"]

SUBMIT_COORDINATION_JUDGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_coordination_judgment",
        "description": "Submit your evaluation of this session's multi-agent coordination and communication quality.",
        "parameters": {
            "type": "object",
            "properties": {
                "communication_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "coordination_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "coordination_feedback": {"type": "string"},
            },
            "required": [*COORDINATION_JUDGE_CRITERIA, "coordination_feedback"],
            "additionalProperties": False,
        },
    },
}

COORDINATION_JUDGE_PROMPT_TEMPLATE = """You are evaluating the COORDINATION and COMMUNICATION quality of a multi-agent database diagnosis session -- NOT whether the final answer was correct (that is scored separately). This system uses a fixed star topology: 5 specialist agents each investigate one candidate root cause independently in parallel, then report structured findings to a single Coordinator agent. The Coordinator may ask ONE follow-up question total to exactly one specialist (ask_specialist), and may run its own direct verification queries (query_db), before submitting a final verdict (submit_verdict).

Task description:
{task_content}

Specialist reports (what the Coordinator received):
{formatted_specialist_evidence}

Coordinator's own activity this session (its tool calls and their results, in order):
{formatted_coordinator_activity}

Final verdict: {predicted_root_causes}
Final reasoning: {reasoning}

Rate the following on a 1-5 integer scale:
1. communication_score: If the Coordinator used its one ask_specialist follow-up, was the question well-targeted (Clarity), did the specialist's answer add real new information (Information Exchange) that meaningfully informed the final verdict (Task Assistance), and was it used purposefully rather than wastefully (Efficiency)? If it did NOT use the follow-up, was that a reasonable call given the specialist reports already available (5), or a missed opportunity to resolve real conflict/ambiguity between specialists (1-2)?
2. coordination_score: Did the final verdict draw on and synthesize ALL FIVE specialists' reports (and the Coordinator's own query_db checks, if any), rather than ignoring some of them or simply adopting the single highest-confidence specialist's conclusion untouched? Does the reasoning show genuine cross-agent synthesis, or is it a rubber-stamp of one report?

Call `submit_coordination_judgment` with your scores and a short (2-3 sentence) feedback string.
"""


def _format_coordinator_activity(messages: List[Dict[str, Any]]) -> str:
    """Pair each of the Coordinator's tool calls with its matching tool-result
    message (by tool_call_id) so the judge sees exactly what it did and what
    came back, in order."""
    tool_results_by_id = {
        m["tool_call_id"]: m.get("content", "")
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    lines = []
    query_count = 0
    for m in messages:
        for tc in m.get("tool_calls") or []:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            result_text = str(tool_results_by_id.get(tc["id"], ""))
            if name == "query_db":
                query_count += 1
                lines.append(f"- query_db #{query_count}: {args[:200]}\n  Result: {result_text[:300]}")
            elif name == "ask_specialist":
                lines.append(f"- ask_specialist: {args[:200]}\n  Answer: {result_text[:500]}")
            elif name == "submit_verdict":
                lines.append(f"- submit_verdict: {args[:300]}")
    return "\n".join(lines) if lines else "(no tool calls recorded)"


def judge_coordination(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    task_content = task["task"]["content"]
    findings = result.get("transcript", {}).get("findings", [])
    coordinator_messages = result.get("transcript", {}).get("coordinator", [])

    prompt = COORDINATION_JUDGE_PROMPT_TEMPLATE.format(
        task_content=task_content,
        formatted_specialist_evidence=_format_specialist_evidence(findings),
        formatted_coordinator_activity=_format_coordinator_activity(coordinator_messages),
        predicted_root_causes=result.get("predicted_root_causes", []),
        reasoning=result.get("reasoning", ""),
    )
    messages = [{"role": "user", "content": prompt}]
    llm_result = call_llm(
        messages,
        tools=[SUBMIT_COORDINATION_JUDGMENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_coordination_judgment"}},
        temperature=0.0,
    )
    if llm_result.tool_calls:
        args = llm_result.tool_calls[0].arguments
        return {
            **{k: args.get(k) for k in COORDINATION_JUDGE_CRITERIA},
            "coordination_feedback": args.get("coordination_feedback", ""),
        }
    return {
        **{k: None for k in COORDINATION_JUDGE_CRITERIA},
        "coordination_feedback": "Judge failed to produce a rating.",
    }


def judge_task(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    task_content = task["task"]["content"]
    gold_root_causes = task["task"]["root_causes"]
    findings = result.get("transcript", {}).get("findings", [])

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_content=task_content,
        gold_root_causes=gold_root_causes,
        predicted_root_causes=result.get("predicted_root_causes", []),
        reasoning=result.get("reasoning", ""),
        formatted_specialist_evidence=_format_specialist_evidence(findings),
    )
    messages = [{"role": "user", "content": prompt}]
    llm_result = call_llm(
        messages,
        tools=[SUBMIT_JUDGMENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_judgment"}},
        temperature=0.0,
    )
    if llm_result.tool_calls:
        args = llm_result.tool_calls[0].arguments
        return {
            **{k: args.get(k) for k in JUDGE_CRITERIA},
            "feedback": args.get("feedback", ""),
        }
    return {**{k: None for k in JUDGE_CRITERIA}, "feedback": "Judge failed to produce a rating."}


def score_task(
    task: Dict[str, Any], result: Dict[str, Any], evaluate_coordination: bool = True
) -> Dict[str, Any]:
    predicted = result.get("predicted_root_causes", [])
    gold = task["task"]["root_causes"]
    det = deterministic_metrics(predicted, gold)
    judge = judge_task(task, result)
    # Separate, optional LLM-judge pass on the coordination/communication process
    # itself (as opposed to `judge` above, which rates the answer) -- kept as its
    # own call so it can be toggled off (--no-coordination-eval) without touching
    # the always-on task-performance judging.
    coordination = judge_coordination(task, result) if evaluate_coordination else None
    return {
        "task_id": task["task_id"],
        "deterministic": det,
        "judge": judge,
        "coordination": coordination,
        "meta": {
            "gold_root_causes": gold,
            "predicted_root_causes": predicted,
            "number_of_labels_pred": task["task"]["number_of_labels_pred"],
            "forced_fallback": result.get("forced_fallback", False),
            "validation_error": result.get("validation_error"),
            "error": result.get("error"),
            "timing": result.get("timing", {}),
            "token_usage": result.get("token_usage", {}),
        },
    }


def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scored)
    if n == 0:
        return {"n_tasks": 0}

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    # Tasks that failed to score (e.g. a judge-call error) have deterministic/judge
    # set to None -- exclude them from means rather than crashing on them. Same for
    # "coordination": None whenever --no-coordination-eval was used or a task's
    # judge_coordination call itself failed.
    valid_det = [s for s in scored if s.get("deterministic") is not None]
    valid_judge = [s for s in scored if s.get("judge") is not None]
    valid_coord = [s for s in scored if s.get("coordination") is not None]

    # Per-label recall: for each label, the fraction of tasks where it was a *true*
    # root cause that got actually predicted. Computed directly from each task's
    # gold/predicted sets -- reusing a task's overall recall for every one of its
    # gold labels (as an earlier version did) would conflate multi-label tasks: a
    # task with 2 gold causes where only one was found has recall 0.5, but that 0.5
    # would wrongly get attributed to the label that WAS found too.
    label_gold_count: Dict[str, int] = {}
    label_found_count: Dict[str, int] = {}
    for s in valid_det:
        gold = set(s["meta"].get("gold_root_causes", []))
        predicted = set(s["meta"].get("predicted_root_causes", []))
        for label in gold:
            label_gold_count[label] = label_gold_count.get(label, 0) + 1
            if label in predicted:
                label_found_count[label] = label_found_count.get(label, 0) + 1
    recall_by_anomaly_type = {
        label: label_found_count.get(label, 0) / count
        for label, count in label_gold_count.items()
    }

    return {
        "n_tasks": n,
        "n_scoring_errors": n - len(valid_det),
        "mean_precision": mean(s["deterministic"]["precision"] for s in valid_det),
        "mean_recall": mean(s["deterministic"]["recall"] for s in valid_det),
        "mean_f1": mean(s["deterministic"]["f1"] for s in valid_det),
        "exact_match_rate": mean(1.0 if s["deterministic"]["exact_match"] else 0.0 for s in valid_det),
        "mean_judge": {crit: mean(s["judge"].get(crit) for s in valid_judge) for crit in JUDGE_CRITERIA},
        "recall_by_anomaly_type": recall_by_anomaly_type,
        "mean_coordination": (
            {crit: mean(s["coordination"].get(crit) for s in valid_coord) for crit in COORDINATION_JUDGE_CRITERIA}
            if valid_coord
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=config.RESULTS_RAW_DIR)
    parser.add_argument("--out-dir", default=config.RESULTS_SCORED_DIR)
    parser.add_argument("--summary-out", default=config.RESULTS_SUMMARY_PATH)
    parser.add_argument(
        "--coordination-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run a separate LLM-judge pass scoring coordination/communication "
        "quality (the Coordinator's use of ask_specialist/query_db and synthesis "
        "across specialists), adapted from MARBLE's Communication/Planning rubrics. "
        "Use --no-coordination-eval to skip it (saves one LLM call per task).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_tasks_by_id = {t["task_id"]: t for t in benchmark.load_all_tasks()}

    scored_all = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        task_id = result["task_id"]
        try:
            task = all_tasks_by_id[task_id]
            scored = score_task(task, result, evaluate_coordination=args.coordination_eval)
        except Exception as e:  # noqa: BLE001 - one bad task must not sink the whole batch
            print(f"[score] task {task_id} FAILED to score: {e}")
            scored = {
                "task_id": task_id,
                "deterministic": None,
                "judge": None,
                "coordination": None,
                "meta": {"scoring_error": str(e)},
            }
        scored_all.append(scored)
        out_path = os.path.join(args.out_dir, f"{task_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)
        print(f"[score] task {task_id} -> {out_path}")

    summary = aggregate(scored_all)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[score] summary -> {args.summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

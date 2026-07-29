"""Step 2 verification: run one task end-to-end (real LLM) and score it."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mas_workflow
import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=1)
    args = parser.parse_args()

    result = mas_workflow.run_task(mas_workflow.benchmark.load_tasks([args.task_id])[0])

    print("\n--- Result ---")
    print(f"predicted_root_causes: {result.predicted_root_causes}")
    print(f"root_causes (gold):    {result.root_causes}")
    print(f"reasoning: {result.reasoning}")
    print(f"forced_fallback: {result.forced_fallback}, validation_error: {result.validation_error}")

    task = mas_workflow.benchmark.load_tasks([args.task_id])[0]
    result_dict = {
        "task_id": result.task_id,
        "predicted_root_causes": result.predicted_root_causes,
        "reasoning": result.reasoning,
        "transcript": result.transcript,
        "forced_fallback": result.forced_fallback,
        "validation_error": result.validation_error,
        "timing": result.timing,
        "token_usage": result.token_usage,
    }
    scored = score.score_task(task, result_dict)
    print("\n--- Score ---")
    import json

    print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()

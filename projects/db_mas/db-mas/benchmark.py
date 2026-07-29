"""Loader for the 100-task database diagnosis benchmark (database_main.jsonl)."""
import json
from typing import Any, Dict, List, Optional

import config


def load_all_tasks() -> List[Dict[str, Any]]:
    tasks = []
    with open(config.BENCHMARK_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_tasks(task_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Load tasks by task_id, preserving the order of `task_ids`. If task_ids is
    None, returns all 100 tasks in file order."""
    all_tasks = load_all_tasks()
    if task_ids is None:
        return all_tasks
    by_id = {t["task_id"]: t for t in all_tasks}
    missing = [tid for tid in task_ids if tid not in by_id]
    if missing:
        raise ValueError(f"task_id(s) not found in benchmark: {missing}")
    return [by_id[tid] for tid in task_ids]

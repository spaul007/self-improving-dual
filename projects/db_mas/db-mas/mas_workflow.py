"""Orchestrates the fixed Coordinator + 5-Specialists star topology over the
database root-cause diagnosis benchmark, one Postgres container per task."""
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import benchmark
import config
from agents.coordinator.workflow import CoordinatorAgent
from agents.specialists import build_specialists
from environment.task_setup import setup_task_environment, teardown_task_environment


@dataclass
class TaskResult:
    task_id: int
    predicted_root_causes: List[str]
    reasoning: str
    root_causes: List[str]
    number_of_labels_pred: int
    transcript: Dict[str, Any]
    token_usage: Dict[str, Any]
    timing: Dict[str, float]
    forced_fallback: bool = False
    validation_error: Optional[str] = None
    error: Optional[str] = None


def _aggregate_usage(specialists, coordinator) -> Dict[str, Any]:
    per_specialist = {s.agent_id: dict(s.usage) for s in specialists}
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for s in specialists:
        for k in total:
            total[k] += s.usage.get(k, 0)
    for k in total:
        total[k] += coordinator.usage.get(k, 0)
    return {"specialists": per_specialist, "coordinator": dict(coordinator.usage), "total": total}


def run_task(
    task: Dict[str, Any], project_name_suffix: str = "", port: Optional[int] = None
) -> TaskResult:
    """Run one task: fixed environment setup (not this function's concern, see
    environment/task_setup.py), then the harness -- build & run the 5
    specialists, build & run the Coordinator, assemble the result. Everything
    from here down to the result assembly *is* the harness: the multi-agent
    system being run (and, if tuning this benchmark, the part that's fair
    game to change) -- as opposed to the environment setup/teardown calls
    bracketing it, which define the task itself and must not be touched.
    """
    task_id = task["task_id"]
    project_name = f"db_mas_{task_id}{project_name_suffix}"
    timing: Dict[str, float] = {}
    t_start = time.time()

    try:
        # --- fixed setup: not part of the harness (environment/task_setup.py) ---
        timing.update(setup_task_environment(task, project_name, port))

        # --- harness starts here ---
        task_content = task["task"]["content"]
        labels = task["task"]["labels"]
        number_of_labels_pred = task["task"]["number_of_labels_pred"]
        root_causes = task["task"]["root_causes"]

        t0 = time.time()
        specialists = build_specialists(task_content)
        with ThreadPoolExecutor(max_workers=len(specialists)) as ex:
            futures = {ex.submit(s.run): s for s in specialists}
            for fut in as_completed(futures):
                fut.result()  # each specialist stores its own .findings
        timing["specialists_s"] = time.time() - t0

        findings = [s.findings for s in specialists]
        specialists_by_id = {s.agent_id: s for s in specialists}

        t0 = time.time()
        coordinator = CoordinatorAgent(
            task_content=task_content,
            labels=labels,
            number_of_labels_pred=number_of_labels_pred,
            specialist_findings=findings,
            specialists_by_id=specialists_by_id,
        )
        verdict = coordinator.run()
        timing["coordinator_s"] = time.time() - t0

        transcript = {
            "specialists": {s.agent_id: s.messages for s in specialists},
            "coordinator": coordinator.messages,
            "findings": [asdict(f) for f in findings],
        }
        token_usage = _aggregate_usage(specialists, coordinator)

        result = TaskResult(
            task_id=task_id,
            predicted_root_causes=verdict.predicted_root_causes,
            reasoning=verdict.reasoning,
            root_causes=root_causes,
            number_of_labels_pred=number_of_labels_pred,
            transcript=transcript,
            token_usage=token_usage,
            timing=timing,
            forced_fallback=verdict.forced_fallback,
            validation_error=verdict.validation_error,
        )
    except Exception as e:  # noqa: BLE001 - record the failure per-task rather than aborting a batch
        result = TaskResult(
            task_id=task_id,
            predicted_root_causes=[],
            reasoning="",
            root_causes=task.get("task", {}).get("root_causes", []),
            number_of_labels_pred=task.get("task", {}).get("number_of_labels_pred", 0),
            transcript={},
            token_usage={},
            timing=timing,
            error=str(e),
        )
    finally:
        timing["total_s"] = time.time() - t_start
        # --- fixed teardown: not part of the harness (environment/task_setup.py) ---
        teardown_task_environment(project_name)

    os.makedirs(config.RESULTS_RAW_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_RAW_DIR, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[mas_workflow] task {task_id} done in {timing.get('total_s', 0):.1f}s -> {out_path}")
    return result


_worker_slot_queue = None  # set once per worker process by _init_worker


def _init_worker(slot_queue) -> None:
    global _worker_slot_queue
    _worker_slot_queue = slot_queue


def _run_task_in_slot(task: Dict[str, Any]) -> TaskResult:
    """ProcessPoolExecutor target: block for a free port slot, run the task on
    that port, then release the slot for the next task. Must be a top-level
    function (not a closure/method) so it's picklable for the pool."""
    slot = _worker_slot_queue.get()
    try:
        return run_task(task, port=config.PARALLEL_DB_PORT_BASE + slot)
    finally:
        _worker_slot_queue.put(slot)


def run_many(
    task_ids: Optional[List[int]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_workers: int = 1,
) -> List[TaskResult]:
    """Run a batch of tasks. max_workers=1 (default) runs strictly sequentially
    on the default port (unchanged from before). max_workers>1 runs up to that
    many tasks concurrently, each in its own OS process with its own Postgres
    container on a distinct port (config.PARALLEL_DB_PORT_BASE + slot)."""
    if tasks is None:
        tasks = benchmark.load_tasks(task_ids)

    if max_workers <= 1:
        return [run_task(t) for t in tasks]

    results: List[TaskResult] = []
    # `with Manager()` ensures the manager process (hosting slot_queue) is always
    # shut down when this function returns -- without it, calling run_many
    # repeatedly in one long-lived process would leak a manager process per call.
    with multiprocessing.Manager() as manager:
        slot_queue = manager.Queue()
        for slot in range(max_workers):
            slot_queue.put(slot)

        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_worker, initargs=(slot_queue,)
        ) as ex:
            futures = {ex.submit(_run_task_in_slot, t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001 - one task's worker crash must not sink the batch
                    print(f"[mas_workflow] task {task['task_id']} raised in worker: {e}")
    return results

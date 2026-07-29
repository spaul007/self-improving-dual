"""Orchestration of the sequential predictor -> reflector math MAS.

    question
       |
       v
  [predictor]  full solution
       |
       v
  compress (mutable tool)     <- skipped if MAS_USE_COMPRESSED_CONTEXT=0
       |
       v
  [reflector]  critique + corrected answer  ==  MAS answer

`run_task` handles one problem; `run_many` runs a batch with bounded
concurrency.
"""

import asyncio
import time
from typing import Any, Callable

import config
from agents.predictor.workflow import PredictorAgent
from agents.reflector.workflow import ReflectorAgent
from tools.mutable.compress import compress


def build_mas() -> tuple[PredictorAgent, ReflectorAgent]:
    """Instantiate the fixed two-agent pipeline."""
    return PredictorAgent(), ReflectorAgent()


async def run_task(item: dict[str, Any]) -> dict[str, Any]:
    """Run the full MAS on one problem and return a raw result record.

    Never raises: a failed task returns a record with `error` set so a batch run
    is not lost to one bad sample.
    """
    unique_id = str(item.get("unique_id", ""))
    question = item[config.PROBLEM_KEY]
    started = time.time()

    predictor, reflector = build_mas()

    try:
        pred_out = await predictor.arun(question)

        if config.USE_COMPRESSED_CONTEXT:
            pred_out.short = await compress(pred_out.raw)
            context = pred_out.short
        else:
            context = pred_out.raw

        refl_out = await reflector.arun(question, context)

        return {
            "unique_id": unique_id,
            "problem": question,
            "gold_answer": item.get(config.ANSWER_KEY, ""),
            "prediction": refl_out.answer,
            "final_raw": refl_out.raw,
            "predictor_answer": pred_out.answer,
            "trajectory": [pred_out.to_dict(), refl_out.to_dict()],
            "elapsed_s": round(time.time() - started, 3),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
        return {
            "unique_id": unique_id,
            "problem": question,
            "gold_answer": item.get(config.ANSWER_KEY, ""),
            "prediction": "",
            "final_raw": "",
            "predictor_answer": "",
            "trajectory": [],
            "elapsed_s": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_many(
    items: list[dict[str, Any]],
    max_concurrent: int | None = None,
    on_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the MAS over many problems, at most `max_concurrent` at a time.

    Results come back in the same order as `items`. `on_done` fires as each task
    finishes (useful for progress output).
    """
    limit = max_concurrent or config.MAX_CONCURRENT_TASKS
    sem = asyncio.Semaphore(limit)

    async def _guarded(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await run_task(item)
            if on_done is not None:
                on_done(result)
            return result

    return await asyncio.gather(*(_guarded(item) for item in items))

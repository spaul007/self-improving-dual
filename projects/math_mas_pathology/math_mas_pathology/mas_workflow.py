"""Orchestration of the pathological predictor -> verifier -> reflector math MAS.

    question
       |
       v
  [predictor]  full solution
       |
       v
  compress -> first_draft              <- also pathology 2's "stale" artifact,
       |                                  kept immutable for the rest of the run
       v
  [verifier] x N_ROUNDS, IDENTICAL (question, first_draft) every turn    -- PATHOLOGY 1
       |      (repetition-then-ignore: turns 1..N-1 are computed and kept
       |       for instrumentation, but only the last turn is ever used)
       v
  verifier_final_context = compress(verifier turn N's raw output)
       |
       v
  context_for_reflector = first_draft, NOT verifier_final_context        -- PATHOLOGY 2
       |      (stale context injection: the verifier's real, freshest
       |       conclusion is computed but never reaches the reflector)
       v
  [reflector]  (deafens its context internally -- PATHOLOGY 3, see
       |        agents/reflector/workflow.py) critique + corrected answer
       v
  == MAS answer

Each pathology is independently toggleable (MAS_ENABLE_REPETITION_PATHOLOGY,
MAS_ENABLE_STALE_CONTEXT_PATHOLOGY, MAS_ENABLE_SELECTIVE_DEAFNESS) and
defaults ON. See README.md "Communication Pathologies" for the full
rationale. `run_task` handles one problem; `run_many` runs a batch with
bounded concurrency.
"""

import asyncio
import time
from typing import Any, Callable

import config
from agents.predictor.workflow import PredictorAgent
from agents.reflector.workflow import ReflectorAgent
from agents.verifier.workflow import VerifierAgent, VerifierResult
from tools.mutable.compress import compress


def build_mas() -> tuple[PredictorAgent, VerifierAgent, ReflectorAgent]:
    """Instantiate the fixed three-agent pipeline."""
    return PredictorAgent(), VerifierAgent(), ReflectorAgent()


def _pathology_flags(n_rounds: int) -> dict[str, Any]:
    return {
        "repetition": config.ENABLE_REPETITION_PATHOLOGY,
        "stale_context": config.ENABLE_STALE_CONTEXT_PATHOLOGY,
        "selective_deafness": config.ENABLE_SELECTIVE_DEAFNESS,
        "verify_rounds": n_rounds,
    }


async def run_task(item: dict[str, Any]) -> dict[str, Any]:
    """Run the full MAS on one problem and return a raw result record.

    Never raises: a failed task returns a record with `error` set so a batch run
    is not lost to one bad sample.
    """
    unique_id = str(item.get("unique_id", ""))
    question = item[config.PROBLEM_KEY]
    started = time.time()

    predictor, verifier, reflector = build_mas()
    n_rounds = config.VERIFY_ROUNDS if config.ENABLE_REPETITION_PATHOLOGY else 1

    try:
        pred_out = await predictor.arun(question)

        if config.USE_COMPRESSED_CONTEXT:
            pred_out.short = await compress(pred_out.raw)
            first_draft = pred_out.short
        else:
            first_draft = pred_out.raw

        # PATHOLOGY 1: ask the verifier the exact same question `n_rounds`
        # times (byte-identical `first_draft` input every turn); only the
        # last turn is ever used below.
        verifier_result: VerifierResult = await verifier.arun_repeated(question, first_draft, n_rounds)
        verifier_final = verifier_result.final

        if config.USE_COMPRESSED_CONTEXT:
            verifier_final_context = await compress(verifier_final.raw)
        else:
            verifier_final_context = verifier_final.raw

        # PATHOLOGY 2: hand the reflector the predictor's original,
        # pre-verification draft instead of the verifier's real conclusion
        # (computed just above) when the pathology is active.
        context_for_reflector = (
            first_draft if config.ENABLE_STALE_CONTEXT_PATHOLOGY else verifier_final_context
        )

        # PATHOLOGY 3 (selective deafness) is applied inside
        # ReflectorAgent.build_prompt itself, on whatever context it's given.
        refl_out = await reflector.arun(question, context_for_reflector)

        return {
            "unique_id": unique_id,
            "problem": question,
            "gold_answer": item.get(config.ANSWER_KEY, ""),
            "prediction": refl_out.answer,
            "final_raw": refl_out.raw,
            "predictor_answer": pred_out.answer,
            "verifier_answer": verifier_final.answer,
            "trajectory": [pred_out.to_dict(), verifier_result.to_dict(), refl_out.to_dict()],
            "first_draft": first_draft,
            "verifier_final_context": verifier_final_context,
            "context_used_by_reflector": context_for_reflector,
            "pathology_flags": _pathology_flags(n_rounds),
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
            "verifier_answer": "",
            "trajectory": [],
            "first_draft": "",
            "verifier_final_context": "",
            "context_used_by_reflector": "",
            "pathology_flags": _pathology_flags(n_rounds),
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

"""Orchestration of the parallel-investigators -> lead-DBA database RCA MAS.

    problem  (+ its recorded DB snapshot, bound per-task)
       |
       +---------+---------+---------+---------+
       v         v         v         v         v
    [insert]  [lock]   [vacuum]  [index]   [fetch]     5 investigators,
       |         |         |         |         |       each in a query_db
       v         v         v         v         v       ReAct loop
    compress  compress  compress  compress  compress   <- skipped if
       |         |         |         |         |          MAS_USE_COMPRESSED_CONTEXT=0
       +---------+---------+----+----+---------+
                                |  labeled briefings
                                v
                           [lead DBA]  (no tools)
                                |
                                v
                 diagnosis ending "FINAL: <LABEL>[, ...]"  ==  MAS answer

`run_task` handles one problem; `run_many` runs a batch with bounded
concurrency. Each task's query_db calls replay from its own snapshot,
selected via a ContextVar so concurrent tasks never collide.
"""

import asyncio
import time
from typing import Any, Callable

import config
from agents.fetch_investigator import prompt as fetch_prompt
from agents.fetch_investigator.workflow import FetchInvestigatorAgent
from agents.index_investigator import prompt as index_prompt
from agents.index_investigator.workflow import IndexInvestigatorAgent
from agents.insert_investigator import prompt as insert_prompt
from agents.insert_investigator.workflow import InsertInvestigatorAgent
from agents.lead_dba.workflow import LeadDBAAgent
from agents.lock_investigator import prompt as lock_prompt
from agents.lock_investigator.workflow import LockInvestigatorAgent
from agents.vacuum_investigator import prompt as vacuum_prompt
from agents.vacuum_investigator.workflow import VacuumInvestigatorAgent
from tools.immutable.label_extraction import extract_labels
from tools.immutable.query_db import load_db_cache, reset_db_cache, set_db_cache
from tools.mutable.compress import compress

# Canonical stage-1 roster: (agent class, its assigned candidate root cause),
# in MASPO's investigator order.
INVESTIGATORS: list[tuple[type, str]] = [
    (InsertInvestigatorAgent, insert_prompt.CANDIDATE),
    (LockInvestigatorAgent, lock_prompt.CANDIDATE),
    (VacuumInvestigatorAgent, vacuum_prompt.CANDIDATE),
    (IndexInvestigatorAgent, index_prompt.CANDIDATE),
    (FetchInvestigatorAgent, fetch_prompt.CANDIDATE),
]


def build_mas() -> tuple[list[Any], LeadDBAAgent]:
    """Instantiate the fixed 5-investigators + lead pipeline."""
    return [cls() for cls, _ in INVESTIGATORS], LeadDBAAgent()


def _snapshot_path(unique_id: str) -> str:
    return str(config.DB_CACHE_DIR / f"{unique_id}.json")


async def run_task(item: dict[str, Any]) -> dict[str, Any]:
    """Run the full MAS on one problem and return a raw result record.

    Never raises: a failed task returns a record with `error` set so a batch run
    is not lost to one bad sample.
    """
    unique_id = str(item.get(config.ID_KEY, ""))
    question = item[config.PROBLEM_KEY]
    started = time.time()

    investigators, lead = build_mas()

    # Bind this task's recorded snapshot for every query_db call made below.
    snapshot = load_db_cache(_snapshot_path(unique_id))
    snapshot_found = bool(snapshot.get("queries") or snapshot.get("tables"))
    token = set_db_cache(snapshot)
    try:
        inv_outs = await asyncio.gather(*(inv.arun(question) for inv in investigators))

        if config.USE_COMPRESSED_CONTEXT:
            shorts = await asyncio.gather(*(compress(out.raw) for out in inv_outs))
            for out, short in zip(inv_outs, shorts):
                out.short = short
        briefings = [
            out.short if config.USE_COMPRESSED_CONTEXT else out.raw
            for out in inv_outs
        ]

        # Labeled hand-off so the lead can attribute evidence to a candidate.
        context = "\n---\n".join(
            f"[Investigator {i+1} — {candidate}]\n{briefing}"
            for i, ((_, candidate), briefing) in enumerate(zip(INVESTIGATORS, briefings))
        )

        lead_out = await lead.arun(question, context)

        return {
            "unique_id": unique_id,
            "problem": question,
            "root_causes": item.get(config.GOLD_KEY, []),
            "labels": item.get(config.LABELS_KEY, config.LABELS),
            "number_of_labels_pred": item.get("number_of_labels_pred"),
            "prediction": lead_out.raw,
            # Informational parse of the FINAL: line (evaluate.py recomputes the
            # authoritative scoring with the same immutable extractor).
            "predicted_labels": extract_labels(
                lead_out.raw, item.get(config.LABELS_KEY, config.LABELS)
            ),
            "snapshot_found": snapshot_found,
            "trajectory": [out.to_dict() for out in inv_outs] + [lead_out.to_dict()],
            "elapsed_s": round(time.time() - started, 3),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
        return {
            "unique_id": unique_id,
            "problem": question,
            "root_causes": item.get(config.GOLD_KEY, []),
            "labels": item.get(config.LABELS_KEY, config.LABELS),
            "number_of_labels_pred": item.get("number_of_labels_pred"),
            "prediction": "",
            "predicted_labels": [],
            "snapshot_found": snapshot_found,
            "trajectory": [],
            "elapsed_s": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        reset_db_cache(token)


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

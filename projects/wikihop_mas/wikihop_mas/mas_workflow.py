"""Orchestration of the wikihop MAS's non-sequential, controller-driven topology.

    question
       |
       v
  [decomposer]  classifies type, emits hop-plan (independent | dependent)
       |
       +-- independent (comparison / bridge_comparison) --> run hop1, then hop2
       |                                                     (order-agnostic, no concurrency)
       +-- dependent (inference / compositional) ----------> run hop1, substitute {hop1_answer}
                                                               into hop2's text, then run hop2
       |
       v
  [concluder] phase 1 -- final_answer + per-hop grounding judgment
       |
       +-- any hop ungrounded? --> rerun ONLY that hop (bounded: config.MAX_HOP_RETRIES)
       |                           --> [concluder] phase 2 (mandatory, final, accepted unconditionally)
       v
  final_answer  ==  MAS answer

Each hop is [retriever] (search_context tool) -> [extractor] -> deterministic
grounding_check.verify_quote. Fully synchronous -- no asyncio anywhere.
`run_task` handles one question; `run_many` runs a batch via a plain
ThreadPoolExecutor for throughput (unrelated to the intra-question topology).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Callable

import config
from agents.concluder.workflow import ConcluderAgent
from agents.decomposer.workflow import DecomposerAgent, parse_hop_plan
from agents.extractor.workflow import ExtractorAgent
from agents.retriever.workflow import RetrieverAgent
from mas_state import ConcluderCall, HopResult, MASState, SubQuestion
from tools.immutable import grounding_check, search_context
from tools.immutable.answer_extraction import extract_final_answer
from tools.mutable import entity_substitution


def build_state(item: dict[str, Any]) -> MASState:
    return MASState(
        unique_id=str(item.get("unique_id", "")),
        question=item[config.QUESTION_KEY],
        gold_answer=item.get(config.ANSWER_KEY, ""),
        gold_type=item.get(config.TYPE_KEY, ""),
        context_paragraphs=item["context_paragraphs"],
        gold_supporting_facts=item.get("supporting_facts", []),
        evidences=item.get("evidences", []),
    )


def run_task(item: dict[str, Any]) -> dict[str, Any]:
    """Run the full MAS on one question and return a raw result record.

    Never raises: a failed task returns a record with `error` set so a batch
    run is not lost to one bad sample.
    """
    state = build_state(item)
    started = time.time()
    try:
        index = search_context.build_index(state.context_paragraphs)

        decomposer_out = DecomposerAgent().run(state.question)
        state.hop_plan = parse_hop_plan(decomposer_out, state.question)
        if config.ORACLE_TYPE and state.gold_type:
            # Ablation: force the independent/dependent dispatch from the
            # dataset's gold `type` instead of the Decomposer's own
            # classification, isolating how much the rest of the pipeline
            # depends on the Decomposer having classified correctly.
            # `predicted_type` (used for decomposer_type_accuracy) is left
            # untouched -- only the *dispatch* is overridden.
            state.hop_plan.dependency = (
                "dependent" if state.gold_type in config.DEPENDENT_TYPES else "independent"
            )

        if state.hop_plan.dependency == "independent":
            for sq in state.hop_plan.sub_questions:          # plain loop, order doesn't matter
                state.hops[sq.hop_id] = _run_hop(state, sq, index)
        else:
            sub_questions = state.hop_plan.sub_questions
            # Select hop1 by its actual "no dependency" marker, not by list
            # position -- trusting position would silently chain the wrong
            # hop if the Decomposer ever orders sub_questions differently.
            independents = [sq for sq in sub_questions if sq.depends_on is None]
            sq1 = independents[0] if independents else sub_questions[0]
            hop1 = _run_hop(state, sq1, index)
            state.hops[sq1.hop_id] = hop1
            for sq2 in sub_questions:
                if sq2.hop_id == sq1.hop_id:
                    continue
                sq2_text = entity_substitution.substitute(sq2.text, hop1.extractor_answer)
                state.hops[sq2.hop_id] = _run_hop(state, replace(sq2, text=sq2_text), index)

        _run_concluder_with_bounded_retry(state, index)
    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
        state.error = f"{type(exc).__name__}: {exc}"

    state.elapsed_s = round(time.time() - started, 3)
    return state.to_record()


def _run_hop(state: MASState, sub_question: SubQuestion, index, retry_hint: str = "") -> HopResult:
    retriever_out = RetrieverAgent().run(sub_question.text, index, retry_hint)
    extractor_out = ExtractorAgent().run(sub_question.text, retriever_out.retrieved_paragraphs)
    parsed = extractor_out.parsed if "_parse_error" not in extractor_out.parsed else {}
    quote = str(parsed.get("quote", ""))
    verified = grounding_check.verify_quote(quote, state.context_paragraphs)

    # Prefer the deterministically-verified location over the model's
    # self-reported one -- verify_quote() found the quote's *actual* place in
    # the context, which is strictly more trustworthy than the model's claim
    # (which can name the right quote but the wrong title/sent_id). Only fall
    # back to the model's claim when the quote couldn't be verified at all,
    # since supporting-fact scoring depends directly on this being accurate.
    source = None
    if verified["verified"]:
        source = {"title": verified["title"], "sent_id": verified["sent_id"]}
    elif parsed.get("source_title") is not None:
        source = {"title": parsed.get("source_title"), "sent_id": parsed.get("source_sent_id")}

    return HopResult(
        hop_id=sub_question.hop_id,
        sub_question=sub_question.text,
        retriever_trace=retriever_out.trace,
        retriever_raw=retriever_out.raw,
        retrieved_paragraphs=retriever_out.retrieved_paragraphs,
        retriever_rounds_used=retriever_out.rounds_used,
        extractor_raw=extractor_out.raw,
        extractor_answer=str(parsed.get("answer", "")),
        extractor_quote=quote,
        extractor_source=source,
        quote_verified=verified["verified"],
    )


def _run_concluder_with_bounded_retry(state: MASState, index) -> None:
    """Calls the Concluder at most twice -- a structural cap, not a while-loop
    gated on model output. Only the specific ungrounded hop(s) are rerun, and
    the second Concluder call's output is accepted unconditionally."""
    concluder = ConcluderAgent()

    out1 = concluder.run(state.question, state.hops)
    state.concluder_rounds = 1
    parsed1 = out1.parsed if "_parse_error" not in out1.parsed else {}
    final_answer_1 = extract_final_answer(out1.raw, parsed1)
    state.final_answer_pre_retry = final_answer_1
    state.concluder_calls.append(ConcluderCall(
        phase=1, raw=out1.raw, tool_trace=out1.trace,
        hop_grounding=parsed1.get("hop_grounding", []) or [],
        final_answer=final_answer_1, reasoning=str(parsed1.get("reasoning", "")),
    ))
    _apply_grounding(state, parsed1.get("hop_grounding", []) or [])

    ungrounded = [h for h in state.hops.values() if h.llm_grounded is False]
    retried_any = False
    for hop in ungrounded:
        if hop.retry_count >= config.MAX_HOP_RETRIES:
            continue
        hop.retry_count += 1
        retried_any = True
        state.hops[hop.hop_id] = _run_hop(
            state, SubQuestion(hop_id=hop.hop_id, text=hop.sub_question), index,
            retry_hint="Previous evidence looked insufficient for this specific question; "
                       "broaden or rephrase the search query.",
        )
        state.hops[hop.hop_id].retry_count = hop.retry_count

    if retried_any:
        out2 = concluder.run(state.question, state.hops)
        state.concluder_rounds = 2
        parsed2 = out2.parsed if "_parse_error" not in out2.parsed else {}
        final_answer_2 = extract_final_answer(out2.raw, parsed2)
        state.concluder_calls.append(ConcluderCall(
            phase=2, raw=out2.raw, tool_trace=out2.trace,
            hop_grounding=parsed2.get("hop_grounding", []) or [],
            final_answer=final_answer_2, reasoning=str(parsed2.get("reasoning", "")),
        ))
        state.final_answer = final_answer_2               # accepted unconditionally
    else:
        state.final_answer = final_answer_1


def _apply_grounding(state: MASState, hop_grounding: list[dict]) -> None:
    for entry in hop_grounding:
        if not isinstance(entry, dict):
            continue
        try:
            # JSON hop_ids can come back as strings ("1") from some models;
            # state.hops is always keyed by int (parse_hop_plan casts it), so
            # an uncoerced lookup would silently miss and never mark the hop
            # ungrounded -- quietly disabling the retry loop for it.
            hop_id = int(entry.get("hop_id"))
        except (TypeError, ValueError):
            continue
        hop = state.hops.get(hop_id)
        if hop is not None:
            hop.llm_grounded = bool(entry.get("grounded", True))


def run_many(
    items: list[dict[str, Any]],
    on_done: Callable[[dict[str, Any]], None] | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Simple batch runner for throughput across independent questions.

    Plain stdlib ThreadPoolExecutor (LLM calls are I/O-bound) -- no asyncio.
    Results come back in the same order as `items`.
    """
    workers = max_workers or config.MAX_CONCURRENT_TASKS
    results: list[dict[str, Any] | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_task, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            if on_done is not None:
                on_done(results[i])
    return results

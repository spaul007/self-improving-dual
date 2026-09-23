"""Context compaction: summarise the old transcript, keep the recent tail verbatim.

WHY THIS REPLACES TRUNCATION
    v7 and earlier bounded a role's conversation with `blackboard.trim_history()`, which
    clipped OLD tool output to 1,500 chars and kept every step forever. That preserves the
    SHAPE of the history at degraded fidelity: the role can see that it ran a build, but not
    what the build actually said. A 1,500-char clip of a compiler log routinely drops the one
    error line that mattered.

    Claude Code does the opposite, and its raw session file says so precisely:

        "compactMetadata": {"trigger": "auto", "preTokens": 167387, "durationMs": 478376,
                            "preservedMessages": {"uuids": [ ...7 records... ]}}

    i.e. at ~167K tokens it summarises the transcript into a structured summary, PRESERVES THE
    LAST ~7 RECORDS VERBATIM, and continues in the same thread. Measured across 60 trials:
    88% compacted, median 1 compaction, peak context 165,904 tokens.

    So: full fidelity recently, a structured reconstruction of everything older. That is the
    trade this module implements.

⚠ COST, MEASURED, NOT ESTIMATED
    That compaction took durationMs = 478,376 -- **8.0 minutes** of LLM time on a 167K-token
    prompt. Against seedling's ~175-minute working window that is ~4.5% per compaction. Budget
    1-2 per PATCH role and LOG EVERY ONE: if compaction starts dominating, the fix is a higher
    threshold or smaller Read windows, not silence.

OBSERVABILITY IS NOT OPTIONAL HERE
    This mechanism only touches `messages`, which no artifact records. That exact shape has
    already produced two unfalsifiable criteria in this project (the forced-finish prompt, and
    v7's `[prior reasoning]` note -- both greppable for zero hits while running correctly). So
    every compaction increments a counter that lands on the role-end log line.
"""
from __future__ import annotations

import json

from . import settings

COMPACT_HEADER = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\nSummary:\n"
)

# Claude Code's own section list, transcribed from a real summary in its session file. The
# sections are chosen so the agent can RESUME rather than merely recall: intent, what was
# already ruled out, what is still pending. "All user messages" is the load-bearing one -- it
# keeps the original requirements verbatim so they survive summarisation intact.
# The summariser is a DIFFERENT job from the agent. Without this, the model reads a transcript
# that ends mid-investigation and CONTINUES it -- emitting the agent's next tool call as XML in
# place of a summary. Run 8: 7 of 20 compactions did exactly that ("Let me verify the array.ts
# path exists.\n\n<tool_call>..."), finish=stop, ~100 tokens, by choice.
SUMMARIZER_SYSTEM = (
    "You are a transcript summarizer. Your ONLY output is a written summary of the transcript "
    "you are given. You never call tools, never emit <tool_call> or <function=...> markup, and "
    "never continue the transcript as if you were a participant in it."
)

SUMMARY_PROMPT = """Your context is full. Write a summary of the conversation so far that will
REPLACE it -- you will continue working from your summary alone, plus the last few messages.

Be specific and technical. Preserve exact file paths, function names, command lines, error
text and test counts. Someone reading only this summary must be able to continue the work
without re-discovering anything.

Use exactly these sections:

1. Primary Request and Intent: what the task requires, in full detail.
2. Key Technical Concepts: the frameworks, build/test commands and idioms of this repository.
3. Files and Code Sections: every file examined or changed, with WHY it matters and the exact
   edits made so far.
4. Errors and fixes: every error encountered and how it was resolved, INCLUDING approaches
   that were tried and ruled out -- so you do not repeat them.
5. Problem Solving: what has been solved and what is still being worked out.
6. All user messages: every non-tool instruction you were given, verbatim.
7. Pending Tasks: what remains.
8. Current Work: precisely what you were doing when the context filled.
9. Next Step: the immediate next action.
"""


def est_tokens(messages: list) -> int:
    """Crude size estimate. Only needs to be right to ~20% -- it gates a threshold, not a
    budget. Counting characters over the serialised messages is stable and costs nothing;
    a real tokenizer call per turn would not pay for itself."""
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c) if isinstance(c, str) else len(json.dumps(c, default=str)) if c else 0
        # Reasoning is now carried on assistant turns and rendered into the prompt, so it
        # MUST be counted here -- otherwise compaction triggers late and the request overflows
        # the 262,144-token window (vLLM hard-rejects prompt + max_tokens beyond it).
        n += len(m.get("reasoning") or "")
        if m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"], default=str))
    return n // settings.CHARS_PER_TOKEN


# The summarisation request must fit the model's window with room for the prompt and the
# summary itself. Qwen3.8 serves 262,144 tokens; reserve ~60k for MAX_TOKENS plus overhead
# and cap the rendered blob at ~200k tokens.
_MODEL_CONTEXT_CHARS = 800_000


def _render_cap() -> int:
    """How much of the conversation the summariser may see.

    DERIVED, never hardcoded. A fixed 400,000 was smaller than what compaction actually holds:
    the threshold is 150,000 tokens ~= 600,000 chars, so a third of the head was dropped before
    the summary was written -- and because the clip kept the OLDEST text, the part discarded was
    the NEWEST, i.e. exactly the work the summary most needs to carry forward. A truncation, in
    the mechanism that exists to replace truncation.

    It was also INVISIBLE TO THE SMOKE: with SEEDLING_SUMMARIZE_AT=30000 the blob is ~120,000
    chars and the cap never engages, so no smoke run at that setting could ever detect it.
    Deriving the cap from the threshold makes the two impossible to drift apart.
    """
    return min(int(settings.SUMMARIZE_AT_TOKENS * settings.CHARS_PER_TOKEN * 1.5),
               _MODEL_CONTEXT_CHARS)


def _render(messages: list, cap: int | None = None) -> str:
    """Flatten the messages into text for the summariser.

    If the blob somehow still exceeds the cap, drop the MIDDLE rather than the tail: the head
    carries the task and the tail carries the most recent work, and losing either defeats the
    summary. Clipping is marked inline so the model is told, not silently deceived."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        c = m.get("content")
        txt = c if isinstance(c, str) else json.dumps(c, default=str) if c else ""
        if m.get("tool_calls"):
            names = [tc.get("function", {}).get("name") or tc.get("name")
                     for tc in m["tool_calls"]]
            txt = (txt or "") + f"  [called: {', '.join(str(n) for n in names)}]"
        r = m.get("reasoning")
        if r:   # the summariser must see what the model was THINKING, not only what it did
            # A neutral label, NOT literal <think> tags: those are Qwen3 SPECIAL TOKENS and
            # placing them inside a user message is a risk class we need not carry. A/B on
            # the live server: [reasoning] and <think> summarised equally well.
            txt = f"[reasoning]\n{r}\n[/reasoning]\n" + (txt or "")
        out.append(f"<{role}>\n{txt}")
    blob = "\n\n".join(out)
    if cap is None:
        cap = _render_cap()
    if len(blob) <= cap:
        return blob
    half = cap // 2
    dropped = len(blob) - cap
    return (blob[:half]
            + f"\n\n[... {dropped:,} characters from the middle of the conversation omitted "
              f"for length; the beginning and the most recent work are shown ...]\n\n"
            + blob[-half:])


async def maybe_compact(messages: list, role: str, h, real_tokens: int = 0,
                        stats: dict | None = None) -> bool:
    """Summarise in place if the conversation has grown past the threshold.

    Returns True if a compaction happened. Mutates `messages` via slice assignment so the
    caller's list object (which IS the persistent per-role conversation on the blackboard)
    is updated in place -- rebinding would silently lose the continuity this design exists
    to provide.

    Never raises: a failed summarisation must not kill a role that is otherwise working. On
    failure the conversation is left untouched and the role continues; the wall budget still
    bounds it, and the entry cap still bounds any single tool result.
    """
    # `real_tokens` is the server's own count for the LAST request. The conversation has grown
    # by one turn since, so take the larger of the two readings: never later than the old
    # estimate, and exact whenever the server has told us.
    if max(int(real_tokens or 0), est_tokens(messages)) < settings.SUMMARIZE_AT_TOKENS:
        return False
    keep = max(2, settings.PRESERVE_TAIL_MSGS)
    if len(messages) <= keep + 2:          # nothing meaningful to summarise
        return False

    system, head, tail = messages[0], messages[1:-keep], messages[-keep:]
    before_msgs, before_tok = len(messages), est_tokens(messages)

    import time as _t
    _t0 = _t.monotonic()
    blob = _render(head)
    # A summary this short is not a summary. Run 7 replaced 30, 46 and 14 messages with 14,
    # 96 and 339 CHARACTERS -- in ~2s each -- and the old `if not summary` guard let them
    # through because 14 chars is not empty. Erasure dressed as compaction. The floor scales
    # with the head so a long conversation cannot be "summarised" into a sentence.
    # head/300 let 440-757-char stubs through for 90K heads. Real summaries ran 3.9K-27K.
    # Stubs sit at ~1:100+ of the head (200-800 chars for 90K); real summaries at 1:4-1:11
    # (3.9K-27K). head/60 separates them for every case seen; the 6000 cap keeps a 600K
    # production head from demanding a 15K summary and rejecting good ones.
    min_len = max(1500, min(len(blob) // 60, 6000))
    summary, fin, rsn_len, ctoks = "", None, 0, None
    try:
        for _try in range(2):
            # Transcript FIRST inside explicit delimiters, instruction AFTER it: with the
            # instruction on top of a 100K-char prompt, the model's attention lands on the
            # live-looking end of the transcript and picks up the agent's next step.
            resp = await h.llm.chat(
                [{"role": "system", "content": SUMMARIZER_SYSTEM},
                 {"role": "user", "content":
                     "<transcript>\n" + blob + "\n</transcript>\n\n"
                     "The transcript above has ENDED. Do NOT continue it and do NOT emit any "
                     "tool call. Your task is only to summarize it.\n\n" + SUMMARY_PROMPT +
                     "\nWrite all nine sections in full prose. Begin with "
                     "'1. Primary Request and Intent:'."}],
                tools=None,
                # THINKING OFF for the summariser (v8.3). EXP-027 yaegi: four consecutive
                # tries deliberated 8-25K chars and emitted 0 chars of content with
                # finish=stop -- the summary was written inside <think> and never surfaced.
                thinking=False,
            )
            summary = (resp.get("content") or "").strip()
            fin = resp.get("finish_reason")
            rsn_len = len(resp.get("reasoning") or "")
            ctoks = (resp.get("usage") or {}).get("completion_tokens")
            # If the model still thought and put the summary THERE, accept the reasoning
            # text iff it carries the mandated structure -- never a bare think block.
            if not summary and rsn_len and "Primary Request and Intent" in (resp.get("reasoning") or ""):
                summary = (resp.get("reasoning") or "").strip()
                h.logger.warning("role %s COMPACT: content empty, structured summary found in the "
                                 "reasoning channel (%d chars) -- using it", role, len(summary))
            # The unambiguous signature of "continued instead of summarised".
            continued = ("<tool_call>" in summary) or ("<function=" in summary)
            # The prompt mandates the nine sections and the opening line; their absence is a
            # more precise invalidity test than length alone.
            malformed = "Primary Request and Intent" not in summary
            if len(summary) >= min_len and not continued and not malformed:
                break
            if continued or malformed:
                summary = ""          # never let a tool-call-shaped blob be accepted as a summary
            if stats is not None:
                stats["stub_attempts"] = stats.get("stub_attempts", 0) + 1
            h.logger.warning("role %s COMPACT produced a STUB summary on try %d: %d chars for a "
                             "%d-char head (finish=%s, reasoning=%d chars, completion_tokens=%s, "
                             "took=%.0fs so far). Stub verbatim: %r", role, _try + 1, len(summary),
                             len(blob), fin, rsn_len, ctoks, _t.monotonic() - _t0, summary[:300])
    except (AttributeError, TypeError) as e:
        # A WIRING error, not a service error. These mean this module is calling something
        # that does not exist or with the wrong signature -- exactly the bug that shipped in
        # the first cut of v8 (`h.llm.call`, when LLM only defines `chat`). Swallowed as a
        # generic failure it is INVISIBLE: the role logs "COMPACT FAILED", continues
        # uncompacted, and the headline feature is silently dead for the whole run while
        # every artifact looks healthy. Re-raise so it fails loudly on the first turn.
        h.logger.error("role %s COMPACT WIRING ERROR (%s: %s) -- this is a BUG, not a "
                       "service failure", role, type(e).__name__, e)
        raise
    except Exception as e:  # noqa: BLE001 -- a failed summary must never kill the role
        h.logger.warning("role %s COMPACT FAILED (%s: %s) -- continuing uncompacted",
                         role, type(e).__name__, e)
        return False

    if len(summary) < min_len:
        # Keep the conversation INTACT. Uncompacted-and-complete beats compacted-and-erased;
        # the entry cap and the wall still bound it, and the next turn will try again.
        if stats is not None:
            stats["rejected"] = stats.get("rejected", 0) + 1
        h.logger.warning("role %s COMPACT REJECTED (%d chars < floor %d) -- continuing "
                         "uncompacted", role, len(summary), min_len)
        return False

    # A tail message may be an `assistant` turn whose tool_calls have their results in the
    # NEXT message. Slicing mid-pair would leave a dangling tool_call_id, which some servers
    # reject outright. Walk forward to the first message that is not an orphaned tool result.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]

    messages[:] = [system,
                   {"role": "user", "content": COMPACT_HEADER + summary},
                   *tail]
    took = _t.monotonic() - _t0
    if stats is not None:
        stats["compacted"] = stats.get("compacted", 0) + 1
    # #6: PERSIST THE SUMMARY. It replaces the entire conversation and was recorded
    # nowhere -- only its length. A 631-char summary of 116 messages in run 3 could not
    # be inspected. Root-stream step so `pier view` (root steps only) shows it.
    try:
        h.traj.add_system_step(f"[compaction:{role}] {before_msgs}->{len(messages)} msgs, "
                               f"~{before_tok}->{est_tokens(messages)} tok, {took:.0f}s\n\n"
                               + summary)
    except Exception as e:  # noqa: BLE001 -- recording must never break compaction
        h.logger.warning("role %s could not record compaction summary: %s", role, e)

    # #8: log the DURATION. Claude Code's compactMetadata records durationMs (8.0 min
    # once); ours was invisible while 9 compactions ran under contention in run 5.
    h.logger.warning("role %s COMPACTED: msgs %d -> %d, ~tokens %d -> %d, summary %d chars, "
                     "tail kept %d, took %.0fs, head %d chars, finish=%s, reasoning=%d chars, "
                     "completion_tokens=%s", role, before_msgs, len(messages), before_tok,
                     est_tokens(messages), len(summary), len(tail), took, len(blob), fin,
                     rsn_len, ctoks)
    return True

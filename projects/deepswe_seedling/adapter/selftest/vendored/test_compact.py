"""Functional tests for v8 context compaction.

The stub below implements `chat` with the SAME name and signature as the real
`seedling.llm.LLM.chat`, and test_wiring.py separately asserts that compact.py only calls
methods the real class defines. Between them, a rename on either side fails a test instead of
turning compaction silently off for a whole run.
"""
import asyncio
import inspect
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from seedling import compact, settings        # noqa: E402
from seedling.llm import LLM                  # noqa: E402


class StubLLM:
    # Long enough to clear the stub floor (max(400, head/300)) for every fixture here; the
    # rejection test passes a 14-char reply explicitly.
    def __init__(self, reply="1. Primary Request and Intent: " + "SUMMARY BODY " * 700,
                 raises=None):
        self.reply, self.raises, self.calls = reply, raises, 0

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice=None, thinking=True) -> dict:
        self.calls += 1
        if self.raises:
            raise self.raises
        return {"content": self.reply, "reasoning": "", "tool_calls": [], "usage": {}}


class H:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger("test")


def _convo(n_pairs: int, chars: int = 4000) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM PROMPT MARKER"}]
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [{"id": f"c{i}", "function": {"name": "Bash"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * chars})
    return msgs


def test_stub_matches_the_real_chat_signature():
    assert (inspect.signature(StubLLM.chat).parameters.keys()
            == inspect.signature(LLM.chat).parameters.keys())


def test_below_threshold_is_a_noop():
    m = _convo(3)
    before = list(m)
    assert asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM()))) is False
    assert m == before


def test_above_threshold_compacts_and_preserves_system_and_tail():
    m = _convo(120)                      # ~240 msgs, ~480K chars -> ~120K tokens
    while compact.est_tokens(m) < settings.SUMMARIZE_AT_TOKENS:
        m.extend(_convo(20)[1:])
    before_tok, tail_before = compact.est_tokens(m), m[-1]["content"]
    h = H(StubLLM())
    assert asyncio.run(compact.maybe_compact(m, "patch", h)) is True
    assert h.llm.calls == 1
    assert m[0]["content"] == "SYSTEM PROMPT MARKER", "system prompt lost"
    assert "SUMMARY BODY" in m[1]["content"]
    assert m[1]["content"].startswith(compact.COMPACT_HEADER)
    assert m[-1]["content"] == tail_before, "tail not preserved verbatim"
    assert compact.est_tokens(m) < before_tok


def test_no_orphaned_tool_result_at_the_seam():
    """A tail starting with role=tool would carry a tool_call_id whose assistant turn was
    summarised away; some servers reject that outright."""
    for pairs in range(60, 200, 7):
        m = _convo(pairs)
        while compact.est_tokens(m) < settings.SUMMARIZE_AT_TOKENS:
            m.extend(_convo(20)[1:])
        fired = asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM())))
        assert fired, f"compaction must actually fire here (pairs={pairs}) or this test is vacuous"
        assert m[2]["role"] != "tool", f"orphaned tool result at seam (pairs={pairs})"


def test_service_failure_leaves_the_conversation_intact():
    m = _convo(400)
    before = list(m)
    h = H(StubLLM(raises=RuntimeError("503 upstream")))
    assert asyncio.run(compact.maybe_compact(m, "patch", h)) is False
    assert m == before, "a failed summary must not damage the conversation"


def test_empty_summary_is_treated_as_failure():
    m = _convo(400)
    before = list(m)
    assert asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM(reply="   ")))) is False
    assert m == before


def test_wiring_error_is_raised_not_swallowed():
    """The regression that shipped in v8's first cut. A missing method must be loud."""
    class Broken:
        pass
    m = _convo(400)
    try:
        asyncio.run(compact.maybe_compact(m, "patch", H(Broken())))
    except AttributeError:
        return
    raise AssertionError("a missing llm method was swallowed instead of raised")


def test_render_cap_always_exceeds_what_compaction_can_hold():
    """The bug: a hardcoded 400,000-char cap against a 150,000-token (~600,000-char) threshold
    dropped 33% of the head before summarising -- and kept the OLDEST text, so what vanished was
    the most recent work. Invisible at the smoke's 30,000-token setting, which is why it must be
    asserted here rather than smoke-tested."""
    import importlib
    from seedling import settings as S
    for threshold in (30_000, 150_000, 200_000):
        S.SUMMARIZE_AT_TOKENS = threshold
        importlib.reload(compact)
        need = threshold * S.CHARS_PER_TOKEN
        cap = compact._render_cap()
        assert cap >= need or cap == compact._MODEL_CONTEXT_CHARS, (
            f"threshold {threshold} needs {need:,} chars but the cap is {cap:,}")
    S.SUMMARIZE_AT_TOKENS = 150_000
    importlib.reload(compact)


def test_render_drops_the_middle_not_the_newest():
    """If the cap must ever bite, the head (task) and tail (recent work) must both survive."""
    head, tail = "HEAD_MARKER " * 10, "TAIL_MARKER " * 10
    msgs = ([{"role": "user", "content": head}]
            + [{"role": "assistant", "content": "x" * 5000} for _ in range(60)]
            + [{"role": "user", "content": tail}])
    out = compact._render(msgs, cap=20_000)
    assert "HEAD_MARKER" in out, "the task at the head was dropped"
    assert "TAIL_MARKER" in out, "the most recent work was dropped -- the original bug"
    assert "omitted" in out, "clipping is not disclosed to the model"
    assert len(out) <= 20_000 + 300


def test_render_returns_everything_when_it_fits():
    msgs = [{"role": "user", "content": "small"}]
    assert compact._render(msgs) == "<user>\nsmall"



def test_reasoning_is_counted_and_rendered():
    """Uncounted reasoning makes compaction fire LATE (requests overflow 262,144); unrendered
    reasoning hides the model's thinking from the summariser."""
    m = [{"role": "user", "content": "task"},
         {"role": "assistant", "content": "ok", "reasoning": "R" * 4000}]
    assert compact.est_tokens(m) >= 1000, "reasoning is not counted by est_tokens"
    out = compact._render(m)
    assert "[reasoning]" in out and "R" * 100 in out, "reasoning is not rendered for the summariser"
    assert "<think>" not in out, "special tokens must not be injected into a user message"


def test_stub_summary_is_rejected_and_conversation_kept():
    """Run 7: 30 messages -> a 14-char 'summary' passed the empty-check. Never again."""
    m = _convo(400); before = list(m)
    h = H(StubLLM(reply="Summary: done."))          # 14 chars, like the real stub
    assert asyncio.run(compact.maybe_compact(m, "patch", h)) is False
    assert m == before, "a stub summary must leave the conversation untouched"
    assert h.llm.calls == 2, "a stub should be retried exactly once before rejection"


def test_continuation_is_detected_and_rejected():
    """Run 8's real failure: the summariser emitted the agent's NEXT TOOL CALL instead of a
    summary -- long enough to pass a length floor, useless as context."""
    m = _convo(400); before = list(m)
    fake = "Let me verify the path.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls /app\n" * 200
    assert asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM(reply=fake)))) is False
    assert m == before


def test_summariser_gets_a_system_message_and_instruction_after_transcript():
    src = __import__("inspect").getsource(compact.maybe_compact)
    assert '"role": "system"' in src and "SUMMARIZER_SYSTEM" in src
    assert src.index("<transcript>") < src.index("SUMMARY_PROMPT +"), "instruction must FOLLOW the transcript"


def test_real_token_count_triggers_compaction_when_estimate_is_low():
    """chars/4 undercounts reasoning by ~25%. The server's real count must be able to trigger
    compaction on its own, so a request cannot creep past the threshold on a bad estimate."""
    m = _convo(20, chars=10)                        # many messages, tiny by the estimate
    assert compact.est_tokens(m) < settings.SUMMARIZE_AT_TOKENS
    fired = asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM()),
                                              real_tokens=settings.SUMMARIZE_AT_TOKENS + 1))
    assert fired is True, "a real over-threshold count did not trigger compaction"


def test_summary_missing_mandated_section_is_rejected():
    m = _convo(400); before = list(m)
    long_but_wrong = "Here is what happened: " + ("the agent read files and ran tests. " * 400)
    assert asyncio.run(compact.maybe_compact(m, "patch", H(StubLLM(reply=long_but_wrong)))) is False
    assert m == before


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)

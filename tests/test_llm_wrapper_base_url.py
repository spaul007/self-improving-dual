"""Tests for ``platform_core.llm_wrapper`` base_url routing.

The wrapper must:
  * Forward ``base_url`` to the OpenAI SDK constructor when set via kwarg.
  * Pick up ``LLM_BASE_URL`` env var when no kwarg is given.
  * Prefer kwarg over env when both are set.
  * Omit ``base_url`` from the client constructor when neither is set
    (so the SDK keeps its default OpenAI endpoint).
  * Relax the ``OPENAI_API_KEY`` requirement when ``base_url`` resolves
    truthy (any value accepted; missing key auto-fills as ``"EMPTY"``).
  * Record the resolved ``base_url`` on the ``llm_call`` trace event.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_llm_wrapper_base_url
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response() -> SimpleNamespace:
    """Minimal Responses-API-shaped object that ``_extract_output`` accepts."""
    return SimpleNamespace(
        output=[],
        output_text="ok",
        status="completed",
        usage=None,
    )


class _FakeResponses:
    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def create(self, **kwargs):
        self._sink["create_kwargs"] = kwargs
        return _fake_response()


class _FakeOpenAI:
    """Captures the constructor kwargs so we can assert on ``base_url``."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.sink: dict = {}
        self.responses = _FakeResponses(self.sink)
        _FakeOpenAI.instances.append(self)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class BaseUrlPlumbingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot env so tests don't leak.
        self._env_snapshot = {
            k: os.environ.get(k)
            for k in ("OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                      "LLM_REASONING_EFFORT", "META_AGENT_TRACE_PATH")
        }
        for k in ("LLM_BASE_URL", "LLM_REASONING_EFFORT",
                  "META_AGENT_TRACE_PATH"):
            os.environ.pop(k, None)
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["LLM_MODEL"] = "gpt-5.4-mini"

        _FakeOpenAI.instances.clear()

        # Patch the OpenAI symbol the wrapper imports.
        self._openai_patch = mock.patch(
            "openai.OpenAI", new=_FakeOpenAI
        )
        self._openai_patch.start()

        self.tmp = Path(tempfile.mkdtemp(prefix="llm_base_url_test_"))

    def tearDown(self) -> None:
        self._openai_patch.stop()
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ----- core forwarding ----------------------------------------------

    def test_kwarg_base_url_forwarded_to_client(self) -> None:
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://example.test:8000/v1",
        )

        self.assertEqual(len(_FakeOpenAI.instances), 1)
        client = _FakeOpenAI.instances[0]
        self.assertEqual(
            client.init_kwargs.get("base_url"),
            "http://example.test:8000/v1",
        )

    def test_env_base_url_forwarded_when_no_kwarg(self) -> None:
        os.environ["LLM_BASE_URL"] = "http://env-host:9000/v1"
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(messages=[{"role": "user", "content": "hi"}])

        client = _FakeOpenAI.instances[0]
        self.assertEqual(
            client.init_kwargs.get("base_url"),
            "http://env-host:9000/v1",
        )

    def test_kwarg_wins_over_env(self) -> None:
        os.environ["LLM_BASE_URL"] = "http://env-loser:9000/v1"
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://kwarg-winner:8000/v1",
        )

        client = _FakeOpenAI.instances[0]
        self.assertEqual(
            client.init_kwargs.get("base_url"),
            "http://kwarg-winner:8000/v1",
        )

    def test_no_base_url_means_no_constructor_kwarg(self) -> None:
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(messages=[{"role": "user", "content": "hi"}])

        client = _FakeOpenAI.instances[0]
        # Important: when base_url isn't set, we must NOT pass it through —
        # the SDK has its own default and we don't want to clobber it.
        self.assertNotIn("base_url", client.init_kwargs)

    # ----- API-key relaxation -------------------------------------------

    def test_missing_api_key_raises_when_no_base_url(self) -> None:
        os.environ.pop("OPENAI_API_KEY", None)
        from platform_core import llm_wrapper

        with self.assertRaises(RuntimeError):
            llm_wrapper.call_llm(messages=[{"role": "user", "content": "hi"}])

    def test_missing_api_key_accepted_when_base_url_set(self) -> None:
        os.environ.pop("OPENAI_API_KEY", None)
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://local:8000/v1",
        )

        client = _FakeOpenAI.instances[0]
        # Wrapper auto-fills "EMPTY" when key is missing + base_url is set.
        self.assertEqual(client.init_kwargs.get("api_key"), "EMPTY")

    def test_explicit_api_key_kept_when_base_url_set(self) -> None:
        os.environ["OPENAI_API_KEY"] = "some-key"
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://local:8000/v1",
        )

        client = _FakeOpenAI.instances[0]
        # We don't overwrite a user-supplied key, even pointing at vLLM.
        self.assertEqual(client.init_kwargs.get("api_key"), "some-key")

    # ----- trace event --------------------------------------------------

    def test_trace_event_records_base_url(self) -> None:
        trace_path = self.tmp / "trace.jsonl"
        os.environ["META_AGENT_TRACE_PATH"] = str(trace_path)

        # platform_core.trace reads the env var when emitting; reload to be
        # safe in case some other test cached state.
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://traced:8000/v1",
        )

        events = [
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if line.strip()
        ]
        llm_calls = [e for e in events if e.get("kind") == "llm_call"]
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(
            llm_calls[0]["payload"].get("base_url"),
            "http://traced:8000/v1",
        )

    def test_trace_event_records_none_base_url_when_unset(self) -> None:
        trace_path = self.tmp / "trace.jsonl"
        os.environ["META_AGENT_TRACE_PATH"] = str(trace_path)
        from platform_core import llm_wrapper

        llm_wrapper.call_llm(messages=[{"role": "user", "content": "hi"}])

        events = [
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if line.strip()
        ]
        llm_calls = [e for e in events if e.get("kind") == "llm_call"]
        self.assertEqual(len(llm_calls), 1)
        self.assertIsNone(llm_calls[0]["payload"].get("base_url"))


class ReasoningFallbackTests(unittest.TestCase):
    """Cover the message/reasoning content extraction in
    `_extract_output`. The wrapper should prefer message-item text when
    it's non-empty (OpenAI path) and fall through to reasoning-item
    text when the message is empty or whitespace-only
    (vLLM/Qwen3.5 path — caught 2026-05-12 when Qwen put the
    `<plan>...</plan>` answer into the reasoning item and the message
    was just `\\n\\n`)."""

    def _msg(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=text)],
        )

    def _reasoning(self, text: str) -> SimpleNamespace:
        # Match the .content shape vLLM emits.
        return SimpleNamespace(
            type="reasoning",
            content=[SimpleNamespace(type="reasoning_text", text=text)],
        )

    def test_message_text_wins_when_present(self) -> None:
        # OpenAI path: real message text + a reasoning item. The
        # reasoning item must NOT bleed into content.
        from platform_core.llm_wrapper import _extract_output

        resp = SimpleNamespace(
            output=[
                self._reasoning("internal thinking — should be hidden"),
                self._msg("Final answer here."),
            ]
        )
        content, _ = _extract_output(resp)
        self.assertEqual(content, "Final answer here.")

    def test_falls_back_to_reasoning_when_message_missing(self) -> None:
        # No message item at all — reasoning becomes the visible content.
        from platform_core.llm_wrapper import _extract_output

        resp = SimpleNamespace(
            output=[self._reasoning("<plan>do thing</plan>")]
        )
        content, _ = _extract_output(resp)
        self.assertEqual(content, "<plan>do thing</plan>")

    def test_falls_back_to_reasoning_when_message_whitespace_only(self) -> None:
        # The exact Qwen3.5 failure mode: message item present but
        # contains only `\n\n`; the real answer is in reasoning.
        from platform_core.llm_wrapper import _extract_output

        resp = SimpleNamespace(
            output=[
                self._reasoning("<plan>kyoto plan</plan>"),
                self._msg("\n\n"),
            ]
        )
        content, _ = _extract_output(resp)
        self.assertIn("<plan>kyoto plan</plan>", content or "")

    def test_no_reasoning_no_message_returns_none(self) -> None:
        from platform_core.llm_wrapper import _extract_output

        resp = SimpleNamespace(output=[])
        content, tool_calls = _extract_output(resp)
        self.assertIsNone(content)
        self.assertEqual(tool_calls, [])

    def test_function_call_extracted_alongside_reasoning_fallback(self) -> None:
        # Edge case: reasoning + function_call but no message item.
        # Wrapper should surface BOTH the tool call AND the reasoning
        # text — earlier no-op (returning None content) was a footgun
        # because the seed would lose the model's commentary.
        from platform_core.llm_wrapper import _extract_output

        resp = SimpleNamespace(
            output=[
                self._reasoning("I need to look up the train schedule first."),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="query_trains",
                    arguments='{"origin":"Tokyo"}',
                ),
            ]
        )
        content, tool_calls = _extract_output(resp)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "query_trains")
        self.assertIn("train schedule", content or "")

    def test_reasoning_summary_shape_also_works(self) -> None:
        # Some servers expose reasoning text under .summary instead of
        # .content. The helper handles either.
        from platform_core.llm_wrapper import _extract_output

        item = SimpleNamespace(
            type="reasoning",
            content=None,
            summary=[SimpleNamespace(type="summary_text", text="hi")],
        )
        resp = SimpleNamespace(output=[item])
        content, _ = _extract_output(resp)
        self.assertEqual(content, "hi")


if __name__ == "__main__":
    unittest.main()

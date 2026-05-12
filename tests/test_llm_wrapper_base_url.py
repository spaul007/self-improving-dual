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


if __name__ == "__main__":
    unittest.main()

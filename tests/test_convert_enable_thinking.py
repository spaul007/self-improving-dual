"""travel_mas_refactored plan converter: opt-in ``TRAVEL_CONVERT_ENABLE_THINKING``.

    PYTHONPATH=. python3 -m unittest tests.test_convert_enable_thinking
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from projects.travel_mas_refactored.adapter import scorer_impl


class _Client:
    def __init__(self, **kw) -> None:
        self.requests: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **req):
                outer.requests.append(req)
                msg = SimpleNamespace(content='<JSON>{"ok": 1}</JSON>')
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=_Completions())


def _convert(env: dict) -> dict:
    clients: list[_Client] = []

    def factory(**kw):
        clients.append(_Client(**kw))
        return clients[-1]

    base = {k: v for k, v in os.environ.items() if not k.startswith(("TRAVEL_CONVERT", "LLM_"))}
    with mock.patch.dict(os.environ, {**base, "TRAVEL_CONVERT_BASE_URL": "http://c/v1",
                                      "TRAVEL_CONVERT_MODEL": "Qwen/Qwen3.8-27B", **env}, clear=True), \
            mock.patch("openai.OpenAI", side_effect=factory):
        parsed, err = scorer_impl._convert_plan_to_json("day 1: fly")
    assert err is None and parsed == {"ok": 1}, (parsed, err)
    return clients[0].requests[0]


class ConvertThinkingTests(unittest.TestCase):
    def test_unset_request_unchanged(self) -> None:
        req = _convert({})
        self.assertEqual(sorted(req), ["messages", "model"])
        self.assertEqual(req["model"], "Qwen/Qwen3.8-27B")

    def test_false_disables_thinking(self) -> None:
        req = _convert({"TRAVEL_CONVERT_ENABLE_THINKING": "false"})
        self.assertEqual(req["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})

    def test_true_enables_thinking(self) -> None:
        req = _convert({"TRAVEL_CONVERT_ENABLE_THINKING": "True"})
        self.assertEqual(req["extra_body"], {"chat_template_kwargs": {"enable_thinking": True}})


class SeedNoThinkTests(unittest.TestCase):
    def test_variant_differs_from_seed_only_by_the_thinking_switch(self) -> None:
        import filecmp
        from pathlib import Path

        root = Path(scorer_impl.__file__).resolve().parents[1]
        seed, variant = root / "seed", root / "seed_qwen27b_nothink"

        def files(d: Path) -> set[str]:
            return {p.relative_to(d).as_posix() for p in d.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}

        self.assertEqual(files(seed), files(variant))
        differing = sorted(r for r in files(seed) if not filecmp.cmp(seed / r, variant / r, shallow=False))
        self.assertEqual(differing, ["agents/llm_backbone.py", "mas_llm_backbone.yaml"])

    def test_backbone_loader_returns_enable_thinking_false(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        root = Path(scorer_impl.__file__).resolve().parents[1]
        out = subprocess.run(
            [sys.executable, "-c",
             "from agents.llm_backbone import get_backbone_config; "
             "print(get_backbone_config('flight')['enable_thinking'])"],
            cwd=root / "seed_qwen27b_nothink", capture_output=True, text=True,
        )
        self.assertEqual(out.stdout.strip(), "False", out.stderr)


if __name__ == "__main__":
    unittest.main()

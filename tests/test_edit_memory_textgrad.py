"""TextGrad instruction optimizer of the edit-memory layer
(``instruction_optimizer: "textgrad"``): the graph and prompts of one step
(scripted model), and the layer running it in place of the one-call
updater, with its fallbacks."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TEXTGRAD_LOG_DIR", os.path.join(tempfile.gettempdir(), "textgrad_logs"))

from textgrad.autograd.llm_backward_prompts import BACKWARD_SYSTEM_PROMPT  # noqa: E402

from meta_agent.edit_memory import prompts as P  # noqa: E402
from meta_agent.edit_memory import textgrad_opt as T  # noqa: E402
from meta_agent.edit_memory.generator import LLMSpec  # noqa: E402

from tests import test_edit_memory_layer as L  # noqa: E402
from tests.test_edit_memory_layer import LayerBase, _memory_doc, _Resp, _ScriptedLLM  # noqa: E402

OPTIMIZER_SYSTEM_START = "You are part of an optimization system that improves text"
LONG_ADDENDUM = ("- Every ranked edit names the file and function it changed, so an editor can open it "
                 "without searching, and states in one clause whether the task agent's trace shows it firing.\n"
                 "- KEEP-THIS-TAIL-MARKER\n")


def _role(messages: list[dict]) -> str:
    sys_msg = messages[0]["content"]
    if sys_msg == T.CRITIC_SYSTEM:
        return "critic"
    if sys_msg == BACKWARD_SYSTEM_PROMPT:
        return "backward"
    if sys_msg.startswith(OPTIMIZER_SYSTEM_START):
        return "optimizer"
    return "other"


class _TGScripted:
    """Plays critic / backward / optimizer; ``optimizer_replies`` are
    consumed in order (the last one repeats)."""

    def __init__(self, optimizer_replies: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.optimizer_replies = optimizer_replies or ["<IMPROVED_VARIABLE>- new bullet</IMPROVED_VARIABLE>"]

    def __call__(self, **kw):
        role = _role(kw["messages"])
        self.calls.append({"role": role, **kw})
        if role == "critic":
            return _Resp("## Verdict\nok\n## Issues\n- [INSTRUCTION] C3 — citations lack files\n"
                         "- [INPUT] C3 — LATER-TOOL was unknown to the curation\n"
                         "## Keep\n- ranking\n## Not attributable to the instruction\n- [INPUT] LATER-TOOL")
        if role == "backward":
            return _Resp(f"FEEDBACK#{sum(c['role'] == 'backward' for c in self.calls)}: name files")
        if role == "optimizer":
            n = sum(c["role"] == "optimizer" for c in self.calls)
            return _Resp(self.optimizer_replies[min(n, len(self.optimizer_replies)) - 1])
        raise AssertionError(f"unexpected call: {kw['messages'][0]['content'][:80]}")


class TestOptimizeAddendum(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tg_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec = LLMSpec(model="m", reasoning_effort="low", base_url="http://x", api_key_env="K",
                            llm_timeout_s=60, extra_body={"provider": {"order": ["P"]}})

    def _sample(self, v: int = 2) -> T.CritiqueSample:
        return T.CritiqueSample(
            memory_version=v, memory=_memory_doc("MEM-V2"), previous_memory=_memory_doc("MEM-V1"),
            curation="CURATION-TEXT {with braces}", system_prompt="GEN-SYSTEM core+addendum",
            user_prompt="GEN-USER window 2", usage="- node 7 (parent 3); changed files: workflow.py")

    def _run(self, llm, **kw):
        args = dict(addendum=LONG_ADDENDUM, samples=[self._sample()], audit="AUDIT-Q",
                    core="FIXED-CORE-TEXT", past_feedback=["PAST-FB-1"], max_chars=4000,
                    out_dir=self.tmp / "tg")
        args.update(kw)
        return T.optimize_addendum(llm, self.spec, **args)

    def test_one_step_graph_and_prompts(self) -> None:
        llm = _TGScripted()
        out, errs, rec = self._run(llm)
        self.assertEqual((out, errs), ("- new bullet\n", []))
        # critic -> backward to the memory -> backward through the generator -> optimizer
        self.assertEqual([c["role"] for c in llm.calls], ["critic", "backward", "backward", "optimizer"])
        # the layer's model kwargs reach every call
        self.assertTrue(all(c["model"] == "m" and c["api_key_env"] == "K" and c["extra_body"]
                            for c in llm.calls))
        critic_in = llm.calls[0]["messages"][1]["content"]
        for s in ("GEN-SYSTEM core+addendum", "MEM-V1", "CURATION-TEXT {with braces}", "AUDIT-Q",
                  "node 7", "MEM-V2", "[INSTRUCTION]", "[INPUT]"):
            self.assertIn(s, critic_in)
        # [INPUT] items are kept in the critique file but never sent backward
        hop1 = llm.calls[1]["messages"][1]["content"]
        self.assertIn("citations lack files", hop1)
        self.assertNotIn("LATER-TOOL", hop1)
        self.assertNotIn("LATER-TOOL", llm.calls[2]["messages"][1]["content"])
        self.assertIn("LATER-TOOL", (self.tmp / "tg" / "critique_v002.md").read_text())
        self.assertNotIn("LATER-TOOL", (self.tmp / "tg" / "critique_v002_propagated.md").read_text())
        # the second hop sees the generator's own conversation
        hop2 = llm.calls[2]["messages"][1]["content"]
        self.assertIn("<LM_SYSTEM_PROMPT> GEN-SYSTEM core+addendum", hop2)
        self.assertIn("GEN-USER window 2", hop2)
        self.assertIn("FEEDBACK#1", hop2)
        # the optimizer sees the WHOLE addendum, the core, momentum, the
        # compose note and the constraints
        opt = llm.calls[3]["messages"][1]["content"]
        self.assertIn(LONG_ADDENDUM.strip(), opt)
        self.assertIn("<FIXED_CORE>\nFIXED-CORE-TEXT", opt)
        self.assertIn("PAST-FB-1", opt)
        self.assertIn("FEEDBACK#2", opt)
        self.assertIn("fixed core instruction (cannot change)", opt)
        self.assertIn("Never ask the memory to predict scores", opt)
        # artifacts
        d = self.tmp / "tg"
        self.assertIn("[INSTRUCTION]", (d / "critique_v002.md").read_text())
        self.assertIn("FEEDBACK#1", (d / "memory_feedback_v002.md").read_text())
        self.assertIn("FEEDBACK#2", (d / T.ADDENDUM_FEEDBACK_FILE).read_text())
        calls = [json.loads(line) for line in (d / T.CALLS_FILE).read_text().splitlines()]
        self.assertEqual([c["phase"] for c in calls],
                         ["critic_v002", "backward_v002", "backward_v002", "optimizer_1"])
        self.assertEqual(json.loads((d / T.SUMMARY_FILE).read_text())["n_llm_calls"], 4)
        self.assertEqual(rec["result"], "written")

    def test_two_samples_accumulate_on_the_addendum(self) -> None:
        llm = _TGScripted()
        out, errs, _ = self._run(llm, samples=[self._sample(3), self._sample(2)])
        self.assertEqual(out, "- new bullet\n")
        self.assertEqual([c["role"] for c in llm.calls],
                         ["critic", "backward", "backward", "critic", "backward", "backward", "optimizer"])
        opt = llm.calls[-1]["messages"][1]["content"]
        # each memory's through-generator feedback reaches the addendum once
        self.assertEqual((opt.count("FEEDBACK#2"), opt.count("FEEDBACK#4")), (1, 1))

    def test_rejected_draft_is_retried_then_kept(self) -> None:
        llm = _TGScripted([f"<IMPROVED_VARIABLE>{P.CORE_SENTINEL}\n- x</IMPROVED_VARIABLE>",
                           "<IMPROVED_VARIABLE>- predict the score</IMPROVED_VARIABLE>"])
        out, errs, rec = self._run(llm)
        self.assertEqual(out, "- predict the score\n")       # final draft kept ...
        self.assertTrue(any("score prediction" in e for e in errs))   # ... findings reported
        retry = [c for c in llm.calls if c["role"] == "optimizer"][1]["messages"][1]["content"]
        self.assertIn("Your previous draft was rejected", retry)
        self.assertIn(LONG_ADDENDUM.strip(), retry)            # retried from the OLD addendum
        self.assertEqual(len(rec["attempts"]), 2)

    def test_unparseable_optimizer_yields_none(self) -> None:
        out, errs, rec = self._run(_TGScripted(["no tags here"]))
        self.assertIsNone(out)
        self.assertIn("IMPROVED_VARIABLE", errs[0])
        self.assertEqual(rec["result"], "failed")

    def test_llm_failure_yields_none(self) -> None:
        def boom(**kw):
            raise RuntimeError("down")
        out, errs, _ = self._run(boom)
        self.assertIsNone(out)
        self.assertIn("down", errs[0])

    def test_empty_addendum_is_shown_as_empty(self) -> None:
        llm = _TGScripted()
        out, _, _ = self._run(llm, addendum="", past_feedback=[])
        self.assertEqual(out, "- new bullet\n")
        opt = llm.calls[-1]["messages"][1]["content"]
        self.assertIn(T.EMPTY_ADDENDUM, opt)
        self.assertNotIn("PAST_FEEDBACK", opt)

    def test_strip_input_items(self) -> None:
        critique = (
            "## Verdict\nGood overall.\n\n## Issues\n\n"
            "- [INSTRUCTION] C3 — no lineage roll-up (evidence: node 13).\n"
            "  continuation of the instruction issue\n\n"
            "- [INPUT] C4 — the curation never named TOOL-X,\n"
            "  so the memory could not cite it\n\n"
            "1. **[INPUT]** numbered input item\n"
            "- [COMPLIANCE] C2 — names not stable\n\n"
            "## Keep\n- the ledger\n\n"
            "## Not attributable to the instruction\n- [INPUT] TOOL-X discovered later\n"
        )
        out = T.strip_input_items(critique)
        self.assertNotIn("TOOL-X", out)
        self.assertNotIn("numbered input item", out)
        self.assertNotIn(T.INPUT_SECTION, out)
        for kept in ("Good overall.", "continuation of the instruction issue",
                     "[COMPLIANCE] C2", "## Keep\n- the ledger"):
            self.assertIn(kept, out)
        # nothing to strip -> unchanged (modulo the final newline)
        clean = "## Verdict\nfine\n## Issues\n- [INSTRUCTION] x\n"
        self.assertEqual(T.strip_input_items(clean), clean)

    def test_facts_rule_allows_illustration(self) -> None:
        rules = " ".join(T.optimizer_constraints(max_chars=100, forbid_case_values=False))
        self.assertIn("may illustrate a rule, but must not stand in for one", rules)
        self.assertIn("satisfiable together", rules)
        self.assertIn("may illustrate the rule, but must not stand in for it", T.CRITIC_FORMAT)
        self.assertIn("never fault it for lacking something absent from", T.CRITIC_FORMAT)

    def test_critic_format_has_only_its_fields(self) -> None:
        import string
        names = {f for _, f, _, _ in string.Formatter().parse(T.CRITIC_FORMAT) if f}
        self.assertEqual(names, set(T.CRITIC_FIELDS))


class _LayerScripted(_ScriptedLLM):
    """The layer test's scripted model plus the three textgrad roles."""

    def __init__(self, optimizer_reply: str = "<IMPROVED_VARIABLE>- tg addendum</IMPROVED_VARIABLE>") -> None:
        super().__init__()
        self.tg = _TGScripted([optimizer_reply])

    def __call__(self, **kw):
        if not kw.get("tools") and _role(kw["messages"]) != "other":
            self.calls.append({k: v for k, v in kw.items() if k != "messages"} | {"messages": list(kw["messages"])})
            return self.tg(**kw)
        return super().__call__(**kw)


class TestLayerWithTextGrad(LayerBase):
    _tree = L.TestCadence._tree
    _child = L.TestCadence._child

    def tg_layer(self, llm: _LayerScripted, **kw):
        from meta_agent.edit_memory.layer import EditMemoryLayer
        opts = dict(window_size=2, instruction_every=1, arm_min_pulls=1, seed=3,
                    curator={"sandbox": "none", "max_llm_calls": 6, "timeout_s": 60},
                    instruction_optimizer="textgrad")
        opts.update(kw)
        lay = EditMemoryLayer(llm, repo_root=self.tmp / "repo", **opts)
        lay.setup(self.exp)
        return lay

    def _windows(self, lay, tree, plan):
        for nid, arm, version in plan:
            self._child(tree, nid, arm=arm, version=version)
            lay.on_event("expand", tree, node_id=nid)
            lay.on_event("expand_eval", tree, node_id=nid)

    def test_textgrad_replaces_the_updater(self) -> None:
        llm = _LayerScripted()
        lay = self.tg_layer(llm)
        tree = self._tree()
        self._windows(lay, tree, [(1, "none", None), (2, "none", None)])
        gi = json.loads((lay.dir / "window_001" / "generation_inputs.json").read_text())
        self.assertEqual((gi["memory_version"], gi["instruction_version"]), (1, 0))
        self.assertTrue(gi["system"].startswith(P.CORE_SENTINEL))
        self._windows(lay, tree, [(3, "with", 1), (4, "without", 1)])
        self.assertEqual((lay.memory_version, lay.instruction_version), (2, 1))
        self.assertEqual((lay.dir / "instruction.md").read_text(), "- tg addendum\n")
        u = lay.dir / "instruction_update_001"
        self.assertFalse((u / "update_call.json").exists())          # the updater did not run
        self.assertTrue((u / "textgrad" / "critique_v001.md").exists())
        # the critic judged v1 against the audit, with node 3 as its reader
        critic = [c for c in llm.tg.calls if c["role"] == "critic"][0]["messages"][1]["content"]
        self.assertIn(_memory_doc("v1").strip(), critic)
        self.assertIn("- node 3 (parent 0)", critic)
        self.assertIn(P.Q_SECTIONS[0], critic)
        # v2 generated under the textgrad addendum, recorded as such
        gi2 = json.loads((lay.dir / "window_002" / "generation_inputs.json").read_text())
        self.assertEqual(gi2["instruction_version"], 1)
        self.assertIn("- tg addendum", gi2["system"])
        ev = [e for e in lay.events if e["event"] == "instruction_written"][0]
        self.assertEqual(ev["optimizer"], "textgrad")
        self.assertEqual(json.loads((lay.dir / "state.json").read_text())["config"]["instruction_optimizer"],
                         "textgrad")
        self.assertTrue(list((lay.dir / "textgrad_logs").glob("*.jsonl")))
        # Window 3: the second step gets the first step's feedback as momentum.
        self._windows(lay, tree, [(5, "with", 2), (6, "with", 2)])
        self.assertEqual(lay.instruction_version, 2)
        opt = [c for c in llm.tg.calls if c["role"] == "optimizer"][-1]["messages"][1]["content"]
        self.assertIn("<PAST_FEEDBACK>", opt)
        self.assertIn((u / "textgrad" / T.ADDENDUM_FEEDBACK_FILE).read_text().strip()[:40], opt)

    def test_falls_back_to_the_updater(self) -> None:
        llm = _LayerScripted(optimizer_reply="no tags")
        lay = self.tg_layer(llm)
        tree = self._tree()
        self._windows(lay, tree, [(1, "none", None), (2, "none", None), (3, "with", 1), (4, "without", 1)])
        self.assertEqual(lay.instruction_version, 1)
        self.assertEqual((lay.dir / "instruction.md").read_text(), "- addendum v1: cite failing checks\n")
        events = [e["event"] for e in lay.events]
        self.assertLess(events.index("textgrad_failed"), events.index("instruction_written"))
        self.assertEqual([e for e in lay.events if e["event"] == "instruction_written"][0]["optimizer"],
                         "updater")

    def test_keep_fallback_and_missing_inputs(self) -> None:
        llm = _LayerScripted()
        lay = self.tg_layer(llm, textgrad={"fallback": "keep"})
        tree = self._tree()
        self._windows(lay, tree, [(1, "none", None), (2, "none", None)])
        (lay.dir / "window_001" / "generation_inputs.json").unlink()   # e.g. a run started without textgrad
        self._windows(lay, tree, [(3, "with", 1), (4, "without", 1)])
        self.assertEqual(lay.instruction_version, 0)
        self.assertEqual([c for c in llm.tg.calls], [])              # nothing to critique
        failed = [e for e in lay.events if e["event"] == "textgrad_failed"][0]
        self.assertIn("no recorded generation inputs", failed["errors"][0])
        self.assertIn("instruction_failed", [e["event"] for e in lay.events])
        self.assertEqual(lay.memory_version, 2)                      # the memory still advanced

    def test_config_validation(self) -> None:
        from meta_agent.edit_memory.layer import EditMemoryLayer
        with self.assertRaises(ValueError):
            EditMemoryLayer(_ScriptedLLM(), instruction_optimizer="adam")
        with self.assertRaises(ValueError):
            EditMemoryLayer(_ScriptedLLM(), textgrad={"max_versions": 1})
        with self.assertRaises(ValueError):
            EditMemoryLayer(_ScriptedLLM(), instruction_optimizer="textgrad", textgrad={"fallback": "x"})


if __name__ == "__main__":
    unittest.main()

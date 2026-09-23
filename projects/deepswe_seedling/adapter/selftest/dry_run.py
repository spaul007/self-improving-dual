"""Dry run of a candidate seedling tree: the REAL multi-role pipeline end-to-end in seconds.

    $PIERPY dry_run.py <candidate task_agent dir>     -> prints one JSON object (last line)

Runs ``SeedlingAgent(mode="multi").run(...)`` exactly as Pier would, but against a fake
container (``FakeExec``: every command succeeds with the canned git-bootstrap markers) and
a fake LLM (``litellm.acompletion`` monkeypatched). The fake model ends every role turn
immediately (plain text, no tool call); when the harness forces the ``finish`` report it
answers with schema-valid arguments built from the offered schema (``verdict: pass`` for
VERIFY, so the pipeline terminates after one PATCH -> VERIFY cycle).

It catches what static checks cannot, before paying 1-3 h per real case:
  * runtime errors anywhere in the edited roles/pipeline/compaction/tools wiring,
  * the ROLE MANDATE -- the run must contain >= 1 PATCH and >= 1 VERIFY role run,
  * observability -- role_stats entries carry the keys the scorer/meta-agent read,
  * A0 -- each role's calls are sent with THAT role's own system prompt as messages[0],
  * settings sanity (reported; judged by the settings_guard validator).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REQUIRED_ROLE_KEYS = ("role", "attempt", "steps", "stop_reason", "wall_terminated", "edits",
                      "tests_run", "llm_calls", "wall_sec")


class FakeExec:
    def agent_process_env(self, env):
        return env

    async def exec(self, command, cwd=None, env=None, user=None, timeout_sec=None):
        class R:
            stdout = ("SEED_BOOTSTRAP_OK\nSEED_BASE=abc123\nSEED_HAS_TIMEOUT=1\n"
                      "SEED_CHECKPOINT_OK\nSEED_PATCH_BYTES=42\nWROTE")
            stderr = None
            return_code = 0
        return R()


def _value(spec: dict):
    t = spec.get("type")
    if spec.get("enum"):
        return "pass" if "pass" in spec["enum"] else spec["enum"][0]
    if t == "array":
        return ["dry-run item"]
    if t in ("integer", "number"):
        return 1
    if t == "boolean":
        return True
    if t == "object":
        return {}
    return "dry-run"


CALLS: list[dict] = []


async def fake_acompletion(messages, tools=None, tool_choice=None, **kw):
    sys0 = (messages[0].get("content") or "") if messages and messages[0].get("role") == "system" else ""
    names = [((t.get("function") or {}).get("name")) for t in (tools or [])]
    CALLS.append({"sys_sha": hashlib.sha256(sys0.encode()).hexdigest()[:12], "tools": names,
                  "n_messages": len(messages)})
    finish = next((t for t in (tools or []) if (t.get("function") or {}).get("name") == "finish"), None)
    if finish is not None:
        props = ((finish["function"].get("parameters") or {}).get("properties") or {})
        args = {k: _value(v) for k, v in props.items()}
        tc = SimpleNamespace(id=f"c{len(CALLS)}", type="function",
                             function=SimpleNamespace(name="finish", arguments=json.dumps(args)))
        msg = SimpleNamespace(content="", reasoning=None, tool_calls=[tc])
        reason = "tool_calls"
    else:
        msg = SimpleNamespace(content="Done.", reasoning=None, tool_calls=None)
        reason = "stop"
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=reason)], usage=usage)


def main(candidate: Path) -> dict:
    sys.path.insert(0, str(candidate))
    sys.dont_write_bytecode = True
    import litellm  # noqa: F401  (must be importable: seedling.llm imports it lazily)
    litellm.acompletion = fake_acompletion
    from pier.models.agent.context import AgentContext
    from seedling import roles, settings
    from seedling.agent import SeedlingAgent

    out: dict = {"raised": None}
    prompts = {}
    for name in ("PATCH", "VERIFY", "BASELINE"):
        r = getattr(roles, name, None)
        if r is not None:
            try:
                prompts[r.name] = hashlib.sha256(r.system_prompt().encode()).hexdigest()[:12]
            except Exception as exc:  # noqa: BLE001
                out.setdefault("prompt_errors", []).append(f"{name}: {exc!r}")
    tmp = tempfile.mkdtemp(prefix="sid_dryrun_")
    agent = SeedlingAgent(logs_dir=Path(tmp), model_name="openai/dry-run",
                          extra_env={"OPENAI_BASE_URL": "http://127.0.0.1:9/v1"},
                          logger=logging.getLogger("dry"), agent_timeout_sec="900", mode="multi")
    t0 = time.time()
    try:
        asyncio.run(asyncio.wait_for(agent.run("Dry run: make any trivial change.", FakeExec(),
                                               AgentContext()), timeout=150))
    except BaseException as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {exc}"[:500]
    out["wall_s"] = round(time.time() - t0, 1)
    rs_path = Path(tmp) / "run_summary.json"
    rs = json.loads(rs_path.read_text()) if rs_path.is_file() else {}
    out["outcome"] = rs.get("outcome")
    out["role_stats"] = [{k: r.get(k) for k in REQUIRED_ROLE_KEYS} | {
        "report_ok": r.get("report_ok"),
        "missing_keys": [k for k in REQUIRED_ROLE_KEYS if k not in r]} for r in rs.get("role_stats") or []]
    out["role_prompt_sha"] = prompts
    out["sys_sha_seen"] = sorted({c["sys_sha"] for c in CALLS})
    out["n_llm_calls"] = len(CALLS)
    budgets = getattr(settings, "BUDGETS", {}) or {}
    out["settings"] = {
        "REASONING_EFFORT": getattr(settings, "REASONING_EFFORT", None),
        "MAX_TOKENS": getattr(settings, "MAX_TOKENS", None),
        "MAX_PATCH_ATTEMPTS": getattr(settings, "MAX_PATCH_ATTEMPTS", None),
        "wall_frac": {k: (v or {}).get("wall_frac") for k, v in budgets.items()},
        "SAMPLING": getattr(settings, "SAMPLING", None),
        "LLM_TIMEOUT_SEC": getattr(settings, "LLM_TIMEOUT_SEC", None),
    }
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    res = main(Path(sys.argv[1]).resolve())
    print(json.dumps(res, default=str))

"""MUTABLE — internal hand-off protocol between the investigators and the lead.

Compresses an investigator's full evidence report into a short briefing so the
lead DBA gets five compact, comparable summaries instead of five full query
transcripts. This is *not* part of the benchmark contract, so an optimizer is
free to retune the prompt in mas_prompt_cfg.yaml (`tools.compress`).
"""

import config
import llm_client


async def compress(raw: str) -> str:
    """Summarize a full investigator report into a <=120-word briefing."""
    if not raw or not raw.strip():
        return ""

    prompt = config.tool_prompt("compress").format(raw=raw)
    short = await llm_client.get_client().acall(prompt)
    return (short or "").strip()

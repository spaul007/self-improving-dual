"""MUTABLE — internal hand-off protocol between the predictor and the reflector.

Compresses the predictor's full solution into a short briefing so the reflector
gets a compact context instead of the whole chain of thought. This is *not* part
of the benchmark contract, so an optimizer is free to retune the prompt in
mas_prompt_cfg.yaml (`tools.compress`).
"""

import config
import llm_client


async def compress(raw: str) -> str:
    """Summarize a full solution into a ~30-word briefing."""
    if not raw or not raw.strip():
        return ""

    prompt = config.tool_prompt("compress").format(raw=raw)
    short = await llm_client.get_client().acall(prompt)
    return (short or "").strip()

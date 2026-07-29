"""IMMUTABLE — the benchmark's answer-extraction contract.

This decides *what counts as the MAS's answer*. Rewriting it would change what
the benchmark measures, not how well the agents perform, so an automated
prompt/tool optimizer must never touch this file.

Precedence: Concluder's parsed JSON `final_answer` field > a
<answer>...</answer> regex fallback (in case JSON parsing failed) > empty
string. Mirrors math_mas's tag>boxed>fallback precedence, just with JSON as
the primary format since wikihop_mas's hand-offs are structured (see
agents/base.py's parse_json_output).
"""

import html
import re

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)


def extract_final_answer(raw: str, parsed: dict) -> str:
    """Pull the Concluder's final answer out of its (raw, parsed) output."""
    answer = parsed.get("final_answer") if isinstance(parsed, dict) else None
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    if not raw:
        return ""
    text = raw.strip()
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped

    matches = _ANSWER_TAG.findall(text)
    if matches:
        return matches[-1].strip()

    return ""

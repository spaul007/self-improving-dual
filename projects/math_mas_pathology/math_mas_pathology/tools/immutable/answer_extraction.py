"""IMMUTABLE — the benchmark's answer-extraction contract.

This decides *what counts as the MAS's answer* for a math problem. Rewriting it
would change what the benchmark measures, not how well the agents perform, so an
automated prompt/tool optimizer must never touch this file.

Logic is kept byte-compatible with MASPO's `utils.extract_answer` so scores are
comparable across the two codebases.
"""

import html
import re

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)


def extract_answer(raw: str) -> str:
    """Pull the final answer out of a model's free-text solution.

    Precedence: <answer>...</answer> (last one) > \\boxed{...} (first one) >
    trailing-sentence fallback.
    """
    if not raw:
        return ""

    raw = raw.strip()
    for _ in range(3):
        unescaped = html.unescape(raw)
        if unescaped == raw:
            break
        raw = unescaped

    matches = _ANSWER_TAG.findall(raw)
    if matches:
        return matches[-1].strip()

    boxed = _BOXED.search(raw)
    if boxed:
        return boxed.group(1).strip()

    sentences = re.split(r"[。\n;]+", raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last

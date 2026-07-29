"""IMMUTABLE — the benchmark's answer-extraction contract.

This decides *which root-cause labels count as the MAS's answer*: the lead DBA
is required to end its diagnosis with a line

    FINAL: <LABEL>[, <LABEL> ...]

and this module reads the labels out of that verdict deterministically (no
LLM). Rewriting it would change what the benchmark measures, not how well the
agents perform, so an automated prompt/tool optimizer must never touch it.

Logic is kept behaviour-compatible with MASPO_v2's `judges.parse_final_labels`
so recall numbers are comparable across the two codebases. (MASPO additionally
falls back to an LLM extractor for unparseable answers; this repo stays fully
deterministic and instead reports extraction failures in the scored summary.)
"""

import re
from typing import List, Optional

# Marker lines the lead DBA is asked to end with (see mas_prompt_cfg.yaml,
# agents.lead_dba.task), plus the de-facto headings models use anyway, so
# answers that drift from the strict format still parse. Ordered most- to
# least-authoritative; the LAST occurrence anchors the verdict.
_VERDICT_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:\*\*|##+\s*)?"
    r"(FINAL\s*:|final diagnosis|final decision|final answer|final verdict)",
    re.IGNORECASE,
)


def extract_labels(raw: str, labels: List[str],
                   n_pred: Optional[int] = None) -> List[str]:
    """Deterministically read the verdict labels out of the lead DBA's answer.

    The label vocabulary is a handful of fixed tokens, so once the verdict
    section is located the extraction is exact. We anchor on the LAST verdict
    marker (the answer discusses every candidate label in its reasoning, so an
    unanchored scan would return all of them), then take labels in order of
    appearance, deduped. With no marker at all, the closing lines are scanned.

    `n_pred=None` returns everything the verdict named — callers truncate
    themselves so that over-naming stays visible rather than silently cut.
    Returns [] when nothing parses.
    """
    text = raw or ""
    if not text.strip() or not labels:
        return []
    marks = list(_VERDICT_MARKERS.finditer(text))
    if marks:
        tail = text[marks[-1].end():]
    else:
        # No marker at all: the verdict is conventionally the closing lines.
        tail = "\n".join(text.strip().splitlines()[-4:])
    pattern = re.compile("|".join(re.escape(lab) for lab in labels), re.IGNORECASE)
    by_norm = {lab.lower(): lab for lab in labels}
    predicted: List[str] = []
    for m in pattern.finditer(tail):
        lab = by_norm[m.group(0).lower()]
        if lab not in predicted:
            predicted.append(lab)
        if n_pred is not None and len(predicted) >= n_pred:
            break
    return predicted

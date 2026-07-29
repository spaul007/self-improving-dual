"""Generator for wikihop_mas_2k's cases.jsonl: a genuinely RANDOM sample of
2000 rows from the full 2WikiMultihopQA dev split (12,576 rows total),
unlike projects/wikihop_mas/benchmark/generate_cases.py's "first 200 rows in
file order" (an arbitrary, non-random selection -- confirmed against the
official Dropbox-hosted release that our source data matches byte-for-byte,
but the SELECTION of which 200 rows was never randomized).

Built to size up the benchmark enough that different random subsamples of
it have low enough score variance to trust generalization claims -- see
aristotle_analysis.md's sampling-variance analysis: at n=200 (the original
wikihop_mas project's full pool), the population std was 0.45, meaning a
50-case train draw had sampling std ~0.055, larger than several of the real
edit-quality signals HGM was trying to detect. 2000 total (with 1000-case
resamples tested for variance) targets a regime where that stops being true.

Reads the same local dev.jsonl the original wikihop_mas project uses
(WIKIHOP_DEV_JSONL env var, same default path) -- never modifies it.
Ground truth handling identical to the original generator: `answer`/`type`/
`supporting_facts`/`evidences` go only into `meta_info`, never `context`.

Rerun with a different SAMPLE_SEED to draw a different 2000-row sample.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

_DEFAULT_DEV_JSONL = "/groups/AIC-MV/v.kulkarni1/wikihop_mas/data/2wikimultihopqa/dev.jsonl"
_DEV_JSONL = Path(os.environ.get("WIKIHOP_DEV_JSONL", _DEFAULT_DEV_JSONL))
_OUT = Path(__file__).resolve().parent / "cases.jsonl"
_SAMPLE_SIZE = int(os.environ.get("WIKIHOP_2K_SAMPLE_SIZE", "2000"))
_SAMPLE_SEED = int(os.environ.get("WIKIHOP_2K_SAMPLE_SEED", "42"))


def main() -> None:
    if not _DEV_JSONL.exists():
        raise FileNotFoundError(
            f"{_DEV_JSONL} not found -- set WIKIHOP_DEV_JSONL to the "
            "2WikiMultihopQA dev.jsonl path (see wikihop_mas/README.md's "
            "download+convert steps)."
        )

    rows = []
    with _DEV_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(_SAMPLE_SEED)
    rng.shuffle(rows)
    sampled = rows[:_SAMPLE_SIZE]

    lines: list[str] = []
    for row in sampled:
        case_id = str(row.get("_id"))
        case = {
            "id": case_id,
            "input": row["question"],
            "context": {"paragraphs": row.get("context", [])},
            "meta_info": {
                "answer": row.get("answer", ""),
                "type": row.get("type", ""),
                "supporting_facts": row.get("supporting_facts", []),
                "evidences": row.get("evidences", []),
            },
        }
        lines.append(json.dumps(case, ensure_ascii=False))

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {len(lines)} cases (random sample, seed={_SAMPLE_SEED}) -> {_OUT}  "
        f"(source: {_DEV_JSONL}, {len(rows)} total rows)"
    )


if __name__ == "__main__":
    main()

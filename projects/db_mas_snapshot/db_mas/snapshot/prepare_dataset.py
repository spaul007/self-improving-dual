"""Flatten MARBLE's database task config into this repo's tasks JSONL.

MARBLE stores each database root-cause-analysis task as a full simulation
config: the natural-language problem under `task.content`, the required answer
template under `task.output_format`, and the ground truth under `task.labels`
(candidate root causes), `task.root_causes` (the gold answer), and
`task.number_of_labels_pred` (how many labels the team must predict). This MAS
consumes a flat record with a single `problem` field, so we concatenate
content + output_format into `problem` and carry the ground-truth fields
through.

Two deliberate transformations (both inherited from MASPO_v2, so numbers stay
comparable between the two repos):

1. Case marker. Many MARBLE database cases share identical content +
   output_format (same scenario + candidate labels) but have DIFFERENT gold
   root causes — distinguishable only via query_db. A `[Diagnosis case #<id>]`
   marker keeps every case's problem string unique (this repo keys snapshots by
   unique_id, but the marker also protects any downstream consumer that keys by
   the problem STRING, e.g. MASPO's train split / gold map). It leaks no
   answer.

2. Prediction-count retarget. MARBLE asks for len(root_causes) + 1 labels,
   which makes precision/F1/exact-match degenerate. We rewrite the two places
   the task text states that count so the team is asked for exactly
   len(root_causes). Numbers produced under this are NOT comparable to
   upstream MARBLE / crewai runs, which keep the n+1 convention.

Usage (from the db_mas repo root):
    python snapshot/prepare_dataset.py

Writes: data/marble-db/database_tasks.jsonl  (one line per MARBLE task)
"""

import json
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_ROOT = os.path.dirname(_THIS_DIR)
if _MAS_ROOT not in sys.path:
    sys.path.insert(0, _MAS_ROOT)

import config  # noqa: E402

_MARBLE_ROOT = os.getenv("MARBLE_ROOT", "/groups/AIC-MV/sudipta.paul/code/rsi/MARBLE")
MARBLE_DATABASE = os.path.join(
    _MARBLE_ROOT, "multiagentbench", "database", "database_main.jsonl"
)
OUT_PATH = str(config.DATASET_PATH)

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# The two places MARBLE's task text states how many labels to name. Upstream both
# carry `number_of_labels_pred`, which is always len(root_causes) + 1.
_COUNT_PATTERNS = (
    re.compile(r"(root cause can be only )(one|two|three|four|five)( of the following)", re.I),
    re.compile(r"(You can ONLY CHOOSE )(one|two|three|four|five)(\.)", re.I),
)


def retarget_prediction_count(problem: str, n_gold: int, task_id: int) -> str:
    """Ask the team for exactly `n_gold` labels instead of MARBLE's n_gold + 1.

    Raises if the expected phrasing is absent, so a silent mismatch between the
    requested and the scored count is impossible.
    """
    word = _NUMBER_WORDS[n_gold]
    for pattern in _COUNT_PATTERNS:
        problem, n_subs = pattern.subn(
            lambda m: f"{m.group(1)}{word}{m.group(3)}", problem
        )
        if n_subs == 0:
            raise ValueError(
                f"task {task_id}: expected phrasing {pattern.pattern!r} not found in "
                "the task text; cannot retarget the prediction count safely."
            )
    return problem


def main():
    n = 0
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(MARBLE_DATABASE, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec["task"]
            content = (task.get("content", "") or "").strip()
            output_format = (task.get("output_format", "") or "").strip()
            problem = (
                content + "\n\n" + output_format
                + f"\n\n[Diagnosis case #{rec['task_id']}]"
            )
            root_causes = task.get("root_causes", [])
            problem = retarget_prediction_count(
                problem, len(root_causes), rec["task_id"]
            )
            out = {
                "problem": problem,
                "unique_id": rec["task_id"],
                "answer": "",
                # ground truth used by eval/metrics.py
                "labels": task.get("labels", []),
                "root_causes": root_causes,
                # What the task text now asks for, and what scoring uses. Kept
                # equal to len(root_causes) by construction; score_record derives
                # the count from the gold set regardless, so the two cannot drift.
                "number_of_labels_pred": len(root_causes),
                # MARBLE's original n+1 value, retained for provenance only.
                "marble_number_of_labels_pred": task.get("number_of_labels_pred", 2),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} database tasks to {OUT_PATH}")


if __name__ == "__main__":
    main()

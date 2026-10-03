"""Generate benchmark/cases.jsonl from the DeepSWE v1.1 task directory.

One case per task in deep-swe/tasks113.txt (the 113 scored tasks; the tasks/
dir also holds README/manifest files). Case contract (runner.Task):
  id          bare task directory name (Pier's task_name is "datacurve/<id>")
  input       instruction.md -- the problem statement; ALSO what the feedback
              gatherer shows the meta-agent as the case "query"
  context     {"task_dir": <abs path>} -- the only thing workflow.py needs
  meta_info   {"language", "category", "repository_url"} -- never reaches the
              agent (runner keeps meta_info off the Task); used for
              split.stratify_by and the per-language breakdown

    python3 projects/deepswe_seedling/adapter/make_cases.py [DEEPSWE_ROOT]
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

DEFAULT_ROOT = Path("/groups/AIC-MV/n.tzou/swe/deep-swe")
OUT = Path(__file__).resolve().parents[1] / "benchmark" / "cases.jsonl"


def main(root: Path = DEFAULT_ROOT) -> int:
    names = [l.strip() for l in (root / "tasks113.txt").read_text().splitlines() if l.strip()]
    rows = []
    for name in names:
        tdir = (root / "tasks" / name).resolve()
        meta = tomllib.loads((tdir / "task.toml").read_text()).get("metadata", {})
        rows.append({
            "id": name,
            "input": (tdir / "instruction.md").read_text(encoding="utf-8"),
            "context": {"task_dir": str(tdir)},
            "meta_info": {
                "language": meta.get("language"),
                "category": meta.get("category"),
                "repository_url": meta.get("repository_url"),
            },
        })
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT))

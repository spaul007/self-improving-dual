"""Import an already-finished experiment's ROOT evaluation into the root cache.

    python scripts/root_cache_import.py <experiment_dir> <config.yaml> [--dry-run]

Computes the key the given config would produce for ``<experiment_dir>/round_000/task_agent``
(same seed tree + train ids + config fingerprint as meta_agent.root_cache) and stores
``round_000/eval_result.json`` + ``round_000/logs`` under it, so the next launch of that
config replays the root instead of re-evaluating it.

Refuses when the experiment's root was run with a different seed than the config resolves
to, or on a different train set -- the key would then describe a result it did not produce.
Cases the manager skipped entirely (LLM-call failures under exclude_llm_call_failures) are
absent from eval_result.json and so absent from the replay.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from meta_agent import config as C, root_cache  # noqa: E402
from meta_agent.models import EvaluationResult  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rd = a.experiment_dir / "round_000"
    agent_dir, logs, ev = rd / "task_agent", rd / "logs", rd / "eval_result.json"
    for p in (agent_dir, logs, ev):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    cfg = C.load(a.config)
    fw = C.build_components(cfg)
    m = fw.manager
    if not getattr(m, "root_cache_dir", None):
        print("ERROR: the config sets no manager.config.root_cache_dir", file=sys.stderr)
        return 2
    seed_digest = root_cache.tree_digest(fw.seed_dir)
    root_digest = root_cache.tree_digest(agent_dir)
    if seed_digest != root_digest:
        print(f"ERROR: experiment root tree {root_digest[:12]} != config seed {seed_digest[:12]} "
              f"({fw.seed_dir})", file=sys.stderr)
        return 3
    result = EvaluationResult.model_validate_json(ev.read_text())
    ran = sorted(c.case_id for c in result.per_case)
    want = sorted(str(c) for c in (fw.train_case_ids or []))
    if not set(ran) <= set(want) or len(ran) < len(want) * 0.9:
        print(f"ERROR: root ran {len(ran)} cases; config train set has {len(want)} "
              f"(missing {len(set(want) - set(ran))}, extra {len(set(ran) - set(want))})", file=sys.stderr)
        return 3
    key, parts = root_cache.cache_key(agent_dir, fw.train_case_ids, m.root_cache_fingerprint)
    print(f"key {key[:12]}  cases {len(ran)}/{len(want)}  score {result.score:.3f}  -> {m.root_cache_dir}")
    if a.dry_run:
        return 0
    stored = root_cache.store(Path(m.root_cache_dir), key, parts, result, logs, source=str(a.experiment_dir))
    print("stored" if stored else "NOT stored (entry exists or write failed)")
    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(main())

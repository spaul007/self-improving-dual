"""Tests for the train/eval split, including the optional level-stratified mode.

    PYTHONPATH=. python3 -m unittest tests.test_compute_split

Focus:
  1. ``stratify_by=None`` is byte-identical to the legacy seeded shuffle/slice,
     so projects that don't opt in are unaffected and runs stay reproducible.
  2. ``stratify_by="context.level"`` produces a proportional, deterministic
     split that keeps every level in BOTH halves (the shopping fix: 25/25/10).
  3. Largest-remainder lands on train_size exactly for non-divisible sizes.
  4. A missing key never crashes (falls into a None stratum).
"""
from __future__ import annotations

import collections
import json
import random
import tempfile
import unittest
from pathlib import Path

from meta_agent.config import REPO_ROOT, compute_split


def _write_cases(rows: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp())
    with (d / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


def _level_hist(benchmark_dir: Path, ids: list[str]) -> dict:
    cases = {
        str(json.loads(l)["id"]): json.loads(l)
        for l in (benchmark_dir / "cases.jsonl").read_text().splitlines()
        if l.strip()
    }
    return dict(
        collections.Counter(cases[i]["context"]["level"] for i in ids)
    )


SHOPPING_BENCH = REPO_ROOT / "projects" / "shopping" / "benchmark"


class UnstratifiedUnchangedTest(unittest.TestCase):
    def test_none_matches_legacy_shuffle(self) -> None:
        rows = [
            {"id": f"L{lvl}-{i}", "context": {"level": lvl}}
            for lvl in (1, 2, 3)
            for i in range(10)
        ]
        d = _write_cases(rows)
        ids = [r["id"] for r in rows]
        # Legacy behavior: seeded shuffle then slice.
        rng = random.Random(7)
        legacy = list(ids)
        rng.shuffle(legacy)
        exp_train, exp_eval = legacy[:12], legacy[12:]

        train, eval_ = compute_split(d, seed=7, train_size=12, stratify_by=None)
        self.assertEqual((train, eval_), (exp_train, exp_eval))


class StratifiedProportionalTest(unittest.TestCase):
    def test_shopping_real_benchmark_25_25_10(self) -> None:
        if not (SHOPPING_BENCH / "cases.jsonl").exists():
            self.skipTest("shopping benchmark not present")
        train, eval_ = compute_split(
            SHOPPING_BENCH, seed=42, train_size=60, stratify_by="context.level"
        )
        self.assertEqual(len(train), 60)
        self.assertEqual(len(eval_), 60)
        # Every level present in BOTH halves, proportional 25/25/10.
        self.assertEqual(_level_hist(SHOPPING_BENCH, train), {1: 25, 2: 25, 3: 10})
        self.assertEqual(_level_hist(SHOPPING_BENCH, eval_), {1: 25, 2: 25, 3: 10})
        # Disjoint and covers everything.
        self.assertEqual(set(train) & set(eval_), set())
        self.assertEqual(len(set(train) | set(eval_)), 120)

    def test_deterministic(self) -> None:
        if not (SHOPPING_BENCH / "cases.jsonl").exists():
            self.skipTest("shopping benchmark not present")
        a = compute_split(SHOPPING_BENCH, seed=42, train_size=60, stratify_by="context.level")
        b = compute_split(SHOPPING_BENCH, seed=42, train_size=60, stratify_by="context.level")
        self.assertEqual(a, b)

    def test_non_divisible_largest_remainder(self) -> None:
        # 50/50/20 == 120 total; train_size=61 is not cleanly proportional.
        rows = (
            [{"id": f"L1-{i}", "context": {"level": 1}} for i in range(50)]
            + [{"id": f"L2-{i}", "context": {"level": 2}} for i in range(50)]
            + [{"id": f"L3-{i}", "context": {"level": 3}} for i in range(20)]
        )
        d = _write_cases(rows)
        train, eval_ = compute_split(d, seed=1, train_size=61, stratify_by="context.level")
        self.assertEqual(len(train), 61)
        self.assertEqual(len(eval_), 59)
        hist = _level_hist(d, train)
        # Proportional targets are 25.4 / 25.4 / 10.2 -> base 25/25/10 = 60,
        # one leftover goes to the largest remainder (L1 or L2, .4 > .2).
        self.assertEqual(sum(hist.values()), 61)
        self.assertEqual(hist[3], 10)
        self.assertEqual(sorted([hist[1], hist[2]]), [25, 26])

    def test_missing_key_does_not_crash(self) -> None:
        rows = [
            {"id": "a", "context": {"level": 1}},
            {"id": "b", "context": {}},          # no level
            {"id": "c"},                          # no context at all
            {"id": "d", "context": {"level": 1}},
        ]
        d = _write_cases(rows)
        train, eval_ = compute_split(d, seed=3, train_size=2, stratify_by="context.level")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(eval_), 2)
        self.assertEqual(set(train) | set(eval_), {"a", "b", "c", "d"})


if __name__ == "__main__":
    unittest.main()

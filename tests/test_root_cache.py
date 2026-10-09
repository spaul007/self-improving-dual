"""Root-evaluation cache: miss -> store -> hit replays an identical result and logs;
any key-part change misses; unset = old behaviour (evaluator always runs)."""
import json
import tempfile
import unittest
from pathlib import Path

from meta_agent import root_cache
from meta_agent.managers.hgm import HGMManager
from meta_agent.models import CaseResult, EvaluationResult


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def run(self, out_dir, benchmark_dir, case_ids=None):
        self.calls += 1
        logs = Path(out_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for c in case_ids:
            (logs / f"case_{c}.json").write_text(json.dumps({"case_id": c}))
        (logs / "trace.jsonl").write_text("")
        per = [CaseResult(case_id=c, passed=(i % 2 == 0), score=float(i % 2 == 0),
                          details={"k": i}) for i, c in enumerate(case_ids)]
        return EvaluationResult(score=sum(p.score for p in per) / len(per), per_case=per,
                                passed=sum(p.passed for p in per), failed=sum(not p.passed for p in per))


def _mgr(exp, cache, **kw):
    m = HGMManager(root_cache_dir=str(cache) if cache else None, **kw)
    m._experiment_dir = Path(exp)
    m._benchmark_dir = Path(exp) / "bench"
    m._train_case_ids = ["a", "b", "c"]
    m.root_cache_fingerprint = {"project": "p", "env": {"X": "1"}}
    return m


class RootCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.agent = self.d / "agent"
        (self.agent / "__pycache__").mkdir(parents=True)
        (self.agent / "w.py").write_text("x = 1\n")
        (self.agent / "__pycache__" / "w.pyc").write_bytes(b"junk")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, name, cache, **kw):
        out = self.d / name / "round_000"
        (out / "logs").mkdir(parents=True)
        m = _mgr(self.d / name, cache, **kw)
        ev = FakeEvaluator()
        res = m._root_eval_cached(out, self.agent, ev)
        return res, ev.calls, out

    def test_off_always_evaluates(self):
        _, calls, _ = self._run("e1", None)
        self.assertEqual(calls, 1)

    def test_miss_store_then_hit_identical(self):
        cache = self.d / "cache"
        r1, c1, _ = self._run("e1", cache)
        r2, c2, out2 = self._run("e2", cache)
        self.assertEqual((c1, c2), (1, 0))
        self.assertEqual(r1.model_dump(), r2.model_dump())
        self.assertTrue((out2 / "logs" / "case_b.json").is_file())
        self.assertTrue((out2 / "logs" / "trace.jsonl").is_file())
        entry = next(p for p in cache.iterdir() if not p.name.startswith("."))
        man = json.loads((entry / "manifest.json").read_text())
        self.assertEqual(man["n_cases"], 3)
        self.assertIn("e1", man["source"])

    def test_read_false_only_writes(self):
        cache = self.d / "cache"
        self._run("e1", cache)
        _, calls, _ = self._run("e2", cache, root_cache_read=False)
        self.assertEqual(calls, 1)

    def test_key_changes_on_tree_cases_fingerprint_not_pycache(self):
        k0, _ = root_cache.cache_key(self.agent, ["a", "b"], {"x": 1})
        self.assertEqual(k0, root_cache.cache_key(self.agent, ["b", "a"], {"x": 1})[0])
        (self.agent / "__pycache__" / "w.pyc").write_bytes(b"other")
        self.assertEqual(k0, root_cache.cache_key(self.agent, ["a", "b"], {"x": 1})[0])
        self.assertNotEqual(k0, root_cache.cache_key(self.agent, ["a", "c"], {"x": 1})[0])
        self.assertNotEqual(k0, root_cache.cache_key(self.agent, ["a", "b"], {"x": 2})[0])
        (self.agent / "skills").mkdir()
        (self.agent / "skills" / "s.md").write_text("new skill")
        self.assertNotEqual(k0, root_cache.cache_key(self.agent, ["a", "b"], {"x": 1})[0])

    def test_crashed_result_not_stored(self):
        cache = self.d / "cache"
        out = self.d / "e1" / "round_000"
        (out / "logs").mkdir(parents=True)
        m = _mgr(self.d / "e1", cache)

        class Crashing(FakeEvaluator):
            def run(self, *a, **k):
                r = super().run(*a, **k)
                return r.model_copy(update={"crashed": True})

        m._root_eval_cached(out, self.agent, Crashing())
        self.assertFalse(cache.exists() and any(not p.name.startswith(".") for p in cache.iterdir()))


class FingerprintTest(unittest.TestCase):
    def test_secret_env_values_hashed(self):
        from types import SimpleNamespace
        from meta_agent.config import _root_cache_fingerprint
        cfg = SimpleNamespace(project="p", evaluator=SimpleNamespace(type="subprocess", config={"parallelism": 16, "scorer": "s"}),
                              task_agent=None, env={"OPENAI_API_KEY": "sk-secret", "LLM_REQUEST_TIMEOUT_S": "1800"})
        with tempfile.TemporaryDirectory() as d:
            fp = _root_cache_fingerprint(cfg, Path(d))
        self.assertNotIn("sk-secret", json.dumps(fp))
        self.assertEqual(fp["env"]["LLM_REQUEST_TIMEOUT_S"], "1800")
        self.assertNotIn("parallelism", fp["evaluator"])


if __name__ == "__main__":
    unittest.main()

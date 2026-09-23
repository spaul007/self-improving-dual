"""Phase-2 acceptance: the validator chain must REJECT harmful edits and ACCEPT benign ones.

    PYTHONPATH=<repo> $SID_PY adapter/selftest/mutation_suite.py

Builds the same validator chain the evolution config uses (framework validators +
the project's four), applies each mutation to a fresh copy of the seed (with the
unedited seed as the base round, exactly as the editor sees it) and checks the verdict.
"""
from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import benchmark.scorer  # noqa: E402,F401  -- registers the project's validators
from meta_agent import editor_validators  # noqa: E402,F401  -- registers framework validators
from meta_agent import registry  # noqa: E402

MUTABLE_EXCLUDE = ["workflow.py", "seedling/agent.py", "seedling/execpool.py", "seedling/gitops.py",
                   "seedling/trajectory.py", "seedling/deadline.py", "seedling/llm.py",
                   "seedling/tools/__init__.py", "seedling/prompts/solo.md"]
CHAIN = [("syntax", {}), ("signature", {"workflow_filenames": ["workflow.py"]}),
         ("immutable_files", {}), ("undefined_names", {}), ("seedling_host_isolation", {}),
         ("seedling_selftest", {}), ("seedling_dry_run", {}), ("seedling_settings_guard", {})]


def build():
    out = []
    for name, cfg in CHAIN:
        cls = registry.get("validator", name)
        kw = dict(cfg)
        if "mutable_exclude" in inspect.signature(cls).parameters:
            kw["mutable_exclude"] = MUTABLE_EXCLUDE
        out.append((name, cls(**kw)))
    return out


def sub(path, old, new, count=1):
    def f(agent: Path):
        p = agent / path
        s = p.read_text()
        assert s.count(old) >= 1, f"anchor not found in {path}: {old[:50]}"
        p.write_text(s.replace(old, new, count))
    return f


MUTATIONS = [
    # (label, mutation, expect_reject)
    ("drop the VERIFY role call", sub("seedling/pipeline.py", "bb.verify = await _run_role(VERIFY, bb, h)",
                                      'bb.verify = {"verdict": "pass"}'), True),
    ("rename the finish tool", sub("seedling/roles.py", '"name": "finish",', '"name": "done",'), True),
    ("NameError in Role.run", sub("seedling/roles.py", "    async def run(self, bb, h: Harness) -> dict:\n",
                                  "    async def run(self, bb, h: Harness) -> dict:\n        _probe = undefined_name_xyz\n"), True),
    ("drop a role_stats key", sub("seedling/roles.py", '"wall_terminated": (not end_turn)', '"_wt": (not end_turn)'), True),
    ("reasoning effort = high", sub("seedling/settings.py", 'REASONING_EFFORT = "medium"', 'REASONING_EFFORT = "high"'), True),
    ("read a host file", sub("seedling/pipeline.py", "async def solve(",
                             "_LEAK = open('/groups/AIC-MV/n.tzou/swe/deep-swe/tasks/x/tests/test.patch').read()\n\n\nasync def solve("), True),
    ("edit DENY_WRITE_GLOBS (frozen)", sub("seedling/tools/__init__.py", "DENY_WRITE_GLOBS = [", "DENY_WRITE_GLOBS = [] and ["), True),
    ("benign: patch prompt wording", sub("seedling/prompts/patch.md", "Your job is to FIND the code and EDIT it.",
                                         "Your job is to FIND the relevant code and EDIT it carefully."), False),
    ("benign: wall_frac tweak", sub("seedling/settings.py", '"patch":   {"wall_frac": 0.22}', '"patch":   {"wall_frac": 0.25}'), False),
    ("benign: unchanged seed", lambda agent: None, False),
]


def main() -> int:
    chain = build()
    fails = 0
    t_max = 0.0
    for label, mutate, expect_reject in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="sid_mut_") as d:
            base, out = Path(d) / "base", Path(d) / "out"
            for r in (base, out):
                shutil.copytree(PROJECT / "seed", r / "task_agent",
                                ignore=shutil.ignore_patterns("__pycache__"))
            mutate(out / "task_agent")
            t0 = time.time()
            errors = []
            for name, v in chain:
                errors += [f"[{name}] {e}" for e in v.validate(out, base)]
            dt = time.time() - t0
            t_max = max(t_max, dt)
        rejected = bool(errors)
        ok = rejected == expect_reject
        fails += not ok
        print(f"{'OK ' if ok else 'BAD'} {'REJECT' if rejected else 'accept'} ({dt:4.1f}s) {label}")
        for e in errors[:3]:
            print(f"      {e[:220]}")
    print(f"\nslowest validator pass: {t_max:.1f}s")
    print("MUTATION SUITE PASS" if not fails else f"{fails} MUTATION(S) MISJUDGED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

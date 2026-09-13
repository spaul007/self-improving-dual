"""Standalone validation: before letting the HGM's llm_backbone_selection
block pick from a catalog slug, check each slug actually helps -- evaluate
the Gemma baseline (no override) and every catalog slug (as the
SIGHTSEEING backbone override, exactly how llm_backbone_selection applies
it) on the SAME fixed 32-case batch, so the comparison is apples-to-apples
and immune to the round-to-round case-sampling variance this session kept
running into.

Each candidate gets a fresh copy of the pristine seed's task_agent (never
mutates the seed itself). Uses the real evaluator.run() -- and therefore
the already-fixed platform_core.llm_wrapper.call_llm retry logic -- so
these numbers aren't confounded by the OpenRouter status="failed" bug the
way earlier in-run scores were.

Usage: source /groups/AIC-MV/v.kulkarni1/.env && python3 validate_backbone_catalog_slugs.py
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from meta_agent import config as cfg_mod
from meta_agent import runtime_env

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = REPO_ROOT / "configs/hgm_travel_gemma_full_scale_block_tagged_X100Y180.yaml"
OUT_ROOT = REPO_ROOT / "backbone_slug_validation_32"

CANDIDATES = [
    {"label": "baseline (no override)", "slug": None, "note": None},
    {"label": "qwen/qwen3.6-27b", "slug": "qwen/qwen3.6-27b",
     "note": "newer generation, smaller than the pipeline default"},
    {"label": "qwen/qwen3.8-27b", "slug": "qwen/qwen3.8-27b",
     "note": "newest Qwen generation available"},
    {"label": "google/gemini-3.6-flash", "slug": "google/gemini-3.6-flash",
     "note": "different provider, newer Gemini generation"},
    {"label": "google/gemini-3.8-flash", "slug": "google/gemini-3.8-flash",
     "note": "different provider, newest Gemini generation"},
]

_NO_PLAN_RE = re.compile(r"^\s*plan conversion failed", re.IGNORECASE)

BACKBONE_YAML_TEMPLATE = """# Per-agent backbone LLM config, read by agents/llm_backbone.py.
default:
  model: null
  base_url: null
  temperature: null
  max_output_tokens: null
  reasoning_effort: null
agents:
  flight: {{}}
  train: {{}}
  sightseeing:{sightseeing_override}
  accounting: {{}}
"""


def _sightseeing_override_text(slug: str | None) -> str:
    if slug is None:
        return " {}"
    return f"\n    model: {slug}\n    base_url: https://openrouter.ai/api/v1"


cfg = cfg_mod.load(str(CONFIG))
runtime_env.apply_all(cfg)
fw = cfg_mod.build_components(cfg)

FIXED_32 = sorted(fw.train_case_ids, key=int)[:32]
print(f"Fixed 32-case batch (same for every candidate): {FIXED_32}", flush=True)

OUT_ROOT.mkdir(exist_ok=True)
results = []

for cand in CANDIDATES:
    label = cand["label"]
    slug = cand["slug"]
    cand_dir_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", label)
    out_dir = (OUT_ROOT / cand_dir_name).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(
        fw.seed_dir, out_dir / "task_agent",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (out_dir / "task_agent" / "mas_llm_backbone.yaml").write_text(
        BACKBONE_YAML_TEMPLATE.format(
            sightseeing_override=_sightseeing_override_text(slug)
        ),
        encoding="utf-8",
    )
    (out_dir / "logs").mkdir(exist_ok=True)

    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    result = fw.evaluator.run(out_dir, fw.benchmark_dir, case_ids=FIXED_32)
    elapsed = time.time() - t0

    no_plan_count = sum(
        1 for c in result.per_case
        if isinstance(c.error, str) and _NO_PLAN_RE.search(c.error)
    )
    no_plan_rate = no_plan_count / len(result.per_case) if result.per_case else None

    per_case = [
        {"case_id": c.case_id, "passed": c.passed, "score": c.score, "error": c.error}
        for c in result.per_case
    ]
    (OUT_ROOT / f"{cand_dir_name}.json").write_text(
        json.dumps(
            {
                "label": label, "slug": slug, "note": cand["note"],
                "composite_score": result.score, "passed": result.passed,
                "failed": result.failed, "crashed": result.crashed,
                "no_plan_rate": no_plan_rate, "elapsed_s": elapsed,
                "per_case": per_case,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"composite_score={result.score:.4f} no_plan_rate={no_plan_rate:.4f} "
        f"passed={result.passed} failed={result.failed} crashed={result.crashed} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )
    results.append({
        "label": label, "slug": slug, "composite_score": result.score,
        "no_plan_rate": no_plan_rate,
    })
    shutil.rmtree(out_dir, ignore_errors=True)

baseline_score = results[0]["composite_score"]
print("\n=== SUMMARY (fixed 32-case batch, all candidates) ===")
print(f"{'candidate':<30} {'score':>8} {'no_plan_rate':>14} {'delta vs baseline':>18}")
for r in results:
    delta = r["composite_score"] - baseline_score
    print(
        f"{r['label']:<30} {r['composite_score']:>8.4f} "
        f"{r['no_plan_rate']:>14.4f} {delta:>+18.4f}"
    )

(OUT_ROOT / "summary.json").write_text(
    json.dumps({"fixed_case_ids": FIXED_32, "results": results}, indent=2),
    encoding="utf-8",
)
print(f"\nOutputs saved under: {OUT_ROOT}")

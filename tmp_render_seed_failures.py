"""Render the failure-analysis report from a round_eval/logs dir, using the
project's categorizer + failure_report builder. Lets us SEE the error logs the
meta-agent would receive for the seed agent, with the updated scorer/categorizer.

    PYTHONPATH=. python3 tmp_render_seed_failures.py <round_eval_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from meta_agent.failure_report import build_failure_report, render_failure_report
from projects.shopping.shopping_error_categorizer import categorize_errors

round_dir = Path(sys.argv[1])
log_dir = round_dir / "logs"
cases = []
for f in sorted(log_dir.glob("case_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    cases.append(
        SimpleNamespace(
            case_id=d.get("case_id"),
            score=float(d.get("score") or 0.0),
            error=d.get("error"),
            details=d.get("details") or {},
        )
    )

print(f"loaded {len(cases)} case logs from {log_dir}\n")

# Per-case quick view of the NEW details fields.
print("=" * 80)
print("PER-CASE details (new fields highlighted)")
print("=" * 80)
for c in sorted(cases, key=lambda x: str(x.case_id)):
    det = c.details
    mfc = det.get("missing_feature_categories") or {}
    bc = det.get("budget_check") or {}
    own = det.get("coupon_ownership") or {}
    print(f"\n[{c.case_id}] score={c.score:.3f} level={det.get('level')}")
    print(f"  missing_products: {det.get('missing_products')}")
    print(f"  extra_products:   {det.get('extra_products')}")
    print(f"  missing_feature_categories -> {list(mfc.keys())}")
    for cat, slot in mfc.items():
        preds = slot.get("predicates") or []
        ps = ", ".join(
            f"{p.get('field')} {p.get('operator')} {p.get('operator_value')}"
            if p.get("operator") else f"{p.get('field')}"
            for p in preds
        )
        print(f"      - {cat}: [{ps}]  pids={slot.get('product_ids')}")
    print(f"  budget_check:     {bc}")
    if own.get("applied_not_owned") or own.get("over_owned_qty"):
        print(f"  coupon_ownership: {own}")

# Full rendered failure report (what the meta-agent's prompt shows).
cats = categorize_errors(cases)
report = build_failure_report(cases, cats)
print("\n" + "=" * 80)
print("RENDERED FAILURE REPORT (meta-agent prompt view)")
print("=" * 80)
print(render_failure_report(report) if report else "(no report — nothing failing)")

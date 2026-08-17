"""shopping_mas scorer — set-intersection match on product ids + coupons.

Copied from projects/shopping/benchmark/scorer.py (the sibling single-agent
project's scorer), renamed for this project's own registration. No logic
changes -- this scorer is agent-topology-agnostic: it only reads
``agent_output.result`` (the final cart, as a dict or JSON string) and
``validation_cases.json`` off disk, neither of which depends on how many
agents produced the cart.

No LLM call at scoring time (the agent's effect is recorded in the
per-case ``cart.json``; the scorer compares directly against
``validation_cases.json``). Ultimately ports
`/users/n.tzou/cl/shopping_agent/evaluation/evaluation_pipeline.py::
evaluate_single_case` verbatim (via the sibling project), just adapted to
read the cart from ``agent_output.result`` instead of disk.

Registered as ``shopping_mas_default`` via the meta-agent's component
registry. Same shape contract as travel's/shopping's scorer:

  ``score(case, agent_output) -> {"score", "passed", "details"}``
  ``aggregate(per_case, trace_events) -> dict``

``aggregate`` lands on ``AgentFeedback.project_metrics`` and reports
the per-level breakdown that the optimizer/strategy can use.
"""
from __future__ import annotations

import json
import re
from functools import reduce
from typing import Any

from meta_agent.registry import register


# Map a requirement-feature ``field`` (as it appears in validation_cases.json's
# meta_info[*].features) to a readable error category. Defined from the field
# names present in the shopping data (brand/color/rating.*/sales_volume.*/...),
# NOT copied from any external categorizer. Used only to enrich diagnostics in
# ``details["failure_causes"]["feature_mismatch"]`` — never affects the score.
_FIELD_TO_CATEGORY = {
    "brand": "brand",
    "color": "color",
    "name": "name",
    "size": "size",
    "stock_quantity": "stock",
    "suitable_season": "season",
    "target_demographic": "demographic",
    "transport_time": "delivery_time",
    "price": "price",
    "rating.average_score": "rating_score",
    "rating.total_reviews": "review_count",
    "sales_volume.monthly": "sales_volume",
    "sales_volume.total": "sales_volume",
}


def _field_to_category(field: str) -> str:
    """Readable category for a requirement-feature field. Prefix rules cover
    the star-distribution / rating / sales families; unknown fields degrade to
    a slugified field name so nothing is silently dropped."""
    if not field:
        return "unknown"
    if field in _FIELD_TO_CATEGORY:
        return _FIELD_TO_CATEGORY[field]
    if field.startswith("rating.distribution."):
        return "review_distribution"
    if field.startswith("sales_volume."):
        return "sales_volume"
    if field.startswith("rating."):
        return "rating_score"
    return field.replace(".", "_")


def _parse_cart(agent_output: Any) -> dict[str, Any]:
    """``agent_output`` may be an ``AgentOutput`` (with a JSON-string
    ``result``), a raw JSON string, or a dict. Return a dict."""
    raw = getattr(agent_output, "result", agent_output)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _case_dir(case: dict[str, Any]):
    """Resolve the on-disk case directory for ``case``. The scorer runs in
    the parent process (so we don't go through ``_db.case_dir``, which reads
    env vars only set in the evaluator's child) — level/sample_id come off the
    case dict's own ``env``, only the database root may come from os.environ.
    Returns a ``Path`` or ``None`` when the case isn't locatable."""
    env = case.get("env") or {}
    level = env.get("SHOPPING_LEVEL")
    sample_id = env.get("SHOPPING_SAMPLE_ID")
    if not level or not sample_id:
        return None
    import os
    from pathlib import Path

    root_env = os.environ.get("SHOPPING_DATABASE_ROOT")
    if root_env:
        root = Path(root_env)
    else:
        # Project-relative default. scorer_impl.py at
        # projects/shopping_mas/adapter/scorer_impl.py -> repo root is three
        # parents up; data lives at projects/shopping_mas/data (a symlink to
        # a shared, verified data source -- see the project's own data/ entry).
        root = Path(__file__).resolve().parents[3] / "projects" / "shopping_mas" / "data"
    return root / f"database_level{level}" / f"case_{sample_id}"


def _load_validation(case: dict[str, Any]) -> dict[str, Any]:
    """Read the case's validation_cases.json off disk."""
    case_dir = _case_dir(case)
    if case_dir is None:
        return {}
    validation = case_dir / "validation_cases.json"
    if not validation.exists():
        return {}
    try:
        return json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_products(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the case's products.jsonl off disk, return ``{product_id: product}``.
    Used to recover full attributes for the agent's (wrong) cart picks — the
    cart itself only carries product_id/name/quantity/price. Degrades to ``{}``
    when the case/file is unavailable (caller falls back to all-features)."""
    case_dir = _case_dir(case)
    if case_dir is None:
        return {}
    path = case_dir / "products.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = p.get("product_id") if isinstance(p, dict) else None
                if pid:
                    out[pid] = p
    except OSError:
        return {}
    return out


def _load_user_info(case: dict[str, Any]) -> dict[str, Any]:
    """Read the case's user_info.json off disk (for owned-coupon checks)."""
    case_dir = _case_dir(case)
    if case_dir is None:
        return {}
    path = case_dir / "user_info.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Predicate evaluation — used only to NARROW the feature_mismatch causes to the
# features the agent's best near-miss product actually violates. Never affects
# the score. ``transport_time`` is intentionally not stored on products (it is
# computed from origin+dest+provider), so _nested returns None for it and it is
# treated as unverifiable → always counted as violated (kept reported).
# --------------------------------------------------------------------------- #
def _nested(obj: Any, key_path: str) -> Any:
    """Resolve a dotted field path (e.g. ``rating.distribution.3_star``)."""
    try:
        return reduce(
            lambda d, k: d.get(k) if isinstance(d, dict) else None,
            key_path.split("."),
            obj,
        )
    except (TypeError, AttributeError):
        return None


def _to_num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _satisfies(
    product: dict[str, Any],
    predicate: dict[str, Any],
    *,
    transport_days: int | None = None,
) -> bool:
    """True iff ``product`` satisfies ``predicate`` ({field, operator,
    operator_value}). Compares against ``operator_value`` (the threshold), NOT
    ``value`` (the gold product's own value). Missing field / unknown operator /
    non-numeric comparison ⇒ False (counts as a violation).

    ``transport_time`` is not stored on products; pass the precomputed
    ``transport_days`` for the candidate so the predicate can be evaluated. When
    ``transport_days is None`` (uncomputable) the transport predicate stays
    conservatively violated (returns False)."""
    field = str(predicate.get("field") or "")
    op = predicate.get("operator")
    target = predicate.get("operator_value")
    if not field or not op:
        return False
    if field == "transport_time":
        pv: Any = transport_days
    else:
        pv = _nested(product, field)
    if pv is None:
        return False  # unresolvable (incl. uncomputed transport_time) -> violated

    if op == "equals":
        nf_pv, nf_t = _to_num(pv), _to_num(target)
        if nf_pv is not None and nf_t is not None:
            return nf_pv == nf_t
        return str(pv).strip().lower() == str(target).strip().lower()
    if op == "contains":
        return str(target).strip().lower() in str(pv).lower()

    nf_pv, nf_t = _to_num(pv), _to_num(target)
    if nf_pv is None or nf_t is None:
        return False
    if op == "greater_than":
        return nf_pv > nf_t
    if op == "greater_than_or_equal":
        return nf_pv >= nf_t
    if op == "less_than":
        return nf_pv < nf_t
    if op == "less_than_or_equal":
        return nf_pv <= nf_t
    return False  # unknown operator -> conservative violation


# Comparator phrase -> operator, longest phrases first so "no more than" wins
# over "more than". Used to recover operators for level-2/3 sub-queries, whose
# meta_info features carry only {field, value} (no operator/operator_value).
_COMPARATORS: list[tuple[str, str]] = [
    ("no less than", "greater_than_or_equal"),
    ("no more than", "less_than_or_equal"),
    ("at least", "greater_than_or_equal"),
    ("at most", "less_than_or_equal"),
    ("greater than", "greater_than"),
    ("more than", "greater_than"),
    ("less than", "less_than"),
    ("fewer than", "less_than"),
    ("over", "greater_than"),
    ("above", "greater_than"),
    ("exceeds", "greater_than"),
    ("under", "less_than"),
    ("below", "less_than"),
]
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _subquery_numeric_clauses(sub_query: str) -> list[tuple[str, float]]:
    """Extract ``(operator, threshold)`` clauses from a sub-query string by
    scanning each comparator phrase and reading the number that follows it."""
    clauses: list[tuple[str, float]] = []
    low = (sub_query or "").lower()
    for phrase, op in _COMPARATORS:
        start = 0
        while True:
            i = low.find(phrase, start)
            if i < 0:
                break
            start = i + len(phrase)
            m = _NUM_RE.search(low, start)
            if m:
                n = _to_num(m.group(0).replace(",", ""))
                if n is not None:
                    clauses.append((op, n))
    return clauses


def _num_op_holds(op: str, lhs: float, thr: float) -> bool:
    """Direct numeric operator check (no field resolution) — for sanity-checking
    a parsed clause against a gold value."""
    if op == "greater_than":
        return lhs > thr
    if op == "greater_than_or_equal":
        return lhs >= thr
    if op == "less_than":
        return lhs < thr
    if op == "less_than_or_equal":
        return lhs <= thr
    return False


# Field -> natural-language phrases, for STRUCTURAL operator↔field linking on
# level-2/3 sub-queries (binds the comparator nearest a field's own phrase, so a
# sibling clause's number can't be stolen by value proximity — see _link_clause).
_FIELD_PHRASES: dict[str, tuple[str, ...]] = {
    "rating.distribution.1_star": ("1-star", "one-star", "1 star", "one star"),
    "rating.distribution.2_star": ("2-star", "two-star", "2 star", "two star"),
    "rating.distribution.3_star": ("3-star", "three-star", "3 star", "three star"),
    "rating.distribution.4_star": ("4-star", "four-star", "4 star", "four star"),
    "rating.distribution.5_star": ("5-star", "five-star", "5 star", "five star"),
    "rating.average_score": ("average rating", "average score", "average review"),
    "rating.total_reviews": ("total reviews", "total number of reviews", "total review"),
    "sales_volume.monthly": ("monthly sales", "monthly sale"),
    "sales_volume.total": ("total sales", "total sale", "total sales volume"),
    "stock_quantity": ("stock",),
    "transport_time": ("transport time", "delivery", "arrive", "ship"),
    "price": ("price", "cost"),
}


def _subquery_clauses_pos(low: str) -> list[tuple[str, float, int]]:
    """Like ``_subquery_numeric_clauses`` but also records the char position of
    each clause's number, for distance-based structural linking."""
    out: list[tuple[str, float, int]] = []
    for phrase, op in _COMPARATORS:
        start = 0
        while True:
            i = low.find(phrase, start)
            if i < 0:
                break
            start = i + len(phrase)
            m = _NUM_RE.search(low, start)
            if m:
                n = _to_num(m.group(0).replace(",", ""))
                if n is not None:
                    out.append((op, n, m.start()))
    return out


def _link_clause(
    low: str, field: str, clauses_pos: list[tuple[str, float, int]]
) -> tuple[str, float] | None:
    """Structurally bind ``field`` to the (operator, threshold) clause whose
    number sits NEAREST the field's own phrase in the sentence. Returns None when
    the field has no known phrase or no phrase occurrence."""
    phrases = _FIELD_PHRASES.get(field)
    if not phrases or not clauses_pos:
        return None
    positions = [low.find(p) for p in phrases]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return None
    anchor = min(positions)
    op, thr, _ = min(clauses_pos, key=lambda c: abs(c[2] - anchor))
    return op, thr


_CONTAINS_RE = re.compile(r"contains\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


def _parse_predicates_from_subquery(
    sub_query: str, features: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Infer ``{field, operator, operator_value}`` predicates for level-2/3
    features (which carry only ``field`` + gold ``value``) from the sub-query
    text. A feature whose operator can't be recovered is emitted WITHOUT an
    operator so the caller treats it as non-narrowable (always reported)."""
    clauses = _subquery_numeric_clauses(sub_query)
    low = (sub_query or "").lower()
    clauses_pos = _subquery_clauses_pos(low)
    preds: list[dict[str, Any]] = []
    for ft in features or []:
        if not isinstance(ft, dict):
            continue
        field = str(ft.get("field") or "")
        if not field:
            continue
        gold = ft.get("value")

        if field == "name":
            # The gold ``value`` for name is a paraphrase, not the threshold;
            # the real substring is quoted in the sub-query ("contains 'X'").
            m = _CONTAINS_RE.search(sub_query or "")
            if m:
                preds.append(
                    {"field": field, "operator": "contains", "operator_value": m.group(1)}
                )
            else:
                preds.append({"field": field, "operator": None, "operator_value": gold})
            continue

        gold_num = _to_num(gold)
        if gold_num is None:
            # Non-name string field (brand/color/size/season/demographic) — the
            # gold value is the exact match target.
            preds.append({"field": field, "operator": "equals", "operator_value": gold})
            continue

        # Numeric field. PRIMARY: structural phrase-adjacency link (bind the
        # comparator nearest the field's own phrase), accepted only if the gold
        # value satisfies it. FALLBACK: value-anchor (nearest satisfied threshold).
        chosen: tuple[str, float] | None = None
        lc = _link_clause(low, field, clauses_pos)
        if lc is not None and _num_op_holds(lc[0], gold_num, lc[1]):
            chosen = lc
        else:
            best_gap: float | None = None
            for op, thr in clauses:
                if _num_op_holds(op, gold_num, thr):
                    gap = abs(gold_num - thr)
                    if best_gap is None or gap < best_gap:
                        chosen, best_gap = (op, thr), gap
        if chosen is not None:
            preds.append(
                {"field": field, "operator": chosen[0], "operator_value": chosen[1]}
            )
        else:
            # Non-narrowable: keep the field but no operator -> always reported.
            preds.append({"field": field, "operator": None, "operator_value": gold})
    return preds


def _predicates_for(req: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalized ``{field, operator, operator_value}`` predicates for one
    meta_info entry. Level-1 features carry operator/operator_value natively;
    level-2/3 features carry only field+value, so operators are recovered from
    the sub_query text (a feature whose operator can't be recovered keeps
    ``operator=None`` and is treated as non-narrowable / always violated)."""
    feats = [
        f
        for f in (req.get("features") or [])
        if isinstance(f, dict) and f.get("field")
    ]
    if any(f.get("operator") for f in feats):
        return [
            {
                "field": str(f.get("field")),
                "operator": f.get("operator"),
                "operator_value": f.get("operator_value"),
            }
            for f in feats
        ]
    return _parse_predicates_from_subquery(str(req.get("sub_query") or ""), feats)


# Budget is stated only in the query's natural language (no structured field),
# e.g. "My budget is between X and Y", "a budget of X to Y", "the total price is
# no less than X and no more than Y", "in the range of X to Y", "keep my total
# spending between X and Y".
_BUDGET_ANCHORS = (
    "budget", "no less than", "the range of", "price is", "spending", "spend",
)


def _parse_budget(query: str) -> tuple[str, tuple[float, float] | None]:
    """Return ``("ok", (min, max))`` when a budget range can be read from the
    query, ``("unparsed", None)`` when a budget phrase is present but two numbers
    can't be recovered, or ``("no_budget", None)`` when no budget phrase exists."""
    low = (query or "").lower()
    anchor = min(
        (low.find(a) for a in _BUDGET_ANCHORS if a in low),
        default=-1,
    )
    if anchor < 0:
        return "no_budget", None
    nums = [
        _to_num(m.group(0).replace(",", ""))
        for m in _NUM_RE.finditer(low, anchor)
    ]
    nums = [n for n in nums if n is not None]
    if len(nums) < 2:
        return "unparsed", None
    lo, hi = sorted(nums[:2])
    return "ok", (lo, hi)


# --------------------------------------------------------------------------- #
# Transport-time computation — ported verbatim from
# projects/shopping/tools/calculate_transport_time.py (that module imports
# platform_core at load, so it can't be imported from the scorer's process).
# Lets us evaluate transport_time predicates against a candidate product instead
# of treating them as always-violated. Reconciles 108/112 (96%) with the GT
# meta_info transport values; the ~4% remainder returns None -> conservative.
# --------------------------------------------------------------------------- #
_TT_PROVINCE_ALIASES = {
    "beijing": ["beijing", "bj", "北京"], "shanghai": ["shanghai", "sh", "上海"],
    "tianjin": ["tianjin", "tj", "天津"], "chongqing": ["chongqing", "cq", "重庆"],
    "hebei": ["hebei", "ji", "河北"], "shanxi": ["shanxi", "jin", "山西"],
    "liaoning": ["liaoning", "liao", "辽宁"], "jilin": ["jilin", "ji_ln", "吉林"],
    "heilongjiang": ["heilongjiang", "hei", "黑龙江"], "jiangsu": ["jiangsu", "su", "江苏"],
    "zhejiang": ["zhejiang", "zhe", "浙江"], "anhui": ["anhui", "wan", "安徽"],
    "fujian": ["fujian", "min", "福建"], "jiangxi": ["jiangxi", "gan", "江西"],
    "shandong": ["shandong", "lu", "山东"], "henan": ["henan", "yu", "河南"],
    "hubei": ["hubei", "e", "湖北"], "hunan": ["hunan", "xiang", "湖南"],
    "guangdong": ["guangdong", "yue", "gd", "广东"], "hainan": ["hainan", "qiong", "海南"],
    "sichuan": ["sichuan", "chuan", "shu", "四川"], "guizhou": ["guizhou", "qian", "gui_gz", "贵州"],
    "yunnan": ["yunnan", "yun", "dian", "云南"], "shaanxi": ["shaanxi", "shan", "qin", "陕西"],
    "gansu": ["gansu", "gan_gs", "甘肃"], "qinghai": ["qinghai", "qing", "青海"],
    "inner mongolia": ["inner mongolia", "neimenggu", "meng", "内蒙古"],
    "guangxi": ["guangxi", "gui", "广西"], "tibet": ["tibet", "xizang", "zang", "西藏"],
    "ningxia": ["ningxia", "ning", "宁夏"], "xinjiang": ["xinjiang", "xin", "新疆"],
    "hongkong": ["hongkong", "hk", "xianggang", "香港"], "macau": ["macau", "mo", "aomen", "澳门"],
    "taiwan": ["taiwan", "tw", "台湾"],
}
_TT_NORM = {a: std for std, al in _TT_PROVINCE_ALIASES.items() for a in al}
_TT_REGION_MAP = {
    "beijing": "NC", "tianjin": "NC", "hebei": "NC", "shanxi": "NC", "inner mongolia": "NC",
    "liaoning": "NE", "jilin": "NE", "heilongjiang": "NE",
    "shanghai": "EC", "jiangsu": "EC", "zhejiang": "EC", "anhui": "EC",
    "fujian": "EC", "jiangxi": "EC", "shandong": "EC",
    "henan": "CC", "hubei": "CC", "hunan": "CC",
    "guangdong": "SC", "guangxi": "SC", "hainan": "SC",
    "hongkong": "SC", "macau": "SC", "taiwan": "SC",
    "sichuan": "SW", "chongqing": "SW", "guizhou": "SW", "yunnan": "SW", "tibet": "SW",
    "shaanxi": "NW", "gansu": "NW", "qinghai": "NW", "ningxia": "NW", "xinjiang": "NW",
}
_TT_BASE = {
    "NC": {"NC": 1, "NE": 2, "EC": 2, "CC": 2, "SC": 3, "SW": 3, "NW": 3},
    "NE": {"NC": 2, "NE": 1, "EC": 3, "CC": 3, "SC": 4, "SW": 4, "NW": 4},
    "EC": {"NC": 2, "NE": 3, "EC": 1, "CC": 2, "SC": 2, "SW": 3, "NW": 4},
    "CC": {"NC": 2, "NE": 3, "EC": 2, "CC": 1, "SC": 2, "SW": 2, "NW": 3},
    "SC": {"NC": 3, "NE": 4, "EC": 2, "CC": 2, "SC": 1, "SW": 3, "NW": 4},
    "SW": {"NC": 3, "NE": 4, "EC": 3, "CC": 2, "SC": 3, "SW": 1, "NW": 3},
    "NW": {"NC": 3, "NE": 4, "EC": 4, "CC": 3, "SC": 4, "SW": 3, "NW": 1},
}
_TT_PROVIDER_MODIFIERS = {
    "sf express": -2, "sf": -2, "sf_express": -2,
    "jd logistics": -1, "jd": -1, "jd_logistics": -1,
    "yto express": 1, "yto": 0, "yto_express": 1,
    "zto express": 1, "zto": 0, "zto_express": 1,
    "sto express": 1, "sto": 0, "sto_express": 1,
    "yunda express": 1, "yunda": 0, "yunda_express": 1,
    "cainiao": 1, "china post": 2, "china_post": 2, "ems": 0,
    "deppon express": 0, "deppon": 0, "deppon_express": 0, "default": 0,
}
_TT_REMOTE = {"tibet", "xinjiang", "qinghai", "inner mongolia"}


def _tt_normalize(address: str | None):
    if not address:
        return None
    s = address.lower().replace(" ", "").replace("province", "").replace("city", "")
    if s in _TT_NORM:
        return _TT_NORM[s]
    for alias, std in _TT_NORM.items():
        if alias in s:
            return std
    return None


def _transport_days(origin, dest_province, provider) -> int | None:
    """Estimated delivery days for (origin, destination province, provider), or
    None when either province can't be normalized/mapped (-> conservative)."""
    o, d = _tt_normalize(origin), _tt_normalize(dest_province)
    if not o or not d:
        return None
    o_r, d_r = _TT_REGION_MAP.get(o), _TT_REGION_MAP.get(d)
    if not o_r or not d_r:
        return None
    base = _TT_BASE[o_r][d_r]
    if o in _TT_REMOTE or d in _TT_REMOTE:
        base += 2
    chosen = (provider or "default").lower()
    base += _TT_PROVIDER_MODIFIERS.get(chosen, _TT_PROVIDER_MODIFIERS["default"])
    return max(1, base)


def _product_transport_days(product: dict[str, Any], dest_province) -> int | None:
    si = product.get("shipping_info") or {}
    return _transport_days(si.get("origin"), dest_province, si.get("provider"))


# --------------------------------------------------------------------------- #
# Profile-derived constraints (gender, size) — the agent must infer missing
# size/gender from the user profile. Used to label "user_info_mismatch" misses.
# --------------------------------------------------------------------------- #
def _gender_to_demographic(gender: str | None) -> str | None:
    return {"male": "Men", "female": "Women"}.get((gender or "").strip().lower())


def _infer_size_category(name: str | None) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("shoe", "sneaker", "boot", "loafer", "slipper", "sandal", "heel")):
        return "shoes"
    if any(k in n for k in ("pant", "jean", "shorts", "trouser", "legging", "skirt", "skort")):
        return "bottoms"
    return "tops"


def _size_category(product: dict[str, Any]) -> str:
    """Which profile size applies. A NUMERIC size (36-46) is always a shoe size
    regardless of the product name; letter sizes fall back to name keywords."""
    if _to_num(product.get("size")) is not None:
        return "shoes"
    return _infer_size_category(product.get("name"))


def _size_eq(a, b) -> bool:
    na, nb = _to_num(a), _to_num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a).strip().lower() == str(b).strip().lower()


def _profile_violations(
    product: dict[str, Any], user_info: dict[str, Any], stated_fields: set[str]
) -> list[dict[str, Any]]:
    """Profile constraints a product violates, gated to fields NOT already stated
    in the query (so a stated size/demographic is only ever a feature_mismatch).
    Returns dicts: {attribute, field, expected, actual, [category]}."""
    out: list[dict[str, Any]] = []
    demo = (user_info.get("demographics") or {})
    sizes = ((user_info.get("body_profile") or {}).get("standard_sizes") or {})
    if "target_demographic" not in stated_fields:
        exp = _gender_to_demographic(demo.get("gender"))
        act = product.get("target_demographic")
        if exp and act is not None and str(act).strip().lower() != exp.lower():
            out.append({"attribute": "gender", "field": "target_demographic",
                        "expected": exp, "actual": act})
    if "size" not in stated_fields:
        cat = _size_category(product)
        exp = sizes.get(cat)
        act = product.get("size")
        if exp is not None and act is not None and not _size_eq(exp, act):
            out.append({"attribute": "size", "field": "size",
                        "expected": exp, "actual": act, "category": cat})
    return out


# --------------------------------------------------------------------------- #
# Coupon discount — full per-scope stacking model (the L3 prompt's rules), used
# to compute the cart's true final price, scope validity, and the optimal
# achievable discount. ``_parse_coupon`` ported from add_coupon_to_cart.py.
# --------------------------------------------------------------------------- #
_COUPON_RE = re.compile(r"¥([\d,]+)\s+off\s+every\s+¥([\d,]+)")


def _parse_coupon(name: str) -> tuple[float, float, str] | None:
    """(discount, threshold, scope) where scope ∈ {cross, same, vip}; None if unparseable."""
    m = _COUPON_RE.search(name or "")
    if not m:
        return None
    try:
        disc = float(m.group(1).replace(",", ""))
        thr = float(m.group(2).replace(",", ""))
    except ValueError:
        return None
    scope = "vip" if name.startswith("VIP") else ("same" if name.startswith("Same-brand") else "cross")
    return disc, thr, scope


def _stacking_discount(
    items: list[tuple[float, str]], coupons: list[tuple[str, int]], is_vip: bool
) -> tuple[float, list[str]]:
    """Total discount under per-scope stacking + list of invalid coupon names.
    ``items`` = [(line_total, brand)]; ``coupons`` = [(name, qty)]. Same-brand
    coupons apply to a single brand's subtotal (floor(subtotal/threshold) times),
    deducted from the cart total before Cross-store/VIP apply to the remaining
    total. A coupon copy that can't meet its threshold is invalid."""
    from collections import defaultdict
    brand_sub: dict[str, float] = defaultdict(float)
    total = 0.0
    for val, br in items:
        total += val
        brand_sub[br] += val
    rem_brand = dict(brand_sub)
    disc = 0.0
    invalid: list[str] = []
    for name, qty in coupons:
        c = _parse_coupon(name)
        if not c or c[2] != "same":
            continue
        d, thr, _ = c
        applied = 0
        for _ in range(int(qty or 0)):
            b = max(rem_brand, key=lambda k: rem_brand[k]) if rem_brand else None
            if b is not None and rem_brand[b] >= thr:
                disc += d
                rem_brand[b] -= thr
                applied += 1
            else:
                break
        if applied < int(qty or 0):
            invalid.append(name)
    rem_total = total - disc
    for name, qty in coupons:
        c = _parse_coupon(name)
        if not c or c[2] == "same":
            continue
        d, thr, scope = c
        if scope == "vip" and not is_vip:
            invalid.append(name)
            continue
        applied = 0
        for _ in range(int(qty or 0)):
            if rem_total >= thr:
                disc += d
                rem_total -= thr
                applied += 1
            else:
                break
        if applied < int(qty or 0) and name not in invalid:
            invalid.append(name)
    return round(disc, 2), invalid


def _optimal_discount(
    items: list[tuple[float, str]], owned: dict[str, int], is_vip: bool
) -> float:
    """Max achievable stacking discount on ``items`` using the user's owned
    coupons (small brute-force over owned quantities)."""
    from itertools import product as iproduct
    names = list(owned.keys())
    if not names or not items:
        return 0.0
    cap = 2 if len(names) > 6 else 3
    ranges = [range(0, min(int(owned[n] or 0), cap) + 1) for n in names]
    best = 0.0
    for combo in iproduct(*ranges):
        cs = [(names[i], combo[i]) for i in range(len(names)) if combo[i] > 0]
        best = max(best, _stacking_discount(items, cs, is_vip)[0])
    return round(best, 2)


def _level_objective(level: Any) -> str | None:
    """A POINTER to the level's governing system prompt — not the prompt text.

    The editor already reads the agent's ``workflow.py`` (which contains
    ``_SYSTEM_PROMPT_LEVEL_{1,2,3}``) in its current-sources section, so the
    error log only needs to name which one governs this case."""
    lv = _to_num(level)
    lv = int(lv) if lv is not None else None
    if lv is None:
        return None
    return f"L{lv} — _SYSTEM_PROMPT_LEVEL_{lv} in workflow.py"


def _render_error_log(details: dict[str, Any]) -> str:
    """The entire per-case error log rendered to text (all fired failure causes,
    budget, coupon and final-price diagnostics). Delegates to the categorizer,
    which owns the message templates. Lazy import to avoid load-order coupling."""
    try:
        from projects.shopping_mas.shopping_mas_error_categorizer import render_case_error_log
        return render_case_error_log(details)
    except Exception:
        return ""


@register("scorer", "shopping_mas_default")
class ShoppingMasScorer:
    """Shopping scorer + round-level aggregator.

    Matches the reference's scoring (`evaluation_pipeline.py:85`):
    score = (matched_products + matched_coupons) / (expected_products + expected_coupons).
    A product is matched when its id appears in both the cart and
    ground truth. A coupon is matched when the (name, quantity) pair
    appears identically. ``case_score = 1.0`` only when ``matched ==
    expected``; otherwise 0.0 (kept as a separate bool inside details).
    """

    def score(self, case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
        cart = _parse_cart(agent_output)
        validation = _load_validation(case)

        cart_items = cart.get("items") or []
        gt_products = validation.get("ground_truth_products") or []
        gt_coupons = validation.get("ground_truth_coupons") or {}
        cart_coupons = cart.get("used_coupons") or []

        if not validation:
            # Without a ground-truth file we can't score anything.
            return {
                "score": 0.0,
                "passed": False,
                "details": {
                    "error": "validation_cases.json not found for this case",
                    "level": (case.get("meta_info") or {}).get("level"),
                    "composite_score": 0.0,
                    "matched_count": 0,
                    "expected_count": 0,
                },
            }

        cart_pids = {
            it.get("product_id") for it in cart_items if it.get("product_id")
        }
        gt_pids = {
            p.get("product_id") for p in gt_products if p.get("product_id")
        }
        matched_products = cart_pids & gt_pids

        cart_coupon_names: set[str] = set()
        matched_coupons = 0
        matched_coupon_names: set[str] = set()
        coupon_details: list[dict[str, Any]] = []
        for c in cart_coupons:
            name = c.get("coupon_name", "")
            qty = int(c.get("quantity", 0))
            cart_coupon_names.add(name)
            expected_qty = int(gt_coupons.get(name, 0))
            ok = name in gt_coupons and qty == expected_qty
            if ok:
                matched_coupons += 1
                matched_coupon_names.add(name)
            coupon_details.append(
                {
                    "coupon_name": name,
                    "quantity": qty,
                    "expected_quantity": expected_qty,
                    "match": ok,
                }
            )

        matched_count = len(matched_products) + matched_coupons
        expected_count = len(gt_pids) + len(gt_coupons)
        composite = matched_count / expected_count if expected_count else 0.0
        passed = matched_count == expected_count and expected_count > 0
        coupon_score = (
            matched_coupons / len(gt_coupons) if gt_coupons else 0.0
        )

        gt_coupon_names = set(gt_coupons.keys())
        extra_products = list(cart_pids - gt_pids)
        missing_products = list(gt_pids - cart_pids)
        extra_coupons = list(cart_coupon_names - gt_coupon_names)
        missing_coupons = list(gt_coupon_names - matched_coupon_names)

        meta_info = validation.get("meta_info") or []
        pid_to_idx = {
            p.get("product_id"): i
            for i, p in enumerate(gt_products)
            if p.get("product_id")
        }

        # Per-missing-product CAUSE classification (diagnostics only; never
        # changes the score). For each MISSING gold product we find the agent's
        # best near-miss (the extra product violating the fewest stated features,
        # transport_time COMPUTED), then assign exactly ONE cause in strict order:
        #   feature_mismatch -> user_info_mismatch -> not_cheapest -> ambiguous,
        # or missing_product when the agent added no comparable substitute at all.
        products_by_id = _load_products(case)
        user_info = _load_user_info(case)
        dest_province = (user_info.get("shipping_addresses") or {}).get("province")
        level = (case.get("meta_info") or {}).get("level")
        candidates = [
            products_by_id[pid] for pid in extra_products if pid in products_by_id
        ]

        def _new_slot() -> dict[str, Any]:
            return {"sub_queries": [], "fields": [], "predicates": [], "product_ids": []}

        def _accumulate(slot: dict[str, Any], sub_q: str, pid: str,
                        preds: list[dict[str, Any]]) -> None:
            if sub_q and sub_q not in slot["sub_queries"]:
                slot["sub_queries"].append(sub_q)
            if pid not in slot["product_ids"]:
                slot["product_ids"].append(pid)
            for p in preds:
                f = p.get("field")
                if f and f not in slot["fields"]:
                    slot["fields"].append(f)
                if p not in slot["predicates"]:
                    slot["predicates"].append(p)

        def _req_for(pid: str) -> dict[str, Any]:
            idx = pid_to_idx.get(pid)
            return (meta_info[idx] if isinstance(idx, int) and idx < len(meta_info)
                    and isinstance(meta_info[idx], dict) else {})

        product_causes: dict[str, Any] = {
            "feature_mismatch": {},  # {feature_category: slot}
            "user_info_mismatch": {"gender": {**_new_slot(), "violations": []},
                                   "size": {**_new_slot(), "violations": []}},
            "not_cheapest": {**_new_slot(), "gaps": []},
            "ambiguous": _new_slot(),
            "missing_product": _new_slot(),
        }
        # Working alias used by the accumulation loop below.
        missing_feature_categories: dict[str, dict[str, Any]] = product_causes["feature_mismatch"]

        for pid in missing_products:
            req = _req_for(pid)
            sub_q = str(req.get("sub_query") or "")
            preds = _predicates_for(req)
            g = products_by_id.get(pid)
            stated = {p["field"] for p in preds}

            # Step 0 — no substitute attempted.
            if not candidates:
                _accumulate(product_causes["missing_product"], sub_q, pid, preds)
                continue

            # Step 1 — best near-miss (fewest violated stated predicates).
            best: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
            for cand in candidates:
                td = _product_transport_days(cand, dest_province)
                v = [p for p in preds if not _satisfies(cand, p, transport_days=td)]
                if best is None or len(v) < len(best[1]):
                    best = (cand, v)
            cand, violated = best

            # Step 2 — feature_mismatch (a stated feature, incl. real transport).
            if violated:
                for p in violated:
                    slot = missing_feature_categories.setdefault(
                        _field_to_category(p["field"]), _new_slot())
                    _accumulate(slot, sub_q, pid, [p])
                continue

            # Step 3 — user_info_mismatch (a profile constraint the GOLD satisfies
            # but the pick fails — the disambiguation guard).
            gold_attrs = {v["attribute"] for v in (_profile_violations(g, user_info, stated) if g else [])}
            real_pv = [v for v in _profile_violations(cand, user_info, stated)
                       if v["attribute"] not in gold_attrs]
            if real_pv:
                for v in real_pv:
                    slot = product_causes["user_info_mismatch"][v["attribute"]]
                    _accumulate(slot, sub_q, pid, [])
                    if v not in slot["violations"]:
                        slot["violations"].append(v)
                continue

            # Step 4 — not_cheapest (pick is a pricier valid substitute than gold).
            # PER-ITEM only at L1/L3. At L2 cost is a cart-level property (cheapest
            # combination within budget), so per-item price comparison is the wrong
            # lens — skip it (the miss falls to ambiguous) and rely on the cart-level
            # `not_cheapest_cart_level` signal below.
            if level != 2:
                pp, gg = _to_num(cand.get("price")), _to_num((g or {}).get("price"))
                if pp is not None and gg is not None and pp > gg:
                    slot = product_causes["not_cheapest"]
                    _accumulate(slot, sub_q, pid, [])
                    slot["gaps"].append({"gold_id": pid, "picked_id": cand.get("product_id"),
                                         "gold_price": gg, "picked_price": pp,
                                         "gap": round(pp - gg, 2)})
                    continue

            # Step 5 — residual ambiguous (meets everything checkable, not pricier).
            _accumulate(product_causes["ambiguous"], sub_q, pid, preds)

        # Budget feedback (levels 2/3) — diagnostics only, never affects score.
        # Budget lives only in the query text; the gold cart total is always
        # in-budget, so we use it to sanity-check the parse.
        gt_total = sum((_to_num(p.get("price")) or 0.0) for p in gt_products)
        cart_total = _to_num((cart.get("summary") or {}).get("total_price"))
        if cart_total is None:
            cart_total = sum(
                (_to_num(it.get("price")) or 0.0) * (_to_num(it.get("quantity")) or 1.0)
                for it in cart_items
            )
        b_status, bounds = _parse_budget(validation.get("query") or "")
        if b_status == "ok" and bounds is not None:
            bmin, bmax = bounds
            if not (bmin <= gt_total <= bmax):
                # Parsed bounds are inconsistent with the known in-budget gold
                # total -> the parse grabbed the wrong numbers; don't trust it.
                budget_check: dict[str, Any] = {"status": "unparsed"}
            else:
                status = (
                    "under" if cart_total < bmin
                    else "over" if cart_total > bmax
                    else "within"
                )
                budget_check = {
                    "budget_min": bmin,
                    "budget_max": bmax,
                    "cart_total": cart_total,
                    "gt_total": gt_total,
                    "status": status,
                    "over_amount": round(cart_total - bmax, 2) if status == "over" else 0.0,
                    "under_amount": round(bmin - cart_total, 2) if status == "under" else 0.0,
                    "cost_gap": round(cart_total - gt_total, 2),
                    # L2 cart-level not-cheapest: the cart costs more than the cheapest
                    # valid combination (gold total), regardless of band. This is the
                    # right cost lens for L2 (cost is a whole-cart property).
                    "frame": ("not_cheapest_cart_level"
                              if level == 2 and cart_total > gt_total else None),
                }
        else:
            budget_check = {"status": b_status}

        # Coupon-ownership feedback (level 3) — flag applied coupons the user
        # doesn't own or over-applies.
        owned_coupons = user_info.get("coupons") or {}
        applied_not_owned: list[str] = []
        over_owned_qty: list[dict[str, Any]] = []
        for c in cart_coupons:
            name = c.get("coupon_name", "")
            qty = int(c.get("quantity", 0))
            if name not in owned_coupons:
                applied_not_owned.append(name)
            elif qty > int(owned_coupons.get(name, 0)):
                over_owned_qty.append(
                    {"name": name, "applied": qty, "owned": int(owned_coupons.get(name, 0))}
                )
        coupon_ownership = {
            "applied_not_owned": applied_not_owned,
            "over_owned_qty": over_owned_qty,
        }

        # User profile the agent was expected to infer missing details from
        # (size/gender) — surfaced in the error log so a user_info mismatch is
        # actionable (shows what the correct gender/size/coupons were).
        _demos = user_info.get("demographics") or {}
        user_profile = {
            "gender": _demos.get("gender"),
            "target_demographic": _gender_to_demographic(_demos.get("gender")),
            "standard_sizes": (user_info.get("body_profile") or {}).get("standard_sizes") or {},
            "is_vip": bool(user_info.get("is_vip")),
            "owned_coupons": owned_coupons,
            "shipping_province": dest_province,
        }

        # L3 final-price (per-scope stacking): the agent's true final price, the
        # GT reference, the best achievable on the agent's own cart, and a
        # product-vs-coupon attribution for the gap. Diagnostics only.
        final_price_check: dict[str, Any] = {}
        if level == 3:
            is_vip = bool(user_info.get("is_vip"))
            agent_items: list[tuple[float, str]] = []
            for it in cart_items:
                pr = _to_num(it.get("price")) or 0.0
                qy = _to_num(it.get("quantity")) or 1.0
                br = (products_by_id.get(it.get("product_id")) or {}).get("brand", "?")
                agent_items.append((pr * qy, br))
            used = [(c.get("coupon_name", ""), int(c.get("quantity", 0))) for c in cart_coupons]
            a_disc, a_invalid = _stacking_discount(agent_items, used, is_vip)
            a_base = round(sum(x for x, _ in agent_items), 2)
            a_final = round(max(0.0, a_base - a_disc), 2)
            gt_items = [(_to_num(p.get("price")) or 0.0, p.get("brand", "?")) for p in gt_products]
            g_disc, _g_inv = _stacking_discount(gt_items, list(gt_coupons.items()), is_vip)
            g_base = round(sum(x for x, _ in gt_items), 2)
            g_final = round(max(0.0, g_base - g_disc), 2)
            opt = _optimal_discount(agent_items, owned_coupons, is_vip)
            product_driven = bool(missing_products or extra_products) or abs(a_base - g_base) > 1e-6
            coupon_driven = bool(missing_coupons or extra_coupons or a_invalid) or (opt - a_disc > 1e-6)
            attribution = ("both" if product_driven and coupon_driven
                           else "product" if product_driven
                           else "coupon" if coupon_driven else "none")
            final_price_check = {
                "model": "stacking",
                "agent": {"base": a_base, "discount": a_disc, "final": a_final,
                          "valid": not a_invalid, "invalid_coupons": a_invalid},
                "gt": {"base": g_base, "discount": g_disc, "final": g_final},
                "optimal_on_agent_cart": {"discount": opt,
                                          "final": round(max(0.0, a_base - opt), 2)},
                "final_price_gap": round(a_final - g_final, 2),
                "coupon_suboptimality": round(max(0.0, opt - a_disc), 2),
                "attribution": attribution,
            }

        # Emit only the failure causes that actually fired (omit empty buckets).
        _ui = {a: product_causes["user_info_mismatch"][a]
               for a in ("gender", "size")
               if product_causes["user_info_mismatch"][a]["product_ids"]}
        failure_causes: dict[str, Any] = {}
        if product_causes["feature_mismatch"]:
            failure_causes["feature_mismatch"] = product_causes["feature_mismatch"]
        if _ui:
            failure_causes["user_info_mismatch"] = _ui
        if product_causes["not_cheapest"]["product_ids"]:
            failure_causes["not_cheapest"] = product_causes["not_cheapest"]
        if product_causes["missing_product"]["product_ids"]:
            failure_causes["missing_product"] = product_causes["missing_product"]
        if product_causes["ambiguous"]["product_ids"]:
            failure_causes["ambiguous"] = product_causes["ambiguous"]

        details: dict[str, Any] = {
            "composite_score": composite,
            "case_score": 1.0 if passed else 0.0,
            "matched_count": matched_count,
            "expected_count": expected_count,
            "coupon_score": coupon_score,
            "matched_products": list(matched_products),
            "missing_products": missing_products,
            "extra_products": extra_products,
            "coupon_details": coupon_details,
            "extra_coupons": extra_coupons,
            "missing_coupons": missing_coupons,
            "failure_causes": failure_causes,
            "user_profile": user_profile,
            "budget_check": budget_check,
            "coupon_ownership": coupon_ownership,
            "final_price_check": final_price_check,
            "level": level,
            "level_objective": _level_objective(level),
        }
        # Full per-case error log (the entire diagnostic rendered to text) for the
        # self-improving prompt dump.
        details["error_log"] = _render_error_log(details)
        return {"score": composite, "passed": passed, "details": details}

    # ----- round-level aggregate ----- #

    def aggregate(
        self, per_case: list[Any], trace_events: list[Any]
    ) -> dict[str, Any]:
        """Per-level breakdown for the round-level project_metrics.

        The gatherer hands ``per_case`` a list of ``CaseResult`` objects.
        Each carries ``.score`` and a ``.details`` dict — the same dict
        ``score()`` emitted (``level``, ``expected_count``, ...). We bucket
        by level and report mean composite + match count per level so the
        strategy prompt sees signal about where the optimizer is improving.
        """
        levels: dict[int, list[float]] = {1: [], 2: [], 3: []}
        all_scores: list[float] = []
        completed = 0
        for r in per_case or []:
            score = float(getattr(r, "score", 0.0) or 0.0)
            # CaseResult exposes the scorer's emitted dict as ``.details``
            # (there is no ``.metrics`` attribute — reading that left
            # per_level / level_n / cases_with_ground_truth silently empty).
            details = getattr(r, "details", {}) or {}
            lvl = details.get("level")
            all_scores.append(score)
            if lvl in levels:
                levels[lvl].append(score)
            if details.get("expected_count"):
                completed += 1
        return {
            "score_overall": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
            "per_level": {
                lvl: (sum(scores) / len(scores)) if scores else 0.0
                for lvl, scores in levels.items()
            },
            "level_n": {lvl: len(scores) for lvl, scores in levels.items()},
            "cases_with_ground_truth": completed,
        }


# Default instance for the evaluator's "load scorer.py" fallback path
# (i.e. when nothing is passed to SubprocessEvaluator(scorer=...)). The
# in-config path runs `build_components` which constructs a fresh
# `ShoppingMasScorer()` and injects it; this module-level instance is what
# the standalone `evaluate.py` invocation uses.
_DEFAULT_SCORER = ShoppingMasScorer()


def score(case: dict[str, Any], agent_output: Any) -> dict[str, Any]:
    """Module-level scorer entry point used by ``SubprocessEvaluator``
    when no registered scorer instance is provided."""
    return _DEFAULT_SCORER.score(case, agent_output)

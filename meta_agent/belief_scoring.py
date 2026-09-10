"""Scoring of registered belief predictions against measured outcomes, plus
the calibration report that the belief maintainer and the guidance optimizer
both read. Pure and deterministic: no I/O, no LLM.

A registered prediction (``round_NNN/belief_prediction.json``, written the
moment a node was expanded) names, per kind, the belief that covered the
node and its ``p``. Each kind pays Brier loss ``(p - y)^2`` once the node
carries the per-node analysis:

* implementation — ``y = 1`` iff the analysis judged the (matched sub-)edit
  *sound*; scored on every analysed node;
* strategy — scored only when the implementation was sound (an unsound
  implementation says nothing about the strategy). The label depends on
  ``label_source``:
    - ``"judge"`` (default): ``y = 1`` iff the analysis's effect verdict for
      the matched sub-edit is ``improved``; ``no_effect`` / ``regressed`` are
      ``y = 0``; ``unclear`` verdicts and evidence below ``min_evidence``
      stay pending (they may firm up after another batch). No shared cases
      with the parent are needed.
    - ``"delta"``: ``y = 1`` iff Δ vs parent ≥ threshold over ≥ ``min_shared``
      shared cases (the pre-v7 rule, kept for ablation).

A node no belief covered is scored at ``p = 0.5`` (loss 0.25), so silence is
never free. Nodes score once, at first eligibility.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from .belief_contract import Belief, ParsedDoc, Violation, record_effects
from .perf_text import EVIDENCE_RANK, perf_summary, score_well_measured

UNCOVERED_P = 0.5
LABEL_SOURCES = ("judge", "delta")
_PERF_LINE_RE = re.compile(r"^- \*\*performance\*\*: (.*)$", re.M)
_WHAT_RE = re.compile(r"^- \*\*what\*\*: (.*)$", re.M)


@dataclass
class Scored:
    node: int
    kind: str
    slug: Optional[str]
    p: float
    y: int
    brier: float
    belief_version: int
    instruction_version: int
    resolved_at_update: int
    n_shared: int
    delta: float
    impl_reason: str = ""
    # v7 additions (judge labels). The judge is the default label source;
    # rows persisted before v7 carry no key and are loaded as "delta" by
    # ``from_dict`` — that is what they were scored against.
    label_source: str = "judge"
    effect: str = ""
    evidence: str = ""
    effect_reason: str = ""
    matched_edit: Optional[int] = None
    delta_all: Optional[float] = None
    se_all: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scored":
        def _opt_float(v: Any) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        me = d.get("matched_edit")
        return cls(node=int(d["node"]), kind=str(d["kind"]), slug=d.get("slug"),
                   p=float(d["p"]), y=int(d["y"]), brier=float(d["brier"]),
                   belief_version=int(d.get("belief_version") or 0),
                   instruction_version=int(d.get("instruction_version") or 0),
                   resolved_at_update=int(d.get("resolved_at_update") or 0),
                   n_shared=int(d.get("n_shared") or 0),
                   delta=float(d.get("delta") or 0.0),
                   impl_reason=str(d.get("impl_reason") or ""),
                   label_source=str(d.get("label_source") or "delta"),
                   effect=str(d.get("effect") or ""),
                   evidence=str(d.get("evidence") or ""),
                   effect_reason=str(d.get("effect_reason") or ""),
                   matched_edit=int(me) if me is not None else None,
                   delta_all=_opt_float(d.get("delta_all")),
                   se_all=_opt_float(d.get("se_all")))


def _p_of(entry: Any) -> tuple[float, Optional[str]]:
    if isinstance(entry, Mapping) and entry.get("p") is not None:
        try:
            return float(entry["p"]), entry.get("slug")
        except (TypeError, ValueError):
            pass
    return UNCOVERED_P, None


def _matched(entry: Any) -> Optional[int]:
    if isinstance(entry, Mapping) and entry.get("matched_edit") is not None:
        try:
            return int(entry["matched_edit"])
        except (TypeError, ValueError):
            return None
    return None


UNCOVERABLE_REASON = ("not coverable at registration — first node of its "
                      "strategy, no belief could have existed")


def is_measurable(rec: Mapping[str, Any], *, label_source: str,
                  min_shared: int) -> bool:
    """Has the node reached the point where predictions about it can be
    scored? Judge mode: the analysis has run (it is gated on the node's own
    evaluations upstream). Delta mode: a paired Δ over enough shared cases."""
    if label_source == "judge":
        return rec.get("impl_sound") is not None
    return rec.get("delta") is not None and (rec.get("n_shared") or 0) >= min_shared


def resolve(predictions: Mapping[int, Mapping[str, Any]],
            records: Mapping[int, Any], *, threshold: float, min_shared: int,
            already: set[tuple[int, str]],
            n_updates: int, label_source: str = "judge",
            min_evidence: str = "moderate") -> tuple[list[Scored], list[dict[str, Any]],
                                                     list[dict[str, Any]]]:
    """Score every prediction whose node became measurable. Returns the new
    ``Scored`` rows, the nodes that are measurable but could not be scored
    yet (no verdict / strategy not scored because unsound / judge unclear or
    evidence too weak), and the predictions SKIPPED for good: an uncovered
    node whose strategy did not exist when it was registered
    (``coverable: false``) is not charged the silence loss — no belief could
    have covered it. Silence costs 0.25 only where speech was possible."""
    judge = label_source == "judge"
    min_rank = EVIDENCE_RANK.get(min_evidence, 1)
    new: list[Scored] = []
    pending: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for node in sorted(predictions):
        pred = predictions[node] or {}
        rec = records.get(node)
        if rec is None:
            continue
        delta, n_shared = rec.get("delta"), rec.get("n_shared") or 0
        open_kinds = [k for k in ("implementation", "strategy")
                      if (node, k) not in already]
        if not open_kinds:
            continue
        if judge:
            if rec.get("impl_sound") is None:
                # The analysis is gated on the node's own evaluations; a
                # node with evaluations but no analysis yet is pending.
                if rec.get("n_abs") or 0:
                    pending.append({"node": node,
                                    "reason": "awaiting the analysis verdict"})
                continue
        else:
            if delta is None or n_shared < min_shared:
                continue
            if rec.get("impl_sound") is None:
                pending.append({"node": node,
                                "reason": "awaiting implementation verdict"})
                continue
        impl = rec.get("impl_sound")
        reason = str(rec.get("impl_reason") or "")
        by_edit = rec.get("impl_by_edit") or {}
        reason_by_edit = rec.get("impl_reason_by_edit") or {}
        effects = rec.get("effect_by_edit") or {}
        evidence = rec.get("evidence_by_edit") or {}
        effect_reason = rec.get("effect_reason_by_edit") or {}
        first_edit = min(effects) if effects else None
        bv = int(pred.get("belief_version") or 0)
        iv = int(pred.get("instruction_version") or 0)
        coverable = bool(pred.get("coverable", True))
        common = dict(n_shared=n_shared, delta=float(delta or 0.0),
                      label_source=label_source,
                      delta_all=rec.get("delta_all"), se_all=rec.get("se_all"))

        def verdict(entry: Any) -> tuple[bool, str]:
            """The verdict that applies to a covered prediction: the matched
            sub-edit's own (analysis v6) when the record has it, else the
            node's. A bundled edit with one broken part no longer vetoes the
            sound part's strategy evidence."""
            e = _matched(entry)
            if e is not None and e in by_edit:
                return bool(by_edit[e]), str(reason_by_edit.get(e) or reason)
            return bool(impl), reason

        def effect_of(entry: Any) -> tuple[Optional[str], str, str, Optional[int]]:
            """The judge's effect for the matched sub-edit, else the first
            sub-edit's (the primary mechanism)."""
            e = _matched(entry)
            if e is None or e not in effects:
                e = first_edit
            if e is None:
                return None, "", "", None
            return (str(effects[e]), str(evidence.get(e) or ""),
                    str(effect_reason.get(e) or ""), e)

        if (node, "implementation") not in already:
            p, slug = _p_of(pred.get("implementation"))
            if slug is None and not coverable:
                skipped.append({"node": node, "kind": "implementation",
                                "reason": UNCOVERABLE_REASON})
            else:
                v, rsn = verdict(pred.get("implementation"))
                y = 1 if v else 0
                new.append(Scored(node, "implementation", slug, p, y,
                                  (p - y) ** 2, bv, iv, n_updates,
                                  impl_reason=rsn,
                                  matched_edit=_matched(pred.get("implementation")),
                                  **common))
        if (node, "strategy") not in already:
            p, slug = _p_of(pred.get("strategy"))
            v, rsn = verdict(pred.get("strategy"))
            if not v:
                pending.append({"node": node, "impl_reason": rsn,
                                "reason": "implementation unsound — strategy "
                                          "belief not scored"})
            elif slug is None and not coverable:
                skipped.append({"node": node, "kind": "strategy",
                                "reason": UNCOVERABLE_REASON})
            elif judge:
                eff, ev, er, e_idx = effect_of(pred.get("strategy"))
                if eff is None:
                    pending.append({"node": node, "reason":
                                    "awaiting the judge's effect verdict "
                                    "(analysis predates v7)"})
                elif eff == "unclear":
                    pending.append({"node": node, "impl_reason": er,
                                    "reason": "judge: effect unclear — not scored"})
                elif EVIDENCE_RANK.get(ev, 0) < min_rank:
                    pending.append({"node": node, "impl_reason": er,
                                    "reason": f"judge: {ev or 'ungraded'} evidence "
                                              f"below the {min_evidence} minimum — "
                                              "not scored"})
                else:
                    y = 1 if eff == "improved" else 0
                    new.append(Scored(node, "strategy", slug, p, y,
                                      (p - y) ** 2, bv, iv, n_updates,
                                      impl_reason=rsn, effect=eff, evidence=ev,
                                      effect_reason=er, matched_edit=e_idx,
                                      **common))
            else:
                y = 1 if float(delta) >= threshold - 1e-9 else 0
                new.append(Scored(node, "strategy", slug, p, y,
                                  (p - y) ** 2, bv, iv, n_updates,
                                  impl_reason=rsn,
                                  matched_edit=_matched(pred.get("strategy")),
                                  **common))
    return new, pending, skipped


def per_version_brier(scored: Sequence[Scored]) -> dict[int, tuple[int, float]]:
    """``{instruction_version: (n, mean_brier)}``."""
    acc: dict[int, list[float]] = {}
    for s in scored:
        acc.setdefault(s.instruction_version, []).append(s.brier)
    return {v: (len(b), sum(b) / len(b)) for v, b in acc.items()}


def per_belief_stats(scored: Sequence[Scored]) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(kind, slug): {n, mean, outcomes: [(node, y, p)]}}``; uncovered
    rows are keyed with slug ``"(uncovered)"``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for s in scored:
        key = (s.kind, s.slug or "(uncovered)")
        e = out.setdefault(key, {"n": 0, "sum": 0.0, "outcomes": []})
        e["n"] += 1
        e["sum"] += s.brier
        e["outcomes"].append((s.node, s.y, s.p))
    for e in out.values():
        e["mean"] = e["sum"] / e["n"]
    return out


def mean_brier(scored: Sequence[Scored]) -> Optional[float]:
    return sum(s.brier for s in scored) / len(scored) if scored else None


def track_lines(beliefs: Sequence[Belief], scored: Sequence[Scored],
                soft_counts: Optional[Mapping[str, int]] = None) -> dict[str, str]:
    """The code-generated ``- track:`` text per belief slug."""
    stats = per_belief_stats(scored)
    out: dict[str, str] = {}
    for b in beliefs:
        e = stats.get((b.kind, b.slug))
        if e:
            tail = ", ".join(f"{n} {'yes' if y else 'no'}"
                             for n, y, _p in e["outcomes"][-5:])
            text = (f"n={e['n']} · Brier {e['mean']:.2f} (0.25 = uninformative)"
                    f" · outcomes: {tail}")
        else:
            text = "no scored predictions yet"
        k = (soft_counts or {}).get(b.slug, 0)
        if k:
            text += f" · {k} misquoted citation(s) — see the calibration report"
        out[b.slug] = text
    return out


def _perf(rec: Mapping[str, Any]) -> str:
    m = _PERF_LINE_RE.search(rec.get("text") or "")
    return m.group(1).strip() if m else "not yet measured"


def _first_what(rec: Mapping[str, Any]) -> str:
    m = _WHAT_RE.search(rec.get("body") or rec.get("text") or "")
    return m.group(1).strip()[:160] if m else ""


def _tags_of(rec: Mapping[str, Any]) -> str:
    tags = rec.get("tags") or []
    return "; ".join(f"{t.get('strategy') or '?'}"
                     + (f"/{t['area']}" if t.get("area") else "")
                     for t in tags) or "(no tags)"


def _current_effect(rec: Mapping[str, Any], s: Scored) -> Optional[str]:
    """What the record says NOW for the sub-edit a strategy row was scored
    on (a later analysis may have changed its verdict)."""
    effects = record_effects(rec)
    if not effects:
        return None
    if s.matched_edit is not None and s.matched_edit in effects:
        return effects[s.matched_edit]
    return effects[min(effects)]


def render_calibration_report(*, parsed: ParsedDoc, scored: Sequence[Scored],
                              pending: Sequence[Mapping[str, Any]],
                              records: Mapping[int, Any],
                              predictions: Mapping[int, Mapping[str, Any]],
                              soft_violations: Sequence[Violation],
                              joins: Sequence[Mapping[str, Any]],
                              version_history: Sequence[Mapping[str, Any]],
                              current_version: int, min_shared: int,
                              n_skipped: int = 0, label_source: str = "judge",
                              threshold: float = 0.02) -> str:
    """Facts only — no judgment. Everything here is derived from the
    records and the registered predictions."""
    judge = label_source == "judge"

    def ctx(rec: Mapping[str, Any]) -> str:
        return perf_summary(rec, threshold=threshold, min_shared=min_shared)

    L = ["## Calibration report"]
    L.append("- labels: strategy y = the judge's effect verdict on the matched "
             "sub-edit (`improved` = yes; `no_effect`/`regressed` = no; `unclear` "
             "or weak evidence = not scored); implementation y = sound. The "
             "score Δ is shown as context only — at ~16-case batches its SE is "
             "near ±0.1." if judge else
             f"- labels: strategy y = Δ vs parent ≥ {threshold:.2f} over ≥ "
             f"{min_shared} shared cases; implementation y = sound.")
    n = len(scored)
    if n:
        by_kind = {k: sum(1 for s in scored if s.kind == k)
                   for k in ("strategy", "implementation")}
        unc = sum(1 for s in scored if s.slug is None)
        L.append(f"- {n} scored prediction(s) (strategy {by_kind['strategy']} / "
                 f"implementation {by_kind['implementation']}); mean Brier "
                 f"{mean_brier(scored):.3f} vs 0.25 uninformative; "
                 f"{unc} of {n} were uncovered (scored at p=0.5)")
    else:
        L.append("- no scored predictions yet — nothing registered has been "
                 "analysed" if judge else
                 "- no scored predictions yet — nothing registered has been "
                 "measured with an implementation verdict")
    if n_skipped:
        L.append(f"- {n_skipped} prediction(s) skipped, not scored: the first "
                 "node of a new strategy, which no belief could have covered")
    pv = per_version_brier(scored)
    if version_history:
        bits = []
        for v in version_history:
            ver = int(v.get("version", 0))
            stat = pv.get(ver)
            s = f"v{ver}: " + (f"n={stat[0]}, Brier {stat[1]:.3f}" if stat
                               else "no scored predictions")
            if ver == current_version:
                s += " (current)"
            bits.append(s)
        L.append("- guidance versions — " + " · ".join(bits))

    L += ["", "### Per belief"]
    stats = per_belief_stats(scored)
    cited = {}
    for j in joins:
        b = j.get("belief_id")
        if b:
            cited[b] = cited.get(b, 0) + 1
    for b in parsed.beliefs:
        e = stats.get((b.kind, b.slug))
        stat = (f"n={e['n']} · Brier {e['mean']:.2f} · "
                + ", ".join(f"{nd} {'yes' if y else 'no'}"
                            for nd, y, _p in e["outcomes"][-6:])
                if e else "no scored predictions")
        extra = f" · cited by {cited[b.slug]} proposal(s)" if b.slug in cited else ""
        L.append(f"- belief:{b.slug} ({b.kind} {b.scope}, p={b.p:.2f}): {stat}{extra}")
    if not parsed.beliefs:
        L.append("- (no beliefs)")

    covered = [s for s in scored if s.slug is not None]
    misses = sorted(covered, key=lambda s: -s.brier)[:8]
    L += ["", "### Worst misses (by Brier)"]
    for s in misses:
        rec = records.get(s.node) or {}
        verdict = ("sound" if rec.get("impl_sound") else "unsound") \
            if rec.get("impl_sound") is not None else "no verdict"
        line = (f"- node {s.node} · belief:{s.slug} ({s.kind}) p={s.p:.2f} → "
                f"{'yes' if s.y else 'no'} (Brier {s.brier:.2f})")
        if s.kind == "strategy" and s.label_source == "judge":
            line += (f" · judge {s.effect or '?'} ({s.evidence or '?'})"
                     + (f' — "{s.effect_reason[:140]}"' if s.effect_reason else "")
                     + f" · score {ctx(rec)}")
        else:
            line += (f" · Δ{s.delta:+.4f}/{s.n_shared} · implementation {verdict}"
                     + (f' — "{s.impl_reason[:120]}"' if s.impl_reason else ""))
        if _first_what(rec):
            line += f' · edit: "{_first_what(rec)}"'
        L.append(line)
    if not misses:
        L.append("- (none)")

    L += ["", "### Uncovered measured nodes (scored at p=0.5)"]
    unc_rows = [s for s in scored if s.slug is None]
    for s in unc_rows:
        rec = records.get(s.node) or {}
        if s.kind == "strategy":
            outcome = ((f"judge {s.effect}" if s.effect else
                        ("helped" if s.y else "not helped"))
                       if s.label_source == "judge" else
                       ("helped" if s.y else "not helped"))
        else:
            outcome = "sound" if s.y else "unsound"
        line = (f"- node {s.node} · tags {_tags_of(rec)} · {s.kind}: {outcome} "
                f"· {ctx(rec)} · no {s.kind} belief covered this scope")
        # Name the near miss: a same-strategy belief scoped to another area
        # would have covered this node had it been strategy-wide.
        node_strats = {t.get("strategy") for t in (rec.get("tags") or [])}
        near = [b for b in parsed.beliefs
                if b.kind == s.kind and b.area and b.strategy in node_strats]
        if near:
            line += (" — " + "; ".join(
                f"belief:{b.slug} covers `{b.strategy}` only in area "
                f"`{b.area}`" for b in near[:3])
                     + "; a strategy-wide scope would have covered it")
        L.append(line)
    if not unc_rows:
        L.append("- (none)")

    L += ["", "### Not scored"]
    for p in pending:
        L.append(f"- node {p['node']} · {p['reason']}"
                 + (f' — "{str(p.get("impl_reason"))[:120]}"'
                    if p.get("impl_reason") else ""))
    if not pending:
        L.append("- (none)")

    done = {(s.node, s.kind) for s in scored}
    pend_nodes = {p["node"] for p in pending}
    L += ["", "### Open predictions (registered, not yet measurable)"]
    open_rows = []
    for node in sorted(predictions):
        if node in pend_nodes or (node, "implementation") in done:
            continue
        rec = records.get(node)
        if rec is None:
            continue
        if is_measurable(rec, label_source=label_source, min_shared=min_shared):
            continue
        pred = predictions[node] or {}
        bits = []
        for kind in ("strategy", "implementation"):
            e = pred.get(kind)
            if isinstance(e, Mapping) and e.get("slug"):
                bits.append(f"belief:{e['slug']} p={float(e.get('p', 0.5)):.2f} ({kind})")
            else:
                bits.append(f"uncovered ({kind})")
        open_rows.append(f"- node {node} · {' · '.join(bits)}")
    L += open_rows[:12] or ["- (none)"]
    if len(open_rows) > 12:
        L.append(f"- (+{len(open_rows) - 12} more)")

    if judge:
        # A diagnostic for the human and the maintainer alike: where a
        # well-measured score contradicts the judge, that node is worth
        # re-reading. The judge stays the label; the criterion is
        # parent-independent (paired ≥ 16 shared, or |unpaired Δ| > 2×SE).
        L += ["", "### Judge vs score — a diagnostic, not a yardstick (strategy "
                  "rows whose score is well measured: ≥ 16 shared cases, or "
                  "|unpaired Δ| > 2×SE)"]
        agree = disagree = 0
        dis_lines = []
        for s in scored:
            if s.kind != "strategy" or s.label_source != "judge" or not s.effect:
                continue
            rec = records.get(s.node) or {}
            d, ns = rec.get("delta"), int(rec.get("n_shared") or 0)
            how = score_well_measured(delta=d, n_shared=ns,
                                      delta_all=rec.get("delta_all"),
                                      se_all=rec.get("se_all"))
            if how is None:
                continue
            value = d if how == "paired" else rec.get("delta_all")
            if (s.effect == "improved") == (float(value) >= threshold - 1e-9):
                agree += 1
            else:
                disagree += 1
                dis_lines.append(
                    f"- node {s.node}: judge {s.effect} ({s.evidence})"
                    + (f' — "{s.effect_reason[:120]}"' if s.effect_reason else "")
                    + f" · score {ctx(rec)}")
        if agree + disagree:
            L.append(f"- {agree} agree / {disagree} disagree — the judge is the "
                     "label; a disagreement flags a node worth re-reading, not a "
                     "wrong verdict")
            L += dis_lines[:6]
        else:
            L.append("- (no strategy row has a well-measured score yet)")

        L += ["", "### Verdict changes since scoring"]
        flips = []
        for s in scored:
            if s.kind != "strategy" or not s.effect:
                continue
            now = _current_effect(records.get(s.node) or {}, s)
            if now and now != s.effect:
                flips.append(f"- node {s.node}: scored on `{s.effect}`; the latest "
                             f"analysis says `{now}` (rows are not re-scored)")
        L += flips[:8] or ["- (none)"]

    L += ["", "### Citation checks"]
    if soft_violations:
        for v in soft_violations:
            L.append(f"- {v.render()}")
    else:
        n_c = sum(len(b.citations) for b in parsed.beliefs)
        L.append(f"- all {n_c} inline citation(s) match the records")
    return "\n".join(L)

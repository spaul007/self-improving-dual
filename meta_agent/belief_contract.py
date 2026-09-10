"""Predictive contract for the belief document — grammar, parser, validator.

Pure module: no I/O, no LLM, and it must not import ``edit_beliefs`` (which
imports this). The document the belief maintainer writes is a list of
``### belief:<slug>`` sections, each a scoped probability that the run later
SCORES against a measured outcome. This module is the single definition of
what such a section must contain, so the generator prompt, the validator,
the registration join and the steering renderer all agree byte-for-byte.

Grammar (anything else is a HARD violation)::

    ## Summary                      optional, must come first, <= 1200 chars
    <one short paragraph>

    ### belief:<slug> — <title>     slug [a-z0-9][a-z0-9_-]*
    - kind: strategy | implementation
    - scope: strategy=<registry id> [area=<registry id>]
    - predict: p=<0.05..0.95>
    - evidence: <prose; cite nodes by the judge's verdict [node N: improved];
                 a Δ only when well measured [node N: improved; Δ+0.0310/12] /
                 [node N: Δ-0.17±0.14]; [node N: unmeasured] when neither>
    - next: <one concrete move>
    - track: <CODE-GENERATED; stripped on parse, never accepted from the model>

HARD violations (structure, ids, ranges, caps) cost the submission; SOFT ones
(citation misquotes) are reported back and flagged but do not reject.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

BELIEF_KINDS = ("strategy", "implementation")
P_MIN, P_MAX = 0.05, 0.95
SUMMARY_HEADER = "## Summary"
# Style guard only — the document cap is the real compactness bound. 600 cost
# a full retry for a 691-char summary in the 2026-09-07 run; 1200 is a short
# paragraph of headroom.
SUMMARY_CHAR_CAP = 1200
TRACK_KEY = "track"
REQUIRED_KEYS = ("kind", "scope", "predict", "evidence", "next")
ALLOWED_KEYS = REQUIRED_KEYS + (TRACK_KEY,)
# How far a quoted delta may sit from the record's before it is a misquote.
DELTA_TOL = 0.005

# Anchor convention: "### belief:<slug> — <title>". Liberal on separators —
# models often write snake_case; truncating at "_" once broke the join.
ANCHOR_RE = re.compile(r"^###\s+belief:([a-z0-9][a-z0-9_-]*)[ \t]*(.*)$", re.M)
# Citation convention, one bracket per node, any of:
#   "[node 17: improved]"                 the judge's effect verdict
#   "[node 17: regressed; Δ-0.17±0.14]"   verdict + unpaired Δ with its SE
#   "[node 17: Δ-0.04/22]"                paired Δ over 22 shared cases
#   "[node 17: Δ+0.05±0.12]"              unpaired Δ ± SE
#   "[node 17: unmeasured]"
# Groups: node, effect, delta, se, n_shared, unmeasured. A bracket with none
# of effect/delta/unmeasured is not a citation (parse_citations drops it).
CITE_RE = re.compile(
    r"\[node\s+(\d+):\s*"
    r"(?:(improved|no_effect|regressed|unclear)\s*(?:[;,]\s*)?)?"
    r"(?:Δ\s*([+-]?\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?(?:\s*/\s*(\d+))?)?"
    r"\s*(unmeasured)?\s*\]")
_BULLET_RE = re.compile(r"^-\s+\**([A-Za-z_]+)\**\s*:\s*(.*)$")
_TRACK_LINE_RE = re.compile(r"^-\s+\**track\**\s*:[^\n]*\n?", re.M)
_KV_RE = re.compile(r"(strategy|area)\s*=\s*`?([A-Za-z0-9][A-Za-z0-9_-]*)`?")
_P_RE = re.compile(r"(?:\bp\s*=\s*)?([01](?:\.\d+)?|\.\d+)")

GRAMMAR = """## Summary            (optional; must come first; at most 1200 chars)
<one short paragraph of cross-scope context>

### belief:<slug> — <title>
- kind: strategy | implementation
- scope: strategy=<registry strategy id> [area=<registry area id>]
- predict: p=<0.05..0.95>
- evidence: <why; cite every node you rely on by the judge's verdict — [node N: improved] / [node N: no_effect] / [node N: regressed] — adding a Δ only when it is well measured ([node N: improved; Δ+0.0310/12] paired over 12 shared, [node N: Δ-0.17±0.14] unpaired ± SE); [node N: unmeasured] when there is neither>
- next: <one concrete move for the editor>"""

EXAMPLE = """### belief:constraint-enforcement-helps — tool-wired constraint gates pay off
- kind: strategy
- scope: strategy=add-constraint-enforcement
- predict: p=0.65
- evidence: two sound implementations improved their targeted checks [node 13: improved] [node 10: improved; Δ+0.1200/25] (the Δ is cited only because 25 shared cases make it well measured); one prompt-only attempt was unsound and is not counted [node 4: no_effect]
- next: extend the wired gate to intercity transfers; keep the remediation loop it already has

(Scope by strategy, as above, so every future edit using it is covered. Add
`area=<id>` only when you expect the effect to differ by area — an area-scoped
belief does NOT cover the same strategy in other areas; those edits are scored
as uncovered at p=0.5.)"""


@dataclass
class Violation:
    code: str
    message: str
    slug: Optional[str] = None
    hard: bool = True

    def render(self) -> str:
        where = f"belief:{self.slug} — " if self.slug else ""
        return f"[{'HARD' if self.hard else 'SOFT'}] {where}{self.message}"


@dataclass
class Belief:
    slug: str
    title: str
    kind: str
    strategy: str
    area: Optional[str]
    p: float
    evidence: str
    next: str
    track: Optional[str]
    section_text: str
    citations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scope(self) -> str:
        return f"strategy={self.strategy}" + (f" area={self.area}" if self.area else "")


@dataclass
class ParsedDoc:
    summary: str = ""
    beliefs: list[Belief] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    body_chars: int = 0

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.hard]

    @property
    def soft(self) -> list[Violation]:
        return [v for v in self.violations if not v.hard]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def parse_anchors(text: str) -> list[str]:
    return [m.group(1) for m in ANCHOR_RE.finditer(text or "")]


def parse_citations(text: str) -> list[dict[str, Any]]:
    """Every inline citation with the slug of the belief section it sits in
    (``None`` when it appears above the first anchor)."""
    anchors = [(m.start(), m.group(1)) for m in ANCHOR_RE.finditer(text or "")]
    out: list[dict[str, Any]] = []
    for m in CITE_RE.finditer(text or ""):
        effect, delta, se, n_shared, unmeasured = m.group(2, 3, 4, 5, 6)
        if effect is None and delta is None and unmeasured is None:
            continue  # "[node 5: ]" — not a citation
        slug = None
        for pos, s in anchors:
            if pos <= m.start():
                slug = s
            else:
                break
        out.append({
            "slug": slug,
            "node": int(m.group(1)),
            "effect": effect,
            "delta": float(delta) if delta is not None else None,
            "se": float(se) if se is not None else None,
            "n_shared": int(n_shared) if n_shared is not None else None,
            "unmeasured": unmeasured is not None,
            "raw": m.group(0),
        })
    return out


def strip_track_lines(text: str) -> str:
    """Remove every ``- track:`` line — the code-generated calibration line
    is never accepted from the model and never re-parsed as content."""
    return _TRACK_LINE_RE.sub("", text or "")


def inject_track_lines(text: str, track_by_slug: Mapping[str, str]) -> str:
    """Append ``- track: <text>`` as the last bullet of each belief section.
    Deterministic and idempotent (existing track lines are stripped first)."""
    base = strip_track_lines(text).rstrip("\n")
    anchors = [(m.start(), m.group(1)) for m in ANCHOR_RE.finditer(base)]
    if not anchors:
        return base + "\n"
    out = [base[:anchors[0][0]]]
    for i, (pos, slug) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(base)
        section = base[pos:end].rstrip("\n")
        track = track_by_slug.get(slug)
        if track:
            section += f"\n- {TRACK_KEY}: {track}"
        out.append(section + "\n\n")
    return "".join(out).rstrip("\n") + "\n"


def _parse_scope(value: str) -> tuple[Optional[str], Optional[str]]:
    found = dict((k, v) for k, v in _KV_RE.findall(value or ""))
    return found.get("strategy"), found.get("area")


def _parse_p(value: str) -> Optional[float]:
    m = _P_RE.search(value or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Parser — structure only
# --------------------------------------------------------------------------- #
def parse_document(text: str) -> ParsedDoc:
    """Structural parse. Reports every HARD structural violation it finds
    but always returns whatever beliefs it could assemble."""
    body = strip_track_lines(text or "")
    doc = ParsedDoc(body_chars=len(body.strip()))
    lines = body.split("\n")
    summary_lines: list[str] = []
    seen_summary = False
    state = "head"
    # (slug, title, fields{key: value}, raw lines)
    sections: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    cur_key: Optional[str] = None
    prose_flagged = False

    for raw in lines:
        line = raw.rstrip()
        am = ANCHOR_RE.match(line)
        if am:
            cur = {"slug": am.group(1),
                   "title": am.group(2).strip().lstrip("—-– ").strip(),
                   "fields": {}, "raw": [line]}
            sections.append(cur)
            cur_key = None
            state = "section"
            continue
        if state == "head":
            if not line.strip():
                continue
            if line.strip() == SUMMARY_HEADER and not seen_summary:
                seen_summary = True
                state = "summary"
                continue
            if not prose_flagged:
                doc.violations.append(Violation(
                    "prose-above-beliefs",
                    "content above the first `### belief:` section that is "
                    "not a `## Summary` block: " + line.strip()[:80]))
                prose_flagged = True
            continue
        if state == "summary":
            if line.startswith("#"):
                doc.violations.append(Violation(
                    "unexpected-section",
                    f"unexpected heading `{line.strip()[:60]}` — only "
                    f"`{SUMMARY_HEADER}` and `### belief:` sections are allowed"))
                state = "head"
                continue
            summary_lines.append(line)
            continue
        # state == "section"
        assert cur is not None
        if not line.strip():
            continue
        if line.startswith("#"):
            doc.violations.append(Violation(
                "unexpected-section",
                f"unexpected heading `{line.strip()[:60]}` inside or after a "
                "belief section", slug=cur["slug"]))
            cur_key = None
            continue
        bm = _BULLET_RE.match(line)
        if bm:
            key = bm.group(1).lower()
            if key not in ALLOWED_KEYS:
                doc.violations.append(Violation(
                    "unknown-field", f"unknown bullet `{key}` (allowed: "
                    f"{', '.join(REQUIRED_KEYS)})", slug=cur["slug"]))
                cur_key = None
                continue
            if key in cur["fields"]:
                doc.violations.append(Violation(
                    "duplicate-field", f"bullet `{key}` given twice",
                    slug=cur["slug"]))
            cur["fields"][key] = bm.group(2).strip()
            cur["raw"].append(line)
            cur_key = key
            continue
        if raw.startswith((" ", "\t")) and cur_key:
            cur["fields"][cur_key] += " " + line.strip()
            cur["raw"].append(line)
            continue
        doc.violations.append(Violation(
            "unexpected-line",
            "a belief section may contain only its bullets (kind/scope/"
            "predict/evidence/next) and their indented continuations; "
            f"got: {line.strip()[:80]}", slug=cur["slug"]))

    doc.summary = "\n".join(summary_lines).strip()
    if len(doc.summary) > SUMMARY_CHAR_CAP:
        doc.violations.append(Violation(
            "summary-cap", f"`{SUMMARY_HEADER}` is {len(doc.summary)} chars; "
            f"the cap is {SUMMARY_CHAR_CAP}"))

    seen_slugs: set[str] = set()
    for s in sections:
        slug, f = s["slug"], s["fields"]
        if slug in seen_slugs:
            doc.violations.append(Violation(
                "duplicate-slug", "slug used by more than one section", slug=slug))
        seen_slugs.add(slug)
        missing = [k for k in REQUIRED_KEYS if not f.get(k)]
        if missing:
            doc.violations.append(Violation(
                "missing-field", "missing bullet(s): " + ", ".join(missing),
                slug=slug))
        kind = (f.get("kind") or "").strip().lower()
        if kind and kind not in BELIEF_KINDS:
            doc.violations.append(Violation(
                "kind", f"kind must be one of {' | '.join(BELIEF_KINDS)}, "
                f"got `{kind}`", slug=slug))
        strategy, area = _parse_scope(f.get("scope", ""))
        if f.get("scope") and not strategy:
            doc.violations.append(Violation(
                "scope", "scope must read `strategy=<id>` optionally followed "
                f"by `area=<id>`; got `{f.get('scope')[:60]}`", slug=slug))
        p = _parse_p(f.get("predict", ""))
        if f.get("predict") and p is None:
            doc.violations.append(Violation(
                "predict", f"predict must read `p=<0.05..0.95>`; got "
                f"`{f.get('predict')[:40]}`", slug=slug))
        elif p is not None and not (P_MIN <= p <= P_MAX):
            doc.violations.append(Violation(
                "p-range", f"p={p} is outside [{P_MIN}, {P_MAX}]", slug=slug))
        section_text = "\n".join(s["raw"]).rstrip()
        doc.beliefs.append(Belief(
            slug=slug, title=s["title"], kind=kind or "?",
            strategy=strategy or "", area=area,
            p=p if p is not None else -1.0,
            evidence=f.get("evidence", ""), next=f.get("next", ""),
            track=None, section_text=section_text,
            citations=[c for c in parse_citations(section_text)]))
    if not sections:
        doc.violations.append(Violation(
            "no-beliefs", "the document contains no `### belief:<slug>` section"))
    return doc


# --------------------------------------------------------------------------- #
# Validation — registry ids, cap, duplicate scopes, citations vs records
# --------------------------------------------------------------------------- #
def record_effects(rec: Mapping[str, Any]) -> dict[int, str]:
    """``{edit: effect}`` from a record dict (analysis v7); the node-level
    effect under key 0 when no per-edit verdicts exist."""
    out: dict[int, str] = {}
    for e, v in (rec.get("effect_by_edit") or {}).items():
        try:
            out[int(e)] = str(v)
        except (TypeError, ValueError):
            continue
    if not out and rec.get("effect"):
        out[0] = str(rec["effect"])
    return out


def _record_score_text(rec: Mapping[str, Any]) -> str:
    r_delta, r_n = rec.get("delta"), rec.get("n_shared") or 0
    if r_delta is not None:
        return f"Δ{r_delta:+.4f} over {r_n} shared"
    r_all = rec.get("delta_all")
    if r_all is not None:
        se = rec.get("se_all")
        return f"unpaired Δ{r_all:+.4f}" + (f" ± {se:.4f}" if se is not None else "")
    return "no measured Δ"


def verify_citations(text: str, records: Mapping[int, Any]) -> list[Violation]:
    """Fact-check every inline citation against the records. SOFT only —
    a misquote is reported and flagged, never grounds for rejection.

    An effect word must be one of the record's verdicts (any sub-edit's);
    a paired Δ must match the record's shared Δ and count; an unpaired Δ
    the record's own-case Δ; ``unmeasured`` is a misquote once the record
    carries any of those."""
    out: list[Violation] = []
    for c in parse_citations(text):
        rec = records.get(c["node"]) if records else None
        slug = c.get("slug")
        raw = c["raw"]
        if rec is None:
            out.append(Violation("no-record", f"quotes `{raw}` but there is "
                                 "no record for that node", slug=slug, hard=False))
            continue
        r_delta, r_n = rec.get("delta"), rec.get("n_shared") or 0
        r_all = rec.get("delta_all")
        effects = record_effects(rec)
        if c.get("effect"):
            if not effects:
                out.append(Violation(
                    "misquote", f"quotes `{raw}` but the record has no effect "
                    "verdict yet", slug=slug, hard=False))
            elif c["effect"] not in effects.values():
                shown = ", ".join(f"edit {e}: {v}" if e else v
                                  for e, v in sorted(effects.items()))
                out.append(Violation(
                    "misquote", f"quotes `{raw}` but the record's verdict is "
                    f"{shown}", slug=slug, hard=False))
        if c.get("unmeasured"):
            if r_delta is not None or r_all is not None or effects:
                out.append(Violation(
                    "misquote", f"quotes `{raw}` but the record shows "
                    + _record_score_text(rec)
                    + (f"; verdict {', '.join(sorted(set(effects.values())))}"
                       if effects else ""), slug=slug, hard=False))
            continue
        if c["delta"] is None:
            continue
        if c.get("n_shared") is not None:
            # Paired form: Δ over N shared cases.
            if r_delta is None:
                out.append(Violation(
                    "misquote", f"quotes `{raw}` but the record has no "
                    "measured Δ yet", slug=slug, hard=False))
            elif (c["n_shared"] != r_n
                  or (c["delta"] > 0) != (r_delta > 0) and abs(r_delta) > 1e-9
                  or abs(c["delta"] - r_delta) > DELTA_TOL):
                out.append(Violation(
                    "misquote", f"quotes `{raw}` but the record shows "
                    f"Δ{r_delta:+.4f} over {r_n} shared", slug=slug, hard=False))
            continue
        # Unpaired form: Δ ± SE over each side's own cases.
        if r_all is None:
            out.append(Violation(
                "misquote", f"quotes `{raw}` but the record has no unpaired Δ "
                "yet", slug=slug, hard=False))
        elif ((c["delta"] > 0) != (r_all > 0) and abs(r_all) > 1e-9
              or abs(c["delta"] - r_all) > DELTA_TOL):
            r_se = rec.get("se_all")
            out.append(Violation(
                "misquote", f"quotes `{raw}` but the record shows unpaired "
                f"Δ{r_all:+.4f}" + (f" ± {r_se:.4f}" if r_se is not None else ""),
                slug=slug, hard=False))
    return out


def validate_document(text: str, *, registry: Optional[Mapping[str, Any]],
                      doc_char_cap: int,
                      records: Optional[Mapping[int, Any]] = None) -> ParsedDoc:
    """Structural parse + everything that needs run state. ``registry`` is
    the run's category registry (``{"strategies": {...}, "areas": {...}}``);
    when it is ``None`` id checks are skipped."""
    doc = parse_document(text)
    if doc.body_chars > doc_char_cap:
        doc.violations.append(Violation(
            "over-cap", f"document is {doc.body_chars} chars; the cap is "
            f"{doc_char_cap} — merge or retire beliefs, drop prose"))
    strategies = set((registry or {}).get("strategies") or {}) if registry is not None else None
    areas = set((registry or {}).get("areas") or {}) if registry is not None else None
    seen_scope: set[tuple[str, str, Optional[str]]] = set()
    for b in doc.beliefs:
        if strategies is not None and b.strategy and b.strategy not in strategies:
            doc.violations.append(Violation(
                "unknown-strategy", f"scope strategy `{b.strategy}` is not a "
                "registry strategy id (use one from the list in the prompt)",
                slug=b.slug))
        if areas is not None and b.area and b.area not in areas:
            doc.violations.append(Violation(
                "unknown-area", f"scope area `{b.area}` is not a registry "
                "area id", slug=b.slug))
        key = (b.kind, b.strategy, b.area)
        if b.strategy and key in seen_scope:
            doc.violations.append(Violation(
                "duplicate-scope", f"another {b.kind} belief already covers "
                f"{b.scope}", slug=b.slug))
        seen_scope.add(key)
    if records is not None:
        doc.violations.extend(verify_citations(strip_track_lines(text), records))
    return doc


def match_belief(beliefs: Sequence[Belief], kind: str,
                 tags: Sequence[Mapping[str, Any]]) -> Optional[tuple[Belief, int]]:
    """The belief of ``kind`` that covers a node whose sub-edits carry
    ``tags`` (``[{edit, strategy, area}]`` in record order). Most specific
    wins: a (strategy, area) match on any sub-edit beats a strategy-only
    match; ties go to sub-edit order. Returns ``(belief, edit_index)``."""
    mine = [b for b in beliefs if b.kind == kind and b.strategy]
    for t in tags:
        for b in mine:
            if b.area and b.strategy == t.get("strategy") and b.area == t.get("area"):
                return b, int(t.get("edit") or 0)
    for t in tags:
        for b in mine:
            if not b.area and b.strategy == t.get("strategy"):
                return b, int(t.get("edit") or 0)
    return None


def render_violations(violations: Sequence[Violation]) -> str:
    return "\n".join(f"{i}. {v.render()}" for i, v in enumerate(violations, 1))

"""Online, TextGrad-style optimization of the belief maintainer's GUIDANCE.

The belief generator's system prompt is two parts: a FIXED contract (format,
scoring rules — owned by ``edit_beliefs``) and a LEARNED guidance text (how
to reason about evidence). Only the guidance is optimized here.

Loop (all inside the live run):
  forward  — the maintainer writes beliefs under guidance version k;
  loss     — registered predictions are scored by Brier as nodes are measured
             (``belief_scoring``);
  backward — every ``every`` newly scored predictions, ONE LLM call sees the
  + step     guidance, its predecessors with their loss, and the scored
             predictions with the belief text that made them, writes a
             critique and a revised guidance (bounded by ``char_cap``).
  rollback — if the current version, with enough samples, is worse than an
             earlier one by more than ``rollback_margin``, that earlier text
             is restored before the next step (hill-climb, best-so-far).

Files (run root): ``belief_instruction.md`` (current), and under
``belief_instruction_archive/``: ``v000.md``…, ``step_NNN_prompt.txt``,
``step_NNN_response.json``. Bookkeeping lives in the belief state sidecar
under ``state["instruction"]``; this module never touches other keys.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .belief_scoring import Scored, mean_brier, per_version_brier
from .perf_text import score_context, score_well_measured

INSTRUCTION_NAME = "belief_instruction.md"
INSTRUCTION_ARCHIVE_DIR = "belief_instruction_archive"

SEED_INSTRUCTION = (
    "Write one strategy belief per registry strategy, scoped to the strategy "
    "alone (add an area only when you expect the effect to differ by area), and "
    "an implementation belief wherever the analyses show a broken mechanism "
    "(dead components, suspect verifiers, gates the scorer disagrees with). "
    "Commit: p=0.5 is scored exactly like having no belief. Use 0.5 only while a "
    "strategy has no measured outcome at all; once evidence points one way, move "
    "p modestly (0.35-0.65 on a single measured node) and further as measured "
    "nodes agree, and say in `next` what would change your mind. Cite every "
    "node you rely on. Keep the document short."
)

OPTIMIZER_SYSTEM = """You tune the GUIDANCE text that a belief maintainer follows when writing scoped
probabilities for a self-improving agent run. Each belief predicts either P(the judge finds
the mechanism improved its target | implementation sound) for a strategy scope, or
P(implementation sound). The judge is the per-node analysis LLM: it reads the edit's
implementation, its runtime traces, the per-check results and the per-case rows, and grades
each mechanism `improved` / `no_effect` / `regressed` with an evidence grade and the checks
it targeted — its verdicts are the labels (in delta-labelled ablation runs the label is
instead Δ ≥ threshold over shared cases). Each prediction is scored by Brier loss (p - y)^2
against that label; 0.25 is what an uninformative p=0.5 scores. The benchmark score Δ
appears below only where it is well measured, and only as context.

You see: the current guidance, its predecessors with their loss, the predictions scored
since the last revision (each with the belief text that made it, the judge's verdict,
evidence grade, reason and targeted checks, and — only when well measured — the score), and
the maintainer's calibration report.

Write a short critique of what went wrong SYSTEMATICALLY (hedging every belief at 0.5 —
which scores exactly like silence — or over-confidence; scoping too broadly or too
narrowly, leaving covered strategies unpredicted, ignoring implementation evidence or the
judge's reasons, trusting the description of an edit over what its traces showed, predicting
the score instead of the judge, chasing single-node noise), then a revised guidance that
fixes it. Constraints: general reasoning rules only — never name node ids or specific
strategies; do not restate the fixed format contract; keep it under {cap} characters.
Submit via `submit_instruction_update` (critique, instruction)."""

INSTRUCTION_TOOL = {"type": "function", "function": {
    "name": "submit_instruction_update",
    "description": "Submit the critique and the revised guidance text.",
    "parameters": {"type": "object", "properties": {
        "critique": {"type": "string"},
        "instruction": {"type": "string"}},
        "required": ["instruction"]}}}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class InstructionOptimizer:
    """``call(system, user, tool, tag) -> Optional[dict]`` is the belief
    store's LLM adapter (already bound to the optimizer's model/effort)."""

    def __init__(self, call: Callable[..., Optional[dict]], *,
                 enabled: bool = True, every: int = 8, min_scored: int = 8,
                 rollback_margin: float = 0.02, char_cap: int = 2500) -> None:
        self.call = call
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.min_scored = max(1, int(min_scored))
        self.rollback_margin = float(rollback_margin)
        self.char_cap = max(200, int(char_cap))

    # ------------------------------------------------------------------ #
    # Files / state
    # ------------------------------------------------------------------ #
    @staticmethod
    def _archive_dir(experiment_dir: Path) -> Path:
        return Path(experiment_dir) / INSTRUCTION_ARCHIVE_DIR

    def ensure_seed(self, experiment_dir: Path, state: dict[str, Any]) -> None:
        """Write v000 + the current file on first use (idempotent)."""
        experiment_dir = Path(experiment_dir)
        info = state.get("instruction")
        current = experiment_dir / INSTRUCTION_NAME
        if isinstance(info, dict) and info.get("versions") and current.exists():
            return
        _atomic_write(self._archive_dir(experiment_dir) / "v000.md",
                      SEED_INSTRUCTION + "\n")
        _atomic_write(current, SEED_INSTRUCTION + "\n")
        state["instruction"] = {
            "version": 0,
            "versions": [{"version": 0, "file": "v000.md",
                          "created_at_update": int(state.get("n_updates", 0)),
                          "parent": None, "critique": "",
                          "chars": len(SEED_INSTRUCTION)}],
            "events": [],
        }

    def current_text(self, experiment_dir: Path) -> str:
        path = Path(experiment_dir) / INSTRUCTION_NAME
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip() or SEED_INSTRUCTION
        except (OSError, UnicodeDecodeError):
            pass
        return SEED_INSTRUCTION

    @staticmethod
    def current_version(state: Mapping[str, Any]) -> int:
        info = state.get("instruction") or {}
        return int(info.get("version", 0) or 0)

    def _version_text(self, experiment_dir: Path, version: int) -> str:
        path = self._archive_dir(experiment_dir) / f"v{version:03d}.md"
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    # ------------------------------------------------------------------ #
    # The step
    # ------------------------------------------------------------------ #
    def maybe_step(self, experiment_dir: Path, state: dict[str, Any], *,
                   scored: Sequence[Scored],
                   predictions: Mapping[int, Mapping[str, Any]],
                   records: Mapping[int, Any], calibration_report: str,
                   n_updates: int) -> bool:
        """Run the rollback check + one optimization step when enough new
        predictions have been scored. Returns True when the guidance text
        changed (step or revert). Never raises past its own logging."""
        if not self.enabled:
            return False
        since = int(state.get("scored_since_step", 0) or 0)
        if since < self.every:
            return False
        experiment_dir = Path(experiment_dir)
        self.ensure_seed(experiment_dir, state)
        info = state["instruction"]
        cur = int(info.get("version", 0))
        changed = False

        pv = per_version_brier(scored)
        cur_stat = pv.get(cur)
        if cur_stat and cur_stat[0] >= self.min_scored:
            others = [(v, s) for v, s in pv.items()
                      if v != cur and s[0] >= self.min_scored]
            if others:
                best_v, best_s = min(others, key=lambda t: t[1][1])
                if best_s[1] < cur_stat[1] - self.rollback_margin:
                    text = self._version_text(experiment_dir, best_v)
                    if text:
                        _atomic_write(experiment_dir / INSTRUCTION_NAME, text + "\n")
                        info["events"].append({
                            "event": "revert", "at_update": n_updates,
                            "from": cur, "to": best_v,
                            "note": f"v{cur} Brier {cur_stat[1]:.3f} (n={cur_stat[0]}) "
                                    f"vs v{best_v} {best_s[1]:.3f} (n={best_s[0]})"})
                        info["version"] = best_v
                        cur = best_v
                        changed = True
                        print(f"[belief_optimizer] reverted guidance to v{best_v}",
                              flush=True)

        window = list(scored)[-since:] if since <= len(scored) else list(scored)
        user = self._build_prompt(experiment_dir, info, cur, pv, window,
                                  predictions, records, calibration_report)
        system = OPTIMIZER_SYSTEM.replace("{cap}", str(self.char_cap))
        step_no = len(info["versions"])
        got = self.call(system, user, INSTRUCTION_TOOL, "guidance step")
        text = self._clean(got)
        retry_note = ""
        if not self._acceptable(text):
            retry_note = ("\n\n## Your previous submission was rejected\n"
                          + (f"The guidance was {len(text)} chars; the cap is "
                             f"{self.char_cap}. " if text else
                             "No guidance text was submitted. ")
                          + "Resubmit a complete guidance text within the cap.")
            got = self.call(system, user + retry_note, INSTRUCTION_TOOL,
                            "guidance step (retry)")
            text = self._clean(got)
        try:
            _atomic_write(self._archive_dir(experiment_dir)
                          / f"step_{step_no:03d}_prompt.txt",
                          "### SYSTEM\n" + system + "\n\n### USER\n" + user
                          + retry_note + "\n")
            _atomic_write(self._archive_dir(experiment_dir)
                          / f"step_{step_no:03d}_response.json",
                          json.dumps(got, indent=1, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[belief_optimizer] prompt dump failed: {exc!r}", flush=True)
        state["scored_since_step"] = 0
        if not self._acceptable(text):
            info["events"].append({"event": "rejected", "at_update": n_updates,
                                   "from": cur, "to": cur,
                                   "note": "no acceptable guidance after retry"})
            print("[belief_optimizer] step rejected; guidance unchanged", flush=True)
            return changed
        new_v = len(info["versions"])
        _atomic_write(self._archive_dir(experiment_dir) / f"v{new_v:03d}.md",
                      text + "\n")
        _atomic_write(experiment_dir / INSTRUCTION_NAME, text + "\n")
        info["versions"].append({
            "version": new_v, "file": f"v{new_v:03d}.md",
            "created_at_update": n_updates, "parent": cur,
            "critique": str((got or {}).get("critique") or "")[:600],
            "chars": len(text)})
        info["events"].append({"event": "step", "at_update": n_updates,
                               "from": cur, "to": new_v,
                               "note": f"window n={len(window)}, Brier "
                                       f"{(mean_brier(window) or 0.0):.3f}"})
        info["version"] = new_v
        print(f"[belief_optimizer] guidance v{new_v} accepted ({len(text)} chars)",
              flush=True)
        return True

    def _acceptable(self, text: str) -> bool:
        return bool(text.strip()) and len(text) <= self.char_cap

    @staticmethod
    def _clean(got: Optional[Mapping[str, Any]]) -> str:
        if not got:
            return ""
        return str(got.get("instruction") or "").strip()

    # ------------------------------------------------------------------ #
    def _build_prompt(self, experiment_dir: Path, info: Mapping[str, Any],
                      cur: int, pv: Mapping[int, tuple[int, float]],
                      window: Sequence[Scored],
                      predictions: Mapping[int, Mapping[str, Any]],
                      records: Mapping[int, Any], calibration_report: str) -> str:
        parts: list[str] = []
        stat = pv.get(cur)
        head = (f"n={stat[0]} scored under it, Brier {stat[1]:.3f}" if stat
                else "no scored predictions under it yet")
        parts.append(f"## Current guidance (v{cur}; {head})\n"
                     + self.current_text(experiment_dir))

        hist = []
        for v in info.get("versions", []):
            ver = int(v.get("version", 0))
            if ver == cur:
                continue
            s = pv.get(ver)
            hist.append(f"- v{ver}: " + (f"n={s[0]}, Brier {s[1]:.3f}" if s
                                        else "no scored predictions"))
        if hist:
            parts.append("## Guidance history (earlier versions)\n" + "\n".join(hist))
            for v in info.get("versions", []):
                ver = int(v.get("version", 0))
                if ver == cur:
                    continue
                text = self._version_text(experiment_dir, ver)
                if text:
                    parts.append(f"### v{ver} text\n{text[:800]}")

        wb = mean_brier(window)
        lines = [f"## Predictions scored since the last revision ({len(window)}; "
                 f"window Brier {wb:.3f} vs 0.25 uninformative)"
                 if wb is not None else
                 "## Predictions scored since the last revision (none)"]
        for s in window:
            rec = records.get(s.node) or {}
            pred = predictions.get(s.node) or {}
            who = f"belief:{s.slug}" if s.slug else "(no belief covered this scope)"
            lines.append(f"### node {s.node} · {s.kind} · {who} · p={s.p:.2f} → "
                         f"y={s.y} · Brier {s.brier:.2f}")
            if s.kind == "strategy" and s.label_source == "judge":
                targets = []
                if s.matched_edit is not None:
                    tb = rec.get("targets_by_edit") or {}
                    targets = [str(t) for t in (tb.get(s.matched_edit) or [])][:4]
                grade = f"{s.evidence or '?'} evidence"
                if targets:
                    grade += "; targets: " + ", ".join(targets)
                lines.append(f"- outcome: judge says {s.effect or '?'} ({grade})"
                             + (f' — "{s.effect_reason[:220]}"'
                                if s.effect_reason else ""))
                # The score is quoted only where it is well measured; a thin
                # paired Δ or a within-noise unpaired one says nothing.
                if score_well_measured(delta=s.delta if s.n_shared else None,
                                       n_shared=s.n_shared,
                                       delta_all=s.delta_all, se_all=s.se_all):
                    ctx = score_context(delta=s.delta if s.n_shared else None,
                                        n_shared=s.n_shared,
                                        delta_all=s.delta_all, se_all=s.se_all)
                    lines.append(f"- score (well measured): {ctx}")
            elif s.kind == "strategy":
                lines.append(f"- outcome: Δ{s.delta:+.4f} over {s.n_shared} shared "
                             "(helped = Δ ≥ threshold)")
            else:
                lines.append(f"- outcome: implementation "
                             f"{'sound' if s.y else 'unsound'}"
                             + (f' — "{s.impl_reason[:200]}"' if s.impl_reason else ""))
            if s.kind == "strategy" and s.impl_reason:
                lines.append(f'- implementation: sound — "{s.impl_reason[:200]}"')
            tags = "; ".join(
                f"strategy={t.get('strategy')}"
                + (f" area={t['area']}" if t.get("area") else "")
                for t in (rec.get("tags") or [])) or "(no tags)"
            what = ""
            for t in rec.get("tags") or []:
                if t.get("what"):
                    what = t["what"][:160]
                    break
            lines.append(f"- edit: {tags}" + (f' · "{what}"' if what else ""))
            entry = pred.get(s.kind) if isinstance(pred, Mapping) else None
            section = (entry or {}).get("section") if isinstance(entry, Mapping) else None
            if section:
                lines.append("- belief text at registration:\n  "
                             + str(section).replace("\n", "\n  "))
        parts.append("\n".join(lines))
        parts.append(calibration_report)
        return "\n\n".join(parts)

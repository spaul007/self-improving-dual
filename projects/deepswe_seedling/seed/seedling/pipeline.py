"""THE MULTI-AGENT ARCHITECTURE.

Roles are CALLED by this pipeline; the model never chooses to delegate. That is measured,
not stylistic: OpenHands' `task` delegation tool fired 0 times in 24 instances unprompted
(EXP-011) and only 5.2% with a prompt engineered to demand it (EXP-015).

THE SEED'S CONTRACT IS ONE SENTENCE: BASELINE once, then PATCH -> VERIFY until VERIFY says
`pass` or the deadline arrives. A `pass` is accepted on SCHEMA alone -- the report call already
enforces the field types (roles.Role.validate) -- and nothing here reads the verdict's content.

WHY THERE IS NO GATE (EXP-027). The previous `_gate()` had six checks, four of them regexes over
prose the model wrote (a "none" word-list, a runner-name match, a test-runner counter, a fuzzy
behaviour-coverage matcher). Each encoded one incident; each then failed on the next: it rejected
genuine katex passes twice (`pnpm test:jest` vs `npx jest` -- same runner, different wrapper) and
accepted a recursive-delegation pass whose 6/6 scratch test exercised PATCH's own interface
choice (f2p 0.286). A seed agent whose acceptance rule is sixty lines of heuristics is a harness
the self-evolution loop cannot reason about. Every quantity the gate used to enforce is still
MEASURED -- as signals on `role_stats` and in the exec ledger -- and the false-pass rate against
the grader (jobs/verify_confusion.py) is the per-role metric the loop optimises. The gate's job
moved to the meta-agent, which is where it belongs.

Keep this file small. Branching logic belongs inside a role, not here.
"""

from __future__ import annotations

import re

from .settings import MAX_PATCH_ATTEMPTS
from .roles import PATCH, VERIFY, SOLO, BASELINE


def _validate_baseline(bl: dict, h) -> None:
    """Annotate (never gate) the baseline. `_unreliable` is shown to VERIFY as context and
    surfaced as a signal; it changes nothing in control flow.

    Accepts a built, green, plausibly-long run WITHOUT a numeric count: geo-shapeindex's
    `go test ./...` (14s, all green, "collected=?") was falsely flagged in EXP-026."""
    _col = re.sub(r"\D", "", str(bl.get("tests_collected") or ""))
    # v8.3: FIRST numeric token, not the digit-stripped whole string. koota's baseline reported
    # `.1..2026021621015801771219887256381808...` and the old stripping produced garbage that
    # float() rejected, fell back to 0.0, and false-flagged a 172/172 green suite.
    _m = re.search(r"\d+(?:\.\d+)?", str(bl.get("duration_sec") or ""))
    _dur = _m.group(0) if _m else "0"
    _failed = re.sub(r"\D", "", str(bl.get("failed") or "")) or "0"
    _built = str(bl.get("build_ok") or "").strip().lower() in ("yes", "none")
    try:
        _durf = float(_dur)
    except ValueError:
        _durf = 0.0
    if not str(bl.get("test_command") or "").strip():
        bl["_unreliable"] = "no test command reported"
    elif _col in ("", "0") and not (_built and _failed == "0" and _durf >= 3.0):
        bl["_unreliable"] = "0 tests collected (or no count) and no green >=3s run to vouch for it"
    elif _durf < 3.0 and _failed == "0":
        bl["_unreliable"] = f"suite green in {_dur}s -- implausible, probably ran nothing"
    if bl.get("_unreliable"):
        h.logger.warning("BASELINE UNRELIABLE: %s -- VERIFY will be told; no regression claims",
                         bl["_unreliable"])
    h.logger.warning("BASELINE cmd=%r collected=%s passed=%s failed=%s build=%s dur=%ss reliable=%s",
                     str(bl.get("test_command"))[:100], _col or "?", bl.get("passed"), _failed,
                     bl.get("build_ok"), _dur, not bl.get("_unreliable"))


async def _run_role(role, bb, h) -> dict:
    """Run one role and attribute its change BY GIT: checkpoint before/after, `git diff --numstat`
    between. Counters can lie (a task once reported 41 edits against a 0-byte patch); the diff
    is what the grader sees. The result lands on the role's own `role_stats` entry."""
    _pre = h.git.head_sha or h.git.base_sha
    out = await role.run(bb, h)
    # Commit immediately: `git diff BASE HEAD` is commit-to-commit, so a cancel between
    # here and the end would otherwise discard everything this role did.
    await h.git.checkpoint(f"seedling: {role.name} attempt {bb.attempt} end")
    try:
        ns = await h.git.numstat(_pre, h.git.head_sha)
        if bb.role_stats and bb.role_stats[-1].get("role") == role.name:
            bb.role_stats[-1]["git"] = ns
        # v8.3: re-flush so run_summary carries this role's git entry NOW, not one flush late.
        if h.on_progress is not None:
            try:
                h.on_progress(force=True)
            except TypeError:
                h.on_progress()
        h.logger.warning("role %s GIT files=%d +%d/-%d source_files=%d test_files=%d",
                         role.name, ns["files"], ns["added"], ns["deleted"],
                         ns["source_files"], ns["test_files"])
        if role.name != "patch" and ns["source_files"] > 0:
            h.logger.warning("role %s CHANGED SOURCE FILES %s -- attribution confound",
                             role.name, ns["paths"][:5])
    except Exception:  # noqa: BLE001 -- observability never breaks the pipeline
        h.logger.exception("numstat failed (non-fatal)")
    return out


def _verify_signals(bb, h) -> None:
    """Everything the old gate read, kept as MEASUREMENTS on VERIFY's role_stats entry."""
    v = bb.verify or {}
    if not (bb.role_stats and bb.role_stats[-1].get("role") == "verify"):
        return
    rs = bb.role_stats[-1]
    _beh = [str(b).strip() for b in (v.get("behaviours") or []) if str(b).strip()]
    _tested = [str(b).strip() for b in (v.get("behaviours_tested") or []) if str(b).strip()]
    rs.update({
        "verdict": v.get("verdict"),
        "incomplete_report": bool(v.get("_incomplete")),
        "n_behaviours": len(_beh), "n_behaviours_tested": len(_tested),
        "evidence_chars": len(str(v.get("evidence") or "")),
        "test_command_chars": len(str(v.get("test_command") or "")),
        "n_issues": len(v.get("issues") or []),
        "baseline_unreliable": bool((bb.baseline or {}).get("_unreliable")),
    })
    h.logger.warning("attempt %d: VERIFY verdict=%s behaviours=%d tested=%d tests_run=%d edits=%d "
                     "evidence_chars=%d incomplete=%s", bb.attempt, v.get("verdict"), len(_beh),
                     len(_tested), rs.get("tests_run", 0), rs.get("edits", 0),
                     rs["evidence_chars"], rs["incomplete_report"])


async def solve(bb, h):
    """BASELINE once; then PATCH -> VERIFY until VERIFY says `pass` or the deadline arrives.

    TWO roles, each with ONE persistent conversation for the whole trial. No exploration
    role: Claude Code gathers context as a phase inside one context -- "these phases blend
    together" -- and our PATCH role already explored on its own anyway (12-38 tool calls
    before its first edit, every attempt).

    NO HARD-CODED TEST OR BUILD COMMANDS. There is no command matrix here, because there
    cannot be a correct one: across these 113 tasks there are 170 distinct test invocations
    (`go test`, `npx vitest/jest/ava`, `cargo nextest`, `pytest`, `stestr`, `deno test`,
    `tox`, bespoke runners), and builds are project-specific too. So BASELINE discovers how THIS
    project tests, VERIFY re-runs it and writes its own test, and its verdict is FINAL.

    State crosses attempts two ways: each role RESUMES its own conversation (so it cannot
    forget what it tried), and PATCH additionally receives `git diff base..HEAD` plus
    VERIFY's report from the previous attempt.
    """
    # BASELINE FIRST, once. It shares VERIFY's conversation, so VERIFY remembers it.
    bb.baseline = await _run_role(BASELINE, bb, h)
    _validate_baseline(bb.baseline or {}, h)

    attempt = -1
    while not h.deadline.expired() and attempt + 1 < MAX_PATCH_ATTEMPTS:
        attempt += 1
        bb.attempt = attempt + 1

        # Refresh the diff so the retry SEES its predecessor's work rather than
        # rediscovering it. Cheap (one git command) and it is the same view the grader gets.
        if bb.attempt > 1:
            bb.repo_diff = await h.git.current_diff()

        bb.patch = await _run_role(PATCH, bb, h)
        bb.verify = await _run_role(VERIFY, bb, h)
        _verify_signals(bb, h)

        # THE ACCEPTANCE RULE. Schema-validated by the report call; a synthesized
        # (incomplete) report is never a pass because _synthesize() picks `fail` for enums.
        if (bb.verify or {}).get("verdict") == "pass":
            bb.verify_passes += 1
            h.logger.warning("attempt %d: VERIFY pass -- task ends", bb.attempt)
            break
        if h.deadline.expired():
            break

    return bb


async def solve_single(bb, h):
    """Single-role control. The multi-vs-single ablation is this one-line difference."""
    bb.patch = await _run_role(SOLO, bb, h)
    return bb

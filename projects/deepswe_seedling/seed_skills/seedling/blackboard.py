"""Shared state between roles, and the inter-agent communication surface.

Each role sees ONLY what its context() function returns. That function IS the comms
design: widening what Patch can see is a one-line change whose effect is measurable.

The baseline deliberately passes SUMMARIES, not transcripts. A 27B model degrades on long
context, and Qwen3.8's 131k window against repos of ~1 GB / 46k files is not the place to
be generous. Passing transcripts is a mutation an evolver can try and we can measure.

EXP-027: `gate_issues` and `verify_rejection` are GONE with the gate. PATCH sees VERIFY's report
as "Reviewer notes" and nothing else from the harness. The old "MEASURED build/test failures you
MUST fix" channel relabelled a reviewer's GREEN evidence as a failure list after a rejected pass
(katex, EXP-026) -- harness text read as instruction, the same family as the report residue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def _fmt(label: str, payload) -> str:
    if not payload:
        return ""
    body = json.dumps(payload, indent=2, default=str) if isinstance(payload, (dict, list)) \
        else str(payload)
    return f"\n## {label}\n{body}\n"


@dataclass
class Blackboard:
    task: str
    patch: dict | None = None
    verify: dict | None = None
    repo_diff: str = ""                 # the change so far, as the grader will see it
    baseline: dict | None = None        # test baseline on the UNMODIFIED repo (BASELINE role)
    # PERSISTENT PER-ROLE CONVERSATION. attempt N+1 RESUMES attempt N's own message list
    # rather than starting fresh. Measured why: kea's patch.3 rediscovered 15 of the 16
    # things earlier attempts had already examined, and all three attempts produced
    # byte-identical gate verdicts. This is not the earlier "fork transcript" -- that copied
    # a truncated tail of OTHER roles into a fresh context and needed three invented
    # constants. This is simply not resetting the list, which is what one session means.
    histories: dict = field(default_factory=dict)
    attempt: int = 0
    history: list = field(default_factory=list)   # append-only audit of every role run
    role_stats: list = field(default_factory=list) # per role-run counters: the failure-class signals
    verify_passes: int = 0                          # verdict == pass count (the only acceptance rule)

    def conversation(self, role: str) -> list:
        """The role's own message list, persisted across attempts."""
        return self.histories.setdefault(role, [])

    # trim_history() REMOVED in v8. It clipped OLD tool output to 1,500 chars and kept
    # every step forever -- the shape of history at degraded fidelity, so a role could see
    # that it ran a build but not what the build said. compact.maybe_compact() replaces it:
    # full fidelity recently, a structured summary of everything older (Claude Code's
    # design, whose own session file shows it preserving the last ~7 records verbatim).

    def record(self, role: str, out: dict) -> None:
        self.history.append({"role": role, "attempt": self.attempt, "output": out})

    def record_stats(self, role: str, stats: dict) -> None:
        """Machine-readable per-role-run signals; surfaced in run_summary.json."""
        self.role_stats.append(dict(stats))

    # -- the communication surface ----------------------------------------------------
    def context_for(self, role: str) -> str:
        """What this role is allowed to see. Edit these lists to change agent comms."""
        if role == "patch":
            return (_fmt("Task", self.task)
                    + _fmt("Tests that ALREADY FAIL on the unmodified repo (not caused by you; "
                           "do not chase them)", (self.baseline or {}).get("failing_tests"))
                    # WHAT YOU ALREADY CHANGED. Without this a retry rediscovers its own
                    # predecessor's edits by reading files -- measured on katex, patch.2
                    # re-read 1 of the 2 files patch.1 had edited.
                    + _fmt("The change you have made SO FAR (git diff base..HEAD)", self.repo_diff)
                    # The reviewer's report, verbatim: verdict, behaviours, evidence, issues.
                    # Empty on attempt 1; the retry signal on later ones.
                    + _fmt("Reviewer notes", self.verify))
        if role == "verify":
            return (_fmt("Task", self.task)
                    + _fmt("Your BASELINE (measured on the UNMODIFIED repo -- run the SAME "
                           "test command; a regression is a test that passed here and fails now)",
                           self.baseline)
                    + _fmt("What the patch role changed", self.patch))
        return _fmt("Task", self.task)   # solo

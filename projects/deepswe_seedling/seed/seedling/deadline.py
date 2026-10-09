"""Monotonic budget arithmetic for the agent run.

Pier enforces the agent budget with `asyncio.wait_for(agent.run(...), agent_timeout_sec)`
(trial/execution.py). Expiry is a HARD CancelledError raised at whatever `await` we are
parked on -- there is no cooperative stop signal. So we keep our own *soft* deadline
strictly inside Pier's hard one and aim to finish normally before it fires.

Two reserves sit between soft and hard:
  finalize_reserve -- time to run the final commit + apply-check
  safety           -- covers the gap between wait_for starting its clock and run()'s
                      first statement executing, plus scheduler jitter
"""

from __future__ import annotations

import time


class Deadline:
    def __init__(
        self,
        t0: float,
        total_sec: float,
        finalize_reserve_sec: float = 180.0,
        safety_sec: float = 120.0,
    ) -> None:
        self.t0 = t0
        self.total_sec = float(total_sec)
        self.finalize_reserve_sec = float(finalize_reserve_sec)
        self.safety_sec = float(safety_sec)
        self.hard = t0 + self.total_sec
        # Never let the reserves invert the window on a very short budget.
        self.soft = max(t0, self.hard - self.finalize_reserve_sec - self.safety_sec)

    def remaining(self) -> float:
        """Seconds until Pier cancels us."""
        return self.hard - time.monotonic()

    def soft_remaining(self) -> float:
        """Seconds of normal working time left."""
        return self.soft - time.monotonic()

    def expired(self) -> bool:
        """True once we should stop starting new work and go finalize."""
        return time.monotonic() >= self.soft

    def clamp(self, want_sec: float, critical: bool = False) -> float:
        """Largest timeout a command may have.

        Normal work is clamped to the SOFT deadline so the pipeline stops starting new
        work in time to finalize. Finalization itself is `critical=True` and clamps to the
        HARD deadline instead -- the finalize_reserve is precisely the budget it should be
        allowed to spend. Clamping finalize to soft_remaining() refuses the final commit
        and scores the trial 0.
        """
        if critical:
            return max(0.0, min(float(want_sec), self.remaining() - 15.0))
        return max(0.0, min(float(want_sec), self.soft_remaining()))

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def snapshot(self) -> dict:
        return {
            "total_sec": self.total_sec,
            "elapsed_sec": round(self.elapsed(), 1),
            "soft_remaining_sec": round(self.soft_remaining(), 1),
            "hard_remaining_sec": round(self.remaining(), 1),
            "soft_deadline_hit": self.expired(),
        }

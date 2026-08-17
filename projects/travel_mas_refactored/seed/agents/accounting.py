"""Stage 4: Accounting Agent -- budget-tally specialist (no tools).

Reads the finished day-by-day itinerary (Sightseeing's output, already
containing the Flight/Train legs) and computes only the itemized Budget
Summary. Does not re-plan, re-type, or audit the itinerary itself -- see
``run_accounting_stage``'s docstring for why the final plan is assembled
in plain Python rather than asking the LLM to reproduce the body.
"""
from __future__ import annotations

import re

from agents.common import run_notool_stage

BUDGET_RULES = """
--------------------------------------------------
BUDGET / PRICING CALCULATION RULES
--------------------------------------------------
travel_city: price shown is total cost per vehicle per trip.
  total = trip price x number of vehicles (taxi = 4 passengers/vehicle, round up).
travel_intercity_public: price shown is per person.
  total = price per person x total passengers.
attraction: price shown is per person.
  total = ticket price x total passengers.
meal: price shown is per person (estimated per-capita consumption).
  total = per capita price x total number of people.
hotel/accommodation: price shown is per room per night.
  total = per-room price x number of rooms x number of nights.

Final Budget Summary format (last thing in the plan, after the last day):
**Budget Summary**:
   **Transportation: <total> RMB**. <one-line arithmetic breakdown>
   **Accommodation: <total> RMB**. <one-line arithmetic breakdown>
   **Meals: <total> RMB**. <one-line arithmetic breakdown>
   **Attractions & Tickets: <total> RMB**. <one-line arithmetic breakdown>
   **Total Estimated Budget: <sum of the four totals above> RMB**
"""

ACCOUNTING_SYSTEM_PROMPT = f"""You are the accounting specialist, the last
role in a team planning a trip. Three other specialists already decided
every flight/train leg and built the complete day-by-day itinerary (given
to you below, for reference only -- you do not need to repeat it back).
You do not re-plan or change any activity, time, name, or per-line price.
Your only job is to compute and report the itemized Budget Summary.

{BUDGET_RULES}

Read every per-line price in the itinerary below and compute the Budget
Summary, applying the passenger-count/vehicle-count/room-count/night-count
multipliers per the rules above. Report the true computed total even if
it exceeds any budget figure mentioned in the traveler's request -- do
not alter or omit anything to hide an overage; that is not your role.

Output ONLY the Budget Summary block (the "**Budget Summary**:" line and
its four sub-totals plus the grand total) wrapped in
<budget_summary></budget_summary> tags. Do not repeat the itinerary
itself -- another step appends your summary to it verbatim."""

_BUDGET_RE = re.compile(r"<budget_summary>(.*?)</budget_summary>", re.DOTALL | re.IGNORECASE)


def _extract_budget_summary(text: str) -> str:
    if not text:
        return ""
    match = _BUDGET_RE.search(text)
    return match.group(1).strip() if match else ""


def run_accounting_stage(task_description: str, sightseeing_body: str) -> str:
    """Compute the Budget Summary only -- never asks an LLM to retype the
    itinerary. An earlier design had this stage "reproduce the day-by-day
    body exactly" before appending the summary; live evaluation showed
    that large-text transcription step silently dropped/paraphrased
    content (Itinerary Structure and Route Consistency dimensions failed
    on ~90%+ of a 120-case run) even though the underlying itinerary was
    fine. Assembling the final <plan> here in plain Python, from the
    Sightseeing stage's untouched text plus this stage's small, focused
    budget computation, removes that whole failure class by construction."""
    user_content = (
        f"Traveler's request (for reference, e.g. party size / room count / "
        f"stated budget):\n{task_description}\n\n"
        f"Day-by-day itinerary (for computing the budget from -- do not "
        f"repeat it back):\n{sightseeing_body}\n"
    )
    text = run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, user_content)
    summary = _extract_budget_summary(text)
    if not summary:
        # One retry: nudge explicitly for the missing tag. If this also
        # fails, fall back to the raw text (still better than dropping
        # the budget section entirely) -- the itinerary body itself is
        # never at risk either way.
        retry_note = (
            user_content
            + "\n\nYour previous answer did not include "
            "<budget_summary></budget_summary> tags. Re-send just the "
            "Budget Summary, wrapped in <budget_summary>...</budget_summary>."
        )
        text = run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, retry_note)
        summary = _extract_budget_summary(text) or text.strip()

    return f"{sightseeing_body.strip()}\n\n{summary.strip()}"

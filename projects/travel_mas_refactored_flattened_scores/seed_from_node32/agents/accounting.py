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
from agents.immutable.message import AgentMessage, from_sender
from platform_core import trace
from platform_core.runner import Task

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


# ---------------------------------------------------------------------------
# Budget consistency verifier: parse line-item prices from the sightseeing
# body, extract multipliers from the budget summary, recompute totals, and
# patch any mismatched numbers. This catches the cost_calculation_correctness
# failures where the LLM's summary totals don't match what the body's own
# prices actually add up to.
# ---------------------------------------------------------------------------

# Regex patterns for parsing line items from the sightseeing body.
_CITY_PRICE_RE = re.compile(r'¥([\d,]+(?:\.[\d]+)?)')
_PER_PERSON_PRICE_RE = re.compile(r'¥([\d,]+(?:\.[\d]+)?)/person')
_HOTEL_PRICE_RE = re.compile(r'[Aa]ccommo?dation:.*?¥([\d,]+(?:\.[\d]+)?)/room/night')
_ACCOMMODATION_LINE_RE = re.compile(r'^[Aa]ccommo?dation:', re.MULTILINE)

# Patterns for extracting multipliers from the budget summary text.
# The LLM often uses the multiplication sign × (U+00D7) instead of ASCII x,
# and may write plural forms (rooms, nights). Both variants are accepted.
_PAX_RE = re.compile(r'[x×]\s*(\d+)\s*(?:person|people)')
_ROOM_RE = re.compile(r'[x×]\s*(\d+)\s*(?:room|rooms)')
_NIGHT_RE = re.compile(r'[x×]\s*(\d+)\s*(?:night|nights)')

# Patterns for extracting stated totals from the budget summary.
_STATED_TRANSPORT_RE = re.compile(
    r'\*\*Transportation:\s*([\d,]+(?:\.[\d]+)?)\s*RMB\*\*'
)
_STATED_ACCOMMODATION_RE = re.compile(
    r'\*\*Accommodation:\s*([\d,]+(?:\.[\d]+)?)\s*RMB\*\*'
)
_STATED_MEALS_RE = re.compile(
    r'\*\*Meals:\s*([\d,]+(?:\.[\d]+)?)\s*RMB\*\*'
)
_STATED_ATTRACTIONS_RE = re.compile(
    r'\*\*Attractions & Tickets:\s*([\d,]+(?:\.[\d]+)?)\s*RMB\*\*'
)
_STATED_TOTAL_RE = re.compile(
    r'\*\*Total Estimated Budget:\s*([\d,]+(?:\.[\d]+)?)\s*RMB\*\*'
)


def _parse_float(text: str) -> float:
    return float(text.replace(',', ''))


def _fmt_money(value: float) -> str:
    """Format a float as integer with comma separators, e.g. 3749 -> '3,749'."""
    return f'{value:,.0f}'


def verify_budget_consistency(
    sightseeing_body: str, budget_summary: str, task_description: str = ""
) -> str:
    """Verify that budget summary totals match line-item prices in the body.

    Parses all per-unit prices from the sightseeing itinerary body,
    extracts multipliers (passengers, rooms, nights) from the budget
    summary text, recomputes each category total, and patches any
    mismatched numbers in the summary. On any parse failure the original
    summary is returned unchanged — this is a best-effort patch, not a
    hard gate.
    """
    try:
        # ── Parse per-unit prices from the sightseeing body ──────────
        city_prices: list[float] = []
        intercity_prices: list[float] = []
        meal_prices: list[float] = []
        attraction_prices: list[float] = []
        hotel_price: float | None = None

        for line in sightseeing_body.split('\n'):
            line = line.strip()
            if not line:
                continue

            if '| travel_city |' in line:
                m = _CITY_PRICE_RE.search(line)
                if m:
                    city_prices.append(_parse_float(m.group(1)))
            elif '| travel_intercity_public |' in line:
                m = _PER_PERSON_PRICE_RE.search(line)
                if m:
                    intercity_prices.append(_parse_float(m.group(1)))
            elif '| meal |' in line:
                m = _PER_PERSON_PRICE_RE.search(line)
                if m:
                    meal_prices.append(_parse_float(m.group(1)))
            elif '| attraction |' in line:
                m = _PER_PERSON_PRICE_RE.search(line)
                if m:
                    attraction_prices.append(_parse_float(m.group(1)))

        # Hotel price — take the first Accommodation line.
        hotel_match = _HOTEL_PRICE_RE.search(sightseeing_body)
        if hotel_match:
            hotel_price = _parse_float(hotel_match.group(1))

        # Number of nights = count of Accommodation: lines in the body.
        num_nights = len(_ACCOMMODATION_LINE_RE.findall(sightseeing_body))

        # ── Extract multipliers from the budget summary ──────────────
        passengers = 1
        rooms = 1

        pax_match = _PAX_RE.search(budget_summary)
        if pax_match:
            passengers = int(pax_match.group(1))
        else:
            trace.log(
                label='verifier_fired',
                verdict='skip',
                name='budget_pax_default',
                reason='no passenger multiplier found in budget summary',
            )

        room_match = _ROOM_RE.search(budget_summary)
        if room_match:
            rooms = int(room_match.group(1))
        else:
            trace.log(
                label='verifier_fired',
                verdict='skip',
                name='budget_room_default',
                reason='no room multiplier found in budget summary',
            )

        night_match = _NIGHT_RE.search(budget_summary)
        if night_match:
            num_nights = int(night_match.group(1))
        else:
            trace.log(
                label='verifier_fired',
                verdict='skip',
                name='budget_night_default',
                reason=f'no night multiplier found; using body count ({num_nights})',
            )

        vehicles = max(1, (passengers + 3) // 4)  # ceil(passengers / 4)

        # ── Compute expected totals ──────────────────────────────────
        transport_city_total = sum(city_prices) * vehicles
        transport_intercity_total = sum(intercity_prices) * passengers
        transport_total = transport_city_total + transport_intercity_total

        accommodation_total = (hotel_price or 0) * rooms * num_nights
        meals_total = sum(meal_prices) * passengers
        attractions_total = sum(attraction_prices) * passengers

        grand_total = (
            transport_total + accommodation_total + meals_total + attractions_total
        )

        # ── Compare against stated totals ────────────────────────────
        def _extract_stated(pattern: re.Pattern, text: str) -> float | None:
            m = pattern.search(text)
            return _parse_float(m.group(1)) if m else None

        stated_transport = _extract_stated(_STATED_TRANSPORT_RE, budget_summary)
        stated_accommodation = _extract_stated(_STATED_ACCOMMODATION_RE, budget_summary)
        stated_meals = _extract_stated(_STATED_MEALS_RE, budget_summary)
        stated_attractions = _extract_stated(_STATED_ATTRACTIONS_RE, budget_summary)
        stated_total = _extract_stated(_STATED_TOTAL_RE, budget_summary)

        mismatches: list[str] = []
        if stated_transport is not None and abs(stated_transport - transport_total) > 0.5:
            mismatches.append(
                f'transport: {stated_transport:.0f} -> {transport_total:.0f}'
            )
        if stated_accommodation is not None and abs(stated_accommodation - accommodation_total) > 0.5:
            mismatches.append(
                f'accommodation: {stated_accommodation:.0f} -> {accommodation_total:.0f}'
            )
        if stated_meals is not None and abs(stated_meals - meals_total) > 0.5:
            mismatches.append(
                f'meals: {stated_meals:.0f} -> {meals_total:.0f}'
            )
        if stated_attractions is not None and abs(stated_attractions - attractions_total) > 0.5:
            mismatches.append(
                f'attractions: {stated_attractions:.0f} -> {attractions_total:.0f}'
            )
        if stated_total is not None and abs(stated_total - grand_total) > 0.5:
            mismatches.append(
                f'total: {stated_total:.0f} -> {grand_total:.0f}'
            )

        # ── Instrument the decision ──────────────────────────────────
        if mismatches:
            trace.log(
                label='verifier_fired',
                verdict='fail',
                name='budget_consistency',
                mismatches=mismatches,
                passengers=passengers,
                rooms=rooms,
                nights=num_nights,
                vehicles=vehicles,
            )
        else:
            trace.log(
                label='verifier_fired',
                verdict='pass',
                name='budget_consistency',
            )

        # ── Patch the summary (always, to ensure consistent formatting) ──
        summary = budget_summary
        summary = _STATED_TRANSPORT_RE.sub(
            f'**Transportation: {_fmt_money(transport_total)} RMB**', summary
        )
        summary = _STATED_ACCOMMODATION_RE.sub(
            f'**Accommodation: {_fmt_money(accommodation_total)} RMB**', summary
        )
        summary = _STATED_MEALS_RE.sub(
            f'**Meals: {_fmt_money(meals_total)} RMB**', summary
        )
        summary = _STATED_ATTRACTIONS_RE.sub(
            f'**Attractions & Tickets: {_fmt_money(attractions_total)} RMB**', summary
        )
        summary = _STATED_TOTAL_RE.sub(
            f'**Total Estimated Budget: {_fmt_money(grand_total)} RMB**', summary
        )

        return summary

    except Exception:
        # Best-effort: any parse failure returns the original summary.
        return budget_summary


def run_accounting_stage(task: Task, inbox: list[AgentMessage]) -> AgentMessage:
    """Compute the Budget Summary only -- never asks an LLM to retype the
    itinerary. An earlier design had this stage "reproduce the day-by-day
    body exactly" before appending the summary; live evaluation showed
    that large-text transcription step silently dropped/paraphrased
    content (Itinerary Structure and Route Consistency dimensions failed
    on ~90%+ of a 120-case run) even though the underlying itinerary was
    fine. Assembling the final <plan> here in plain Python, from the
    Sightseeing stage's untouched text plus this stage's small, focused
    budget computation, removes that whole failure class by construction.
    Never fails today -- ok stays at its AgentMessage default (True)."""
    sightseeing_body = from_sender(inbox, "sightseeing").content
    user_content = (
        f"Traveler's request (for reference, e.g. party size / room count / "
        f"stated budget):\n{task.description}\n\n"
        f"Day-by-day itinerary (for computing the budget from -- do not "
        f"repeat it back):\n{sightseeing_body}\n"
    )
    text = run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, user_content, "accounting")
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
        text = run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, retry_note, "accounting")
        summary = _extract_budget_summary(text) or text.strip()

    # ── Budget consistency verifier ──────────────────────────────────
    # The LLM sometimes produces summary totals that don't match the
    # body's own line-item prices (cost_calculation_correctness failed in
    # 22 parent cases). Parse both, recompute, and patch any mismatches
    # deterministically — the body's prices are already correct; we just
    # fix the summary.
    if summary and sightseeing_body:
        summary = verify_budget_consistency(
            sightseeing_body, summary, task.description
        )

    plan = f"{sightseeing_body.strip()}\n\n{summary.strip()}"
    return AgentMessage(sender="accounting", content=plan)

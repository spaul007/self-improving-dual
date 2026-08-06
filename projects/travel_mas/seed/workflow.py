"""Travel-MAS agent -- a 4-role sequential pipeline over the same benchmark
as ``projects/travel/seed``: Flight Agent -> Train Agent -> Sightseeing
Agent -> Accounting Agent. Each role owns a distinct slice of the task and
(except Accounting) a distinct subset of the 9 tools; each role's final text
becomes a "note" handed to the next role. There is no step that re-reads the
finished plan and audits/patches it against the scoring rubric -- Accounting
does the itemized budget tally its role would do anyway, nothing more.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput
"""
from __future__ import annotations

import json
import os
import re

from platform_core.llm_wrapper import call_llm
from platform_core.runner import AgentOutput, Task

from tool_wrapper import ToolWrapper

MAX_ITERATIONS_PER_STAGE = 25
# Sightseeing is the heaviest role (hotel + N attractions + N restaurants +
# road-route lookups, then composing the full multi-day body) -- 25 was
# sometimes too tight: live evaluation showed a case where it ran out of
# budget mid-tool-loop and its trailing "final" text was just leftover
# reasoning prose ("Let me pick another attraction for Day 2...") instead
# of a real itinerary, which the Accounting stage then tried to compute a
# budget from anyway, fabricating numbers. Higher cap here reduces how
# often that happens; the <itinerary> tag + retry below is the actual
# safety net.

FLIGHT_TOOLS = {"query_flight_info", "search_location"}
TRAIN_TOOLS = {"query_train_info"}
SIGHTSEEING_TOOLS = {
    "query_hotel_info",
    "recommend_attractions",
    "query_attraction_details",
    "recommend_restaurants",
    "query_restaurant_details",
    "query_road_route_info",
    "search_location",
}

# ================================================================
# Shared format/pricing rules, quoted from the reference plan
# format so all four roles emit numbers and lines the scorer
# actually accepts. Split per-role below rather than repeating the
# single agent's full monolithic prompt.
# ================================================================

COMMON_RULES = """
All information must come exclusively from tool query results. Do not
fabricate, guess, or use any data outside of tool results. Names must
match tool query results exactly -- do not abbreviate, rename, or add
extra descriptions.
"""

INTERCITY_LINE_SPEC = """
Output one line per intercity leg you book, in this exact format:
  HH:MM-HH:MM | travel_intercity_public | <flight/train> <No.>, <Departure Stop> - <Arrival Stop>, ¥<price>/person
Example: 07:00-09:00 | travel_intercity_public | flight CA1234, Shanghai Hongqiao International Airport - Beijing Capital International Airport, ¥650/person
Times/duration must match the tool result exactly, without adjustment.
Price shown is per person; do not multiply it yourself -- state the raw
per-person price from the tool result verbatim (the Accounting stage
will apply passenger-count multiplication).
"""

DAY_STRUCTURE_RULES = """
--------------------------------------------------
DAILY STRUCTURE
--------------------------------------------------
Each day begins with:
Day [N]:
Current City: [see rule below]
Accommodation: [Hotel name], ¥[price]/room/night   (omit this line on the final day if departing)

**Current City format is not cosmetic -- it determines how this day is
scored.** If this day's activity list contains ANY travel_intercity_public
leg (arrival or departure), the Current City line MUST be written as
"from X to Y" (e.g. "from Harbin to Dalian"), even if the leg only takes
up part of the day and the rest is spent in one city. Writing just "Y" or
just "X" on such a day causes it to be scored as a full ordinary day
(requiring 2 meals and 2 attractions) instead of a transfer day -- this
is true for the FINAL day too (the one that only departs, doesn't
arrive). Only write a single city name on a day with no intercity leg at
all.

Activity line formats (each is one line, chronological, no gaps/overlaps):
1. Intercity transport (flight/train) -- see the format given to you above;
   insert the Flight/Train agents' exact lines at the correct time slot.
2. Intracity transport:
   HH:MM-HH:MM | travel_city | [Start] - [End], [distance], [duration], ¥[price]
   (price is the total per-vehicle cost for that hop; taxi seats 4, round up
   vehicle count from passenger count when computing totals later)
3. Attraction visit:
   HH:MM-HH:MM | attraction | [Attraction Name], ¥[price]/person
4. Meal:
   HH:MM-HH:MM | meal | [Lunch/Dinner], [Restaurant Name], ¥[price]/person
5. Hotel:
   HH:MM-HH:MM | hotel | [Check-in/Check-out/Rest], [Hotel Name]
6. Buffer (waiting/prep time around intercity transport, or short rest):
   HH:MM-HH:MM | buffer | [description]

Rules:
- Geospatial continuity: insert a travel_city or travel_intercity_public
  line whenever the end location of one activity differs from the start
  of the next. The full trip is a closed loop (starts and ends in the
  origin city).
- Attraction/meal times must respect the tool-reported opening hours and
  min/max visit-hour ranges.
- No breakfast needs scheduling (assumed at the hotel). At least 2 hours
  between lunch and dinner. Full sightseeing days need both lunch and
  dinner; transfer days depend on arrival/departure time (arrive before
  10:00 -> both meals; 10:00-15:00 -> dinner, lunch optional; after 15:00
  -> at most one meal; symmetric logic for departure).
- Except on the final day, the last activity of each day is returning to
  the hotel to rest. On the final day, the last activity is arriving at
  the departure airport/station.
- Avoid repeating the same restaurant or attraction across days.
- A full sightseeing day needs at least 2 attractions (or >=4h at one
  major attraction). A transfer day needs at least 1 attraction if there
  is a meaningful arrival/departure window (arrive before 12:00, or leave
  after 16:00).
"""

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

_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def _extract_plan(text: str) -> str:
    """Same extraction rule as the single-agent seed: strip everything up
    to the last ``</think>``, join all ``<plan>...</plan>`` blocks, return
    ``""`` if none are present (no lenient fallback)."""
    if not text:
        return ""
    think_ends = list(_THINK_END_RE.finditer(text))
    if think_ends:
        text = text[think_ends[-1].end():]
    matches = _PLAN_RE.findall(text)
    cleaned = [m.strip() for m in matches if m.strip()]
    return "\n\n".join(cleaned) if cleaned else ""


def _item_type(item) -> str:
    t = getattr(item, "type", None)
    if t is None and isinstance(item, dict):
        t = item.get("type")
    return t or ""


def _strip_reasoning(raw_output):
    """See projects/travel/seed/workflow.py for the full rationale: local
    vLLM-served models misread echoed `reasoning` items as a wrap-up signal.
    Controlled by the same META_AGENT_STRIP_REASONING env var/convention."""
    items = list(raw_output or [])
    if os.environ.get("META_AGENT_STRIP_REASONING") != "1":
        return items
    return [item for item in items if _item_type(item) != "reasoning"]


def _filter_schema(schema: list[dict], allowed_names: set[str]) -> list[dict]:
    return [t for t in schema if t.get("function", {}).get("name") in allowed_names]


def _run_tool_stage(
    system_prompt: str,
    user_content: str,
    schema: list[dict],
    wrapper: ToolWrapper,
    max_iterations: int = MAX_ITERATIONS_PER_STAGE,
) -> tuple[str, int, bool, list]:
    """Run one role's bounded tool-calling loop. Returns
    (final_text, iterations_used, budget_exhausted, messages) -- the
    accumulated ``messages`` lets a caller do a follow-up nudge call with
    full context (see ``_run_sightseeing_stage``'s retry) instead of
    starting over."""
    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_text = ""
    for i in range(max_iterations):
        response = call_llm(messages=messages, tools=schema)
        last_text = response.content or last_text

        raw = getattr(response, "raw", None)
        raw_output = getattr(raw, "output", None) or []
        messages.extend(_strip_reasoning(raw_output))

        if not response.tool_calls:
            return response.content or "", i + 1, False, messages

        for tc in response.tool_calls:
            try:
                result = wrapper.execute(tc.name, tc.arguments)
            except Exception as e:  # noqa: BLE001 - surface tool errors to the model
                result = json.dumps({"error": str(e)}, ensure_ascii=False)
            messages.append({
                "type": "function_call_output",
                "call_id": tc.id,
                "output": result,
            })

    return last_text, max_iterations, True, messages


def _run_notool_stage(system_prompt: str, user_content: str) -> str:
    response = call_llm(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])
    return response.content or ""


# ================================================================
# Stage 1: Flight Agent
# ================================================================

FLIGHT_SYSTEM_PROMPT = f"""You are an airline booking specialist, one role
in a team planning a trip. You do not plan the rest of the itinerary --
another specialist handles hotels/attractions/trains, and a third
assembles the final budget. Your only job: for every intercity leg of
this trip, decide whether it should be booked by air, and if so book it.

{COMMON_RULES}

Read the traveler's request. It may describe one leg (a simple round trip)
or several (a multi-city trip). For each leg:
- If the request names a specific flight number, or otherwise makes clear
  air travel is wanted/required for that leg, query flights (respecting
  any stated seat-class constraint via the seatClassName filter) and pick
  one whose remaining-seats field confirms availability.
- If the request clearly wants a different mode (e.g. names a specific
  train number, or says "by train") for a leg, do NOT book a flight for
  that leg -- state plainly "No flight needed for <leg>" instead.
- If the mode isn't specified for a leg, use judgement (distance/duration)
  or book a reasonable flight if one is plausible; state your assumption.

{INTERCITY_LINE_SPEC}

When you are done deciding every leg, stop calling tools and write your
final note as plain text: one line per leg, either the formatted booking
line above or an explicit "No flight needed for <leg>" statement. Do not
invent a whole itinerary -- only report on intercity air travel."""


def _run_flight_stage(task_description: str, wrapper: ToolWrapper, full_schema: list[dict]):
    schema = _filter_schema(full_schema, FLIGHT_TOOLS)
    text, iters, exhausted, _messages = _run_tool_stage(
        FLIGHT_SYSTEM_PROMPT, task_description, schema, wrapper
    )
    return text.strip() or "No flight information produced.", iters, exhausted


# ================================================================
# Stage 2: Train Agent
# ================================================================

TRAIN_SYSTEM_PROMPT = f"""You are a rail booking specialist, one role in a
team planning a trip. You do not plan the rest of the itinerary -- another
specialist handles hotels/attractions/flights, and a third assembles the
final budget. Your only job: for every intercity leg of this trip, decide
whether it should be booked by train, and if so book it.

{COMMON_RULES}

Read the traveler's request. For each intercity leg:
- If the request names a specific train number, or otherwise makes clear
  rail travel is wanted/required for that leg, query trains (respecting
  any stated seat-class constraint via the seatClassName filter) and pick
  one whose remaining-seats field confirms availability.
- If the request clearly wants a different mode (e.g. names a specific
  flight number, or says "by flight"/"by air") for a leg, do NOT book a
  train for that leg -- state plainly "No train needed for <leg>" instead.
- If the mode isn't specified for a leg, use judgement, or state that a
  train is not your recommendation for that leg and leave it to air travel.

{INTERCITY_LINE_SPEC}

When you are done deciding every leg, stop calling tools and write your
final note as plain text: one line per leg, either the formatted booking
line above or an explicit "No train needed for <leg>" statement. Do not
invent a whole itinerary -- only report on intercity rail travel."""


def _run_train_stage(task_description: str, wrapper: ToolWrapper, full_schema: list[dict]):
    schema = _filter_schema(full_schema, TRAIN_TOOLS)
    text, iters, exhausted, _messages = _run_tool_stage(
        TRAIN_SYSTEM_PROMPT, task_description, schema, wrapper
    )
    return text.strip() or "No train information produced.", iters, exhausted


# ================================================================
# Stage 3: Sightseeing Agent
# ================================================================

SIGHTSEEING_SYSTEM_PROMPT = f"""You are the sightseeing and logistics
specialist, one role in a team planning a trip. Two other specialists
already decided the trip's intercity flights/trains (their notes are given
to you below, verbatim) -- do not change or re-decide those legs, just
weave their exact lines into your day-by-day itinerary at the correct
time. A fourth specialist will later add up the budget -- your job is to
build the complete day-by-day content, not to total costs.

{COMMON_RULES}

Your responsibilities for this trip:
- Pick a hotel matching every stated constraint (star rating, brand,
  services, decoration-time, area, etc.) via query_hotel_info.
- For each day, build the full chronological activity list: the given
  intercity leg(s) at the right time, buffer time around them (deplaning/
  boarding, security, layovers), travel_city hops connecting every
  consecutive pair of locations (airport/station <-> hotel, hotel <->
  attraction, attraction <-> restaurant, etc.) via query_road_route_info,
  attraction visits via recommend_attractions/query_attraction_details
  (including every explicitly named must-visit attraction and any
  attraction-type requirement), and meals via recommend_restaurants/
  query_restaurant_details (including any named restaurant/tag/area
  requirement). Use search_location to get coordinates when a tool needs
  them and you don't already have them from a prior result.

{DAY_STRUCTURE_RULES}

Do NOT include a Budget Summary section -- another specialist adds that.
When you are done gathering information, stop calling tools and write out
the complete day-by-day body (Day 1 through the final day), with every
activity line's price included inline exactly as the format above
requires, wrapped in <itinerary></itinerary> tags. The <itinerary> tags
are mandatory -- another specialist parses your output looking for them."""

_ITINERARY_RE = re.compile(r"<itinerary>(.*?)</itinerary>", re.DOTALL | re.IGNORECASE)


def _extract_itinerary(text: str) -> str:
    if not text:
        return ""
    match = _ITINERARY_RE.search(text)
    return match.group(1).strip() if match else ""


# Bumped from 40 -- live evaluation on the 122B endpoint found every single
# no-plan failure (5/120 on a full run) had the identical signature:
# sightseeing_iters == 40 (hit the cap exactly), sightseeing_failed=True.
# Reasoning: the single agent gets 100 iterations for ALL of its work
# (flights+trains+hotels+attractions+restaurants+roads+composing the plan);
# Flight/Train here typically finish in 1-3 iterations each, so the single
# agent effectively has ~90+ of its 100 iterations available for the same
# scope of work Sightseeing alone is responsible for -- more than double
# the 40 it had. This wasn't a model-capability problem (it happened even
# at 122B) -- it was this cap being under-sized for complex multi-day,
# many-constraint trips.
MAX_SIGHTSEEING_ITERATIONS = 80


def _run_sightseeing_stage(
    task_description: str,
    flight_note: str,
    train_note: str,
    wrapper: ToolWrapper,
    full_schema: list[dict],
):
    """Returns (itinerary_body, iterations, budget_exhausted). ``itinerary_body``
    is "" if the stage never produced a real <itinerary> block even after a
    retry nudge -- callers must not feed that fallback text onward (an
    earlier design let the Accounting stage compute a budget from a
    Sightseeing stage's leftover reasoning prose when it ran out of
    iterations mid-tool-loop; it dutifully fabricated numbers instead of
    failing, which is worse than an honest empty result)."""
    schema = _filter_schema(full_schema, SIGHTSEEING_TOOLS)
    user_content = (
        f"Traveler's request:\n{task_description}\n\n"
        f"Flight specialist's note (do not change these legs):\n{flight_note}\n\n"
        f"Train specialist's note (do not change these legs):\n{train_note}\n"
    )
    text, iters, exhausted, messages = _run_tool_stage(
        SIGHTSEEING_SYSTEM_PROMPT, user_content, schema, wrapper,
        max_iterations=MAX_SIGHTSEEING_ITERATIONS,
    )
    itinerary = _extract_itinerary(text)
    if itinerary:
        return itinerary, iters, exhausted

    # One retry: force a text-only wrap-up call (no tools) with the full
    # accumulated context, explicitly asking for the missing tag.
    messages.append({
        "role": "user",
        "content": (
            "Stop gathering more data now. Output your complete day-by-day "
            "itinerary so far, wrapped in <itinerary></itinerary> tags."
        ),
    })
    response = call_llm(messages=messages)
    itinerary = _extract_itinerary(response.content or "")
    return itinerary, iters, (exhausted or not itinerary)


# ================================================================
# Stage 4: Accounting Agent (no tools)
# ================================================================

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


def _run_accounting_stage(task_description: str, sightseeing_body: str) -> str:
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
    text = _run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, user_content)
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
        text = _run_notool_stage(ACCOUNTING_SYSTEM_PROMPT, retry_note)
        summary = _extract_budget_summary(text) or text.strip()

    return f"{sightseeing_body.strip()}\n\n{summary.strip()}"


# ================================================================
# Entry point
# ================================================================

def run_task(task: Task) -> AgentOutput:
    wrapper = ToolWrapper()
    full_schema = wrapper.get_schema()

    flight_note, flight_iters, flight_exhausted = _run_flight_stage(
        task.description, wrapper, full_schema
    )
    train_note, train_iters, train_exhausted = _run_train_stage(
        task.description, wrapper, full_schema
    )
    sightseeing_body, sightseeing_iters, sightseeing_exhausted = _run_sightseeing_stage(
        task.description, flight_note, train_note, wrapper, full_schema
    )

    metadata = {
        "stage_iterations": {
            "flight": flight_iters,
            "train": train_iters,
            "sightseeing": sightseeing_iters,
        },
        "budget_exhausted": (
            flight_exhausted or train_exhausted or sightseeing_exhausted
        ),
    }

    if not sightseeing_body:
        # Sightseeing never produced a real <itinerary> block, even after
        # its retry nudge. Do not hand this to Accounting -- it has no
        # real itinerary to compute a budget from and would fabricate
        # numbers (confirmed live: it invented a plausible-looking budget
        # summary from a Sightseeing stage's leftover reasoning prose).
        # Mirrors the single agent's own "no lenient fallback" contract:
        # an honest empty result, not a fabricated one.
        metadata["sightseeing_failed"] = True
        return AgentOutput(result="", metadata=metadata)

    plan = _run_accounting_stage(task.description, sightseeing_body)
    return AgentOutput(result=plan, metadata=metadata)

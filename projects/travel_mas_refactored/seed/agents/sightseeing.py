"""Stage 3: Sightseeing Agent -- hotel/attractions/restaurants/logistics
specialist.

Receives the Flight and Train agents' notes verbatim (must not change
those legs), picks the hotel, and builds the full day-by-day activity
list. Does not compute the budget -- that is the Accounting agent's job.
"""
from __future__ import annotations

import re

from platform_core.llm_wrapper import call_llm

from agents.common import COMMON_RULES, filter_schema, run_tool_stage
from tool_wrapper import ToolWrapper

SIGHTSEEING_TOOLS = {
    "query_hotel_info",
    "recommend_attractions",
    "query_attraction_details",
    "recommend_restaurants",
    "query_restaurant_details",
    "query_road_route_info",
    "search_location",
}

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


def run_sightseeing_stage(
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
    schema = filter_schema(full_schema, SIGHTSEEING_TOOLS)
    user_content = (
        f"Traveler's request:\n{task_description}\n\n"
        f"Flight specialist's note (do not change these legs):\n{flight_note}\n\n"
        f"Train specialist's note (do not change these legs):\n{train_note}\n"
    )
    text, iters, exhausted, messages = run_tool_stage(
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

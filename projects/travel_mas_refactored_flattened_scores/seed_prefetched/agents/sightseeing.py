"""Stage 3: Sightseeing Agent -- hotel/attractions/restaurants/logistics
specialist.

PREFETCHED-CONTEXT VARIANT (option "b" of the tool-reliability experiment):
instead of one long tool-calling loop that both SELECTS venues and COMPUTES
every transfer time itself (the original design), this stage is split into
three phases:

  Phase 1 (LLM + tools, no query_road_route_info): the model only decides
  WHAT to include -- hotel, and an ordered per-day list of attraction/meal
  stops -- and reports each one's exact name/coordinates/price/notes as a
  small structured <stops> JSON block. It never has to compute or report a
  transfer time.

  Phase 2 (harness code, no LLM): given that ordered stop list plus the
  Flight/Train agents' notes, the harness itself walks every consecutive
  pair of anchors per day (arrival point -> hotel -> stop -> stop -> ... ->
  hotel/departure point) and calls query_road_route_info (and, for the
  arrival/departure station or airport, search_location) directly through
  the ToolWrapper -- bypassing the LLM entirely for this fact-lookup. This
  produces an authoritative "already looked up" transfer-time table.

  Phase 3 (LLM, no tools): a single no-tool call that is handed the Phase 1
  selections verbatim and the Phase 2 transfer-time table, and asked only
  to compose the final formatted day-by-day itinerary using those given
  facts -- it cannot skip a lookup it never had the option to make.

Rationale: the baseline's dominant failure mode (see reasonable_transfer_time
and validated_meals/validated_attractions in eval logs) was the model
skipping query_road_route_info / recommend_restaurants entirely and
guessing a plausible-looking number or name instead of calling the tool,
even though the prompt already told it not to. Removing the *option* to
skip the lookup (by having the harness do it unconditionally) tests whether
that is more reliable than instructing a 35B model harder (see
seed_promptmandate, the sibling "option a" experiment).
"""
from __future__ import annotations

import json
import re

from platform_core.llm_wrapper import call_llm

from agents.common import COMMON_RULES, filter_schema, run_tool_stage
from agents.immutable.message import AgentMessage, from_sender
from agents.llm_backbone import get_backbone_config
from platform_core.runner import Task
from tool_wrapper import ToolWrapper

# query_road_route_info deliberately excluded -- Phase 2 (harness code) is
# the only caller of that tool in this variant.
PHASE1_TOOLS = {
    "query_hotel_info",
    "recommend_attractions",
    "query_attraction_details",
    "recommend_restaurants",
    "query_restaurant_details",
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
  min/max visit-hour ranges (given to you below in each stop's notes).
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

PHASE1_SYSTEM_PROMPT = f"""You are the sightseeing and logistics
specialist, one role in a team planning a trip. Two other specialists
already decided the trip's intercity flights/trains (their notes are given
to you below, verbatim) -- do not change or re-decide those legs.

{COMMON_RULES}

In THIS step you only decide WHAT the traveler will do -- you do not write
the final itinerary and you do not compute any travel/transfer time between
locations (a separate step, with its own tools, handles that). Your job:

- Pick a hotel matching every stated constraint (star rating, brand,
  services, decoration-time, area, etc.) via query_hotel_info.
- For each day of the trip, pick an ordered list of stops (attractions and
  meals, in the sequence you intend to visit them) via
  recommend_attractions/query_attraction_details (including every
  explicitly named must-visit attraction and any attraction-type
  requirement) and recommend_restaurants/query_restaurant_details
  (including any named restaurant/tag/area requirement). A full
  sightseeing day needs at least 2 attractions; a transfer day needs at
  least 1 attraction if there is a meaningful arrival/departure window.
  Full sightseeing days need both lunch and dinner. Avoid repeating the
  same restaurant or attraction across days.
- Use search_location to get coordinates when a tool needs them and you
  don't already have them from a prior result.

When you are done, stop calling tools and output ONLY a single JSON block
wrapped in <stops></stops> tags (no other text inside the tags), in exactly
this shape:

<stops>
{{
  "hotel": {{"name": "<exact tool name>", "latitude": "<from tool result>", "longitude": "<from tool result>", "price": "¥<price>/room/night"}},
  "days": [
    {{
      "day": 1,
      "stops": [
        {{"type": "attraction", "name": "<exact tool name>", "latitude": "...", "longitude": "...", "price": "¥<price>/person", "notes": "<opening hours / min-max visit duration from the tool result, if any>"}},
        {{"type": "meal", "slot": "Lunch", "name": "<exact tool name>", "latitude": "...", "longitude": "...", "price": "¥<price>/person", "notes": "<service hours from the tool result, if any>"}}
      ]
    }}
  ]
}}
</stops>

Every "stops" list must be in the exact order the traveler will visit them
that day. Every name/latitude/longitude/price must be copied verbatim from
a tool result -- never estimate or invent one. Do not include hotel
check-in/check-out or any travel_city/travel_intercity_public lines in this
JSON -- those are added later from data you don't need to produce."""

_STOPS_RE = re.compile(r"<stops>(.*?)</stops>", re.DOTALL | re.IGNORECASE)
_CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_ITINERARY_RE = re.compile(r"<itinerary>(.*?)</itinerary>", re.DOTALL | re.IGNORECASE)
_INTERCITY_LINE_RE = re.compile(
    r"(\d{2}:\d{2})-(\d{2}:\d{2})\s*\|\s*travel_intercity_public\s*\|[^,]*,\s*([^-]+?)\s*-\s*([^,]+?)\s*,",
)

MAX_PHASE1_ITERATIONS = 60


def _extract_stops(text: str) -> dict | None:
    if not text:
        return None
    match = _STOPS_RE.search(text)
    if not match:
        return None
    raw = _CODEFENCE_RE.sub("", match.group(1).strip())
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_itinerary(text: str) -> str:
    if not text:
        return ""
    match = _ITINERARY_RE.search(text)
    return match.group(1).strip() if match else ""


def _intercity_legs(*notes: str) -> list[tuple[str, str, str]]:
    """Every (start_hhmm, departure_stop, arrival_stop) found across the
    Flight/Train agents' notes, sorted chronologically by start time. A
    round trip's outbound leg (whichever agent booked it) always sorts
    first, the return leg last -- true regardless of which specialist
    booked which direction."""
    legs: list[tuple[str, str, str]] = []
    for note in notes:
        for m in _INTERCITY_LINE_RE.finditer(note or ""):
            start, _end, dep_stop, arr_stop = m.groups()
            legs.append((start, dep_stop.strip(), arr_stop.strip()))
    legs.sort(key=lambda t: t[0])
    return legs


def _lookup_coords(wrapper: ToolWrapper, place_name: str) -> tuple[str, str] | None:
    try:
        raw = wrapper.execute("search_location", {"place_name": place_name})
        data = json.loads(raw)
        lat, lon = data.get("latitude"), data.get("longitude")
        if lat and lon:
            return str(lat), str(lon)
    except Exception:  # noqa: BLE001 - best-effort harness lookup, never fatal
        pass
    return None


def _lookup_duration_min(wrapper: ToolWrapper, origin: tuple[str, str], destination: tuple[str, str]) -> int | None:
    try:
        raw = wrapper.execute("query_road_route_info", {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
        })
        data = json.loads(raw)
        minutes = data.get("duration_in_minutes")
        if minutes is not None:
            return int(minutes)
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_transfer_tables(stops: dict, wrapper: ToolWrapper, flight_note: str, train_note: str) -> str:
    """Phase 2: the harness (not the LLM) resolves every consecutive-anchor
    transfer time per day, directly through the ToolWrapper, and renders it
    as an authoritative text table for Phase 3 to consume verbatim."""
    hotel = stops.get("hotel") or {}
    hotel_coords = (hotel.get("latitude"), hotel.get("longitude")) if hotel.get("latitude") else None
    hotel_name = hotel.get("name", "the hotel")

    legs = _intercity_legs(flight_note, train_note)
    arrival_stop_name = legs[0][2] if legs else None  # first leg's arrival stop
    departure_stop_name = legs[-1][1] if legs else None  # last leg's departure stop
    arrival_coords = _lookup_coords(wrapper, arrival_stop_name) if arrival_stop_name else None
    departure_coords = _lookup_coords(wrapper, departure_stop_name) if departure_stop_name else None

    days = stops.get("days") or []
    lines: list[str] = []
    for i, day in enumerate(days):
        day_no = day.get("day", i + 1)
        is_first_day = i == 0
        is_last_day = i == len(days) - 1

        anchors: list[tuple[str, tuple[str, str] | None]] = []
        if is_first_day and arrival_coords:
            anchors.append((arrival_stop_name, arrival_coords))
        if hotel_coords:
            anchors.append((hotel_name, hotel_coords))
        for stop in day.get("stops", []) or []:
            lat, lon = stop.get("latitude"), stop.get("longitude")
            name = stop.get("name", "stop")
            anchors.append((name, (lat, lon) if lat and lon else None))
        if is_last_day and departure_coords:
            anchors.append((departure_stop_name, departure_coords))
        elif hotel_coords:
            anchors.append((hotel_name, hotel_coords))

        day_lines = [f"Day {day_no} transfers (already looked up -- use exactly, do not recompute or guess):"]
        any_resolved = False
        for (name_a, coords_a), (name_b, coords_b) in zip(anchors, anchors[1:]):
            if not coords_a or not coords_b:
                day_lines.append(f"  {name_a} -> {name_b}: [not available, use your best judgement]")
                continue
            minutes = _lookup_duration_min(wrapper, coords_a, coords_b)
            if minutes is None:
                day_lines.append(f"  {name_a} -> {name_b}: [not available, use your best judgement]")
            else:
                day_lines.append(f"  {name_a} -> {name_b}: {minutes} min")
                any_resolved = True
        if any_resolved or len(day_lines) > 1:
            lines.append("\n".join(day_lines))
    return "\n\n".join(lines)


def _render_stops_for_prompt(stops: dict) -> str:
    hotel = stops.get("hotel") or {}
    parts = [f"Hotel: {hotel.get('name', '?')}, {hotel.get('price', '?')}"]
    for day in stops.get("days") or []:
        parts.append(f"Day {day.get('day')} stops, in visiting order:")
        for stop in day.get("stops", []) or []:
            if stop.get("type") == "meal":
                parts.append(f"  - Meal ({stop.get('slot', '?')}): {stop.get('name', '?')}, {stop.get('price', '?')}. Notes: {stop.get('notes', 'none')}")
            else:
                parts.append(f"  - Attraction: {stop.get('name', '?')}, {stop.get('price', '?')}. Notes: {stop.get('notes', 'none')}")
    return "\n".join(parts)


def run_sightseeing_stage(
    task: Task,
    inbox: list[AgentMessage],
    wrapper: ToolWrapper,
    full_schema: list[dict],
) -> AgentMessage:
    flight_note = from_sender(inbox, "flight").content
    train_note = from_sender(inbox, "train").content
    schema = filter_schema(full_schema, PHASE1_TOOLS)

    user_content = (
        f"Traveler's request:\n{task.description}\n\n"
        f"Flight specialist's note (do not change these legs):\n{flight_note}\n\n"
        f"Train specialist's note (do not change these legs):\n{train_note}\n"
    )
    text, iters, exhausted, messages = run_tool_stage(
        PHASE1_SYSTEM_PROMPT, user_content, schema, wrapper, "sightseeing",
        max_iterations=MAX_PHASE1_ITERATIONS,
    )
    stops = _extract_stops(text)
    if stops is None:
        messages.append({
            "role": "user",
            "content": (
                "Stop gathering more data now. Output ONLY the <stops>...</stops> "
                "JSON block described earlier, with no other text."
            ),
        })
        response = call_llm(messages=messages, **get_backbone_config("sightseeing"))
        stops = _extract_stops(response.content or "")

    if stops is None:
        return AgentMessage(
            sender="sightseeing", content="", ok=False, iterations=iters, budget_exhausted=exhausted,
            error="phase1 (stop selection) never produced a valid <stops> JSON block",
        )

    transfer_table = _build_transfer_tables(stops, wrapper, flight_note, train_note)

    phase3_prompt = f"""You are the sightseeing and logistics specialist on a
trip-planning team. A colleague has already chosen every stop (hotel,
attractions, meals) for this trip, and a routing system has already looked
up every transfer time between consecutive stops. Your ONLY job now is to
compose the final formatted day-by-day itinerary using exactly the facts
given below -- you have no tools in this step, and none are needed: every
name, price, and transfer time you could need is already provided.

{COMMON_RULES}

Traveler's request:
{task.description}

Flight specialist's note (insert these exact lines at the correct time):
{flight_note}

Train specialist's note (insert these exact lines at the correct time):
{train_note}

Chosen stops (already selected -- use these exact names/prices verbatim,
do not rename, substitute, or add any stop not listed here):
{_render_stops_for_prompt(stops)}

{transfer_table}

{DAY_STRUCTURE_RULES}

Do NOT include a Budget Summary section -- another specialist adds that.
Write out the complete day-by-day body (Day 1 through the final day), with
every activity line's price included inline exactly as the format above
requires, wrapped in <itinerary></itinerary> tags. The <itinerary> tags are
mandatory -- another specialist parses your output looking for them."""

    # Phase 3 is pure formatting over facts that are all already given --
    # but live sanity runs showed the model burns its ENTIRE output budget
    # as reasoning_tokens on an enumerated "Thinking Process" preamble and
    # never emits any visible content at all (confirmed via trace.jsonl:
    # output_tokens == reasoning_tokens == the cap, exactly, on both the
    # 16384 and a doubled 32768 budget) -- implicit thinking mode has no
    # cap, and doubling max_output_tokens just bought it more unproductive
    # thinking, not any actual answer. This is a runaway-reasoning problem,
    # not a budget-sizing one. Force reasoning_effort down for this call
    # only -- unlike Phase 1's tool loop (where the project's own config
    # comments document that forcing reasoning_effort="medium" causes a
    # catastrophic TOOL-CALLING breakdown on this model), Phase 3 has no
    # tools at all, so there is no tool-calling behavior to break; it is a
    # mechanical transcription task that should need minimal deliberation.
    # Keep max_output_tokens generously raised too, now that it should
    # mostly go to real content instead of runaway thinking.
    phase3_backbone = {
        **get_backbone_config("sightseeing"),
        "max_output_tokens": 32768,
        "reasoning_effort": "low",
    }

    response = call_llm(messages=[{"role": "user", "content": phase3_prompt}], **phase3_backbone)
    itinerary = _extract_itinerary(response.content or "")
    if itinerary:
        return AgentMessage(sender="sightseeing", content=itinerary, ok=True, iterations=iters, budget_exhausted=exhausted)

    # One retry: same nudge pattern as the original single-loop design.
    response2 = call_llm(messages=[
        {"role": "user", "content": phase3_prompt},
        {"role": "assistant", "content": response.content or ""},
        {"role": "user", "content": (
            "Output your complete day-by-day itinerary now, wrapped in "
            "<itinerary></itinerary> tags."
        )},
    ], **phase3_backbone)
    itinerary = _extract_itinerary(response2.content or "")
    if itinerary:
        return AgentMessage(sender="sightseeing", content=itinerary, ok=True, iterations=iters, budget_exhausted=exhausted)

    truncated = response2.stop_reason == "incomplete"
    return AgentMessage(
        sender="sightseeing",
        content="",
        ok=False,
        iterations=iters,
        budget_exhausted=exhausted,
        output_truncated=truncated,
        error=(
            "phase3 (composition) retry produced no <itinerary> tag "
            f"(stop_reason={response2.stop_reason!r}); raw response: "
            f"{(response2.content or '')[:500]!r}"
        ),
    )

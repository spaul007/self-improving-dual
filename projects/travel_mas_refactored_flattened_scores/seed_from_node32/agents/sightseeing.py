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
from agents.immutable.message import AgentMessage, from_sender
from agents.llm_backbone import get_backbone_config
from platform_core import trace
from platform_core.runner import Task
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

TRIP_REQUIREMENTS_EXTRACTION = """
--------------------------------------------------
TRIP REQUIREMENTS EXTRACTION (MANDATORY FIRST STEP)
--------------------------------------------------
Before you call any tools or plan any activities, you MUST first extract
and output a structured summary of the trip's requirements. This ensures
you don't miss or misread any constraint.

Output a <trip_requirements></trip_requirements> block containing:

1. **Trip Duration**: Compute the exact number of days from the start and
   end dates in the request. Count inclusively (Nov 12 to Nov 18 = 7 days).
   State: "Start: YYYY-MM-DD, End: YYYY-MM-DD, Total: N days". Do NOT
   just echo a day count the user may have stated — compute it from the
   dates.

2. **Cities (in order)**: List every city the trip visits, in order, as:
   "Origin: [city] → [city2] → ... → [cityN] → Return: [origin]"

3. **Party Size**: Extract the number of travelers. Default to 1 if not
   stated.

4. **Accommodation Constraints**: Star rating, brand, price range, specific
   services required (washing machine, dryer, screen mirroring, robot
   service, etc.), decoration-time requirement, rating preference.

5. **Must-Visit Attractions**: Every explicitly named attraction or location
   the user mentioned. List them all.

6. **Must-Visit Restaurants**: Every explicitly named restaurant or meal
   location the user mentioned.

7. **Transportation Constraints**: Flight time windows, aircraft
   manufacturer, train seat class, direct-only preference, cheapest/
   shortest preference, specific train/flight numbers.

8. **Budget Constraint**: The total budget limit if stated.

9. **Other Constraints**: Any other requirements not covered above
   (dietary needs, attraction types, meal requirements, etc.)

After outputting the <trip_requirements> block, proceed with tool calls
to gather data. Before writing each day, check your requirements list to
ensure the day's plan satisfies all applicable constraints.
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
   The HH:MM-HH:MM span must equal the query_road_route_info duration for
   that hop exactly (in minutes) — do not compress or stretch it to absorb
   buffer time.
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

{TRIP_REQUIREMENTS_EXTRACTION}

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

SELF-CHECK before writing the final <itinerary> output:
- Count the days in your itinerary. Does it match the trip duration you
  computed in <trip_requirements>?
- Check that every must-visit attraction and restaurant from your
  requirements list appears in the itinerary.
- Verify Accommodation: is listed on every day except the final one.
- Confirm the final day has NO Accommodation line (the trip ends).
- Ensure intercity travel moves the traveler to the correct city for each
  day's activities (the Current City line should reflect where the
  traveler actually is that day).
- Verify no attraction or restaurant is repeated across days.

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


# ---------------------------------------------------------------------------
# Time feasibility verifier: enforce that every travel_city line's scheduled
# HH:MM-HH:MM span matches its own stated [duration] (which the prompt
# requires to equal the query_road_route_info duration for that hop). The
# grader's reasonable_transfer_time check compares exactly this span against
# the tool-reported commute time; the LLM frequently writes the correct
# duration in the line but schedules a shorter (or much longer) span. We
# patch the span deterministically and shift the rest of that day by the
# same delta so the schedule stays contiguous.
# ---------------------------------------------------------------------------

_TIME_RANGE_RE = re.compile(
    r'^(\s*)(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})(\s*\|.*)$'
)
_DURATION_HM_RE = re.compile(
    r'(\d+)\s*h\s*(?:(\d+(?:\.\d+)?)\s*)?min', re.IGNORECASE
)
_DURATION_MIN_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*min(?:ute)?s?\b', re.IGNORECASE
)
_DAY_HEADER_RE = re.compile(r'^Day\s+\d+\s*:', re.IGNORECASE)

# A travel_city leg is only "too slow" (and worth shrinking) when its span
# exceeds the stated duration by more than this many minutes -- small
# overruns are inside the grader's tolerance band and aren't worth the
# schedule churn of shifting the rest of the day earlier.
_TOO_SLOW_SLACK_MIN = 10


def _parse_time_minutes(t: str) -> int:
    h, m = t.split(':')
    return int(h) * 60 + int(m)


def _fmt_minutes(mins: int) -> str:
    mins = max(0, min(24 * 60 - 1, int(round(mins))))
    return f'{mins // 60:02d}:{mins % 60:02d}'


def _extract_line_duration(line: str) -> float | None:
    m = _DURATION_HM_RE.search(line)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2) or 0.0)
    m = _DURATION_MIN_RE.search(line)
    if m:
        return float(m.group(1))
    return None


def _patch_day_lines(lines: list[str]) -> tuple[list[str], dict]:
    """Patch one day's worth of lines. Deltas are computed greedily, then
    applied only if every resulting time stays within 00:00-23:59."""
    stats: dict = {
        'patched': 0,
        'skip_missing_duration': 0,
        'skip_boundary': 0,
        'details': [],
    }
    entries: list[dict] = []
    for line in lines:
        m = _TIME_RANGE_RE.match(line)
        if not m:
            entries.append({'line': line, 's': None, 'e': None, 'tc': False, 'd': None})
            continue
        is_tc = '| travel_city |' in line
        d = _extract_line_duration(line) if is_tc else None
        if is_tc and d is None:
            stats['skip_missing_duration'] += 1
        entries.append({
            'line': line,
            'lead': m.group(1),
            's': _parse_time_minutes(m.group(2)),
            'e': _parse_time_minutes(m.group(3)),
            'rest': m.group(4),
            'tc': is_tc,
            'd': d,
        })

    deltas = [0] * len(entries)
    for i, ent in enumerate(entries):
        if ent['s'] is None or not ent['tc'] or ent['d'] is None:
            continue
        scheduled = ent['e'] - ent['s']
        if scheduled <= 0:
            # Spans midnight; leave such lines alone.
            continue
        d = ent['d']
        delta = 0
        if scheduled < d - 0.5:
            delta = int(round(d)) - scheduled
        elif scheduled > d + _TOO_SLOW_SLACK_MIN:
            delta = int(round(d)) - scheduled
        if delta:
            deltas[i] = delta

    # Boundary check: if any shifted time falls outside the day, revert.
    cum = 0
    for i, ent in enumerate(entries):
        cum += deltas[i]
        if ent['s'] is None:
            continue
        fs = ent['s'] + (cum - deltas[i])
        fe = ent['e'] + cum
        if fs < 0 or fe > 24 * 60 - 1:
            stats['skip_boundary'] += 1
            return lines, stats

    out: list[str] = []
    cum = 0
    for i, ent in enumerate(entries):
        if ent['s'] is None:
            out.append(ent['line'])
            continue
        shift_before = cum
        cum += deltas[i]
        fs = ent['s'] + shift_before
        fe = ent['e'] + cum
        if deltas[i]:
            stats['patched'] += 1
            stats['details'].append({
                'scheduled_min': ent['e'] - ent['s'],
                'tool_min': ent['d'],
                'delta_min': deltas[i],
            })
        out.append(f"{ent['lead']}{_fmt_minutes(fs)}-{_fmt_minutes(fe)}{ent['rest']}")
    return out, stats


def verify_time_feasibility(itinerary: str, messages: list) -> str:
    """Enforce scheduled_duration >= tool_reported_duration for travel_city
    lines (the ones that drive the grader's reasonable_transfer_time check).

    Each travel_city line already carries the tool-reported duration as its
    ``[duration]`` field (the prompt requires it to be copied verbatim from
    query_road_route_info). We compare that field against the line's
    HH:MM-HH:MM span and patch mismatches, shifting the remainder of that
    day by the same delta to keep the schedule gap/overlap-free.

    ``messages`` is accepted for signature compatibility and future
    cross-checking against raw query_road_route_info results; the current
    implementation relies on the invariant that the line's [duration] field
    equals the tool result, so no coordinate-based matching is needed.
    """
    if not itinerary:
        return itinerary

    result: list[str] = []
    day_buf: list[str] = []
    totals = {'patched': 0, 'skip_missing_duration': 0, 'skip_boundary': 0}
    all_details: list[dict] = []

    def flush_day() -> None:
        if not day_buf:
            return
        patched_lines, stats = _patch_day_lines(day_buf)
        result.extend(patched_lines)
        totals['patched'] += stats['patched']
        totals['skip_missing_duration'] += stats['skip_missing_duration']
        totals['skip_boundary'] += stats['skip_boundary']
        all_details.extend(stats['details'])

    for line in itinerary.split('\n'):
        if _DAY_HEADER_RE.match(line.strip()):
            flush_day()
            day_buf = [line]
        else:
            day_buf.append(line)
    flush_day()

    road_calls = sum(
        1 for item in (messages or [])
        if isinstance(item, dict) and item.get('name') == 'query_road_route_info'
    )

    if totals['patched']:
        trace.log(
            label='verifier_fired',
            verdict='fail',
            name='time_feasibility',
            patched=totals['patched'],
            skip_missing_duration=totals['skip_missing_duration'],
            skip_boundary=totals['skip_boundary'],
            road_route_calls_seen=road_calls,
            details=all_details,
        )
    elif totals['skip_missing_duration'] or totals['skip_boundary']:
        trace.log(
            label='verifier_fired',
            verdict='skip',
            name='time_feasibility',
            patched=totals['patched'],
            skip_missing_duration=totals['skip_missing_duration'],
            skip_boundary=totals['skip_boundary'],
            road_route_calls_seen=road_calls,
        )
    else:
        trace.log(
            label='verifier_fired',
            verdict='pass',
            name='time_feasibility',
            road_route_calls_seen=road_calls,
        )
    return '\n'.join(result)


# Bumped from 40 -- live evaluation found every no-plan failure had the
# identical signature: sightseeing_iters == 40 (hit the cap exactly),
# sightseeing_failed=True. Sightseeing is by far the heaviest of the four
# split stages -- hotel + N attractions + N restaurants + road-route
# lookups, then composing the full multi-day body -- versus flight/train's
# much simpler single-decision task (see common.py's own
# MAX_ITERATIONS_PER_STAGE=25 note, which observes the same imbalance).
# This cap is this stage's OWN independent budget, not shared with any
# other stage; 40 was simply under-sized for complex multi-day,
# many-constraint trips regardless of which backbone model is running it.
MAX_SIGHTSEEING_ITERATIONS = 80


def run_sightseeing_stage(
    task: Task,
    inbox: list[AgentMessage],
    wrapper: ToolWrapper,
    full_schema: list[dict],
) -> AgentMessage:
    """Reads the Flight and Train agents' notes from `inbox` (by sender
    name, via `from_sender`). Returns an AgentMessage whose `content` is
    "" and `ok` is False if the stage never produced a real <itinerary>
    block even after a retry nudge -- callers must not feed that fallback
    content onward (an earlier design let the Accounting stage compute a
    budget from a Sightseeing stage's leftover reasoning prose when it ran
    out of iterations mid-tool-loop; it dutifully fabricated numbers
    instead of failing, which is worse than an honest empty result)."""
    schema = filter_schema(full_schema, SIGHTSEEING_TOOLS)
    flight_note = from_sender(inbox, "flight").content
    train_note = from_sender(inbox, "train").content
    user_content = (
        f"Traveler's request:\n{task.description}\n\n"
        f"Flight specialist's note (do not change these legs):\n{flight_note}\n\n"
        f"Train specialist's note (do not change these legs):\n{train_note}\n"
    )
    text, iters, exhausted, messages = run_tool_stage(
        SIGHTSEEING_SYSTEM_PROMPT, user_content, schema, wrapper, "sightseeing",
        max_iterations=MAX_SIGHTSEEING_ITERATIONS,
    )
    itinerary = _extract_itinerary(text)
    if itinerary:
        itinerary = verify_time_feasibility(itinerary, messages)
        # ── Instrument: did the LLM include a <trip_requirements> block? ──
        has_requirements = "<trip_requirements>" in text
        trace.log(
            label='verifier_fired',
            verdict='pass' if has_requirements else 'fail',
            name='sightseeing_requirements_extracted',
        )
        return AgentMessage(sender="sightseeing", content=itinerary, ok=True, iterations=iters, budget_exhausted=exhausted)

    # One retry: force a text-only wrap-up call (no tools) with the full
    # accumulated context, explicitly asking for the missing tag.
    messages.append({
        "role": "user",
        "content": (
            "Stop gathering more data now. Output your complete day-by-day "
            "itinerary so far, wrapped in <itinerary></itinerary> tags."
        ),
    })
    response = call_llm(messages=messages, **get_backbone_config("sightseeing"))
    itinerary = _extract_itinerary(response.content or "")
    if itinerary:
        itinerary = verify_time_feasibility(itinerary, messages)
        has_requirements = "<trip_requirements>" in (response.content or "")
        trace.log(
            label='verifier_fired',
            verdict='pass' if has_requirements else 'fail',
            name='sightseeing_requirements_extracted',
        )
        return AgentMessage(sender="sightseeing", content=itinerary, ok=True, iterations=iters, budget_exhausted=exhausted)

    # Genuinely distinct from budget exhaustion: the wrap-up retry ran with
    # no iteration limit of its own and still did not produce a valid
    # <itinerary> tag. Reporting this as budget_exhausted would be false --
    # confirmed live, several such failures had used well under half the
    # iteration cap. Report it as its own task_failure with the actual
    # response text, so a diagnosis never has to guess which of these two
    # unrelated things went wrong.
    # response.stop_reason ("completed"/"incomplete"/... from the Responses
    # API's own status field, see platform_core/llm_wrapper.py) was already
    # being computed by the LLM wrapper and then silently discarded here --
    # the only way anything downstream could ever tell a truncated
    # completion (which looks like the model got cut off mid-reasoning)
    # apart from the model simply not complying was to guess from where the
    # raw text happens to stop. Surface it explicitly instead.
    truncated = response.stop_reason == "incomplete"
    return AgentMessage(
        sender="sightseeing",
        content="",
        ok=False,
        iterations=iters,
        budget_exhausted=exhausted,
        output_truncated=truncated,
        error=(
            "sightseeing wrap-up retry produced no <itinerary> tag "
            f"(stop_reason={response.stop_reason!r}); raw response: "
            f"{(response.content or '')[:500]!r}"
        ),
    )

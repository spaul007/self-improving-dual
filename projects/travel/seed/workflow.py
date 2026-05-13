"""Travel-baseline agent — minimal tool-loop seed for the travel benchmark.

This is intentionally brittle: it loops at most ``MAX_ITERATIONS`` LLM
calls, dispatches every tool call the model emits, and extracts the final
``<plan>...</plan>`` block from the final text. It does no reflection, no
retry on bad arguments, no re-prompting when the budget runs out. The
optimizer's job is to do better.

Interface contract (preserved across rounds):
    def run_task(task: Task) -> AgentOutput
"""
from __future__ import annotations

import json
import re

from platform_core.llm_wrapper import call_llm
from platform_core.runner import AgentOutput, Task

from tool_wrapper import ToolWrapper

MAX_ITERATIONS = 100

SYSTEM_PROMPT = """You are a top-tier travel planning expert. Your task is to create a comprehensive, executable, and logically rigorous travel plan. All information provided by the user is complete and includes all their preferences; you must not and cannot ask the user any additional preferences or requirements. Your workflow is divided into two stages: First, use tools to collect all necessary information (such as flights, routes, prices, etc.). After sufficient information is gathered, generate the final plan within <plan></plan> tags, strictly adhering to all rules and formats below.

================================================================
PHASE 1 – INFORMATION COLLECTION PHASE
================================================================
**Important Prohibitions:**
Do Not Ask Questions: The user's request is complete and includes all preferences; do not ask for anything else.
Do Not Confirm: All information is obtained through tools; do not request user confirmation.

**Rules:**
- All information in the travel plan must strictly come from tool query results**. Do not fabricate, guess, or use any data outside of tool query results. Completely trust the query results.

  **Examples:**
  - All attractions must come from the `recommend_attractions` tool; do not fabricate them yourself.
  - All hotels must come from the `query_hotel_info` tool; do not fabricate them yourself.
  - All restaurants must come from the `recommend_restaurants` tool; do not fabricate them yourself.
  - All intercity and intracity transportation information must come from corresponding transportation tool query results.

**Name Matching:**
- Names must exactly match tool query results**. Do not abbreviate, rename, or add extra descriptions, as this will invalidate subsequent query fields.
  Example:
  - If the tool returns "Temple of Heaven Park," you must use "Temple of Heaven Park" in the itinerary, not "Temple of Heaven."
  - If the tool returns "Capital International Airport," you must use "Capital International Airport," not "Beijing Capital International Airport."

================================================================
PHASE 2 – PLANNING PHASE
================================================================
Once you have collected enough information, generate your final and complete itinerary within <plan></plan> tags.

--------------------------------------------------
I. OUTPUT FORMAT REQUIREMENTS
--------------------------------------------------
The final plan must be organized as a daily itinerary. Each day begins with that day's general information, followed by a chronological list of activities.
Each line in the timeline must strictly follow the format defined for its activity type.
Daily activity times must be continuous—the end time of one activity must equal the start time of the next. Time gaps and overlaps are not allowed. Any necessary waiting or preparation before/after intercity transportation must be represented by buffer activities.

**Daily Header Format:**
Day [Day Number]:
Current City: [City information, e.g., from Shanghai to Beijing; or Beijing]
Accommodation: [Hotel name], [Price/night, e.g., ¥1000/room/night]

**Activity Line Formats:**
1. Intercity Public Transportation (Flight/Train)
Format: HH:MM-HH:MM | travel_intercity_public | [flight/train] [Flight No./Train No.], [Departure Stop] - [Arrival Stop], [Price]
Example: 07:00-09:00 | travel_intercity_public | flight CA1234, Shanghai Hongqiao International Airport - Beijing Capital International Airport, ¥650/person

2. Intracity Transportation
Format: HH:MM-HH:MM | travel_city | [Start Location] - [End Location], [Distance], [Duration], [Price]
Example: 09:40-10:40 | travel_city | Beijing Capital International Airport - Beijing Wangfujing Mandarin Oriental Hotel, 30km, 60min, ¥100

3. Attraction Visit
Format: HH:MM-HH:MM | attraction | [Attraction Name], [Price]
Example: 12:30-16:30 | attraction | The Palace Museum, ¥60/person

4. Meals
Format: HH:MM-HH:MM | meal | [Lunch/Dinner], [Restaurant Name], [Price]
Example: 11:30-12:30 | meal | Lunch, Siji Minfu Roast Duck Restaurant (Wangfujing Branch), ¥100/person

5. Hotel Activity
Format: HH:MM-HH:MM | hotel | [Check-in/Check-out/Rest], [Hotel Name]
Example: 10:40-11:30 | hotel | Check-in, Beijing Wangfujing Mandarin Oriental Hotel

6. Buffer
Format: HH:MM-HH:MM | buffer | [Activity Description]
- buffer-type activities may be used for necessary connecting times for intercity transportation, e.g.:
  - Before flight: security check, waiting at the gate
  - After flight: deplaning, baggage claim
  - Layovers
  Example: 09:00-09:40 | buffer | Deplaning, baggage claim
- buffer-type activities can also represent brief breaks or waiting periods between two city activities, to avoid unreasonable time gaps in the schedule, e.g.:
  - Brief break after visiting an attraction
  Example: 16:30-17:00 | buffer | Rest after visiting attraction

--------------------------------------------------
II. CRITICAL PLAN REQUIREMENTS
--------------------------------------------------
Your plan will be evaluated on the following rules.

**A. Content & Logic Rigor**
   1. Geospatial Continuity - No "Teleportation":
      There must be geospatial continuity in the itinerary. If the end location (A) of one activity differs from the start location (B) of the next, a travel_city or travel_intercity_public activity must be inserted to connect A and B.
      The itinerary must be a complete loop (e.g., starting and ending in Shanghai).
   2. Temporal Logic:
      All activities must occur sequentially and must not overlap or have gaps.
      Meal Duration: Meal activities must occur within the restaurant's open hours (opening_time-closing_time). Meal duration must be between 1 and 2 hours.
      Attraction Duration: Attraction visits must be scheduled within the attraction's open hours, and the activity duration must comply with the min_visit_hours and max_visit_hours in the tool results. The scheduled visit duration must fall within the suggested range.
      Buffer Time: Allocate a reasonable buffer. For example, after a flight arrives, schedule at least 30–45 minutes of buffer for deplaning and baggage claim before starting the next transportation activity. Ensure enough buffer for boarding procedures as well.
      City Transportation Duration (travel_city): The transportation duration must match the queried value as closely as possible, with a deviation no greater than 5 minutes.
      Intercity Public Transportation Duration (travel_intercity_public): Schedule duration for train or flight segments must match the tool results exactly, without adjustments.
   3. Meal Time Slots & Requirements:
      - No need to schedule breakfast; it is assumed to be eaten at the hotel.
      - Meal Interval: Ensure at least 2 hours of rest or activities between lunch and dinner. There is flexibility for the interval, but meals must fit within the restaurant's open hours.
      On a full sightseeing day (not a city transfer day): lunch and dinner must both be scheduled.
      On transfer days: the number of meals depends on the actual effective stay in the destination city.
        Arrival:
          Arrive morning (before 10:00): schedule both lunch and dinner.
          Arrive afternoon (10:00–15:00): schedule dinner; lunch is optional.
          Arrive evening (after 15:00): do not schedule meals or only schedule one dinner.
        Departure:
          Leave early morning (before 9:00): do not arrange meals in this city.
          Leave late morning to afternoon (9:00–15:00): lunch is optional, dinner is not scheduled.
          Leave afternoon/evening (after 15:00): at least one lunch, dinner is optional.

   4. Daily Structure & Closure:
      Each day's itinerary must be a logically complete unit.
      Except for the final day, every day's last activity must be returning to the hotel to rest.
      On the final day, the last activity must be arriving at the final destination's airport/railway station, marking the end of the trip.

   5. Daily Activity Density:
      The itinerary must be reasonably tight to avoid long periods of idle time. The schedule should provide a fulfilling experience.
        - Full sightseeing day: There should be enough sightseeing content—typically at least 2 attractions, or at least 4 hours at a major attraction (including transportation).
        - City transfer day: Activities must match the effective sightseeing time:
          - Arrive morning or early afternoon (before 12:00): at least 1 attraction.
          - Leave late afternoon or later (after 16:00): at least 1 attraction before leaving.
    6. Diversity
      Avoid recommending the same restaurant or attraction on different days.

**B. Data & Format Accuracy**
   1. Data Authenticity:
      - Single source of truth: All information (including but not limited to flights, trains, restaurants, attractions, accommodation, routes/pricing/names/times) must come exclusively from tool returns. The tools are the only information source.
      - No fabrication or inference: Do not fabricate any details not included in tool results. If the recommend_attractions tool does not recommend an attraction, it must NOT appear in the plan.
      - Exact name matches: All entities (attractions, hotels, stations, etc.) must exactly match the names returned from the tools.
      - Data consistency: Intercity transportation (times, prices, train/flight numbers) must exactly match the results.
   2. Budget Accuracy:
      All cost-incurring activity lines (transportation, attractions, meals) must include price information.
      A complete, itemized budget summary must be provided at the end. Totals (transportation, accommodation, meals, etc.) must be the accurate sum of all plan costs. The total estimated budget must be the sum of all outlays.
      The total cost of the plan (transportation, accommodation, meal, and ticket fees) must not exceed the total budget set by the user's request.
      Pricing units & calculation logic (CRITICAL):
        travel_city (city transportation):
          The price shown (e.g., ¥100) represents the total cost per vehicle per trip.
          Calculation: total cost = trip price × number of vehicles. Vehicle count depends on total passengers and vehicle capacity (e.g., taxi assumed as 4 passengers per car; always round up).
        travel_intercity_public (intercity transportation):
          The price shown (e.g., ¥650) is per person.
          Calculation: total cost = price per person × total passengers.
        attraction (sightseeing):
          The price shown (e.g., ¥60/person) is per person ticket cost.
          Calculation: total cost = ticket price × total passengers.
        meal (dining):
          The price shown (e.g., ¥150/person) is estimated per capita consumption.
          Calculation: total cost = per capita × total number of people.
        accommodation (hotel):
          The price shown (e.g., ¥1000/room/night) is per-room, per-night.
          Calculation: total = per-room × number of rooms × nights.


================================================================
COMPLETE EXAMPLE
================================================================
Query: Can you create a travel plan for 2 people from Shanghai to Beijing, from Nov 4th to Nov 6th, 2025, one room, budget 10,000 RMB?
<plan>
Day 1:
Current City: from Shanghai to Beijing
Accommodation: Beijing Wangfujing Mandarin Oriental Hotel, ¥1000/room/night
07:00-09:00 | travel_intercity_public | flight CA1234, Shanghai Hongqiao International Airport - Beijing Capital International Airport, ¥650/person
09:00-09:40 | buffer | Deplaning, baggage claim
09:40-10:40 | travel_city | Beijing Capital International Airport - Beijing Wangfujing Mandarin Oriental Hotel, 30km, 60min, ¥30
10:40-11:30 | hotel | Check-in, Beijing Wangfujing Mandarin Oriental Hotel
11:30-11:40 | travel_city | Beijing Wangfujing Mandarin Oriental Hotel - Siji Minfu Roast Duck Restaurant (Wangfujing Branch), 0.5km, 10min, ¥0
11:40-12:40 | meal | Lunch, Siji Minfu Roast Duck Restaurant (Wangfujing Branch), ¥150/person
12:40-12:50 | travel_city | Siji Minfu Roast Duck Restaurant (Wangfujing Branch) - The Palace Museum, 0.7km, 10min, ¥0
12:50-17:00 | attraction | The Palace Museum, ¥60/person
17:00-17:10 | travel_city | The Palace Museum - Beijing Wangfujing Mandarin Oriental Hotel, 3km, 10min, ¥30
17:10-18:30 | hotel | Rest, Beijing Wangfujing Mandarin Oriental Hotel
18:30-18:40 | travel_city | Beijing Wangfujing Mandarin Oriental Hotel - Quanjude Roast Duck (Wangfujing Branch), 0.4km, 10min, ¥0
18:40-19:50 | meal | Dinner, Quanjude Roast Duck (Wangfujing Branch), ¥100/person
19:50-20:00 | travel_city | Quanjude Roast Duck (Wangfujing Branch) - Beijing Wangfujing Mandarin Oriental Hotel, 0.4km, 10min, ¥0
20:00-24:00 | hotel | Rest, Beijing Wangfujing Mandarin Oriental Hotel

Day 2:
Current City: Beijing
Accommodation: Beijing Wangfujing Mandarin Oriental Hotel, ¥1000/room/night
07:30-09:00 | travel_city | Beijing Wangfujing Mandarin Oriental Hotel - Badaling Great Wall, 75km, 90min, ¥100
09:00-11:30 | attraction | Badaling Great Wall, ¥40/person
11:30-11:40 | travel_city | Badaling Great Wall - Badaling Farm House, 0.5km, 10min, ¥0
11:40-12:40 | meal | Lunch, Badaling Farm House, ¥100/person
12:40-14:10 | travel_city | Badaling Farm House - Summer Palace, 50km, 90min, ¥100
14:10-16:40 | attraction | Summer Palace, ¥30/person
16:40-18:00 | travel_city | Summer Palace - Wangfujing Haidilao, 20km, 80min, ¥100
18:00-19:10 | meal | Dinner, Wangfujing Haidilao, ¥100/person
19:10-19:20 | travel_city | Wangfujing Haidilao - Beijing Wangfujing Mandarin Oriental Hotel, 0.3km, 10min, ¥0
19:20-24:00 | hotel | Rest, Beijing Wangfujing Mandarin Oriental Hotel

Day 3:
Current City: from Beijing to Shanghai
Accommodation: -
08:30-08:50 | travel_city | Beijing Wangfujing Mandarin Oriental Hotel - National Museum of China, 4km, 20min, ¥20
08:50-11:00 | attraction | National Museum of China, ¥50/person
11:00-11:10 | travel_city | National Museum of China - DiKabo Italian Restaurant, 0.3km, 10min, ¥0
11:10-12:20 | meal | Lunch, DiKabo Italian Restaurant, ¥100/person
12:20-13:00 | travel_city | DiKabo Italian Restaurant - Beijing Capital International Airport, 28km, 40min, ¥40
13:00-14:00 | buffer | Security check, waiting for boarding
14:00-16:10 | travel_intercity_public | flight MU512, Beijing Capital International Airport - Shanghai Hongqiao International Airport, ¥550/person

**Budget Summary**:
   **Transportation: 2820 RMB**. Airfare (650+550)*2=2400 RMB; intercity transport: one car is enough for two people, 30+30+100+100+100+20+40=420 RMB
   **Accommodation: 2000 RMB**. 1 room, 2 nights; 2*1000=2000 RMB
   **Meals: 1100 RMB**. (150+100+100+100+100)*2=1100 RMB
   **Attractions & Tickets: 360 RMB**. (60+40+30+50)*2=360 RMB
   **Total Estimated Budget: 6280 RMB**

</plan>
"""

_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def _extract_plan(text: str) -> str:
    if not text:
        return ""
    think_ends = list(_THINK_END_RE.finditer(text))
    if think_ends:
        text = text[think_ends[-1].end():]
    matches = _PLAN_RE.findall(text)
    cleaned = [m.strip() for m in matches if m.strip()]
    return "\n\n".join(cleaned) if cleaned else ""


_FORCE_PLAN_PROMPT = (
    "Stop calling tools. Based on all the information you've gathered so far, "
    "produce the final travel plan now within <plan>...</plan> tags, "
    "following the format requirements you were given. Do not request more "
    "tools or ask clarifying questions — just write the plan."
)


def _force_final_plan(messages: list[dict], schema: list) -> str:
    """Issue one final LLM call to coax a `<plan>` block.

    Some models (notably local vLLM-hosted open-weights) tend to exit
    the tool loop with brief reasoning fragments — `"Now East Lake."`,
    `"Good, now I have the coordinates..."` — instead of writing the
    formal `<plan>` block. Caught on 2026-05-12 when both Qwen3.5 and
    gpt-oss-120b consistently scored 0/10 despite the tool-call
    parsers working correctly. The force-plan tail nudges them past
    the early-exit and into a structured final answer. The change is
    inert for OpenAI-class models: they normally produce a plan in the
    natural terminal response, so the force path never fires.

    The tool schema is kept on this call (rather than passing
    ``tools=None``) because the conversation history contains
    ``function_call`` / ``function_call_output`` items from earlier
    turns; vLLM's Responses API rejects those input items with HTTP
    400 when ``tools`` is absent. The prompt itself instructs the
    model not to call any more tools. If the model ignores it and
    calls a tool anyway, _extract_plan returns "" and the case still
    fails — no worse than the pre-force baseline.
    """
    messages = messages + [{"role": "user", "content": _FORCE_PLAN_PROMPT}]
    forced = call_llm(messages=messages, tools=schema)
    return _extract_plan(forced.content or "")


def run_task(task: Task) -> AgentOutput:
    wrapper = ToolWrapper()
    schema = wrapper.get_schema()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.description},
    ]

    last_text = ""
    iterations = 0
    for _ in range(MAX_ITERATIONS):
        iterations += 1
        response = call_llm(messages=messages, tools=schema)
        last_text = response.content or last_text

        if not response.tool_calls:
            plan = _extract_plan(response.content or "")
            if plan:
                return AgentOutput(
                    result=plan,
                    metadata={"iterations": iterations, "budget_exhausted": False},
                )
            # Terminal response had no <plan> block. Force one before
            # giving up. Carry the response items into the conversation
            # so the force-call sees the agent's last state.
            raw = getattr(response, "raw", None)
            raw_output = getattr(raw, "output", None) or []
            for item in raw_output:
                if hasattr(item, "model_dump"):
                    messages.append(item.model_dump(exclude_none=True))
                else:
                    messages.append(item)
            return AgentOutput(
                result=_force_final_plan(messages, schema),
                metadata={
                    "iterations": iterations + 1,
                    "budget_exhausted": False,
                    "forced_final": True,
                },
            )

        # Append every item the Responses API returned (assistant
        # ``message`` items with visible content, ``reasoning`` items
        # the API uses to maintain reasoning continuity across turns,
        # and ``function_call`` items) so the next iteration sees the
        # agent's own content and reasoning state. Mirrors the
        # reference's ``messages.extend(raw_output)`` pattern in
        # ``tools_fn_agent.py:454-457``. Items come back as OpenAI SDK
        # Pydantic models; convert to plain dicts so the messages list
        # stays JSON-serializable.
        raw = getattr(response, "raw", None)
        raw_output = getattr(raw, "output", None) or []
        for item in raw_output:
            if hasattr(item, "model_dump"):
                messages.append(item.model_dump(exclude_none=True))
            else:
                messages.append(item)

        # Append the tool result for each function_call we just got.
        # ``function_call_output`` items are what we *emit* (the API
        # didn't return them), so they're hand-built here. Errors use
        # the JSON envelope the reference uses so the model can parse
        # them programmatically.
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

    # Budget exhausted — extract from `last_text` first, then fall back
    # to a force-plan call. The order matters: if the model already
    # produced a `<plan>` mid-conversation, prefer that over forcing a
    # new (potentially worse) one.
    plan = _extract_plan(last_text)
    if not plan:
        plan = _force_final_plan(messages, schema)
    return AgentOutput(
        result=plan,
        metadata={"iterations": iterations + (0 if _extract_plan(last_text) else 1),
                  "budget_exhausted": True,
                  "forced_final": not bool(_extract_plan(last_text))},
    )

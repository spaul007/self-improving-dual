"""Stage 2: Train Agent -- rail-booking specialist.

Decides, per intercity leg, whether it should be booked by train, and if
so books it. Runs independently of the Flight agent (a trip may need one
mode, the other, both, or neither) -- no coordinator decides this in
advance; each agent makes its own relevance call from the request text.
"""
from __future__ import annotations

from agents.common import COMMON_RULES, INTERCITY_LINE_SPEC, filter_schema, run_tool_stage
from tool_wrapper import ToolWrapper

TRAIN_TOOLS = {"query_train_info"}

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


def run_train_stage(task_description: str, wrapper: ToolWrapper, full_schema: list[dict]):
    schema = filter_schema(full_schema, TRAIN_TOOLS)
    text, iters, exhausted, _messages = run_tool_stage(
        TRAIN_SYSTEM_PROMPT, task_description, schema, wrapper
    )
    return text.strip() or "No train information produced.", iters, exhausted

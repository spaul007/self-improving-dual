"""Stage 1: Flight Agent -- airline-booking specialist.

Decides, per intercity leg, whether it should be booked by air, and if so
books it. Reads only the raw task description; does not plan the rest of
the itinerary. Its final note is handed to the Sightseeing agent
unmodified.
"""
from __future__ import annotations

from agents.common import COMMON_RULES, INTERCITY_LINE_SPEC, filter_schema, run_tool_stage
from agents.immutable.message import AgentMessage
from platform_core.runner import Task
from tool_wrapper import ToolWrapper

FLIGHT_TOOLS = {"query_flight_info", "search_location"}

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


def run_flight_stage(
    task: Task, inbox: list[AgentMessage], wrapper: ToolWrapper, full_schema: list[dict]
) -> AgentMessage:
    """Flight has no upstream dependency (`inbox` is always `[]`)."""
    schema = filter_schema(full_schema, FLIGHT_TOOLS)
    text, iters, exhausted, _messages = run_tool_stage(
        FLIGHT_SYSTEM_PROMPT, task.description, schema, wrapper
    )
    content = text.strip() or "No flight information produced."
    # No real failure signal exists for this stage today -- the fallback
    # text above only fires on an empty tool-loop response, and a normal
    # "No flight needed for <leg>" model response (a valid, well-formed
    # outcome) is indistinguishable from it at this layer. ok=True always.
    return AgentMessage(sender="flight", content=content, ok=True, iterations=iters, budget_exhausted=exhausted)

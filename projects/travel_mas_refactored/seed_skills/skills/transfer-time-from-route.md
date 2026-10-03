# transfer-time-from-route

**When:** you are about to write a `travel_city` line, or schedule the next activity after moving between two places.

**Why:** a plan fails when the time between two activities does not match the queried travel time between them
(e.g. "query got commute time 26min, plan shows gap 39min").

**Steps**
1. Get both places' coordinates (`search_location`, or the coordinates returned by the attraction/restaurant/hotel tools).
2. Call `query_road_route_info` for that exact pair, in travel order.
3. Write the `travel_city` line with exactly the returned duration: its end time = start time + duration.
4. Start the next activity at the `travel_city` end time. If you need slack (e.g. waiting for opening time), make it an
   explicit `buffer` line with its reason, never a silently longer travel line.

**Check:** for every pair of consecutive lines on a day, the gap equals the queried duration or is an explicit buffer line.

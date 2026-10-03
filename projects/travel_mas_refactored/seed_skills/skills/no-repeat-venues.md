# no-repeat-venues

**When:** you are adding a restaurant or attraction to an itinerary longer than one day.

**Why:** repeating the same restaurant or attraction across the trip is marked as a lack of diversity.

**Steps**
1. Keep a running list of every restaurant and attraction already used on earlier days (and earlier on the same day).
2. Before adding a venue, check the list; if it is there, choose another option from the tool results
   (`recommend_restaurants` near the current location, `recommend_attractions`).
3. Venues the traveler explicitly asked for are placed once, as requested; do not repeat them either.

**Check:** before emitting the itinerary, scan all days: no restaurant or attraction name appears twice.

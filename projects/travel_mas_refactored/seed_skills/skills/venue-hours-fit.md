# venue-hours-fit

**When:** you are about to schedule a restaurant meal or an attraction visit.

**Why:** plans fail when a visit falls outside the venue's hours
(e.g. "Jinyuan Restaurant (00:10-01:10 not within 11:00-21:00)", "Museum (08:43-10:43 not within 09:00-16:30)").

**Steps**
1. Read `opening_time` / `closing_time` (and closing dates for attractions) from `query_restaurant_details`,
   `recommend_restaurants`, `query_attraction_details` or `recommend_attractions`.
2. The whole visit must fit: start >= opening time AND end <= closing time, on a day the venue is open.
3. If arrival would be early, insert a `buffer` line until opening; if the visit cannot end before closing, move it to
   another slot or day, or choose another venue.
4. Late arrivals: do not schedule a meal after the restaurant closes -- pick one open at that hour or skip the meal.

**Check:** for every meal and attraction line, compare its HH:MM range with the tool's hours before finalising the day.

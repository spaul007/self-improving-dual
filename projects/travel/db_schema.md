# Travel database schema

Per-sample database the task-agent tools query (read-only). One directory per
sample under `database_en/id_<sample_id>/`, resolved by the tools from
`TRAVEL_DATABASE_ROOT` + the sample id. Each category is a CSV table loaded by
the corresponding tool (`flight.py`, `train.py`, `hotel.py`, `restaurant.py`,
`attraction.py`, `location.py`, `roadroute.py`). Columns below are the CSV
headers; all values are strings as read (tools parse numbers as needed).

## flights/flights.csv
`origin_city, destination_city, dep_date, dep_station_code, dep_station_name,
arr_station_code, arr_station_name, dep_datetime, arr_datetime, duration,
flight_no, airline, seat_class, seat_status, equip_type, equip_size,
manufacturer, price, segment_index, route_index`
- Multi-segment routes share a `route_index`; `segment_index` orders legs.
- `seat_status` is a hard-constraint signal in scoring (availability).

## trains/trains.csv
`origin_city, destination_city, dep_date, dep_station_code, dep_station_name,
arr_station_code, arr_station_name, dep_datetime, arr_datetime, duration,
train_no, train_type, seat_class, seat_status, price, segment_index, route_index`
- Same route/segment scheme as flights. `train_type`/`seat_class` matter for
  seat-status hard constraints.

## hotels/hotels.csv
`city, name, address, latitude, longitude, decoration_time, hotel_star, price,
score, brand`
- `decoration_time` feeds "newest decoration" constraints; `hotel_star`,
  `brand`, `price` feed star/brand/budget constraints.

## restaurants/restaurants.csv
`restaurant_name, city, latitude, longitude, price_per_person, cuisine,
opening_time, closing_time, nearby_attraction_name, nearby_attraction_coords,
query_latitude, query_longitude, rating, tags`
- `opening_time`/`closing_time` feed business-hours checks; `tags`/`cuisine`
  feed "specific tag nearby" constraints; coords feed transfer-time checks.

## attractions/attractions.csv
`city, attraction_name, attraction_id, description, attraction_type, address,
latitude, longitude, rating, opening_time, closing_time, closing_dates,
min_visit_hours, max_visit_hours, ticket_price`
- `attraction_type` feeds "all of type" constraints; opening/closing feed
  business-hours + duration-rationality checks.

## locations/locations_coords.csv
`poi_name, latitude, longitude, address, poi_type`
- Coordinate lookup for any point of interest by name.

## transportation/distance_matrix.csv
`origin, destination, distance_meters, duration_minutes, cost`
- Pairwise commute time/cost between POIs; drives transfer-time feasibility and
  cost-calculation checks. Keyed by POI name (`origin`/`destination`).

## Notes for editing tools / workflow
- Tools return "database not loaded" sentinels when `TRAVEL_DATABASE_ROOT` is
  unset — never hardcode paths.
- The grader converts the agent's textual plan to JSON and checks route
  consistency, time feasibility, business hours, duration rationality, cost
  accuracy, activity diversity, plus hard constraints (seat status, hotel
  decoration/star, restaurant tags, attraction type). Align tool outputs with
  the fields those checks read (times, coords, prices, statuses, types).

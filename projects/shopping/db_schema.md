# Shopping database schema

Per-case database the task-agent tools query. One directory per case under
`database_level{1,2,3}/case_<sample_id>/`, resolved by the tools (`_db.py`)
from `SHOPPING_DATABASE_ROOT` + `SHOPPING_LEVEL` + `SHOPPING_SAMPLE_ID`.

## products.jsonl  (read-only product catalog; one JSON object per line)
- `product_id: str`, `name: str`, `price: float`, `brand: str`, `color: str`,
  `size: str`, `stock_quantity: int`
- `material_composition: list`, `thickness: str`, `elasticity: str`,
  `version_type: str`, `collar_type: str`, `suitable_season: str`,
  `target_demographic: str`, `details_craftsmanship: str`,
  `washing_instructions: str`
- `sales_volume: {monthly: int, total: int}`
- `rating: {average_score: float, total_reviews: int, distribution: {…star→pct}}`
- `review_summary: list`, `shipping_info: {origin: str, provider: str}`
- Filter/search tools query these fields (brand, color, size, price range,
  rating distribution, etc.). Requirement predicates in scoring reference these
  exact field paths (e.g. `rating.distribution.2_star`).

## user_info.json  (read-only)
- `user_id: str`, `username: str`, `phone_number: str`, `is_vip: bool`
- `demographics: dict`, `body_profile: dict`
- `coupons: dict` — coupons the user owns (name → details); coupon checks read
  these.
- `shipping_addresses: dict`

## cart.json  (MUTATED by the agent's tools — the deliverable)
- `user_id: str`, `user_name: str`
- `items: list` — products the agent added (add/delete product tools mutate)
- `used_coupons: list` — coupons the agent applied (add/delete coupon tools)
- `summary: dict` — totals
- Under concurrent evaluation the mutable cart lives in the per-eval scratch
  dir (`META_AGENT_SCRATCH_DIR`), not the shared read-only tree.

## Notes for editing tools / workflow
- Tools return "database not loaded" sentinels when the env vars are unset —
  never hardcode paths.
- The grader compares the cart's products against ground-truth requirements by
  feature category (brand / price / rating distribution / etc.) and checks
  coupons by name+quantity. `missing_products` / `missing_feature_categories`
  in feedback name the requirement predicates the cart failed to satisfy —
  align search/filter tool behavior with those field paths.

# Product-Scout Agent

**Role (frozen):** for ONE line item at a time, search the catalog and
return every product satisfying all of that item's constraints, so the
Cart-Optimizer can choose. Missing a qualifying cheap product, or
returning non-qualifying ones, directly costs accuracy.

**Tools:** the 9 read-only benchmark catalog tools — `search_products`
(BM25), `filter_by_brand/color/size/range/applicable_coupons`,
`sort_products`, `get_product_details`, `calculate_transport_time`.

**Workflow (`workflow.py`):** two tool-calling turns per line item.

1. **Search.** Chains filters over the whole catalog (a `filter_by_range`
   call without `product_ids` scans everything), then `get_product_details`
   on the survivors to check fields no filter covers (demographic, season,
   name fragment) and `calculate_transport_time` for delivery bounds. The
   prompt requires it to verify every candidate itself before answering.
2. **Verify.** An independent pass that re-checks the first turn's
   candidates field by field, hunts for products the search missed, and
   explicitly discards assumptions the search invented.

**Rules the prompt enforces**, each added in response to a measured failure:
- Never invent a constraint the item does not state — above all never
  assume a product *category* ("footwear"), which was the single largest
  source of lost products.
- The user's `standard_sizes` has separate tops/bottoms/shoes values; a
  product matching any of them is profile-eligible.
- Never return an empty candidate list: relax exactly one constraint in a
  fixed order (size → demographic → season → name fragment → loosest
  numeric bound) and search again, reporting what was relaxed.

**Python does not decide anything here** — it only checks that returned
ids exist in the catalog and caps the pool at `top_k_candidates` for
context size. The per-line-item scouts of one case run concurrently.

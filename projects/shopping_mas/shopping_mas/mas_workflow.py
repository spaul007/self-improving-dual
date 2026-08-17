"""Orchestrates ONE task (harness) + run_many for batches.  (E)

Control flow per case:
    requirement_parser                  (1 LLM call, profile fetched via tool)
    product_scout    x line items       (search + verify turns, run concurrently)
    cart_optimizer                      (optimize + audit turns, minimal retry)
    cart_executor                       (execute + self-verify)
    repair loop: products that failed at execution are excluded and
    optimize -> execute reruns, up to cfg.max_repair_iterations.

Every decision belongs to an agent. Python here is orchestration only:
fetching data through the benchmark tools, running independent scouts in
parallel, enforcing the per-case LLM budget, and recording the trace.
"""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents import base
from agents.cart_executor import workflow as executor
from agents.cart_optimizer import workflow as optimizer
from agents.product_scout import workflow as scout
from agents.requirement_parser import workflow as parser
from config import MASConfig
from llm_client import LLMBudgetExceeded, CallCounter, LLMClient
from tools import loader


def _fetch_user_info(toolset) -> dict:
    """Fetch the case's user profile through the benchmark's get_user_info
    tool — the same (and only) channel the original single agent has. The
    MAS input proper is just the query; this mirrors the mandatory first
    tool call the benchmark's system prompt demands of the baseline agent."""
    return json.loads(toolset["get_user_info"].call("{}"))


def _fetch_products(toolset) -> dict:
    """Enumerate the catalog exclusively through the benchmark tools:
    filter_by_range with no product_ids scans the entire catalog (every
    product has a price), and get_product_details returns full records."""
    ids = json.loads(toolset["filter_by_range"].call(json.dumps(
        {"condition_key": "price", "operator": ">", "value": -1})))["filtered_products_ids"]
    products = json.loads(toolset["get_product_details"].call(json.dumps(
        {"product_ids": ids})))["products"]
    return {p["product_id"]: p for p in products}


class MASWorkflow:
    def __init__(self, cfg: MASConfig | None = None):
        self.cfg = cfg or MASConfig()
        self.llm = LLMClient(self.cfg)

    # ---------------------------------------------------------------- one task
    def run_task(self, case_id: str, query: str, database_path: str | Path) -> dict:
        cfg = self.cfg
        t0 = time.time()
        database_path = Path(database_path)
        toolset = loader.make_toolset(database_path)
        products_map = _fetch_products(toolset)
        user_info = _fetch_user_info(toolset)

        trace = []          # benchmark-format message mirror -> messages.json
        agent_log = []      # structured per-agent outputs -> result record
        counter = CallCounter(cfg.max_llm_calls)
        state = base.State(query=query, user_info=user_info)
        status, issues, plan = "failed", [], None

        def record(agent, out):
            agent_log.append({"agent": agent, "output": out})

        try:
            # 1. Parse ------------------------------------------------------
            items, budget, raw = parser.run(self.llm, cfg.level, query, user_info,
                                            trace=trace, counter=counter)
            if not items:  # one retry on a failed/empty parse
                items, budget, raw = parser.run(self.llm, cfg.level, query, user_info,
                                                trace=trace, counter=counter)
            record("requirement_parser", raw)
            state.line_items, state.budget = items, budget

            # 2. Scout per line item ---------------------------------------
            # Line items are independent and the catalog tools are read-only,
            # so their scouts run concurrently; each gets its own trace slice
            # which is spliced back in item order for a deterministic
            # messages.json. The CallCounter is lock-guarded across threads.
            def scout_one(it):
                sub_trace = []
                try:
                    cands, note = scout.run(self.llm, cfg, it, state, toolset,
                                            products_map, trace=sub_trace,
                                            counter=counter)
                except LLMBudgetExceeded:
                    raise
                except Exception as e:           # one item must not kill the case
                    cands, note = [], f"scout crashed: {type(e).__name__}: {e}"
                return it.item_id, cands, note, sub_trace

            results_by_item = {}
            if items:
                with ThreadPoolExecutor(max_workers=min(len(items), cfg.scout_workers)) as ex:
                    futs = {ex.submit(scout_one, it): it for it in items}
                    budget_error = None
                    for fut in as_completed(futs):
                        try:
                            iid, cands, note, sub_trace = fut.result()
                        except LLMBudgetExceeded as e:
                            budget_error = e
                            continue
                        results_by_item[iid] = (cands, note, sub_trace)
                if budget_error is not None and not results_by_item:
                    raise budget_error
            for it in items:                      # deterministic ordering
                cands, note, sub_trace = results_by_item.get(
                    it.item_id, ([], "scout did not complete", []))
                trace.extend(sub_trace)
                state.candidates[it.item_id] = cands
                state.scout_notes[it.item_id] = note
                record("product_scout", {
                    "item_id": it.item_id,
                    "candidates": [p["product_id"] for p in cands],
                    "notes": note})

            # 3+4. Optimize -> execute, with repair loop -------------------
            exclude_ids = set()
            repair_notes = []
            for attempt in range(cfg.max_repair_iterations + 1):
                plan = optimizer.run(self.llm, cfg, items, state, products_map,
                                     repair_notes=repair_notes or None,
                                     trace=trace, counter=counter)
                if plan is None:
                    state.issues.append("optimizer: no usable plan")
                    break
                plan["quantities"] = {it.item_id: it.quantity for it in items}
                for iid in plan.get("skipped_items") or []:
                    state.issues.append(f"item {iid}: no candidates, left unfilled")
                record("cart_optimizer", {
                    "source": plan.get("source"),
                    "assignments": {i: p["product_id"]
                                    for i, p in plan["assignments"].items()},
                    "coupons": plan.get("coupons"),
                    "skipped_items": plan.get("skipped_items"),
                    "final_price": plan.get("final_price")})

                status, issues, raw_exec = executor.run(
                    self.llm, cfg, plan, state, toolset,
                    trace=trace, counter=counter)
                record("cart_executor", {"status": status, "issues": issues,
                                         "agent_output": raw_exec})
                if status == "ok" or attempt == cfg.max_repair_iterations:
                    break
                failed_pids = {i["id"] for i in issues if i.get("kind") == "product"}
                if not failed_pids:
                    break
                exclude_ids |= failed_pids
                repair_notes = [f"product {pid} could not be added (see issues); "
                                "choose a different product for that item"
                                for pid in sorted(exclude_ids)]

        except LLMBudgetExceeded as e:
            state.issues.append(str(e))
            record("budget_exhausted", {"error": str(e)})
        except Exception as e:  # a single broken case must not kill the batch
            state.issues.append(f"{type(e).__name__}: {e}")
            record("crash", {"error": str(e), "traceback": traceback.format_exc()})

        # The trace must end with a plain assistant message (benchmark
        # completion criterion).
        if not trace or trace[-1].get("role") != "assistant" or trace[-1].get("tool_calls"):
            trace.append({"role": "assistant",
                          "content": json.dumps({"status": status, "issues": issues},
                                                ensure_ascii=False)})
        with open(database_path / "messages.json", "w", encoding="utf-8") as f:
            json.dump({"step": counter.count, "description": "MAS pipeline trace",
                       "messages": trace}, f, ensure_ascii=False, indent=2)

        return {
            "id": case_id,
            "query": query,
            "level": cfg.level,
            "status": status,
            "issues": issues + state.issues,
            "plan": {
                "assignments": {i: p["product_id"]
                                for i, p in (plan or {}).get("assignments", {}).items()},
                "coupons": (plan or {}).get("coupons"),
                "base_total": (plan or {}).get("base_total"),
                "final_price": (plan or {}).get("final_price"),
                "source": (plan or {}).get("source"),
            } if plan else None,
            "budget": state.budget,
            "llm_calls": counter.count,
            "elapsed_seconds": round(time.time() - t0, 1),
            "agent_log": agent_log,
        }

    # ----------------------------------------------------------------- batches
    def run_many(self, samples: list[dict], database_root: str | Path) -> list[dict]:
        """samples: [{"id":..., "query":...}]; database_root holds case_{id}/
        dirs (already copied fresh by run_inference)."""
        results = []
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futures = {
                ex.submit(self.run_task, str(s["id"]), s["query"],
                          Path(database_root) / f"case_{s['id']}"): s
                for s in samples}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"id": str(s["id"]), "query": s["query"],
                         "status": "crashed", "issues": [str(e)]}
                results.append(r)
                print(f"[{r['status']:>6}] case {r['id']}  "
                      f"calls={r.get('llm_calls', '?')}  "
                      f"{r.get('elapsed_seconds', '?')}s", flush=True)
        results.sort(key=lambda r: int(r["id"]))
        return results

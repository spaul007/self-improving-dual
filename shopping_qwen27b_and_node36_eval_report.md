# Shopping Qwen-27B baseline + node 36 (`nocollab_node5`) re-eval

## Shopping Qwen-27B (thinking-on) baseline

- **Config:** `configs/eval_local_shopping_mas_refactored_qwen27b_thinking.yaml`
- **Repeat 1 log/output:** `shopping_mas_refactored_thinking_r1.log` → `runs/adhoc_eval/qwen27b_thinking_r1_full_benchmark/run_1/`
- **Repeats 2+3 log/output:** `shopping_mas_refactored_thinking_r2and3.log` → `runs/adhoc_eval/qwen27b_thinking_r2and3_full_benchmark/run_1/` (repeat 2), `run_2/` (repeat 3)
- **Auto-launch decision log:** `shopping_mas_refactored_thinking_autolaunch.log`

| Repeat | Mean score (n=120) |
|---|---|
| 1 | 0.9165 |
| 2 | 0.8724 |
| 3 | 0.8999 |
| **3-run average** | **0.8963** |

## Node 36 (`nocollab_node5`) full-120×3 independent eval

- **Node source:** `runs/20261007_011833_travel_mas_refactored_deepseek27b_fromseed_agentic_X100Y300_newtools_nocollab_node5/round_036/task_agent/`
- **Original HGM run config:** `configs/hgm_travel_deepseek27b_fromseed_agentic_X100Y300_newtools_nocollab_node5.yaml`
- **Original HGM run log:** `hgm_gated_full_X100Y300_newtools_nocollab_node5.log` (live process PID 4518, still running, round 47+)
- **Re-eval script:** `eval_node36_nocollab_full120_3x.py`
- **Re-eval log:** `eval_node36_nocollab_full120_3x.log`
- **Re-eval output:** `eval_node36_nocollab_full120_3x_out/` (`repeat_1/`, `repeat_2/`, `repeat_3/`, `summary.json`)

| Repeat | Composite | Case accuracy (full-pass) | no_plan_rate | conversion_error_rate | LLM-call fail% |
|---|---|---|---|---|---|
| 1 | 0.7188 | 6.7% | 2.5% | 0% | 0.29% |
| 2 | 0.7406 | 4.2% | 1.7% | 0% | 0.02% |
| 3 | 0.7078 | 4.2% | 2.5% | 0% | 0.87% |
| **Mean** | **0.7224** | **5.0%** | **2.2%** | **0%** | **0.40%** |

`conversion_error_rate` is a clean 0% across all 3 repeats — no real plan→JSON
conversion-infra failures. Case accuracy (full-pass rate) is notably low
(~5%) relative to the composite score, meaning most of the composite is
partial credit rather than outright wins.

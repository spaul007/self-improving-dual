To run hgm dual travel with qwen-122b model:

```
cd /groups/AIC-MV/sudipta.paul/code/rsi/self-improving-dual
conda activate hgm-dual
PYTHONPATH=. python3 main_loop.py --config configs/hgm_dual_travel_4000_qwen122b_node6.yaml
```

To run the agentic editor (bash + str_replace editor tool loop, bwrap-sandboxed, no docker):

```
cd /groups/AIC-MV/sudipta.paul/code/rsi/edit-memory-sep11-agentic/self-improving-dual
conda activate hgm-dual
source /groups/AIC-MV/sudipta.paul/code/random/api.sh   # editor = DeepSeek V4 Pro on OpenRouter (OpenRouter_API_KEY)
# smoke (96 evals, verbose prompts under round_*/verbose/, session logs under round_*/agentic/)
PYTHONPATH=. META_AGENT_VERBOSE=1 python3 main_loop.py --config configs/hgm_travel_smoke_agentic.yaml
# 1000-eval arm, identical to hgm_travel_1000_qwen122b_node5_editmem.yaml except editor.type
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_1000_qwen122b_node5_agentic_editmem.yaml
# tests
PYTHONPATH=. python3 -m unittest tests.test_agentic_policy tests.test_agentic_tools tests.test_agentic_editor
```

Sanity-check pair (2026-09-12): agentic editor + DeepSeek V4 Pro 0813 for meta AND task agent
(task reasoning "none"), eval_budget 100, 60-case train split, no top-k fill-up, meta-agent
trajectory saved under round_NNN/agentic/ (+ verbose/). Arms differ only in edit_memory.
Run directly on the node, one tmux session per arm:

```
cd /groups/AIC-MV/sudipta.paul/code/rsi/edit-memory-sep11-agentic/self-improving-dual
for arm in editmem no_editmem; do
  tmux new-session -d -s agentic_$arm -c "$PWD" "source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh \
    && conda activate hgm-dual && source /groups/AIC-MV/sudipta.paul/code/random/api.sh \
    && PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_100_dsv4pro_agentic_$arm.yaml 2>&1 | tee runs/console_agentic_$arm.log"
done
tmux attach -t agentic_editmem      # or: tail -f runs/console_agentic_editmem.log
```
Expected bill per arm ≈ $30 (≈160 task-agent cases × $0.15 + ~7 editor sessions × $0.3-0.5).

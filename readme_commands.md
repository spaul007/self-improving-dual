# Launch cheat-sheet

Agentic editor (bash + str_replace editor tool loop, bwrap-sandboxed, no docker),
HGM manager; the edit-memory layer is optional (`edit_memory:` block, see README). `editor.config.read_scope` picks what the
meta-agent may read: `"run"` (every node of the run) or `"parent"` (only the
parent node and its own workspace).

```
cd /groups/AIC-MV/sudipta.paul/code/rsi/edit-memory-sep14-agentic/self-improving-dual
conda activate hgm-dual
source /groups/AIC-MV/sudipta.paul/code/random/api.sh   # editor + task agent = DeepSeek V4 Pro on OpenRouter (OpenRouter_API_KEY)
# smoke (96 evals, task agent on node-5 Qwen; verbose prompts under round_*/verbose/, session logs under round_*/agentic/)
PYTHONPATH=. META_AGENT_VERBOSE=1 python3 main_loop.py --config configs/hgm_travel_smoke_agentic.yaml
# tiny component check (48 evals, everything on DeepSeek V4 Pro)
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_tiny_dsv4pro_agentic_no_editmem.yaml
# 1000-eval run (the 2026-09-13 setup; ~30-35 h, ~$300)
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_1000_dsv4pro_agentic_no_editmem.yaml
# same + the edit-memory layer (agentic curators every 4 expansions, bandit over with/without arms)
PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_1000_dsv4pro_agentic_editmem.yaml
# edit-memory smoke (window 2, instruction every 1): check edit_memory/ under the run dir
PYTHONPATH=. META_AGENT_VERBOSE=1 python3 main_loop.py --config configs/hgm_travel_smoke_agentic_editmem.yaml
# tests
PYTHONPATH=. python3 -m unittest discover -s tests
# dashboard (Run / Edit memory / Compare views; base python has streamlit, hgm-dual does not)
/users/sudipta.paul/miniconda3/bin/python3 -m streamlit run hgm_dashboard.py --server.port 8502 --server.address 0.0.0.0 --server.headless true
```

`seed_round_dir` in the 100/1000/tiny configs points at a finished run's
`round_000` (`runs/20260912_061539_hgm_travel_100_dsv4pro_agentic_no_editmem`):
symlink/copy it into `runs/` or drop the key to pay for a fresh 60-case seed
pre-eval (`..._no_editmem_t2.yaml` already does the latter).

Run in tmux on the node:

```
tmux new-session -d -s agentic -c "$PWD" "source /users/sudipta.paul/miniconda3/etc/profile.d/conda.sh \
  && conda activate hgm-dual && source /groups/AIC-MV/sudipta.paul/code/random/api.sh \
  && PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_1000_dsv4pro_agentic_no_editmem.yaml 2>&1 | tee runs/console_agentic.log"
tmux attach -t agentic      # or: tail -f runs/console_agentic.log
```

Full-benchmark re-evaluation of a finished run's LCB-selected node (2 repeats, parallelism 40):

```
scripts/scheduled_lcb_full_eval.sh runs/<experiment> configs/eval_travel_dsv4pro_agentic_node.yaml "2026-09-14 07:00" 2 40
```

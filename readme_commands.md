To run hgm dual travel with qwen-122b model:

```
cd /groups/AIC-MV/sudipta.paul/code/rsi/self-improving-dual
conda activate hgm-dual
PYTHONPATH=. python3 main_loop.py --config configs/hgm_dual_travel_4000_qwen122b_node6.yaml
```
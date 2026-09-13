#!/bin/bash
set -a
source /groups/AIC-MV/v.kulkarni1/.env
set +a
cd /groups/AIC-MV/v.kulkarni1/unified_framework/self-improving-dual
exec env PYTHONPATH=. python3 main_loop.py --config configs/hgm_travel_gemma_full_scale_block_tagged_X100Y180.yaml

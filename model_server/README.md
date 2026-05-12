# model_server

vLLM-backed OpenAI-compatible inference server for the three locally
hosted models the meta-agent can route to. Each model is loaded by a
separate long-running SLURM job; one model is hot at a time.

## Hosted models

| Config                                | HF repo                          | Active / total params | GPUs (TP) |
|---------------------------------------|----------------------------------|-----------------------|-----------|
| `configs/gpt_oss_120b.yaml`           | `openai/gpt-oss-120b`            | 5.1B / 117B (MXFP4)   | 1× H100   |
| `configs/qwen3_5_122b_a10b.yaml`      | `Qwen/Qwen3.5-122B-A10B`         | 10B / 122B            | 8× H100   |
| `configs/qwen3_5_397b_a17b.yaml`      | `Qwen/Qwen3.5-397B-A17B`         | 17B / 397B (FP8)      | 8× H100   |

## What lives where

- Model weights cached at `/groups/AIC-MV/n.tzou/hf_cache/` (`HF_HOME`).
- Server venv(s) at `/groups/AIC-MV/n.tzou/vllm_venv*` (one mainline +
  one pre-release for `gpt-oss`).
- Endpoint discovery file at
  `/groups/AIC-MV/n.tzou/model_server/endpoint.json`. Written by
  `launch.sh` when the server starts and deleted on EXIT/TERM.
- SLURM stdout/stderr at `/groups/AIC-MV/n.tzou/server_logs/<jobid>.{out,err}`.

`vllm` is **not** added to the meta-agent's `requirements.txt`. It is
only installed inside the server venv and is never imported from the
meta-agent (parent) process.

## Lifecycle

### Start

```
bash model_server/slurm_submit.sh model_server/configs/gpt_oss_120b.yaml
```

This wraps the repo-wide `slurm/submit.sh`, which writes the workload
to a temp script and submits via `sbatch` under `srun` so the
job-step cgroup tears down every descendant on job end. The SLURM
job runs `model_server/launch.sh` against the given config; that
script

  1. activates / creates the venv specified by `venv:` in the YAML,
  2. `pip install`s the packages listed in `pip_install_args:` if
     it had to create the venv,
  3. pre-downloads the HF repo into `$HF_HOME` (so the first request
     doesn't pay the download cost),
  4. writes the endpoint discovery JSON, installs the cleanup trap,
  5. `exec`s `vllm serve …`.

### Check

```
python3 model_server/health.py             # full status + /v1/models
python3 model_server/health.py --print-base-url
# → http://<node>:<port>/v1   (or empty + exit 2 if dead/stale)
```

`health.py` reads the discovery file and shells out to `squeue` to
detect a stale file from a job that has finished but didn't run its
trap.

### Stop

```
bash model_server/stop.sh
```

Reads the discovery file's `slurm_job_id`, runs `scancel`, then
verifies the file is gone (force-removes after 10s if the in-job
trap didn't fire).

Plain `scancel <jobid>` also works — the EXIT/TERM trap inside
`launch.sh` removes the discovery file either way.

## Pointing the meta-agent at the server

The meta-agent reads `LLM_BASE_URL` (env) or `base_url:` (YAML field
on `LLMSpec`) per call site. Two equivalent flows:

### env vars

```
export LLM_BASE_URL=$(python3 model_server/health.py --print-base-url)
export OPENAI_API_KEY=EMPTY   # any string; vLLM ignores auth
PYTHONPATH=. python3 main_loop.py --config configs/travel_local.yaml
```

### YAML (per-call-site)

Add `base_url: "http://<node>:<port>/v1"` to any of:

- `task_agent.base_url` — the task-agent's `call_llm` inside the
  evaluator subprocess.
- `editor.config.base_url` — the agent editor's LLM call.
- `manager.config.strategy_base_url` — the hill-climbing strategy
  proposer.

Each call site can target a different server (or stay on OpenAI by
leaving the field unset).

## Common failures

- **`/v1/responses` returns a 400 about an unknown field.** vLLM's
  Responses-API support is newer than its Chat-Completions support
  and may not cover every field the wrapper sends (`instructions`,
  `reasoning.effort`, Responses-shape `tools`). Capture the request
  body from the trace event and the vLLM access log; either drop
  the offending field in the YAML or upgrade vLLM.
- **OOM during model load.** The 397B config defaults to FP8
  quantization to fit a single 8× H100 node. If FP8 still OOMs,
  fall back to AWQ/INT4 or split across two nodes
  (`--pipeline-parallel-size 2`).
- **`launch.sh` blocks at the `huggingface-cli download` step.**
  First run downloads ~250GB+ — that takes a while. Subsequent
  runs hit the cache. If the cache disk fills up, point `HF_HOME`
  at a larger volume via the env var.

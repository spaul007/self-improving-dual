# model_server

A standalone vLLM-backed OpenAI-compatible inference server. The
meta-agent in this repo is one client; **any application that speaks
the OpenAI API can call this server with no meta-agent code on the
classpath**. Each model is loaded by its own long-running SLURM job;
one model is hot at a time.

The server speaks two OpenAI-compatible endpoints:

  - `/v1/responses` — used by the meta-agent.
  - `/v1/chat/completions` — vLLM's mature, well-tested path; what
    most third-party clients reach for.

Both return standard OpenAI-shaped JSON, so the official Python
`openai` SDK, `curl`, `langchain`, `litellm`, etc. all work
unchanged — you just point them at the server's base URL.

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

### Pre-download (optional but recommended for slow shares)

The first `launch.sh` against a fresh model pays the HF download cost
inline before `vllm serve` runs — for the 122B and 397B models that
means a GPU node sits idle for minutes/hours while ~250 GB / ~800 GB
streams in. Skip that by running a CPU-only pre-download first:

```
bash model_server/predownload.sh model_server/configs/gpt_oss_120b.yaml
```

This submits a CPU sbatch that activates the model's `venv:`,
installs the HF CLI if missing, and runs `huggingface-cli download
<repo>` into `$HF_HOME`. Idempotent — re-running against a fully
cached model is a no-op. Logs at
`$SLURM_LOG_DIR/predl_<jobid>.{out,err}` (default
`/groups/AIC-MV/n.tzou/server_logs/`).

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
python3 model_server/health.py                          # full status + /v1/models
python3 model_server/health.py --print-base-url         # base URL only
python3 model_server/health.py --name gpt_oss_120b      # per-model lookup
python3 model_server/health.py --name gpt_oss_120b --print-base-url
```

`health.py` reads the discovery file, shells out to `squeue` to
detect a stale file from a job that has finished but didn't run its
trap, **and** HTTP-probes `/v1/models` with a 5 s timeout — vLLM
takes minutes to initialize after `launch.sh` writes the discovery
file, so a SLURM-only check would falsely report alive during that
window. Connection refused / timeout / DNS error → reported stale;
any HTTP response (including 4xx) → reported alive.

### Stop

```
bash model_server/stop.sh                          # single-slot endpoint.json
bash model_server/stop.sh --name gpt_oss_120b      # per-model lookup
```

Reads the discovery file's `slurm_job_id`, runs `scancel`, then
verifies the file is gone (force-removes after 10s if the in-job
trap didn't fire). Plain `scancel <jobid>` also works — the EXIT/TERM
trap inside `launch.sh` removes the discovery file either way.

### Concurrent servers

Each model YAML has a `discovery_name:` field (defaults to the config
basename). `launch.sh` writes to
`<server_root>/endpoint_<discovery_name>.json`, so multiple servers
can coexist without clobbering each other's files. The shipped configs
use:

  - `gpt_oss_120b`
  - `qwen3_5_122b_a10b`
  - `qwen3_5_397b_a17b`

When running just one server, set nothing — `health.py` / `stop.sh`
both fall back to the single-slot `endpoint.json` path the older
documentation references.

## Calling the server from external applications

The server is intentionally vanilla vLLM — there is no
meta-agent-specific protocol, auth, or wrapper. Anything that speaks
OpenAI-compatible HTTP works.

### Minimum from any Python codebase

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<server-node>:8000/v1",
    api_key="EMPTY",   # vLLM ignores auth; SDK constructor needs a string
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-122B-A10B",
    messages=[{"role": "user", "content": "Plan a 2-day trip to Kyoto"}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

The Responses API works the same way (`client.responses.create(...)`)
when you want the endpoint the meta-agent uses.

### Worked examples in this repo

  - `examples/simple_client.py` — drop-in Python script. Auto-discovers
    the server via the endpoint file (or honours `--base-url` /
    `LLM_BASE_URL`), picks the right model id, and calls either
    `/v1/responses` or `/v1/chat/completions`. Requires only
    `pip install openai`.
  - `examples/curl_example.sh` — same flow with nothing but
    `bash` + `curl` + `python3` (the python is only there to format
    JSON). Useful for shell integrations or non-Python apps.

Run them:

```
# Python — uses the discovery file when available
python3 model_server/examples/simple_client.py --prompt "hello"

# Python — explicit URL, doesn't need the shared filesystem
python3 model_server/examples/simple_client.py \
    --base-url http://node-7:8000/v1 --prompt "hello"

# Shell + curl
bash model_server/examples/curl_example.sh "hello"
BASE_URL=http://node-7:8000/v1 bash model_server/examples/curl_example.sh "hello"
```

### Discovering the URL from any client

Two equally-supported paths:

  - **Shared filesystem.** Clients on the cluster (any user with read
    access to `/groups/AIC-MV/n.tzou/model_server/`) can read
    `endpoint.json` directly, or shell out to `health.py` for the
    stale-aware version. Schema:
    ```json
    {"host": "...", "port": 8000, "model": "...",
     "slurm_job_id": "...", "started_at": "...", "pid": ...}
    ```
  - **Out-of-band.** Anything running off-cluster doesn't see that
    file. Grab the URL once (`python3 model_server/health.py
    --print-base-url`) and hand it to the client however your stack
    normally configures endpoints (env var, secret, config file).

### What clients should NOT assume

  - **The URL is stable.** SLURM picks a node per submit; restarting
    the server lands on a different host. Re-read the discovery file
    (or re-poll `health.py`) on reconnect.
  - **Auth.** There is none. Don't expose this server beyond a trusted
    network; treat the URL as a credential.
  - **Tool calling.** vLLM's tool-use support varies by model. Check
    the per-model card in `configs/` before assuming function-calling
    works.

## Pointing the meta-agent at the server

### One-shot helper: `submit_eval.sh`

The cleanest path for running a meta-agent experiment against a local
server is the `model_server/submit_eval.sh` wrapper:

```
# Hit a local server (LLM_BASE_URL resolved from per-model discovery):
bash model_server/submit_eval.sh \
  configs/eval_local_gpt_oss_120b.yaml gpt_oss_120b

# Hit OpenAI (no discovery name → LLM_BASE_URL stays unset):
bash model_server/submit_eval.sh configs/eval_openai_medium.yaml
```

It calls `health.py --name <name> --print-base-url`, exports
`LLM_BASE_URL` into the SLURM job's env, and defers to
`slurm/submit.sh` for the sbatch wrapping. Logs go to
`$SLURM_LOG_DIR/<jobid>.{out,err}` (default
`/groups/AIC-MV/n.tzou/server_logs/`).

Three eval configs ship in this repo (`configs/eval_local_gpt_oss_120b.yaml`,
`configs/eval_local_qwen3_5_122b_a10b.yaml`,
`configs/eval_openai_medium.yaml`) — all 10-case smokes with
`runs_root: /groups/AIC-MV/n.tzou/evaluations`.

### Manual env + YAML wiring

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

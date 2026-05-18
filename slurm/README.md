# SLURM execution wrapper

The meta-agent loop spawns Python subprocesses (one per evaluation case).
If those subprocesses ran on the login node, an interrupted SSH session
could leave them as orphans. To prevent that, run everything through
SLURM: `srun` puts the workload into a job-step cgroup and the
controller kills every descendant process when the job ends — for any
reason, including session loss, `scancel`, OOM, or wall-time.

## Files

- `submit.sh` — generic wrapper. Submits any command as a SLURM job.
- `smoke.sh` — runs `tests.test_smoke` (or any unittest suite you pass).
- `run.sh` — runs `python3 main_loop.py --config <config>`. Sources the
  OpenAI key from `/users/n.tzou/api.sh` inside the job.
- `run_hgm.sh` — convenience wrapper over `run.sh` for Huxley-Gödel-Machine
  runs: `run_hgm.sh <travel|shopping|math>` picks `configs/hgm_<name>.yaml`
  and pre-sets resource defaults sized for an HGM run (14 h wall time).

## Defaults

| Knob              | Default                  | Override env var   |
|-------------------|--------------------------|--------------------|
| Partition         | `cpu-prepro-queue-02`    | `SLURM_PARTITION`  |
| Time              | 4 h (smoke: 30 min)      | `SLURM_TIME`       |
| CPUs per task     | 4 (smoke: 2)             | `SLURM_CPUS`       |
| Memory            | 16 G (smoke: 4 G)        | `SLURM_MEM`        |
| Job name          | `meta-agent`             | `SLURM_JOB_NAME`   |
| Log directory     | `$REPO_ROOT/runs/slurm`  | `SLURM_LOG_DIR`    |

The CPU partition is the right default in principle — every meta-agent
component is CPU-bound (the heavy work is API calls to OpenAI), so
allocating a GPU just to run Python wastes quota.

**Output location.** SLURM job logs default to `$REPO_ROOT/runs/slurm`
and experiment run folders to `$REPO_ROOT/runs` (the `runs_root` config
field). On this cluster the local `/users` filesystem is small, so for
real runs redirect both to the group filesystem — set `SLURM_LOG_DIR`
for the job logs and `runs_root:` in the YAML for the run folders:

```bash
SLURM_LOG_DIR=/groups/AIC-MV/n.tzou/meta-agent/slurm \
  slurm/run.sh configs/hgm_travel.yaml      # + runs_root: in the YAML
```

In practice on this cluster `cpu-prepro-queue-02` is frequently in
`DOWN` / `DRAINED` state. The known-good fallback is `gpu-aic-mv-01`
(or `gpu-aic-mv-02`) with `SLURM_GRES=none` so the scheduler skips
GPU allocation:

```bash
SLURM_PARTITION=gpu-aic-mv-01 SLURM_GRES=none \
  SLURM_TIME=04:00:00 SLURM_CPUS=16 SLURM_MEM=32G \
  slurm/run.sh configs/travel.yaml
```

Check current partition health with `sinfo -p <partition>` before
submitting. Earlier sessions also used `gpu-aisystem-queue` with the
same `SLURM_GRES=none` trick.

## Common usage

```bash
# Static smoke (no API key needed).
slurm/smoke.sh

# Travel smoke once test_travel_smoke is added.
slurm/smoke.sh tests.test_travel_smoke

# Full evolution run (math benchmark).
slurm/run.sh configs/default.yaml

# Travel benchmark (uses train/eval split from the YAML — train drives
# optimization, eval is a sidecar score per round).
slurm/run.sh configs/travel.yaml

# Bigger memory and longer time.
SLURM_MEM=32G SLURM_TIME=12:00:00 slurm/run.sh configs/travel.yaml

# HGM (Huxley-Gödel-Machine) optimization run — the wrapper applies
# HGM-sized SLURM defaults (gpu-aic-mv-01, GRES=none, 16 CPU, 32 G, 14 h).
slurm/run_hgm.sh travel
slurm/run_hgm.sh shopping
# equivalent to:
SLURM_PARTITION=gpu-aic-mv-01 SLURM_GRES=none SLURM_TIME=14:00:00 \
  SLURM_CPUS=16 SLURM_MEM=32G slurm/run.sh configs/hgm_travel.yaml
```

## Tailing job logs

`sbatch` prints `Submitted batch job 12345`. Then:

```bash
tail -f runs/slurm/12345.out      # stdout (or $SLURM_LOG_DIR if overridden)
tail -f runs/slurm/12345.err      # stderr
squeue -u "$USER"                 # job state
sacct -j 12345 --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Cancelling cleanly

```bash
scancel 12345
```

This propagates to every process the job started — both the parent
`main_loop.py` and every per-case Python subprocess the
`SubprocessEvaluator` forked. After cancellation, `pgrep -f main_loop.py`
on the compute node should return nothing.

## Troubleshooting

- **Job pending forever**: the partition may be full. Check
  `sinfo -p $SLURM_PARTITION`.
- **Job exits 0 immediately with no output**: usually a quoting bug in
  the wrapped command. The job log shows the exact command line.
- **Job leaves processes alive**: should not happen because we run under
  `srun`, but if it does, check that the workload is not detaching
  (e.g. `nohup &` or `setsid`). Detaching escapes the cgroup.

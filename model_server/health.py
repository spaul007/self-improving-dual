"""Operator helper for the model server.

Reads ``/groups/AIC-MV/n.tzou/model_server/endpoint.json`` and tells
you whether the server is actually alive (the file alone isn't proof:
SLURM jobs that get OOM-killed or expire on wall-time may leave a
stale file if the in-job EXIT trap couldn't run). Stale detection is
a ``squeue`` lookup on ``slurm_job_id``.

Two modes:

    python3 model_server/health.py
        Print the full endpoint JSON plus the SLURM job state. Exits 0
        if the server is alive, 2 if the file is missing or the job is
        no longer running.

    python3 model_server/health.py --print-base-url
        Print just the OpenAI-compatible base URL on stdout
        (``http://<host>:<port>/v1``). Suitable for shell substitution:

            export LLM_BASE_URL=$(python3 model_server/health.py --print-base-url)

        Exits 2 (with empty stdout) if the server is not alive.

Importable too — :func:`resolve_endpoint` returns ``Optional[str]``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_SERVER_ROOT = Path(
    os.environ.get(
        "MODEL_SERVER_ROOT",
        "/groups/AIC-MV/n.tzou/model_server",
    )
)
DEFAULT_DISCOVERY_PATH = Path(
    os.environ.get(
        "MODEL_SERVER_DISCOVERY_PATH",
        str(DEFAULT_SERVER_ROOT / "endpoint.json"),
    )
)


def discovery_path_for(name: Optional[str]) -> Path:
    """Resolve the discovery-file path for a server name.

    When ``name`` is given, returns ``<server_root>/endpoint_<name>.json``
    (matching what ``launch.sh`` writes). Falls back to the
    single-slot ``endpoint.json`` (or ``MODEL_SERVER_DISCOVERY_PATH``)
    when no name is given, so single-server installs still work.
    """
    if name is None:
        return DEFAULT_DISCOVERY_PATH
    return DEFAULT_SERVER_ROOT / f"endpoint_{name}.json"

# SLURM states we treat as "the server may still come up or be up".
# Anything else (COMPLETED, FAILED, CANCELLED, TIMEOUT, …) means stale.
_ALIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "RESIZING"}

# Timeout for the HTTP probe in ``is_alive``. Short — `/v1/models` is a
# cheap synchronous call once vLLM is up; if it takes >5s the server is
# probably still loading the model and we should report not-alive.
_HTTP_PROBE_TIMEOUT_S = 5.0


def _read_discovery(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _squeue_state(job_id: str) -> Optional[str]:
    """Return the SLURM state of ``job_id``, or ``None`` if squeue
    doesn't list it (the job has finished or doesn't exist). Returns
    ``None`` rather than raising when squeue is missing — that just
    means we can't tell, and the caller should treat as alive."""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    state = result.stdout.strip()
    return state or None


def _http_probe(base_url: str) -> bool:
    """GET ``<base_url>/models`` with a short timeout. Returns True when
    we get any HTTP response (2xx, 4xx — anything means the server is
    listening). Returns False on connection refused, timeout, or any
    other socket error.

    Why a single probe instead of e.g. checking `/health`: vLLM's
    OpenAI-compatible server doesn't ship a `/health` endpoint on all
    versions, but `/v1/models` is required by the OpenAI spec and
    always present once the engine has finished initializing. Status
    code doesn't matter — `/v1/models` returning 200 vs 404 both
    prove the engine is past startup.
    """
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
    except ImportError:  # pragma: no cover — stdlib always present
        return False

    url = base_url.rstrip("/") + "/models"
    req = Request(url, method="GET")
    try:
        urlopen(req, timeout=_HTTP_PROBE_TIMEOUT_S).read()
        return True
    except HTTPError:
        # 4xx/5xx still means the server is listening — only network
        # errors mean it isn't.
        return True
    except (URLError, OSError, TimeoutError):
        return False


def is_alive(discovery: dict, *, probe_http: bool = True) -> bool:
    """True when the SLURM job recorded in ``discovery`` is alive AND
    the HTTP endpoint actually answers.

    The combination matters: ``launch.sh`` writes the discovery file
    BEFORE it `exec`s ``vllm serve``, so the file appears minutes
    before the server is reachable. The SLURM state alone would say
    the job is RUNNING but a client would get ``Connection refused``.
    The HTTP probe closes that gap.

    A ``slurm_job_id`` prefixed with ``local-`` (set by ``launch.sh``
    when ``SLURM_JOB_ID`` isn't in the env) skips the squeue lookup;
    we still HTTP-probe.

    When squeue itself is unavailable (e.g. running from a machine
    without slurm tools on PATH), we report stale rather than
    fabricating a live URL — the operator probably wants the helper
    to refuse than to hand them a bad endpoint.

    Pass ``probe_http=False`` to fall back to the legacy SLURM-only
    behaviour (mostly useful for unit tests that don't want to mock
    urlopen).
    """
    job_id = str(discovery.get("slurm_job_id") or "")
    if not job_id or job_id.startswith("local-"):
        # No SLURM job to query (running outside SLURM).
        slurm_ok = True
    else:
        state = _squeue_state(job_id)
        if state is None:
            return False  # squeue ran and didn't list it → finished/stale
        slurm_ok = state in _ALIVE_STATES

    if not slurm_ok:
        return False
    if not probe_http:
        return True

    host = discovery.get("host")
    port = discovery.get("port")
    if not host or not port:
        return False
    return _http_probe(f"http://{host}:{port}/v1")


def resolve_endpoint(
    discovery_path: Optional[Path] = None,
    *,
    name: Optional[str] = None,
) -> Optional[str]:
    """Return the OpenAI-compatible base URL for the live server, or
    ``None`` if no live server is reachable. Doesn't network: trusts the
    discovery file plus a squeue check.

    Pass ``name`` to look up a per-model discovery file (matches what
    ``launch.sh`` writes for concurrent servers); pass ``discovery_path``
    to override the path explicitly; pass neither for the single-slot
    default.
    """
    if discovery_path is None:
        discovery_path = discovery_path_for(name)
    discovery = _read_discovery(discovery_path)
    if discovery is None:
        return None
    if not is_alive(discovery):
        return None
    host = discovery.get("host")
    port = discovery.get("port")
    if not host or not port:
        return None
    return f"http://{host}:{port}/v1"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-base-url",
        action="store_true",
        help="Print only the base URL on stdout (for shell substitution).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Per-model server name (matches `discovery_name` in the "
        "model YAML, or the YAML basename). Resolves to "
        "`<server_root>/endpoint_<name>.json`. Mutually exclusive with "
        "`--path`.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override the discovery-file location.",
    )
    args = parser.parse_args(argv)

    if args.path is not None and args.name is not None:
        parser.error("--path and --name are mutually exclusive")

    resolved_path = args.path if args.path is not None else discovery_path_for(args.name)
    discovery = _read_discovery(resolved_path)
    if discovery is None:
        if not args.print_base_url:
            print(f"no discovery file at {resolved_path}", file=sys.stderr)
        return 2

    alive = is_alive(discovery)
    base_url = resolve_endpoint(resolved_path) if alive else None

    if args.print_base_url:
        if base_url:
            print(base_url)
            return 0
        return 2

    print(json.dumps(discovery, indent=2))
    job_id = discovery.get("slurm_job_id")
    if job_id and not str(job_id).startswith("local-"):
        state = _squeue_state(str(job_id)) or "(not in squeue)"
        print(f"\nSLURM job {job_id} state: {state}")
    print(f"alive: {alive}")
    if base_url:
        print(f"base_url: {base_url}")
    return 0 if alive else 2


if __name__ == "__main__":
    raise SystemExit(main())

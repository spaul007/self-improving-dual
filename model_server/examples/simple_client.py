"""Minimal standalone client for the vLLM model server.

Demonstrates that the server is just a vanilla OpenAI-compatible
endpoint — no meta-agent imports needed. The only knob is the
base URL.

Three ways to point this at the server, in priority order:
  1. ``--base-url http://host:port/v1`` on the CLI
  2. ``OPENAI_BASE_URL`` (or ``LLM_BASE_URL``) env var
  3. Read the discovery file written by ``launch.sh`` (default path:
     ``/groups/AIC-MV/n.tzou/model_server/endpoint.json``). The
     helper imports ``model_server.health`` from the sibling
     directory if available; falls back to a raw json read so this
     script is usable from a checkout that doesn't have the meta-agent
     repo on sys.path.

Both endpoints vLLM exposes are demonstrated:
  * ``/v1/responses`` — what the meta-agent uses.
  * ``/v1/chat/completions`` — the more widely-supported path.

Run:
    pip install openai
    python3 simple_client.py --prompt "Plan a 2-day trip to Kyoto"
    python3 simple_client.py --endpoint chat --prompt "Hello"

Exits non-zero if the server can't be reached.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_DISCOVERY_PATH = Path(
    os.environ.get(
        "MODEL_SERVER_DISCOVERY_PATH",
        "/groups/AIC-MV/n.tzou/model_server/endpoint.json",
    )
)


def discover_base_url() -> Optional[str]:
    """Resolve a base URL from CLI / env / discovery file, in that order."""
    # 1. env vars (LLM_BASE_URL takes precedence over the SDK's
    #    OPENAI_BASE_URL because it's what the meta-agent uses).
    for env in ("LLM_BASE_URL", "OPENAI_BASE_URL"):
        val = os.environ.get(env)
        if val:
            return val

    # 2. discovery file. Try the sibling health.py first (it knows about
    #    stale jobs); fall back to a bare JSON read.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "model_server_health",
            Path(__file__).resolve().parent.parent / "health.py",
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.resolve_endpoint(DEFAULT_DISCOVERY_PATH)
    except Exception:  # pragma: no cover — fallback path
        pass

    if DEFAULT_DISCOVERY_PATH.exists():
        try:
            payload = json.loads(
                DEFAULT_DISCOVERY_PATH.read_text(encoding="utf-8")
            )
            host = payload.get("host")
            port = payload.get("port")
            if host and port:
                return f"http://{host}:{port}/v1"
        except (json.JSONDecodeError, OSError):
            pass

    return None


def call_responses(client, model: str, prompt: str) -> str:
    """Hit /v1/responses — the endpoint the meta-agent uses."""
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=512,
    )
    # output_text is a convenience accessor the SDK adds; fall back to
    # walking the structured output if the server didn't populate it.
    return getattr(response, "output_text", None) or json.dumps(
        getattr(response, "output", []), default=str, indent=2
    )


def call_chat(client, model: str, prompt: str) -> str:
    """Hit /v1/chat/completions — the more universally supported path."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        help="Override the discovered base URL (e.g. http://host:port/v1).",
    )
    parser.add_argument(
        "--model",
        help="Model id to send. Defaults to the model in the discovery "
        "file (or `local` if no file is found).",
    )
    parser.add_argument(
        "--endpoint",
        choices=("responses", "chat"),
        default="responses",
        help="Which OpenAI-compatible endpoint to call (default: responses).",
    )
    parser.add_argument(
        "--prompt",
        default="In one sentence: why is the sky blue?",
        help="User prompt to send.",
    )
    args = parser.parse_args(argv)

    base_url = args.base_url or discover_base_url()
    if not base_url:
        print(
            "Could not discover a server URL. Pass --base-url or set "
            "LLM_BASE_URL. (Did you start the server with "
            "model_server/slurm_submit.sh?)",
            file=sys.stderr,
        )
        return 2

    # Read the discovery file to pick a default model id when --model
    # wasn't supplied. Skipped silently if the file isn't readable.
    model = args.model
    if not model:
        try:
            payload = json.loads(
                DEFAULT_DISCOVERY_PATH.read_text(encoding="utf-8")
            )
            model = payload.get("model", "local")
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            model = "local"

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "openai SDK is not installed. Run: pip install openai",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(
        base_url=base_url,
        # vLLM ignores auth, but the SDK constructor requires a string.
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )

    print(f"# base_url:  {base_url}")
    print(f"# model:     {model}")
    print(f"# endpoint:  /v1/{args.endpoint}")
    print(f"# prompt:    {args.prompt!r}")
    print()

    if args.endpoint == "chat":
        text = call_chat(client, model, args.prompt)
    else:
        text = call_responses(client, model, args.prompt)

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

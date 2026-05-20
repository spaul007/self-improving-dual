from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

TRACE_PATH_ENV = "META_AGENT_TRACE_PATH"

_lock = threading.Lock()


def _resolve_path() -> str | None:
    return os.environ.get(TRACE_PATH_ENV)


def emit(kind: str, payload: dict[str, Any]) -> None:
    """Append one JSONL TraceEvent line to the file named by META_AGENT_TRACE_PATH.

    Silently no-ops when the env var is unset (e.g. unit tests not driven by the
    evaluator). Concurrency-safe within a single process via a module lock.
    """
    path = _resolve_path()
    if not path:
        return
    event = {
        "timestamp": time.time(),
        "kind": kind,
        "payload": payload,
    }
    # ``default=str`` so Pydantic ``BaseModel`` instances (e.g. Responses-API
    # output items echoed back into the conversation by reasoning-model
    # agents) serialize as their ``repr`` rather than raising — they're
    # informational here, not part of the consumed schema.
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

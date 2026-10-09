"""Crash- and quota-safe file writes for run state.

A plain ``Path.write_text`` truncates the target first, so a kill or a full
filesystem mid-write leaves a 0-byte or half-written file -- which is exactly how a
resume once lost its counters (loop_state.json truncated when /groups hit quota).
``atomic_write_text`` writes a sibling temp file and ``os.replace``s it over the
target: readers see either the old content or the new, never a partial file.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (same-directory temp + ``os.replace``).
    The temp file is removed if anything fails; the exception propagates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

"""Immutable travel tools.

Each sub-module exports ``NAME``, ``SCHEMA``, ``run(**kwargs) -> str`` and
calls :func:`platform_core.tools.register_tool` at import time. This package's
``__init__`` triggers the auto-discovery side effect by importing every
sub-module.

Per-case database access uses two environment variables:

* ``TRAVEL_DATABASE_ROOT`` — points at the directory containing per-sample
  ``id_<sample_id>/`` sub-directories with the CSV files. Set globally before
  running the meta-agent.
* ``TRAVEL_SAMPLE_ID`` — the sample under evaluation. Set per-case by the
  benchmark via the evaluator's per-case env merge.

When either is missing the tools return a "database not loaded" sentinel
string, mirroring the reference behaviour so the agent loop still terminates.
"""
from __future__ import annotations

# Importing the modules registers each tool via @register_tool side effects.
from . import (  # noqa: F401
    flight,
    hotel,
    train,
    attraction,
    restaurant,
    location,
    roadroute,
)

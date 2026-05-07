"""Immutable tools for the math project.

Each sub-module exports ``NAME``, ``SCHEMA``, ``run(**kwargs) -> str`` and
calls :func:`platform_core.tools.register_tool` at import time. This
package's ``__init__`` triggers the side effect by importing every
sub-module."""
from __future__ import annotations

from . import calculate  # noqa: F401

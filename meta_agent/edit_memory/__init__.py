"""Edit memory for the agentic HGM loop.

    C_{i+1} <- Expand(C_i, L_i, B_j)  or  Expand(C_i, L_i)     bandit over the two arms
    Z       <- Curation(last m nodes and their logs)          agentic curator
    B_{j+1} <- Generation(Z, B_j, I_k)                        one LLM call
    Q       <- Curation(tau_meta, tau_task+/-, B_j, I_k)      agentic curator
    I_{k+1} <- InstructionUpdate(Q, I_k)                      one LLM call

``layer.EditMemoryLayer`` (registered ``edit_memory: {type: agentic}``) is
the manager-facing object: it chooses the arm before every expansion,
observes the loop's events, and writes ``<experiment_dir>/edit_memory/``.
The memory never contains score predictions; the curators judge usefulness
from logs, diffs and per-case outcomes.
"""
from . import layer  # noqa: F401  (registers the component)

__all__ = ["layer"]

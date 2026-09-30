"""Agentic editor support: an HGM-seed-style coding agent (bash + file editor
tool loop) that edits the task agent in place, without docker.

Modules:

- ``policy``   — what the coding agent may read and write (absolute-path
                 allow-list; mirrored by the sandbox binds and enforced
                 in-process by the editor tool).
- ``sandbox``  — bubblewrap (``bwrap``) allow-list sandbox for the ``bash``
                 tool, with a cwd-only fallback when bwrap is unavailable.
- ``tools``    — the ``bash`` / ``editor`` / ``validate`` /
                 ``submit_self_improvement`` tool schemas and functions.
- ``session``  — the tool-use loop (a port of HGM's ``chat_with_agent_openai``
                 over ``platform_core.llm_wrapper.call_llm``), prompts, and
                 per-session artifacts (``transcript.jsonl``, ``session.json``).

The editor class itself lives in ``meta_agent.agent_editor_agentic`` so it is
registered next to the other editor kinds.
"""

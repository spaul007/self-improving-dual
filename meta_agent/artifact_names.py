"""File and directory names of the agentic editor's and the edit-memory
layer's run artifacts, in a module with no dependencies so the dashboard's
loaders (``run_inspect_agentic``) can import them without pulling in the
editor stack (validators, pyflakes, the LLM wrapper)."""

# <run>/edit_memory/ -- the edit-memory layer's directory under the run root.
MEMORY_DIR_NAME = "edit_memory"
# <round>/agentic/ -- one agentic session's transcript and summary.
TRANSCRIPT_NAME = "transcript.jsonl"
SESSION_NAME = "session.json"
# Inside edit_memory/: the latest memory copy, the current addendum, the state.
MEMORY_FILE = "edit_memory.md"
INSTRUCTION_FILE = "instruction.md"
STATE_FILE = "state.json"

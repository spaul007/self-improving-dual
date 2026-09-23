"""Host-side tools over safe_exec.

CONTENT NEVER GOES THROUGH A HEREDOC. It is base64-encoded on the host and decoded in the
container. This project has already been burned by runaway 2.7M-character heredocs from
exactly this pattern, and quoting bugs in generated shell are near-impossible to debug.

We give the model a real file editor, not bash alone. OpenHands telemetry we collected
showed `file_editor` invoked 3,370 times across 60 instances -- second only to the
terminal -- so a bash-only surface is a measured handicap, not a simplification.

v8 MIRRORS CLAUDE CODE'S TOOL SURFACE: Read, Bash, Edit, Write -- names, parameters and
defaults taken from its raw session file, not from documentation. Two measured reasons:

  1. FOUR TOOLS, NOT SIX. Claude Code has no `grep` or `list_dir`; both go through Bash.
     seedling's own numbers agree they are noise: grep 14,720 chars and list_dir 25,299
     across 8 trials, against read_file's 1,443,976.

  2. THE READ WINDOW IS THE WHOLE BALLGAME. seedling's read_file defaulted to 400 LINES and
     its p90 output was 12,026 chars -- sitting exactly on the 12,000 cap, i.e. >=10% of
     reads returned a truncated wall. Claude Code passes an explicit window in 21 of 30
     calls at 30-130 lines, and its median tool_result is 249 chars against seedling's
     thousands. It does not truncate because it does not over-fetch. Same lesson, applied:
     default to a focused slice and let the model widen it deliberately.

`Bash` additionally takes a REQUIRED `description`, exactly as Claude Code's does. It costs
one short string and buys two things: the model must state each command's intent before
running it, and every command becomes self-labelling in the trajectory.
"""

from __future__ import annotations

import base64
import shlex

from . import Tool, denied_path
from .. import settings


def _clip(s: str, n: int = settings.MAX_TOOL_OUTPUT_CHARS) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n...[{len(s)-n} chars elided]...\n" + s[-n // 2 :]


async def Bash(pool, command: str = "", description: str = "",
               timeout: int = settings.EXEC_TIMEOUT_SEC) -> str:
    # `description` is REQUIRED in the schema, exactly as Claude Code's Bash is. It is not
    # used for control flow -- its value is that the model must state intent before acting,
    # and every command becomes self-labelling in the trajectory.
    out = await pool.run(command, timeout_sec=min(int(timeout or 60),
                                                  settings.EXEC_TIMEOUT_SEC),
                         label=f"tool:Bash:{(description or command)[:40]}")
    head = f"[rc={out.return_code}{' TIMED OUT' if out.timed_out else ''}]\n"
    return head + _clip(out.stdout)


async def Read(pool, file_path: str = "", offset: int = 1,
               limit: int = 0) -> str:
    # limit=0 means "unspecified" -> settings.READ_DEFAULT_LIMIT (120 lines,
    # matching the top of Claude Code's observed 30-130 range). The old 400-line
    # default is what pinned p90 output to the 12,000-char cap.
    off = max(1, int(offset or 1))
    lim = max(1, int(limit) if limit else settings.READ_DEFAULT_LIMIT)
    # NR, not `cat -n` + index($0,$2): on a BLANK line $2 is unset, index($0,"")==0,
    # and substr($0,0) returns the WHOLE line -- so blank lines rendered as
    # "     2\t     2\t". edit_file needs an exact match and blank lines are
    # everywhere in code, so quoted-back blocks would silently fail to match.
    cmd = (f"sed -n '{off},{off+lim-1}p' {shlex.quote(file_path)} | "
           f"awk -v o={off-1} '{{printf \"%6d\\t%s\\n\", NR+o, $0}}'")
    out = await pool.run(cmd, timeout_sec=60, label="tool:read")
    if not out.ok:
        return f"[rc={out.return_code}] could not read {file_path}\n" + _clip(out.stdout)
    return _clip(out.stdout) or "(empty or past end of file)"


async def Write(pool, file_path: str = "", content: str = "") -> str:
    bad = denied_path(file_path)
    if bad:
        return bad
    b64 = base64.b64encode((content or "").encode()).decode()
    if len(b64) > 240_000:
        return "refused: content too large for one write; split it"
    cmd = (f"mkdir -p \"$(dirname {shlex.quote(file_path)})\" && "
           f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(file_path)} && "
           f"wc -c < {shlex.quote(file_path)}")
    out = await pool.run(cmd, timeout_sec=120, label="tool:write")
    return f"wrote {file_path} ({out.stdout.strip()} bytes)" if out.ok else \
           f"[rc={out.return_code}] write failed\n" + _clip(out.stdout)


async def Edit(pool, file_path: str = "", old_string: str = "", new_string: str = "",
               replace_all: bool = False) -> str:
    """Unique-match replace, or replace_all. Fails loudly on 0 matches, and on >1 when
    replace_all is false -- guessing which occurrence was meant is how silent mis-edits
    happen. Content never goes through a heredoc; it is base64'd on the host."""
    bad = denied_path(file_path)
    if bad:
        return bad
    if not old_string:
        return "refused: 'old_string' must be non-empty (use Write to create a file)"
    ob = base64.b64encode(old_string.encode()).decode()
    nb = base64.b64encode((new_string or "").encode()).decode()
    py = (
        "import base64,sys\n"
        f"p={file_path!r}\n"
        f"o=base64.b64decode({ob!r}).decode()\n"
        f"n=base64.b64decode({nb!r}).decode()\n"
        f"ra={bool(replace_all)!r}\n"
        "s=open(p,encoding='utf-8',errors='surrogateescape').read()\n"
        "c=s.count(o)\n"
        "if c==0: print('EDIT_FAIL no match'); sys.exit(1)\n"
        "if c>1 and not ra: print(f'EDIT_FAIL {c} matches; add surrounding context or set "
        "replace_all=true'); sys.exit(1)\n"
        "open(p,'w',encoding='utf-8',errors='surrogateescape').write("
        "s.replace(o,n) if ra else s.replace(o,n,1))\n"
        "print(f'EDIT_OK {c}')\n"
    )
    b64 = base64.b64encode(py.encode()).decode()
    out = await pool.run(
        f"printf '%s' {shlex.quote(b64)} | base64 -d | python3 -", timeout_sec=120,
        label="tool:Edit")
    if out.has("EDIT_OK"):
        return f"edited {file_path}"
    return f"[rc={out.return_code}] " + _clip(out.stdout)


ALL = [
    Tool("Bash",
         "Run a shell command in /app. Merged stdout+stderr, truncated. Use this for search "
         "(grep/rg), listing (ls), builds and tests -- there are no separate grep/list tools.",
         {"type": "object",
          "properties": {"command": {"type": "string"},
                         "description": {"type": "string",
                                         "description": "5-10 words: what this command is for."},
                         "timeout": {"type": "integer"}},
          "required": ["command", "description"]}, Bash),
    Tool("Read",
         "Read a file with line numbers. Prefer a targeted window: pass `offset` and `limit` "
         f"rather than reading a whole file. Defaults to {settings.READ_DEFAULT_LIMIT} lines.",
         {"type": "object",
          "properties": {"file_path": {"type": "string"},
                         "offset": {"type": "integer",
                                    "description": "1-based first line."},
                         "limit": {"type": "integer",
                                   "description": "How many lines. Keep it tight."}},
          "required": ["file_path"]}, Read),
    Tool("Edit",
         "Replace old_string with new_string in a file. Fails if old_string is not unique "
         "unless replace_all is true, so include surrounding context.",
         {"type": "object",
          "properties": {"file_path": {"type": "string"},
                         "old_string": {"type": "string"},
                         "new_string": {"type": "string"},
                         "replace_all": {"type": "boolean"}},
          "required": ["file_path", "old_string", "new_string"]}, Edit, mutates=True),
    Tool("Write",
         "Create or overwrite a file with exact content.",
         {"type": "object",
          "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
          "required": ["file_path", "content"]}, Write, mutates=True),
]

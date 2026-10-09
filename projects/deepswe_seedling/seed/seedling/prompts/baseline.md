You are the VERIFY role on your FIRST turn for this task. The repository at `/app` is
**unmodified** -- nobody has changed anything yet. Your only job right now is to establish the
**test baseline** that you will judge against later. Do NOT judge the task, do NOT write tests,
and do NOT change any file.

## What to do

1. Find the project's **canonical, full** test command. Every project is different: look at CI
   config (`.github/workflows`, `.travis.yml`), `Makefile`, `tox.ini`, `pyproject.toml`,
   `package.json` scripts, `go.mod`, `Cargo.toml`, an existing `test.sh`. Prefer the command CI
   runs. Read targeted windows (`Read` with `offset`/`limit`; locate with `grep -n` first).
2. If there is a build step, run it and note whether it succeeds.
3. **Run the full test command once.** For JavaScript/TypeScript runners use a CI, non-watch
   invocation (`CI=true`, jest `--ci --watchAll=false`, vitest `--run`) and skip coverage --
   a watch-mode or coverage run can consume your whole budget and report nothing. Capture: how many tests ran, how many passed, how many
   failed or errored, and the **names of every failing test**. Note how long it took.
4. If the suite cannot be run (missing deps, broken build, no tests), say so precisely -- that is
   a valid baseline finding, not a failure on your part.

## Why this matters

Later, after a change is made, you will run **the same command** and compare. A regression is a
test that passes NOW and fails THEN. Tests that already fail now are not regressions and must not
be blamed on the change. If you pick a narrow or wrong command here, every later judgement is
built on sand -- so prefer the project's real, full suite over a quick subset.

**Never change git state** (`checkout`, `switch`, `reset`, `stash`, `commit`). Do not fetch or
apply anything from the internet.

When done, stop making tool calls and state the baseline briefly. You will be asked once, in a
separate turn, to report it formally: the exact command, counts, failing test names, build
status and duration.

"""Commit discipline -- turning work into a score.

DeepSWE's collect hook is, in every one of the 113 task.toml files:
    git diff --binary <BASE_SHA> HEAD > /logs/artifacts/model.patch

That is a COMMIT-TO-COMMIT diff. Uncommitted and untracked work is invisible and scores 0,
and Pier never commits for you (zero `git commit` calls in the package).

Corpus facts verified across all 113 tasks:
  *   0/113 images set a git identity  -> commit fails hard without -c overrides
  * 113/113 set core.hooksPath=/dev/null (their comment: hooks "block the agent's commits")
  *  40/113 assert a clean worktree at build time -- so 73 MAY hand us a PRE-DIRTY tree
    (e.g. adaptix runs `pip install -e .`, leaving *.egg-info/). A naive `git add -A`
    would commit the image's own dirt into model.patch.

THE LETHAL ONE: the verifier's grader resets only paths appearing in model.patch or
test.patch. A leftover agent-authored `tests/test_scratch.py` therefore SURVIVES into the
verifier, gets collected by `pytest tests/` or `go test ./...`, and can fail the whole p2p
bucket (median 165 tests, max 66,265) -- reward 0 from a file we never meant to submit.
Hence the quarantine sweep, and the hard rule that scratch lives in /tmp/seedling/scratch.

We also write /logs/artifacts/model.patch OURSELVES at every checkpoint. The collect hook
is an && chain whose redirect binds to the last command, so an earlier failure means the
file is never written at all; and on the generic-Exception path Pier still runs
_collect_artifacts() even though it skips the collect hooks, so our copy reaches the host
and the run stays re-gradable offline.
"""

from __future__ import annotations

import re
import shlex

ARTIFACT = "/logs/artifacts/model.patch"
WORK = "/tmp/seedling"
IDENT = "-c user.email=agent@seedling.local -c user.name='Seedling Agent'"

# Untracked paths we must never commit. .git/info/exclude only suppresses UNTRACKED files,
# so a blanket entry can never mask a real edit to a tracked file of the same name.
NEVER_COMMIT = [
    "node_modules/", "target/", "dist/", "out/", ".gocache/", ".cache/",
    "__pycache__/", "*.pyc", "*.egg-info/", ".pytest_cache/", ".mypy_cache/",
    ".ruff_cache/", ".tox/", ".venv/", "venv/", "coverage/", ".nyc_output/",
    "*.orig", "*.rej", "*.log", ".DS_Store",
    "/test.sh",   # test.patch owns this path in 113/113 tasks
]

# Untracked files that look like tests the agent wrote. Committing one can zero the run.
TEST_SHAPED = (
    r'(^|/)(tests?|__tests__|spec)/|'
    r'(_test\.go|\.test\.[cm]?[jt]sx?|\.spec\.[cm]?[jt]sx?|test_[^/]*\.py|[^/]*_test\.py|test\.sh)$'
)

BOOTSTRAP = r"""
cd /app || { echo "SEED_BOOTSTRAP_FAIL=cd"; exit 90; }
mkdir -p WORKDIR /logs/artifacts

git config --global --add safe.directory /app 2>/dev/null || true
git config --local user.email "agent@seedling.local"
git config --local user.name  "Seedling Agent"
git config --local commit.gpgsign false
git config --local core.hooksPath /dev/null
git config --local core.autocrlf false
git config --local core.safecrlf false
git config --local gc.auto 0

# BASE is self-validating: the image checks the default branch out AT the base commit and
# never commits, so HEAD before our first commit IS the base commit.
echo "SEED_BASE=$(git rev-parse HEAD)"

# Baseline dirt, captured BEFORE we touch anything (73/113 images may be dirty).
git status --porcelain=v1 -z --untracked-files=all 2>/dev/null | tr '\0' '\n' > WORKDIR/baseline.status
sed -n 's/^?? //p' WORKDIR/baseline.status > WORKDIR/baseline_untracked.txt
grep -v '^?? ' WORKDIR/baseline.status | sed 's/^...//' > WORKDIR/baseline_tracked_dirty.txt
echo "SEED_DIRTY_TRACKED=$(wc -l < WORKDIR/baseline_tracked_dirty.txt)"
echo "SEED_DIRTY_UNTRACKED=$(wc -l < WORKDIR/baseline_untracked.txt)"

git submodule status --recursive 2>/dev/null | awk '{print $2}' > WORKDIR/submodules.txt || true

{
  echo "# --- seedling: never commit ---"
  printf '%s\n' NEVER_COMMIT_LIST
  cat WORKDIR/baseline_untracked.txt
} >> /app/.git/info/exclude

# instruction.md asks for a new branch; -B is idempotent and preserves worktree state.
git checkout -B seedling/solution >/dev/null 2>&1 || echo "SEED_BRANCH_FAIL(non-fatal)"

command -v timeout >/dev/null 2>&1 && echo "SEED_HAS_TIMEOUT=1" || echo "SEED_HAS_TIMEOUT=0"
echo "SEED_BOOTSTRAP_OK"
"""

CHECKPOINT = r"""
cd /app || exit 90
BASE="$1"; MSG="$2"

# 1. quarantine agent-authored test-shaped untracked files (see module docstring)
git status --porcelain=v1 --untracked-files=all 2>/dev/null | sed -n 's/^?? //p' \
  | grep -Ei 'TEST_SHAPED_RE' > WORKDIR/quarantine.txt || true

# 2. stage everything, including untracked source (new files matter)
git add -A -- . || { echo "SEED_ADD_FAIL"; exit 91; }

# 3. un-stage image dirt, submodule gitlinks and quarantined scratch tests
for L in WORKDIR/baseline_tracked_dirty.txt WORKDIR/submodules.txt WORKDIR/quarantine.txt; do
  [ -s "$L" ] && tr '\n' '\0' < "$L" | xargs -0 -r git reset -q -- 2>/dev/null
done

# 4. commit only if something is actually staged (empty commit would fail)
if git diff --cached --quiet; then
  echo "SEED_NO_CHANGES"
else
  git IDENT commit -q --no-verify -m "$MSG" || { echo "SEED_COMMIT_FAIL"; exit 92; }
fi

# 5. materialise model.patch ourselves, atomically
mkdir -p /logs/artifacts
if git diff --binary "$BASE" HEAD > ARTIFACT.tmp 2>/dev/null; then
  mv -f ARTIFACT.tmp ARTIFACT
else
  rm -f ARTIFACT.tmp; echo "SEED_DIFF_FAIL"
fi

echo "SEED_HEAD=$(git rev-parse HEAD)"
echo "SEED_PATCH_BYTES=$(wc -c < ARTIFACT 2>/dev/null || echo 0)"
echo "SEED_QUARANTINED=$(wc -l < WORKDIR/quarantine.txt 2>/dev/null || echo 0)"
echo "SEED_FILES_BEGIN"; git diff --name-only "$BASE" HEAD 2>/dev/null | head -300; echo "SEED_FILES_END"
echo "SEED_CHECKPOINT_OK"
"""

# Models the grader exactly: reset the touched files to base, then git apply.
APPLY_CHECK = r"""
cd /app || exit 90
BASE="$1"
rm -rf WORKDIR/applycheck
git worktree add --detach -f WORKDIR/applycheck "$BASE" >/dev/null 2>&1 || { echo "SEED_APPLY_WT_FAIL"; exit 0; }
if [ -s ARTIFACT ]; then
  git -C WORKDIR/applycheck apply --check --whitespace=nowarn ARTIFACT >/dev/null 2>&1 \
    && echo "SEED_APPLY_OK" || echo "SEED_APPLY_FAIL"
else
  echo "SEED_APPLY_EMPTY"
fi
git worktree remove --force WORKDIR/applycheck >/dev/null 2>&1; git worktree prune >/dev/null 2>&1
"""


def _render(script: str) -> str:
    return (script
            .replace("WORKDIR", WORK)
            .replace("ARTIFACT", ARTIFACT)
            .replace("IDENT", IDENT)
            .replace("TEST_SHAPED_RE", TEST_SHAPED)
            .replace("NEVER_COMMIT_LIST", " ".join(shlex.quote(p) for p in NEVER_COMMIT)))


class GitOps:
    def __init__(self, pool, logger) -> None:
        self._pool = pool
        self._log = logger
        self.base_sha: str | None = None
        self.head_sha: str | None = None
        self.n_checkpoints = 0
        self.patch_bytes = 0
        self.patch_files: list[str] = []
        self.quarantined = 0
        self.apply_check_result = "skipped"
        self.bootstrap_ok = False
        self.last_error: str | None = None

    async def bootstrap(self) -> dict:
        out = await self._pool.git(f"bash -c {shlex.quote(_render(BOOTSTRAP))}",
                                   timeout_sec=180, label="git:bootstrap")
        self.bootstrap_ok = out.has("SEED_BOOTSTRAP_OK")
        self.base_sha = out.sentinel("SEED_BASE")
        self._pool.set_has_timeout(out.sentinel("SEED_HAS_TIMEOUT") == "1")
        if not self.bootstrap_ok:
            self.last_error = "bootstrap_failed"
            self._log.error("git bootstrap FAILED: %s", out.tail(1500))
        else:
            self._log.info("git bootstrap ok base=%s dirty_tracked=%s dirty_untracked=%s",
                           self.base_sha, out.sentinel("SEED_DIRTY_TRACKED"),
                           out.sentinel("SEED_DIRTY_UNTRACKED"))
        return {
            "ok": self.bootstrap_ok,
            "base_sha": self.base_sha,
            "dirty_tracked": out.sentinel("SEED_DIRTY_TRACKED"),
            "dirty_untracked": out.sentinel("SEED_DIRTY_UNTRACKED"),
        }

    async def checkpoint(self, reason: str) -> dict:
        """Commit whatever exists and refresh model.patch. Never raises."""
        if not self.base_sha:
            return {"ok": False, "reason": "no_base_sha"}
        cmd = (f"bash -c {shlex.quote(_render(CHECKPOINT))} seedling "
               f"{shlex.quote(self.base_sha)} {shlex.quote(reason[:200])}")
        out = await self._pool.git(cmd, timeout_sec=600, label="git:checkpoint", critical=True)
        ok = out.has("SEED_CHECKPOINT_OK")
        if ok:
            self.n_checkpoints += 1
            self.head_sha = out.sentinel("SEED_HEAD") or self.head_sha
            try:
                self.patch_bytes = int(out.sentinel("SEED_PATCH_BYTES") or 0)
                self.quarantined = int(out.sentinel("SEED_QUARANTINED") or 0)
            except ValueError:
                pass
            body = out.stdout
            if "SEED_FILES_BEGIN" in body and "SEED_FILES_END" in body:
                seg = body.split("SEED_FILES_BEGIN", 1)[1].split("SEED_FILES_END", 1)[0]
                self.patch_files = [x for x in (l.strip() for l in seg.splitlines()) if x]
            # Log EVERY successful checkpoint. Commit cadence is safety-critical -- it is
            # what bounds how much work a hard cancel destroys -- and previously nothing was
            # logged on success, so from the job log you could not tell whether commits were
            # happening at all. Silence must not be ambiguous between "committing fine" and
            # "never committed".
            self._log.info("checkpoint ok: %s | head=%s patch=%dB files=%d commits=%d",
                           reason, (self.head_sha or "?")[:12], self.patch_bytes,
                           len(self.patch_files), self.n_checkpoints)
            if self.quarantined:
                self._log.warning("quarantined %d agent-authored test-shaped file(s)",
                                  self.quarantined)
        else:
            self.last_error = "checkpoint_failed"
            self._log.error("checkpoint(%s) FAILED: %s", reason, out.tail(1500))
        return {"ok": ok, "reason": reason, "head": self.head_sha,
                "patch_bytes": self.patch_bytes, "n_files": len(self.patch_files)}

    async def current_diff(self, max_chars: int = 12000) -> str:
        """The change so far, as the GRADER will see it: `git diff --binary base..HEAD`.

        Added because a retry could not see what its predecessor changed. Measured on katex:
        `patch.2` re-READ 1 of the 2 files `patch.1` had edited -- it had to go look to find
        out what had already been done. The harness knew and threw the information away.
        Stat first (always small), then as much of the body as fits.
        """
        if not self.base_sha:
            return ""
        out = await self._pool.git(
            f"cd /app && git diff --stat {self.base_sha} HEAD && echo '--- DIFF ---' && "
            f"git diff {self.base_sha} HEAD | head -c {max_chars}",
            timeout_sec=120, label="git:diff")
        return out.stdout if out.ok else ""

    async def apply_check(self) -> str:
        if not self.base_sha:
            return "skipped"
        out = await self._pool.git(
            f"bash -c {shlex.quote(_render(APPLY_CHECK))} seedling {shlex.quote(self.base_sha)}",
            timeout_sec=300, label="git:applycheck", critical=True)
        for token, val in (("SEED_APPLY_OK", "ok"), ("SEED_APPLY_FAIL", "fail"),
                           ("SEED_APPLY_EMPTY", "empty")):
            if out.has(token):
                self.apply_check_result = val
                break
        else:
            self.apply_check_result = "skipped"
        if self.apply_check_result == "fail":
            self._log.error("APPLY CHECK FAILED -- the grader will score this 0 (apply_failed=1)")
        return self.apply_check_result

    async def finalize(self) -> dict:
        res = await self.checkpoint("seedling: final")
        await self.apply_check()
        if self.patch_bytes == 0:
            self._log.error("FINAL PATCH IS EMPTY -- this trial is a guaranteed zero")
        return {**res, "apply_check": self.apply_check_result}

    async def numstat(self, a: str | None, b: str | None = None) -> dict:
        """What changed between two commits, as the grader would see it: per-role attribution
        by GIT, not by counters. Counters lied once (a task reported 41 edits with a 0-byte
        patch); `git diff --numstat` cannot. Never raises."""
        a = a or self.base_sha
        b = b or self.head_sha or "HEAD"
        if not a:
            return {"files": 0, "added": 0, "deleted": 0, "paths": [], "source_files": 0,
                    "test_files": 0, "from": None, "to": None}
        out = await self._pool.git(f"cd /app && git diff --numstat {shlex.quote(a)} {shlex.quote(b)} | head -400",
                                   timeout_sec=120, label="git:numstat")
        files, added, deleted, paths = 0, 0, 0, []
        for line in (out.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            files += 1
            paths.append(parts[2][:200])
            try:
                added += int(parts[0]); deleted += int(parts[1])
            except ValueError:
                pass                        # binary files show '-'
        _test = re.compile(TEST_SHAPED, re.I)
        n_test = sum(1 for p in paths if _test.search(p))
        return {"files": files, "added": added, "deleted": deleted, "paths": paths[:60],
                "source_files": files - n_test, "test_files": n_test, "from": a, "to": b}

    def snapshot(self) -> dict:
        return {
            "base_sha": self.base_sha, "head_sha": self.head_sha,
            "branch": "seedling/solution", "bootstrap_ok": self.bootstrap_ok,
            "n_checkpoints": self.n_checkpoints, "patch_bytes": self.patch_bytes,
            "patch_files": self.patch_files[:100], "n_patch_files": len(self.patch_files),
            "quarantined": self.quarantined, "apply_check": self.apply_check_result,
            "last_error": self.last_error,
        }

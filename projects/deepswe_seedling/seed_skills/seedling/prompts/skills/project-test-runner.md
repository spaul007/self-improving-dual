# project-test-runner

**When:** before running tests or a build in the repository.

**Steps**
1. Find how the project itself runs tests: `package.json` scripts, `Makefile`, `pyproject.toml`/`setup.cfg`/`tox.ini`,
   `go.mod` (`go test ./...`), `Cargo.toml` (`cargo test`), CI files under `.github/workflows`.
2. Use that runner and its config (the same wrapper, flags and working directory) -- not a different runner that happens
   to be installed.
3. Run the narrowest relevant scope first (one package/file/test name), then the wider suite for the touched packages.
4. Give long commands an explicit timeout below your remaining time; read failures from the first error, not the tail.

**Check:** you can name the exact command the project uses and you ran it on the touched code.

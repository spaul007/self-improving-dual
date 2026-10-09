# build-before-handoff

**When:** before you finish, if you changed compiled or type-checked code (Go, Rust, TypeScript, Java, ...).

**Why:** one type error makes the whole package fail to build, so EVERY test in it fails -- a near-miss becomes a zero.

**Steps**
1. Build/type-check everything you touched: Go `go build ./... && go vet ./<pkg>/...`; Rust `cargo build --all-targets`;
   TypeScript the project's `tsc --noEmit` / build script; others their compile step.
2. Also compile the existing tests of the touched packages (Go `go test -run XXX_NONE ./<pkg>/...`, Rust
   `cargo test --no-run`), since changed signatures break them.
3. Fix every error before handing over; do not leave commented-out or half-applied code.

**Check:** the build and the test compilation both exit 0 after your last edit.

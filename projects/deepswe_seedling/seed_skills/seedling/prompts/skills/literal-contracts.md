# literal-contracts

**When:** the task names exact identifiers, error messages, output formats, CLI flags, config keys or file names.

**Why:** hidden tests compare these literally. "Almost the same" (different casing, wording, field order, pluralisation,
trailing newline, a renamed function) scores as a failure even when the behaviour is right.

**Steps**
1. List every literal the task states, verbatim.
2. PATCH: use each literal exactly as written -- do not rename, paraphrase or "improve" it.
3. Search the diff for each literal (`git diff | grep -F '<literal>'`) and confirm it appears where the task says.
4. VERIFY: write at least one assertion per literal that compares the exact string/format, not a substring or a regex
   that would also accept a near-miss.

**Check:** every listed literal is found verbatim in the change and asserted verbatim by a test.

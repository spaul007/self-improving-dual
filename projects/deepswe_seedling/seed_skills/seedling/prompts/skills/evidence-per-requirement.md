# evidence-per-requirement

**When:** before you give a verdict.

**Why:** a `pass` that tested something adjacent to what the task stated is the most common way a wrong patch is
accepted.

**Steps**
1. Take the requirement checklist (see `requirements-checklist`).
2. For each item, record: the test/command that exercises it AFTER the patch, and its observed result.
3. An item with no test, or tested only indirectly, counts as NOT verified.
4. Verdict: `pass` only if every item is verified and passing; otherwise `fail`, listing each unverified or failing
   item with what PATCH must change.

**Check:** your report maps every checklist item to a command you ran and the output you saw.

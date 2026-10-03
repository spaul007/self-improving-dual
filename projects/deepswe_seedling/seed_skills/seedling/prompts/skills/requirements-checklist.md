# requirements-checklist

**When:** at the start of your work, before editing (PATCH) or testing (VERIFY).

**Why:** most near-misses implement the main behaviour and miss one stated detail -- an edge case, a default, an error
message, a precedence rule written in a "Notes"/"Constraints" section.

**Steps**
1. Re-read the whole task statement, including notes, constraints and examples.
2. Write a numbered checklist of atomic, observable behaviours: one API / output format / error / edge case / default /
   precedence rule per item. Split "X and Y" into two items.
3. Copy exact literals into each item (names, messages, flags, formats, numbers).
4. PATCH: work through the list and tick each item with the file/function that implements it.
   VERIFY: use the list as your test plan -- every item gets its own check.

**Check:** no sentence of the task statement is left without a checklist item.

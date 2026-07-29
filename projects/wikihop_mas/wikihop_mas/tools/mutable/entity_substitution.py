"""MUTABLE — internal hand-off: fills hop2's template with hop1's answer.

Not part of the benchmark contract (compare tools/immutable/), so an optimizer
is free to change how the substitution is phrased if `{hop1_answer}` templating
turns out to need more than a literal replace. Kept as a named, auditable step
(mirrors math_mas's compress()) so the trajectory log records exactly what
hop2 was asked.
"""


def substitute(template: str, hop1_answer: str) -> str:
    if not hop1_answer:
        return template
    return template.replace("{hop1_answer}", hop1_answer)

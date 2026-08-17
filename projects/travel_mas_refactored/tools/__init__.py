"""Thin hook package -- exists purely so the framework's standard tool
discovery (``platform_core.tools.load_project``'s default branch:
``importlib.import_module(f"projects.{project_name}.tools")``, keyed off
``project: "travel_mas"`` in the YAML) finds something real here.

No tool code lives in this file. The actual 9 tool implementations are
``projects/travel/tools/*.py``, reused verbatim (same benchmark, same
per-sample CSV data) -- see ``projects/travel_mas/seed/tool_wrapper.py``'s
own docstring for why (``tool_source_dirs`` doesn't work for that
package's relative imports).

Before this file existed, ``projects/travel_mas/seed/tool_wrapper.py``'s
own ``import projects.travel.tools`` line registered the tools fine at
*runtime* (the evaluator subprocess executes ``tool_wrapper.py``, which
triggers the import as a side effect) -- but the framework's editor
validators (``SchemaWrapperConsistencyValidator``, checked live in a cold
process) call ``platform_core.tools.is_immutable(name)``, which relies on
``_discover()``'s standard ``project.<name>.tools`` lookup, NOT on
``tool_wrapper.py`` having been imported. With no ``projects.travel_mas.tools``
package, that lookup found nothing (silently swallowed as "no immutable
tools"), so every validator run saw all 9 tools as unprovided --
confirmed live: this caused every single edited round of a real hgm_dual
run to fail validation (the editor kept trying to "fix" the false
violation by generating invalid ``mutable_tools/*.py`` files), never
producing one successfully-evaluated edit across 27+ rounds.
"""
from __future__ import annotations

import projects.travel.tools  # noqa: F401 -- registers the real tool implementations

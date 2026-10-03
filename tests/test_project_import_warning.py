"""A failing project scorer import is reported, not swallowed."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ProjectImportWarningTests(unittest.TestCase):
    def test_broken_scorer_import_is_reported(self) -> None:
        from meta_agent import config

        tmp = Path(tempfile.mkdtemp(prefix="projimp_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        bench = tmp / "projects" / "broken" / "benchmark"
        bench.mkdir(parents=True)
        (bench / "scorer.py").write_text("import adapter_that_does_not_exist\n")
        buf = io.StringIO()
        with mock.patch.object(config, "REPO_ROOT", tmp), contextlib.redirect_stdout(buf):
            config._load_project_components("broken")
        self.assertIn("WARNING: importing", buf.getvalue())
        self.assertIn("adapter_that_does_not_exist", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

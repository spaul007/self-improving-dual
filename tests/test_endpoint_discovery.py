"""Tests for model_server.health endpoint-discovery logic.

The helper has to handle three states without ever touching the network:
  * Discovery file missing → no endpoint.
  * Discovery file present + SLURM job alive (RUNNING/PENDING/…) → endpoint.
  * Discovery file present + SLURM job gone → no endpoint (stale file).

We don't actually shell out to squeue — the tests monkey-patch
``subprocess.run`` to return a canned state.

Run from the repo root:
    PYTHONPATH=. python -m unittest tests.test_endpoint_discovery
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util

# model_server isn't a Python package (no __init__.py — it's a script
# directory), so import health.py via spec_from_file_location.
_HEALTH_PATH = REPO_ROOT / "model_server" / "health.py"
_spec = importlib.util.spec_from_file_location("model_server_health", _HEALTH_PATH)
health = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(health)  # type: ignore[union-attr]


def _fake_run(stdout: str, returncode: int = 0):
    """Build a subprocess.run replacement that returns canned output."""

    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(
            stdout=stdout, stderr="", returncode=returncode
        )

    return _run


def _write_discovery(path: Path, **overrides) -> None:
    payload = {
        "host": "compute-node-07",
        "port": 8000,
        "model": "openai/gpt-oss-120b",
        "slurm_job_id": "12345",
        "started_at": "2026-05-12T14:00:00Z",
        "pid": 31415,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ResolveEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="endpoint_discovery_test_"))
        self.discovery = self.tmp / "endpoint.json"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(health.resolve_endpoint(self.discovery))

    def test_alive_job_yields_base_url(self) -> None:
        _write_discovery(self.discovery)
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")):
            self.assertEqual(
                health.resolve_endpoint(self.discovery),
                "http://compute-node-07:8000/v1",
            )

    def test_pending_job_treated_as_alive(self) -> None:
        # The server isn't yet listening, but the YAML config flow only
        # asks "is the job there at all". The caller decides whether to
        # actually retry.
        _write_discovery(self.discovery)
        with mock.patch.object(subprocess, "run", _fake_run("PENDING\n")):
            self.assertEqual(
                health.resolve_endpoint(self.discovery),
                "http://compute-node-07:8000/v1",
            )

    def test_completed_job_yields_none(self) -> None:
        # Stale file: SLURM job has finished but the in-job EXIT trap
        # didn't run (e.g. OOM-killed before the cleanup).
        _write_discovery(self.discovery)
        # squeue prints nothing for a completed job and returns 1.
        with mock.patch.object(subprocess, "run", _fake_run("", returncode=1)):
            self.assertIsNone(health.resolve_endpoint(self.discovery))

    def test_local_run_skips_squeue(self) -> None:
        # When the discovery file was written outside SLURM (e.g.
        # someone ran launch.sh by hand for debugging), slurm_job_id is
        # ``local-<pid>``. squeue isn't queried — assume alive.
        _write_discovery(self.discovery, slurm_job_id="local-99999")
        # Even with a fake squeue that would say DEAD, this short-circuits.
        with mock.patch.object(subprocess, "run", _fake_run("", returncode=1)):
            self.assertEqual(
                health.resolve_endpoint(self.discovery),
                "http://compute-node-07:8000/v1",
            )

    def test_squeue_missing_means_stale(self) -> None:
        # When squeue itself doesn't exist (e.g. invoking from a laptop
        # with the discovery file mounted from a share), we conservatively
        # report stale rather than fabricating a live URL.
        _write_discovery(self.discovery)

        def _missing(*_args, **_kwargs):
            raise FileNotFoundError("squeue not on PATH")

        with mock.patch.object(subprocess, "run", _missing):
            self.assertIsNone(health.resolve_endpoint(self.discovery))

    def test_corrupt_file_returns_none(self) -> None:
        self.discovery.write_text("not valid json", encoding="utf-8")
        self.assertIsNone(health.resolve_endpoint(self.discovery))

    def test_missing_host_port_returns_none(self) -> None:
        _write_discovery(self.discovery, host="")
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")):
            self.assertIsNone(health.resolve_endpoint(self.discovery))


class PerNameDiscoveryTests(unittest.TestCase):
    """Per-model discovery file lookup (concurrent-server case)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="per_name_test_"))
        # Redirect the default server root at the tmpdir for the
        # duration of the test. health.py reads MODEL_SERVER_ROOT once
        # at import time so we have to patch the module-level constant.
        self._orig_root = health.DEFAULT_SERVER_ROOT
        health.DEFAULT_SERVER_ROOT = self.tmp

    def tearDown(self) -> None:
        health.DEFAULT_SERVER_ROOT = self._orig_root
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_discovery_path_for_name(self) -> None:
        self.assertEqual(
            health.discovery_path_for("gpt_oss_120b"),
            self.tmp / "endpoint_gpt_oss_120b.json",
        )

    def test_discovery_path_for_none_keeps_default(self) -> None:
        # Pre-existing single-slot path stays as-is (governed by
        # DEFAULT_DISCOVERY_PATH, not the redirected server root).
        self.assertEqual(
            health.discovery_path_for(None),
            health.DEFAULT_DISCOVERY_PATH,
        )

    def test_resolve_endpoint_by_name(self) -> None:
        _write_discovery(self.tmp / "endpoint_qwen3_5_122b_a10b.json")
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")):
            self.assertEqual(
                health.resolve_endpoint(name="qwen3_5_122b_a10b"),
                "http://compute-node-07:8000/v1",
            )

    def test_resolve_endpoint_by_name_missing_returns_none(self) -> None:
        # No file for this name → None, even if other names have files.
        _write_discovery(self.tmp / "endpoint_gpt_oss_120b.json")
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")):
            self.assertIsNone(
                health.resolve_endpoint(name="qwen3_5_122b_a10b")
            )

    def test_explicit_path_overrides_name(self) -> None:
        # When both name and path are passed, path wins (implementation
        # checks discovery_path first). Useful for tests + adhoc paths.
        custom = self.tmp / "custom.json"
        _write_discovery(custom)
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")):
            self.assertEqual(
                health.resolve_endpoint(custom, name="ignored"),
                "http://compute-node-07:8000/v1",
            )


class MainCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="endpoint_main_test_"))
        self.discovery = self.tmp / "endpoint.json"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_print_base_url_alive(self) -> None:
        _write_discovery(self.discovery)
        with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")), \
             mock.patch.object(sys, "stdout") as fake_stdout:
            rc = health.main([
                "--print-base-url",
                "--path",
                str(self.discovery),
            ])
        self.assertEqual(rc, 0)
        # The single print() emits the URL plus a newline.
        written = "".join(c.args[0] for c in fake_stdout.write.call_args_list)
        self.assertIn("http://compute-node-07:8000/v1", written)

    def test_print_base_url_missing_exits_2(self) -> None:
        rc = health.main([
            "--print-base-url",
            "--path",
            str(self.discovery),
        ])
        self.assertEqual(rc, 2)

    def test_print_base_url_stale_exits_2(self) -> None:
        _write_discovery(self.discovery)
        with mock.patch.object(subprocess, "run", _fake_run("", returncode=1)):
            rc = health.main([
                "--print-base-url",
                "--path",
                str(self.discovery),
            ])
        self.assertEqual(rc, 2)

    def test_name_mode_prints_url(self) -> None:
        # --name <foo> should resolve to <server_root>/endpoint_<foo>.json.
        # Redirect the server root at our tmpdir so the test is hermetic.
        named_file = self.tmp / "endpoint_qwen.json"
        _write_discovery(named_file)
        orig = health.DEFAULT_SERVER_ROOT
        health.DEFAULT_SERVER_ROOT = self.tmp
        try:
            with mock.patch.object(subprocess, "run", _fake_run("RUNNING\n")), \
                 mock.patch.object(sys, "stdout") as fake_stdout:
                rc = health.main(["--print-base-url", "--name", "qwen"])
            self.assertEqual(rc, 0)
            written = "".join(c.args[0] for c in fake_stdout.write.call_args_list)
            self.assertIn("http://compute-node-07:8000/v1", written)
        finally:
            health.DEFAULT_SERVER_ROOT = orig

    def test_name_and_path_are_mutually_exclusive(self) -> None:
        # argparse error exits non-zero via SystemExit; we catch it.
        with self.assertRaises(SystemExit):
            health.main([
                "--name", "x",
                "--path", str(self.discovery),
            ])


if __name__ == "__main__":
    unittest.main()

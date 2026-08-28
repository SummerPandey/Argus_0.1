"""Behavior tests for jetson-performance.sh, run against fake nvpmodel /
jetson_clocks binaries and a fake nvpmodel.conf - no Jetson required.

What must never regress: the script hardcodes no mode id (the MAXN-class
mode differs per Jetson model and some devices have none), --check is
quiet unless this device really has a faster mode than the current one
(so run_nanoowl.sh's advisory never nags a board already at its best or
one with no better mode to offer), --max picks the device's own
MAXN-class mode and locks clocks, --set validates the id against the
device's table, and on a machine without nvpmodel everything is a clean
no-op.
"""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "jetson-performance.sh"

FAKE_NVPMODEL = """#!/bin/bash
case "${1:-}" in
  -q)
    echo "NV Power Mode: ${FAKE_MODE_NAME:-15W}"
    echo "${FAKE_MODE_ID:-0}"
    ;;
  -m)
    echo "nvpmodel -m $2" >> "$FAKE_LOG"
    ;;
esac
"""

FAKE_JETSON_CLOCKS = """#!/bin/bash
echo "jetson_clocks $*" >> "$FAKE_LOG"
"""

CONF_WITH_MAXN = """\
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=7W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
"""

CONF_WITHOUT_MAXN = """\
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=7W >
"""


class JetsonPerformanceScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bin_dir = Path(self.tmp.name) / "bin"
        self.bin_dir.mkdir()
        self.log = Path(self.tmp.name) / "calls.log"
        self.log.touch()
        self.conf = Path(self.tmp.name) / "nvpmodel.conf"
        self.conf.write_text(CONF_WITH_MAXN)
        self._write_tool("nvpmodel", FAKE_NVPMODEL)
        self._write_tool("jetson_clocks", FAKE_JETSON_CLOCKS)

    def _write_tool(self, name, content):
        tool = self.bin_dir / name
        tool.write_text(content)
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)

    def _run(self, *args, mode_name="15W", mode_id="0", jetson=True):
        env = dict(os.environ)
        if jetson:
            env["PATH"] = f"{self.bin_dir}:{env['PATH']}"
        env["NVPMODEL_CONF"] = str(self.conf)
        env["FAKE_MODE_NAME"] = mode_name
        env["FAKE_MODE_ID"] = mode_id
        env["FAKE_LOG"] = str(self.log)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True, text=True, env=env, timeout=30,
        )

    def _calls(self):
        return self.log.read_text().splitlines()

    def test_non_jetson_machine_is_a_clean_noop(self):
        # Without the fake bin dir on PATH there is no nvpmodel (this dev
        # machine is not a Jetson), for any action.
        result = self._run(jetson=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("not a Jetson", result.stdout)
        result = self._run("--check", jetson=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_check_hints_when_a_faster_mode_exists(self):
        result = self._run("--check", mode_name="15W", mode_id="0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Power mode is 15W", result.stdout)
        self.assertIn("--max", result.stdout)
        self.assertIn("MAXN_SUPER", result.stdout)

    def test_check_is_silent_at_the_best_mode(self):
        result = self._run("--check", mode_name="MAXN_SUPER", mode_id="2")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_check_is_silent_when_the_device_has_no_maxn(self):
        self.conf.write_text(CONF_WITHOUT_MAXN)
        result = self._run("--check", mode_name="15W", mode_id="0")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_report_lists_the_devices_own_modes(self):
        result = self._run()
        self.assertEqual(result.returncode, 0)
        self.assertIn("id 0 15W", result.stdout)
        self.assertIn("id 2 MAXN_SUPER", result.stdout)
        self.assertIn("--max", result.stdout)

    @unittest.skipUnless(os.geteuid() == 0, "--max requires root")
    def test_max_picks_this_devices_maxn_mode_and_locks_clocks(self):
        result = self._run("--max", mode_name="15W", mode_id="0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nvpmodel -m 2", self._calls())
        self.assertTrue(any(c.startswith("jetson_clocks") for c in self._calls()))
        self.assertIn("MAXN_SUPER", result.stdout)
        self.assertIn("reboot", result.stdout)  # jetson_clocks caveat stated

    @unittest.skipUnless(os.geteuid() == 0, "--max requires root")
    def test_max_refuses_to_guess_without_a_maxn_mode(self):
        self.conf.write_text(CONF_WITHOUT_MAXN)
        result = self._run("--max")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--set ID", result.stdout)
        self.assertEqual([], [c for c in self._calls() if c.startswith("nvpmodel")])

    @unittest.skipUnless(os.geteuid() == 0, "--set requires root")
    def test_set_validates_the_id_against_the_devices_table(self):
        result = self._run("--set", "9")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([], [c for c in self._calls() if c.startswith("nvpmodel")])
        result = self._run("--set", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nvpmodel -m 1", self._calls())

    def test_unknown_action_shows_usage_and_fails(self):
        result = self._run("--bogus")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

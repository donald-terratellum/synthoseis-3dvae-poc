import os
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "scripts" / "geoaware_classifier_suite.sh"


def _help_options(script):
    out = subprocess.run([sys.executable, str(REPO / script), "--help"], cwd=REPO, capture_output=True, text=True, check=True)
    return set(re.findall(r"--[A-Za-z0-9_]+", out.stdout))


class TestClassifierSuiteScript(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SUITE)], check=True)

    def test_every_flag_exists_in_target_cli(self):
        env = dict(os.environ, PRINT_COMMANDS="1", PYTHON=sys.executable, BENCH_EPOCHS="10 20")
        out = subprocess.run(["bash", str(SUITE), "all"], cwd=REPO, env=env, capture_output=True, text=True, check=True)
        commands = [shlex.split(line) for line in out.stdout.splitlines() if line.strip()]
        scripts = [next(t for t in cmd if t.endswith(".py")) for cmd in commands]
        self.assertEqual(scripts, [
            "scripts/sample_patches.py", "scripts/sample_patches.py", "scripts/train.py",
            "scripts/evaluate_geology_benchmark.py", "scripts/evaluate_geology_benchmark.py",
        ])
        known = {script: _help_options(script) for script in set(scripts)}
        for script, cmd in zip(scripts, commands):
            flags = {t for t in cmd if t.startswith("--")}
            self.assertEqual(flags - known[script], set(), f"unknown flags for {script}")

    def test_unknown_stage_fails(self):
        env = dict(os.environ, PRINT_COMMANDS="1")
        result = subprocess.run(["bash", str(SUITE), "bogus"], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_missing_inputs_fail_fast(self):
        env = dict(os.environ, TRAIN_DATA="/nonexistent/train.zarr")
        result = subprocess.run(["bash", str(SUITE), "train"], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing required input", result.stderr)


if __name__ == "__main__":
    unittest.main()

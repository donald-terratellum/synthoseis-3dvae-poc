import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
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
        self.assertEqual(scripts, ["scripts/sample_patches.py"] * 3 + ["scripts/train.py"] * 4 + ["scripts/evaluate_geology_benchmark.py"] * 8)
        known = {script: _help_options(script) for script in set(scripts)}
        for script, cmd in zip(scripts, commands):
            flags = {t for t in cmd if t.startswith("--")}
            self.assertEqual(flags - known[script], set(), f"unknown flags for {script}")

        train_cmds = [cmd for s, cmd in zip(scripts, commands) if s == "scripts/train.py"]
        out_dirs = [cmd[cmd.index("--out_dir") + 1] for cmd in train_cmds]
        self.assertEqual(out_dirs, [f"checkpoints/geoaware_v4_{r}" for r in ("r1ctrl", "r1", "r2", "r3")])
        data = [cmd[cmd.index("--data") + 1] for cmd in train_cmds]
        self.assertTrue(data[0].endswith("geoscore_noval_32-32-64.zarr"))
        self.assertTrue(all(d.endswith("anchored_32-32-64.zarr") for d in data[1:]))
        # One primary variable per run: classifier from r2, presence strata + quotas only in r3.
        self.assertEqual(["--geology_classifier" in c for c in train_cmds], [False, False, True, True])
        self.assertEqual(["--geology_strata_source" in c for c in train_cmds], [False, False, False, True])
        bench = [cmd for s, cmd in zip(scripts, commands) if s.endswith("evaluate_geology_benchmark.py")]
        self.assertEqual(["--classifier_data" in c for c in bench], [False] * 4 + [True] * 4)

    def test_runs_subset(self):
        env = dict(os.environ, PRINT_COMMANDS="1", RUNS="r2", BENCH_EPOCHS="20")
        out = subprocess.run(["bash", str(SUITE), "train"], cwd=REPO, env=env, capture_output=True, text=True, check=True)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("geoaware_v4_r2", lines[0])
        env["RUNS"] = "bogus"
        result = subprocess.run(["bash", str(SUITE), "train"], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_unknown_stage_fails(self):
        env = dict(os.environ, PRINT_COMMANDS="1")
        result = subprocess.run(["bash", str(SUITE), "bogus"], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_completed_stores_are_skipped_and_partial_ones_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src" / "validation").mkdir(parents=True)
            stores = {name: tmp / f"{name}.zarr" for name in ("train", "ctrl", "val")}
            for path in stores.values():
                path.mkdir()
                (path / "zarr.json").write_text(json.dumps({"attributes": {"n_written": 10}}))
            env = dict(
                os.environ, SOURCE=str(tmp / "src"), PYTHON="false", LOG_DIR=str(tmp / "logs"),
                TRAIN_DATA=str(stores["train"]), CTRL_DATA=str(stores["ctrl"]), VAL_UNIFORM=str(stores["val"]),
            )
            done = subprocess.run(["bash", str(SUITE), "sample"], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stderr.count("Skipping (complete)"), 3)
            (stores["ctrl"] / "zarr.json").write_text(json.dumps({"attributes": {}}))
            partial = subprocess.run(["bash", str(SUITE), "sample"], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertNotEqual(partial.returncode, 0)
            self.assertFalse(stores["ctrl"].exists())

    def test_missing_inputs_fail_fast(self):
        env = dict(os.environ, TRAIN_DATA="/nonexistent/train.zarr")
        result = subprocess.run(["bash", str(SUITE), "train"], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing required input", result.stderr)


if __name__ == "__main__":
    unittest.main()

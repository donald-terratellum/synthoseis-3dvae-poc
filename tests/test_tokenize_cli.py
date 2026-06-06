import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZE_SCRIPT = REPO_ROOT / "scripts" / "tokenize.py"


class TokenizeCliTests(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run(
            [sys.executable, str(TOKENIZE_SCRIPT), *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_main_help(self):
        result = self.run_cmd("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("build-token", result.stdout)
        self.assertIn("search-volume", result.stdout)
        self.assertIn("ui", result.stdout)

    def test_subcommand_help(self):
        for subcommand in ("build-token", "search-volume", "ui"):
            result = self.run_cmd(subcommand, "--help")
            self.assertEqual(result.returncode, 0, msg=f"failed help for {subcommand}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()

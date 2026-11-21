import contextlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import streamed_overlap_runner as runner
from test_streamed_overlap_audit import make_source
from scripts.streamed_overlap_audit import DEFAULT_SEISMIC_KEY


FIXTURE_SERVER = """
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument('--logdir', required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', type=int, required=True)
a = p.parse_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'logdir': a.logdir}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
print('fixture server starting', flush=True)
HTTPServer((a.host, a.port), Handler).serve_forever()
"""


class TensorBoardLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.suite = self.root / "suite with spaces"
        self.logs = self.root / "logs"
        self.suite.mkdir()
        self.logs.mkdir()
        self.server = self.root / "fixture_server.py"
        self.server.write_text(FIXTURE_SERVER)
        self.children = []
        real_popen = subprocess.Popen

        def capture_child(command, *args, **kwargs):
            child = real_popen(command, *args, **kwargs)
            if str(self.server) in command:
                self.children.append(child)
            return child

        self.addCleanup(self.cleanup_children)
        self.popen_patch = mock.patch.object(runner.subprocess, "Popen", side_effect=capture_child)
        self.popen_patch.start()
        self.addCleanup(self.popen_patch.stop)
        self.command_patch = mock.patch.object(
            runner, "tensorboard_command",
            side_effect=lambda suite, port: [
                sys.executable, "-u", str(self.server),
                "--logdir", str(suite), "--host", runner.HOST, "--port", str(port),
            ],
        )
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)

    @staticmethod
    def free_port():
        with socket.socket() as sock:
            sock.bind((runner.HOST, 0))
            return sock.getsockname()[1]

    def test_start_reuse_owned_stop_and_restart_preserve_history(self):
        port = self.free_port()
        owner = runner.start_tensorboard(self.suite, self.logs, port=port, timeout=5)
        self.assertTrue(runner.verify_owner(owner, self.suite))
        self.assertTrue(runner.ready(owner))
        reused = runner.start_tensorboard(self.suite, self.logs, port=port, timeout=5)
        self.assertEqual(reused["pid"], owner["pid"])
        self.assertEqual(len(self.children), 1)
        runner.stop_tensorboard(self.suite, self.logs)
        self.children[0].wait(timeout=5)
        first_log = (self.logs / "tensorboard.log").read_bytes()
        restarted = runner.start_tensorboard(self.suite, self.logs, port=port, timeout=5)
        self.assertNotEqual(restarted["pid"], owner["pid"])
        self.assertTrue((self.logs / "tensorboard.log").read_bytes().startswith(first_log))
        events = [json.loads(row)["event"] for row in
                  (self.logs / "tensorboard_history.jsonl").read_text().splitlines()]
        self.assertEqual(events.count("ready"), 2)
        self.assertIn("stopped", events)
        runner.stop_tensorboard(self.suite, self.logs)

    def test_occupied_port_is_not_appropriated(self):
        with socket.socket() as sock:
            sock.bind((runner.HOST, 0))
            sock.listen()
            port = sock.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                runner.start_tensorboard(self.suite, self.logs, port=port, timeout=1)
            self.assertEqual(len(self.children), 0)
            self.assertFalse((self.logs / "tensorboard.pid").exists())
            self.assertEqual(sock.getsockname()[1], port)

    def test_changed_pid_identity_and_wrong_suite_are_never_signaled(self):
        owner = runner.start_tensorboard(self.suite, self.logs, port=self.free_port(), timeout=5)
        altered = {**owner, "identity": {**owner["identity"], "started": "different process"}}
        runner.save_json(self.logs / "tensorboard_owner.json", altered, replace=True)
        with mock.patch.object(runner.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                runner.stop_tensorboard(self.suite, self.logs)
            kill.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "different suite"):
            runner.verify_owner(owner, self.root)
        self.assertTrue(runner.ready(owner))
        runner.save_json(self.logs / "tensorboard_owner.json", owner, replace=True)
        runner.stop_tensorboard(self.suite, self.logs)

    def test_http_endpoint_mismatch_cleans_up_only_new_child(self):
        with mock.patch.object(runner, "ready", side_effect=RuntimeError("wrong endpoint")):
            with self.assertRaisesRegex(RuntimeError, "wrong endpoint"):
                runner.start_tensorboard(self.suite, self.logs, port=self.free_port(), timeout=5)
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())

    def test_readiness_timeout_and_log_write_failure_propagate(self):
        with mock.patch.object(runner, "ready", return_value=False):
            with self.assertRaises(TimeoutError):
                runner.start_tensorboard(self.suite, self.logs, port=self.free_port(), timeout=.2)
        self.assertIsNotNone(self.children[0].poll())
        with mock.patch.object(runner, "service_event", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                runner.start_tensorboard(self.suite, self.logs, port=self.free_port(), timeout=5)
        self.assertIsNotNone(self.children[-1].poll())

    def test_pid_file_disagreement_is_rejected(self):
        owner = runner.start_tensorboard(self.suite, self.logs, port=self.free_port(), timeout=5)
        (self.logs / "tensorboard.pid").write_text(f"{owner['pid'] + 1}\n")
        with self.assertRaisesRegex(RuntimeError, "records disagree"):
            runner.load_owner(self.logs)
        (self.logs / "tensorboard.pid").write_text(f"{owner['pid']}\n")
        runner.stop_tensorboard(self.suite, self.logs)

    def test_service_lifecycle_lock_is_exclusive(self):
        with runner.service_lock(self.logs):
            with self.assertRaisesRegex(RuntimeError, "owns this run's lock"):
                with runner.service_lock(self.logs):
                    self.fail("Concurrent lifecycle action acquired an existing lock.")
        with runner.service_lock(self.logs):
            pass


class AuditAttemptTests(unittest.TestCase):
    def test_real_fixture_attempt_records_hashes_and_never_starts_tensorboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            make_source(sources)
            make_source(sources / "validation")
            config = root / "config.json"
            config.write_text(json.dumps({
                "source_root": str(sources),
                "seismic_key": DEFAULT_SEISMIC_KEY,
                "sample_sources_per_split": 1,
                "label_z_offset_hypothesis": 1,
                "seed": 20260926,
            }))
            logs = root / "logs"
            logs.mkdir()
            with mock.patch.object(runner, "start_tensorboard") as server:
                code = runner.audit_attempt(config, root / "output", logs, "fixture")
                server.assert_not_called()
            self.assertEqual(code, 0)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "audit_complete_target_approval_pending")
            self.assertFalse(manifest["production_training_allowed"])
            self.assertEqual(manifest["config_sha256"], runner.file_hash(config))
            self.assertEqual(manifest["artifact_sha256"]["target_contract.json"],
                             runner.file_hash(root / "output/target_contract.json"))
            self.assertIn("tensorboard", manifest["versions"])
            with self.assertRaises(FileExistsError):
                runner.audit_attempt(config, root / "output", logs, "fixture")

    def test_preflight_failure_and_child_failure_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            config = root / "config.json"
            config.write_text("invalid json")
            with self.assertRaises(json.JSONDecodeError):
                runner.audit_attempt(config, root / "output", logs, "bad-config")
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertNotEqual(manifest["returncode"], 0)
            config.write_text("{}")
            logs2 = root / "logs2"
            logs2.mkdir()
            child = mock.Mock()
            child.wait.return_value = 7
            with mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                    mock.patch.object(runner.subprocess, "check_output",
                                      side_effect=["fixture", b"fixture", ""]), \
                    mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
                code = runner.audit_attempt(config, root / "output", logs2, "child-failure")
            self.assertEqual(code, 7)
            manifest = json.loads((logs2 / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["returncode"], 7)

    def test_interruption_preserves_attempt_status_and_terminates_only_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            child = mock.Mock()
            child.wait.side_effect = [SystemExit(143), 0]
            with mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                    mock.patch.object(runner.subprocess, "check_output",
                                      side_effect=["fixture", b"fixture", ""]), \
                    mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
                with self.assertRaises(SystemExit) as error:
                    runner.audit_attempt(config, root / "output", logs, "interrupted")
            self.assertEqual(error.exception.code, 143)
            child.send_signal.assert_called_once_with(runner.signal.SIGTERM)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["returncode"], 143)


if __name__ == "__main__":
    unittest.main()

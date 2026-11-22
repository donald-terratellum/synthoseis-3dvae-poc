import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import streamed_overlap_runner as runner
from scripts import streamed_overlap_integration_smoke as integration_smoke
from scripts import streamed_overlap_evaluate_child as evaluate_child
from scripts.streamed_overlap_smoke import _validate_config
from test_streamed_overlap_audit import make_source
from scripts.streamed_overlap_audit import DEFAULT_SEISMIC_KEY
from scripts.streamed_overlap_audit import content_hash


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


class EvaluationAttemptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.suite = self.root / "suite"
        self.suite.mkdir()
        self.record = {
            "schema": "streamed-suite-inputs-v1", "suite_dir": str(self.suite),
            "manifest_path": str(self.root / "manifest"), "source_root": str(self.root / "source"),
            "matching_config": {}, "epoch_validation": {}, "search_manifest": {}, "diagnostic_manifest": {},
            "formal_config": {}, "guardrail_manifest": {}, "guardrail_config": {},
            "historical": {
                "report_path": str(self.root / "report"), "report_sha256": "a" * 64,
                "checkpoint_path": str(self.root / "checkpoint"), "checkpoint_sha256": "b" * 64,
                "checkpoint_label": "fixture",
            },
            "input_hashes": {"source": "a" * 64, "target": "b" * 64, "validation_manifest": "c" * 64},
        }
        self.config_hash = content_hash(self.record)
        self.path = self.root / "record.json"
        self.path.write_text(json.dumps(self.record))
        (self.suite / "resolved_suite_inputs.json").write_text(json.dumps(self.record))
        self.child = mock.Mock()
        self.child.wait.return_value = 0

    def write_result(self, command):
        self.assertIn("scripts.streamed_overlap_evaluate_child", command)
        attempt = command[command.index("--attempt-id") + 1]
        self.assertEqual(command[command.index("--mps-memory-fraction") + 1], "0.4")
        final = {}
        for arm in ("S1", "S2", "S3"):
            directory = self.suite / arm
            directory.mkdir()
            checkpoint = directory / "last.pt"
            checkpoint.write_bytes(f"{arm} checkpoint".encode())
            final[arm] = {"arm": arm, "config_sha256": self.config_hash,
                          "checkpoint_sha256": runner.file_hash(checkpoint)}
        cosine = {"sources": {f"{arm}_final": {"checkpoint": final[arm]} for arm in final}}
        head = {"checks": {"final": {"source": {"checkpoint": final["S3"]}}}}
        hashes = {}
        for key, report in (("mu_cosine_comparisons", cosine), ("pair_head_metric_checks", head)):
            report["sha256"] = content_hash(report)
            hashes[key] = report["sha256"]
            (self.suite / f"{key}.json").write_text(json.dumps(report))
        receipt = {
            "schema": "streamed-evaluation-child-result-v1", "status": "REPORTS_COMPUTED_NOT_ADOPTION",
            "suite_config_sha256": self.config_hash, "suite_dir": str(self.suite), "attempt_id": attempt,
            "mps_memory_fraction": 0.4, "final_checkpoints": final, "report_sha256": hashes,
        }
        Path(command[command.index("--receipt") + 1]).write_text(json.dumps(receipt))
        return self.child

    def attempt(self):
        return runner.evaluation_attempt(
            self.path, self.suite, self.logs, "evaluation", config_sha256=self.config_hash,
            mps_memory_fraction=0.4, min_suite_free_bytes=1, min_log_free_bytes=1,
        )

    def test_completed_child_requires_unchanged_artifacts_and_synced_log(self):
        with mock.patch.object(runner.subprocess, "Popen", side_effect=self.write_result), \
             mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]), \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
            self.assertEqual(self.attempt(), 0)
        manifest_path = self.logs / "run_manifest.json"
        self.assertEqual(json.loads(manifest_path.read_text())["status"], "evaluation_child_complete_log_unconfirmed")
        with self.assertRaisesRegex(ValueError, "completed log"):
            runner.confirm_evaluation_log(self.logs)
        (self.logs / "training.log").write_text("captured fixture log")
        report = self.suite / "pair_head_metric_checks.json"
        original = report.read_text()
        report.write_text("{}")
        with self.assertRaisesRegex(ValueError, "report is missing or changed"):
            runner.confirm_evaluation_log(self.logs)
        report.write_text(original)
        self.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "inputs or receipt"):
            runner.confirm_evaluation_log(self.logs)
        self.path.write_text(json.dumps(self.record))
        receipt = self.logs / "evaluation_receipt.json"
        original_receipt = receipt.read_text()
        altered = json.loads(original_receipt)
        altered["attempt_id"] = "different-attempt"
        receipt.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "receipt does not match"):
            runner.confirm_evaluation_log(self.logs)
        receipt.write_text(original_receipt)
        checkpoint = self.suite / "S3" / "last.pt"
        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
            runner.confirm_evaluation_log(self.logs)
        checkpoint.write_bytes(original_checkpoint)
        with mock.patch.object(runner.os, "fsync", side_effect=OSError("log sync failed")):
            with self.assertRaisesRegex(OSError, "log sync failed"):
                runner.confirm_evaluation_log(self.logs)
        self.assertEqual(json.loads(manifest_path.read_text())["status"], "evaluation_child_complete_log_unconfirmed")
        runner.confirm_evaluation_log(self.logs)
        final = json.loads(manifest_path.read_text())
        self.assertEqual(final["status"], "evaluation_complete_training_not_authorized")
        self.assertFalse(final["production_training_allowed"])
        self.assertGreaterEqual(final["free_space_preflight"]["suite"]["available_bytes"], 1)
        self.assertGreaterEqual(final["free_space_preflight"]["log"]["available_bytes"], 1)
        self.assertEqual(final["receipt_sha256"], runner.file_hash(self.logs / "evaluation_receipt.json"))
        with self.assertRaises(FileExistsError):
            self.attempt()

    def test_invalid_inputs_missing_receipt_and_child_failure_are_not_success(self):
        self.path.write_text("{}")
        with mock.patch.object(runner.subprocess, "Popen") as start, \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
            with self.assertRaisesRegex(ValueError, "pinned configuration hash"):
                self.attempt()
            start.assert_not_called()
        self.assertFalse((self.logs / "run_manifest.json").exists())
        self.path.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "logs must be separate"):
            runner.evaluation_attempt(
                self.path, self.suite, self.suite, "evaluation", config_sha256=self.config_hash,
                mps_memory_fraction=0.4, min_suite_free_bytes=1, min_log_free_bytes=1,
            )
        self.assertFalse((self.suite / "run_manifest.json").exists())
        for returncode, error in ((0, "receipt"), (7, None)):
            logs = self.root / f"logs-{returncode}"
            logs.mkdir()
            self.child.wait.return_value = returncode
            with self.subTest(returncode=returncode), \
                 mock.patch.object(runner.subprocess, "Popen", return_value=self.child), \
                 mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]), \
                 mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        runner.evaluation_attempt(
                            self.path, self.suite, logs, "evaluation", config_sha256=self.config_hash,
                            mps_memory_fraction=0.4, min_suite_free_bytes=1, min_log_free_bytes=1,
                        )
                else:
                    self.assertEqual(runner.evaluation_attempt(
                        self.path, self.suite, logs, "evaluation", config_sha256=self.config_hash,
                        mps_memory_fraction=0.4, min_suite_free_bytes=1, min_log_free_bytes=1,
                    ), 7)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertNotEqual(manifest["returncode"], 0)
            self.assertEqual(manifest["child_returncode"], returncode)

    def test_free_space_floors_block_child_and_record_measurements(self):
        for suite_free, log_free, failed in ((99, 200, "suite"), (100, 199, "log")):
            logs = self.root / f"logs-{failed}"
            logs.mkdir()
            with self.subTest(failed=failed), \
                 mock.patch("scripts.streamed_overlap_epoch_runner._filesystem_available_bytes",
                            side_effect=[suite_free, log_free]) as available, \
                 mock.patch.object(runner.subprocess, "Popen") as start, \
                 mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
                with self.assertRaisesRegex(OSError, f"Insufficient free space for evaluation {failed}"):
                    runner.evaluation_attempt(
                        self.path, self.suite, logs, "evaluation", config_sha256=self.config_hash,
                        mps_memory_fraction=0.4, min_suite_free_bytes=100, min_log_free_bytes=200,
                    )
                self.assertEqual(available.call_args_list, [mock.call(self.suite), mock.call(logs)])
                start.assert_not_called()
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["free_space_preflight"]["suite"]["available_bytes"], suite_free)
            self.assertEqual(manifest["free_space_preflight"]["log"]["available_bytes"], log_free)

        with mock.patch.object(runner.subprocess, "Popen") as start:
            for value in (0, -1, True, 1.5):
                with self.subTest(invalid=value), self.assertRaisesRegex(ValueError, "free-space floors"):
                    runner.evaluation_attempt(
                        self.path, self.suite, self.logs, "evaluation", config_sha256=self.config_hash,
                        mps_memory_fraction=0.4, min_suite_free_bytes=value, min_log_free_bytes=1,
                    )
            start.assert_not_called()
        self.assertFalse((self.logs / "run_manifest.json").exists())

    def test_free_space_equal_to_floors_permits_child(self):
        self.child.wait.return_value = 7
        with mock.patch("scripts.streamed_overlap_epoch_runner._filesystem_available_bytes",
                        side_effect=[100, 200]), \
             mock.patch.object(runner.subprocess, "Popen", return_value=self.child) as start, \
             mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]), \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
            self.assertEqual(runner.evaluation_attempt(
                self.path, self.suite, self.logs, "evaluation", config_sha256=self.config_hash,
                mps_memory_fraction=0.4, min_suite_free_bytes=100, min_log_free_bytes=200,
            ), 7)
        start.assert_called_once()

    def test_interruption_records_failure_and_signals_only_child(self):
        self.child.wait.side_effect = [SystemExit(143), 0]
        with mock.patch.object(runner.subprocess, "Popen", return_value=self.child), \
             mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]), \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"):
            with self.assertRaises(SystemExit) as error:
                self.attempt()
        self.assertEqual(error.exception.code, 143)
        self.child.send_signal.assert_called_once_with(runner.signal.SIGTERM)
        manifest = json.loads((self.logs / "run_manifest.json").read_text())
        self.assertEqual((manifest["status"], manifest["returncode"]), ("interrupted", 143))

    def test_child_receipt_is_exclusive_before_evaluation(self):
        receipt = self.logs / "receipt.json"
        args = [
            "child", "--suite-record", str(self.path), "--suite-dir", str(self.suite),
            "--config-sha256", self.config_hash, "--attempt-id", "fixture", "--mps-memory-fraction", "0.4",
            "--receipt", str(receipt),
        ]
        with mock.patch.object(sys, "argv", args), \
             mock.patch.object(evaluate_child, "evaluate", return_value={"status": "fixture"}) as evaluate, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate_child.main()
            self.assertEqual(json.loads(receipt.read_text()), {"status": "fixture"})
            evaluate.reset_mock()
            with self.assertRaisesRegex(ValueError, "new absolute file"):
                evaluate_child.main()
            evaluate.assert_not_called()

    def test_internal_cli_routes_attempt_and_confirmation(self):
        args = [
            "runner", "--action", "evaluate-attempt", "--log-dir", str(self.logs), "--config", str(self.path),
            "--run-dir", str(self.suite), "--run-id", "evaluation", "--config-sha256", self.config_hash,
            "--mps-memory-fraction", "0.4", "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
        ]
        with mock.patch.object(sys, "argv", args), mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner, "evaluation_attempt", return_value=0) as attempt:
            with self.assertRaises(SystemExit) as result:
                runner.main()
            self.assertEqual(result.exception.code, 0)
            attempt.assert_called_once_with(
                self.path, self.suite, self.logs, "evaluation", config_sha256=self.config_hash,
                mps_memory_fraction=0.4, min_suite_free_bytes=100, min_log_free_bytes=200,
            )
        with mock.patch.object(sys, "argv", args[:-4]), mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner, "evaluation_attempt") as attempt, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runner.main()
            self.assertEqual(result.exception.code, 2)
            attempt.assert_not_called()
        with mock.patch.object(sys, "argv", ["runner", "--action", "confirm-evaluation-log",
                                             "--log-dir", str(self.logs)]), \
             mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner, "confirm_evaluation_log") as confirm:
            runner.main()
            confirm.assert_called_once_with(self.logs)

    def test_training_attempt_and_resume_use_pinned_suite_and_confirm_logs(self):
        self.record["matching_config"] = {"run_id": "training", "final_epoch": 2, "batches_per_epoch": 3}
        self.config_hash = content_hash(self.record)
        self.path.write_text(json.dumps(self.record))
        (self.suite / "resolved_suite_inputs.json").unlink()
        self.suite.rmdir()
        options = {
            "config_sha256": self.config_hash, "epochs": 2, "batches_per_epoch": 3,
            "mps_memory_fraction": 0.4, "min_suite_free_bytes": 1, "min_log_free_bytes": 1,
        }
        calls = []

        def execute(record_path, suite_dir, **kwargs):
            self.assertEqual(record_path, self.path)
            self.assertEqual(suite_dir, self.suite)
            calls.append(kwargs)
            if not kwargs["resume"]:
                suite_dir.mkdir()
                (suite_dir / "resolved_suite_inputs.json").write_text(json.dumps(self.record))
            kwargs["on_suite_bound"]()
            final = {}
            for arm in ("S1", "S2", "S3"):
                directory = suite_dir / arm
                directory.mkdir(exist_ok=True)
                checkpoint = directory / "last.pt"
                if not checkpoint.exists():
                    checkpoint.write_bytes(f"{arm} checkpoint".encode())
                final[arm] = {
                    "arm": arm, "config_sha256": self.config_hash, "epoch": 2, "step": 6,
                    "checkpoint_sha256": runner.file_hash(checkpoint),
                }
            reports = {
                "mu_cosine_comparisons": {"sources": {f"{arm}_final": {"checkpoint": final[arm]}
                                                       for arm in final}},
                "pair_head_metric_checks": {"checks": {"final": {"source": {"checkpoint": final["S3"]}}}},
            }
            for key, report in reports.items():
                report["sha256"] = content_hash(report)
                (suite_dir / f"{key}.json").write_text(json.dumps(report))
            result = {arm: {"last_epoch": {"checkpoint": final[arm]}} for arm in final}
            result["S3"].update(reports)
            return result

        with mock.patch("scripts.streamed_overlap_epoch_runner.run_suite_from_record", side_effect=execute), \
             mock.patch("torch.backends.mps.is_available", return_value=True), \
             mock.patch("torch.mps.set_per_process_memory_fraction") as set_limit, \
             mock.patch.object(runner, "start_tensorboard") as start_service, \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"), \
             mock.patch.object(runner.subprocess, "check_output",
                               side_effect=["fixture", b"fixture", "", "fixture", b"fixture", ""]), \
             mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            runner.training_attempt(
                self.path, self.suite, self.logs, self.logs, "training", resume=False, **options,
            )
            manifest = json.loads((self.logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "suite_complete_log_unconfirmed")
            (self.logs / "training.log").write_text("first captured log")
            checkpoint = self.suite / "S3" / "last.pt"
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
                runner.confirm_training_log(self.logs)
            checkpoint.write_bytes(original)
            runner.confirm_training_log(self.logs)
            self.assertEqual(json.loads((self.logs / "run_manifest.json").read_text())["status"],
                             "suite_complete_not_adoption")
            resume_logs = self.root / "resume-logs"
            resume_logs.mkdir()
            runner.training_attempt(
                self.path, self.suite, resume_logs, self.logs, "retry", resume=True, **options,
            )
            (resume_logs / "training.log").write_text("resume captured log")
            runner.confirm_training_log(resume_logs)
            self.assertEqual([call["resume"] for call in calls], [False, True])
            self.assertNotEqual(calls[0]["attempt_id"], calls[1]["attempt_id"])
            self.assertEqual(set_limit.call_count, 2)
            self.assertEqual(start_service.call_args_list, [mock.call(self.suite, self.logs)] * 2)

    def test_training_attempt_rejects_changed_budget_and_insufficient_space(self):
        self.record["matching_config"] = {"run_id": "training", "final_epoch": 2, "batches_per_epoch": 3}
        self.config_hash = content_hash(self.record)
        self.path.write_text(json.dumps(self.record))
        options = {
            "config_sha256": self.config_hash, "epochs": 2, "batches_per_epoch": 3,
            "mps_memory_fraction": 0.4, "min_suite_free_bytes": 100, "min_log_free_bytes": 200,
        }
        (self.suite / "resolved_suite_inputs.json").unlink()
        self.suite.rmdir()
        with mock.patch("scripts.streamed_overlap_epoch_runner.run_suite_from_record") as run:
            with self.assertRaisesRegex(ValueError, "budget"):
                runner.training_attempt(
                    self.path, self.suite, self.logs, self.logs, "training", resume=False,
                    **{**options, "epochs": 3},
                )
            self.assertFalse((self.logs / "run_manifest.json").exists())
            with mock.patch("scripts.streamed_overlap_epoch_runner._filesystem_available_bytes",
                            side_effect=[99, 200]):
                with self.assertRaisesRegex(OSError, "Insufficient free space"):
                    runner.training_attempt(
                        self.path, self.suite, self.logs, self.logs, "training", resume=False, **options,
                    )
            run.assert_not_called()
        self.assertEqual(json.loads((self.logs / "run_manifest.json").read_text())["status"], "failed")
        self.assertFalse(self.suite.exists())

    def test_training_service_failure_never_reports_completion(self):
        self.record["matching_config"] = {"run_id": "training", "final_epoch": 2, "batches_per_epoch": 3}
        self.config_hash = content_hash(self.record)
        self.path.write_text(json.dumps(self.record))
        (self.suite / "resolved_suite_inputs.json").unlink()
        self.suite.rmdir()

        def execute(record_path, suite_dir, **kwargs):
            suite_dir.mkdir()
            (suite_dir / "resolved_suite_inputs.json").write_text(json.dumps(self.record))
            kwargs["on_suite_bound"]()
            self.fail("Training must not start when TensorBoard fails.")

        with mock.patch("scripts.streamed_overlap_epoch_runner.run_suite_from_record", side_effect=execute), \
             mock.patch("torch.backends.mps.is_available", return_value=True), \
             mock.patch("torch.mps.set_per_process_memory_fraction"), \
             mock.patch.object(runner, "start_tensorboard", side_effect=RuntimeError("port occupied")), \
             mock.patch.object(runner.platform, "platform", return_value="fixture-platform"), \
             mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]), \
             mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            with self.assertRaisesRegex(RuntimeError, "port occupied"):
                runner.training_attempt(
                    self.path, self.suite, self.logs, self.logs, "training", config_sha256=self.config_hash,
                    resume=False, epochs=2, batches_per_epoch=3, mps_memory_fraction=0.4,
                    min_suite_free_bytes=1, min_log_free_bytes=1,
                )
        self.assertEqual(json.loads((self.logs / "run_manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.logs / "evaluation_receipt.json").exists())
        self.assertTrue((self.suite / "resolved_suite_inputs.json").is_file())
        self.assertFalse((self.suite / "S1").exists())

    def test_training_cli_requires_approval_and_routes_resume(self):
        args = [
            "runner", "--action", "resume-attempt", "--log-dir", str(self.logs), "--config", str(self.path),
            "--run-dir", str(self.suite), "--run-id", "retry", "--config-sha256", self.config_hash,
            "--mps-memory-fraction", "0.4", "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
            "--epochs", "2", "--batches-per-epoch", "3", "--tensorboard-log-dir", str(self.logs),
        ]
        with mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner, "training_attempt") as attempt:
            with mock.patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)
            attempt.assert_not_called()
            with mock.patch.object(sys, "argv", args + [
                "--acknowledge-training-approval", "--acknowledge-training-resources",
            ]):
                runner.main()
            attempt.assert_called_once_with(
                self.path, self.suite, self.logs, self.logs, "retry", config_sha256=self.config_hash,
                resume=True, epochs=2, batches_per_epoch=3, mps_memory_fraction=0.4,
                min_suite_free_bytes=100, min_log_free_bytes=200,
            )


class AuditAttemptTests(unittest.TestCase):
    def test_incomplete_execution_modes_remain_blocked_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("train", "resume"):
                result = subprocess.run(
                    ["bash", "scripts/streamed_overlap_experiment.sh", "--mode", mode,
                     "--run-id", "blocked-fixture", "--output-root", temporary],
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TRAINING_NOT_READY", result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_training_shell_previews_route_but_blocks_actual_launch(self):
        for mode in ("train", "resume"):
            with self.subTest(mode=mode):
                run_id = f"training-preview-{mode}-fixture"
                args = [
                    "bash", "scripts/streamed_overlap_experiment.sh", "--mode", mode,
                    "--run-id", run_id, "--config", "record.json", "--run-dir", "suite",
                    "--config-sha256", "a" * 64, "--epochs", "2", "--batches-per-epoch", "3",
                    "--mps-memory-fraction", "0.4", "--min-suite-free-bytes", "100",
                    "--min-log-free-bytes", "200", "--acknowledge-training-approval",
                    "--acknowledge-training-resources",
                ]
                owner = run_id
                if mode == "resume":
                    owner = "original-training"
                    args += ["--tensorboard-run-id", owner]
                preview = subprocess.run(args + ["--print-commands"], capture_output=True, text=True, check=False)
                self.assertEqual(preview.returncode, 0, preview.stderr)
                self.assertIn("PREVIEW ONLY", preview.stdout)
                self.assertIn(f"--action {mode}-attempt", preview.stdout)
                self.assertIn("--action confirm-training-log", preview.stdout)
                self.assertIn(f"--tensorboard-log-dir logs/{owner}", preview.stdout)
                self.assertIn("--epochs 2 --batches-per-epoch 3", preview.stdout)
                self.assertIn("--acknowledge-training-approval", preview.stdout)
                self.assertIn("--acknowledge-training-resources", preview.stdout)
                without_acknowledgements = subprocess.run(
                    [arg for arg in args if arg not in (
                        "--acknowledge-training-approval", "--acknowledge-training-resources",
                    )] + ["--print-commands"], capture_output=True, text=True, check=False,
                )
                self.assertEqual(without_acknowledgements.returncode, 0, without_acknowledgements.stderr)
                result = subprocess.run(args, capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TRAINING_NOT_READY", result.stderr)
                self.assertFalse((Path("logs") / run_id).exists())

    def test_training_shell_preview_requires_resume_owner(self):
        args = [
            "bash", "scripts/streamed_overlap_experiment.sh", "--mode", "resume", "--print-commands",
            "--run-id", "training-rejected-fixture", "--config", "record.json", "--run-dir", "suite",
            "--config-sha256", "a" * 64, "--epochs", "2", "--batches-per-epoch", "3",
            "--mps-memory-fraction", "0.4", "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
        ]
        for flags, message in (
            ([], "--tensorboard-run-id"),
            (["--tensorboard-run-id", "../other"], "--tensorboard-run-id"),
        ):
            with self.subTest(flags=flags):
                result = subprocess.run(args + flags, capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse((Path("logs") / "training-rejected-fixture").exists())

    def test_evaluate_shell_routes_and_confirms_only_after_successful_capture(self):
        Path("logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = root / "record.json"
            config.write_text("{}")
            suite = root / "suite"
            suite.mkdir()
            calls = root / "calls"
            python = root / "python-fixture"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('fixture python'); sys.exit(0)\n"
                "action = sys.argv[sys.argv.index('--action') + 1]\n"
                "with open(os.environ['FIXTURE_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "print('stdout ' + action, flush=True)\n"
                "print('stderr ' + action, file=sys.stderr, flush=True)\n"
                "sys.exit(int(os.environ.get('FIXTURE_FAILURE', '0')) if action == 'evaluate-attempt' else 0)\n"
            )
            python.chmod(0o755)
            env = {**os.environ, "PYTHON": str(python), "FIXTURE_CALLS": str(calls)}
            for failure in ("none", "child", "tee"):
                with self.subTest(failure=failure), \
                     tempfile.TemporaryDirectory(prefix="streamed-eval-test-", dir="logs") as log_name:
                    log_dir = Path(log_name)
                    run_id = log_dir.name
                    log_dir.rmdir()
                    args = [
                        "bash", "scripts/streamed_overlap_experiment.sh", "--mode", "evaluate",
                        "--run-id", run_id, "--config", str(config), "--run-dir", str(suite),
                        "--config-sha256", "a" * 64, "--mps-memory-fraction", "0.4",
                        "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
                        "--acknowledge-evaluation-resources",
                    ]
                    calls.unlink(missing_ok=True)
                    attempt_env = dict(env)
                    if failure == "child":
                        attempt_env["FIXTURE_FAILURE"] = "7"
                    if failure == "tee":
                        bin_dir = root / "bin"
                        bin_dir.mkdir(exist_ok=True)
                        tee = bin_dir / "tee"
                        tee.write_text("#!/bin/sh\ncat >/dev/null\nexit 23\n")
                        tee.chmod(0o755)
                        attempt_env["PATH"] = f"{bin_dir}:{env['PATH']}"
                    result = subprocess.run(args + ["--print-commands"], env=attempt_env,
                                            capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--action evaluate-attempt", result.stdout)
                    self.assertIn("--action confirm-evaluation-log", result.stdout)
                    self.assertFalse(log_dir.exists())
                    self.assertFalse(calls.exists())
                    result = subprocess.run(args, env=attempt_env, capture_output=True, text=True, check=False)
                    commands = [json.loads(line) for line in calls.read_text().splitlines()]
                    actions = [command[command.index("--action") + 1] for command in commands]
                    first = commands[0]
                    for flag, value in (("--config", str(config)), ("--run-dir", str(suite)),
                                        ("--config-sha256", "a" * 64), ("--mps-memory-fraction", "0.4"),
                                        ("--min-suite-free-bytes", "100"), ("--min-log-free-bytes", "200")):
                        self.assertEqual(first[first.index(flag) + 1], value)
                    if failure == "none":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(actions, ["evaluate-attempt", "confirm-evaluation-log"])
                        self.assertIn("Evaluation log confirmed", result.stdout)
                        log = (log_dir / "training.log").read_text()
                        self.assertIn("stdout evaluate-attempt", log)
                        self.assertIn("stderr evaluate-attempt", log)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(actions, ["evaluate-attempt"])
                        self.assertNotIn("Evaluation log confirmed", result.stdout)

    def test_evaluate_requires_explicit_resources_before_creating_logs(self):
        args = [
            "bash", "scripts/streamed_overlap_experiment.sh", "--mode", "evaluate",
            "--run-id", "evaluation-rejected-fixture", "--config", "record.json", "--run-dir", "suite",
            "--config-sha256", "a" * 64, "--mps-memory-fraction", "0.4",
            "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
        ]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--acknowledge-evaluation-resources", result.stderr)
        self.assertFalse((Path("logs") / "evaluation-rejected-fixture").exists())

    def test_evaluate_shell_real_runner_rejects_invalid_record(self):
        Path("logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary, \
             tempfile.TemporaryDirectory(prefix="streamed-eval-test-", dir="logs") as log_name:
            root = Path(temporary).resolve()
            config = root / "record.json"
            config.write_text("{}")
            suite = root / "suite"
            suite.mkdir()
            log_dir = Path(log_name)
            run_id = log_dir.name
            log_dir.rmdir()
            result = subprocess.run(
                ["bash", "scripts/streamed_overlap_experiment.sh", "--mode", "evaluate",
                 "--run-id", run_id, "--config", str(config), "--run-dir", str(suite),
                 "--config-sha256", "a" * 64, "--mps-memory-fraction", "0.4",
                 "--min-suite-free-bytes", "100", "--min-log-free-bytes", "200",
                 "--acknowledge-evaluation-resources"],
                env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned configuration hash", (log_dir / "training.log").read_text())
            self.assertFalse((log_dir / "run_manifest.json").exists())
            self.assertEqual(list(suite.iterdir()), [])

    def test_smoke_attempt_requires_no_optimizer_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            output = root / "output"
            child = mock.Mock()
            child.wait.return_value = 0

            def start_child(command):
                self.assertIn("scripts.streamed_overlap_smoke", command)
                output.mkdir()
                (output / "smoke_report.json").write_text(json.dumps({
                    "status": "NO_OPTIMIZER_SMOKE_PASSED", "optimizer_step": False,
                    "unweighted_task_encoder_gradient_l2": {"overlap": 1.0, "coverage": 2.0},
                }))
                return child

            with mock.patch.object(runner.subprocess, "Popen", side_effect=start_child), \
                 mock.patch.object(runner.platform, "platform", return_value="fixture-platform"), \
                 mock.patch.object(runner.subprocess, "check_output",
                                   side_effect=["fixture", b"fixture", ""]):
                self.assertEqual(runner.smoke_attempt(config, output, logs, "fixture"), 0)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "smoke_complete_training_not_authorized")
            self.assertFalse(manifest["production_training_allowed"])
            self.assertEqual(manifest["artifact_sha256"]["smoke_report.json"],
                             runner.file_hash(output / "smoke_report.json"))

    def test_smoke_attempt_rejects_missing_or_invalid_task_gradients(self):
        for gradients in (None, {"overlap": 1.0}, {"overlap": 1.0, "coverage": 0.0},
                          {"overlap": True, "coverage": 1.0},
                          {"overlap": float("inf"), "coverage": 1.0}):
            with self.subTest(gradients=gradients), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                logs = root / "logs"
                logs.mkdir()
                config = root / "config.json"
                config.write_text("{}")
                output = root / "output"
                child = mock.Mock()
                child.wait.return_value = 0

                def start_child(command):
                    output.mkdir()
                    (output / "smoke_report.json").write_text(json.dumps({
                        "status": "NO_OPTIMIZER_SMOKE_PASSED", "optimizer_step": False,
                        "unweighted_task_encoder_gradient_l2": gradients,
                    }))
                    return child

                with mock.patch.object(runner.subprocess, "Popen", side_effect=start_child), \
                     mock.patch.object(runner.platform, "platform", return_value="fixture-platform"), \
                     mock.patch.object(runner.subprocess, "check_output",
                                       side_effect=["fixture", b"fixture", ""]):
                    with self.assertRaisesRegex(ValueError, "downstream encoder gradients"):
                        runner.smoke_attempt(config, output, logs, "fixture")
                self.assertEqual(json.loads((logs / "run_manifest.json").read_text())["status"], "failed")

    def test_prepare_print_only_and_missing_limits_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = [
                "bash", "scripts/streamed_overlap_experiment.sh", "--mode", "prepare",
                "--run-id", "prepare-print-fixture", "--output-root", temporary,
                "--config", "not-read.json", "--mps-memory-fraction", "0.5",
                "--max-wall-seconds", "30", "--print-commands",
            ]
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--action prepare", result.stdout)
            self.assertIn("--action confirm-prepare-log", result.stdout)
            self.assertIn("--mps-memory-fraction 0.5", result.stdout)
            self.assertIn("--max-wall-seconds 30", result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            result = subprocess.run(args[:-3] + ["--print-commands"], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --mps-memory-fraction", result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_prepare_attempt_records_pinned_output_and_propagates_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text('{"settings":"fixture"}')
            output = root / "input-record.json"
            logs = root / "logs"
            logs.mkdir()
            record = {"schema": "streamed-suite-inputs-v1", "fixture": "not a training result"}
            child = mock.Mock()
            child.wait.return_value = 0

            def start_child(command):
                self.assertIn("scripts.streamed_overlap_prepare", command)
                self.assertEqual(command[command.index("--mps-memory-fraction") + 1], "0.5")
                output.write_text(json.dumps(record))
                return child

            with mock.patch.object(runner.subprocess, "Popen", side_effect=start_child), \
                 mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]):
                self.assertEqual(runner.prepare_attempt(
                    config, output, logs, "fixture", mps_memory_fraction=0.5, max_wall_seconds=30,
                ), 0)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "preparation_child_complete_log_unconfirmed")
            self.assertFalse(manifest["production_training_allowed"])
            self.assertEqual(manifest["suite_config_sha256"], content_hash(record))
            self.assertEqual(manifest["artifact_sha256"]["suite_input_record"], runner.file_hash(output))
            log = logs / "training.log"
            log.write_text("fixture combined log")
            output.write_text('{"schema":"changed"}')
            with self.assertRaisesRegex(ValueError, "changed before log confirmation"):
                runner.confirm_prepare_log(logs)
            self.assertEqual(json.loads((logs / "run_manifest.json").read_text())["status"],
                             "preparation_child_complete_log_unconfirmed")
            output.write_text(json.dumps(record))
            runner.confirm_prepare_log(logs)
            self.assertEqual(json.loads((logs / "run_manifest.json").read_text())["status"],
                             "preparation_complete_training_not_authorized")
            with self.assertRaises(FileExistsError):
                runner.prepare_attempt(config, output, logs, "fixture", mps_memory_fraction=0.5, max_wall_seconds=30)

            logs2 = root / "logs2"
            logs2.mkdir()
            child.wait.return_value = 7
            with mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                 mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]):
                self.assertEqual(runner.prepare_attempt(
                    config, root / "failed", logs2, "failure", mps_memory_fraction=0.5, max_wall_seconds=30,
                ), 7)
            failed = json.loads((logs2 / "run_manifest.json").read_text())
            self.assertEqual((failed["status"], failed["returncode"]), ("failed", 7))
            self.assertNotIn("suite_config_sha256", failed)
            logs3 = root / "logs3"
            logs3.mkdir()
            child.wait.return_value = -15
            with mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                 mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]):
                self.assertEqual(runner.prepare_attempt(
                    config, root / "interrupted", logs3, "signal", mps_memory_fraction=0.5, max_wall_seconds=30,
                ), 143)
            failed = json.loads((logs3 / "run_manifest.json").read_text())
            self.assertEqual((failed["status"], failed["child_returncode"], failed["returncode"]),
                             ("failed", -15, 143))

    def test_prepare_attempt_rejects_changed_config_after_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            output = root / "input-record.json"
            child = mock.Mock()
            child.wait.return_value = 0

            def start_child(command):
                config.write_text('{"changed":true}')
                output.write_text('{"schema":"streamed-suite-inputs-v1"}')
                return child

            with mock.patch.object(runner.subprocess, "Popen", side_effect=start_child), \
                 mock.patch.object(runner.subprocess, "check_output", side_effect=["fixture", b"fixture", ""]):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    runner.prepare_attempt(config, output, logs, "fixture", mps_memory_fraction=0.5,
                                           max_wall_seconds=30)
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["returncode"]), ("failed", 1))

    def test_smoke_config_rejects_source_output_and_validation_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources"
            source.mkdir()
            config = {
                "source_root": str(source), "seed": 1, "sparse_probability": 0.5,
                "mps_memory_fraction": 0.5, "max_wall_seconds": 30,
            }
            self.assertEqual(_validate_config(config, root / "output"), source.resolve())
            with mock.patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
                with self.assertRaisesRegex(ValueError, "fallback"):
                    _validate_config(config, root / "output")
            with self.assertRaisesRegex(ValueError, "outside"):
                _validate_config(config, source / "output")
            forbidden = root / "fake_data_validation_v2"
            forbidden.mkdir()
            config["source_root"] = str(forbidden)
            with self.assertRaisesRegex(ValueError, "validation-v2"):
                _validate_config(config, root / "output")

    def test_integration_smoke_rejects_invalid_inputs_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources"
            source.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.touch()
            config = {
                "source_root": str(source), "seed": 1, "sparse_probability": 0.5,
                "mps_memory_fraction": 0.5, "max_wall_seconds": 30,
                "validation_manifest": str(manifest_path), "validation_manifest_sha256": "a" * 64,
                "reconstruction_query_ids": ["eligible"], "encode_batch_size": 4,
                "reconstruction_batch_size": 1,
            }
            with mock.patch.object(integration_smoke, "load_manifest",
                                   return_value={"manifest_sha256": "a" * 64}), \
                 mock.patch.object(integration_smoke, "_eligible_queries",
                                   return_value=[{"id": "eligible"}]), \
                 mock.patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
                self.assertEqual(integration_smoke._config(config, root / "output")[0], source.resolve())
                for changes in (
                    {"validation_manifest_sha256": "b" * 64},
                    {"reconstruction_query_ids": ["missing"]},
                    {"reconstruction_query_ids": ["eligible", "eligible"]},
                    {"encode_batch_size": 0},
                    {"reconstruction_batch_size": True},
                ):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        integration_smoke._config({**config, **changes}, root / "output")
                with self.assertRaisesRegex(ValueError, "validation-v2"):
                    integration_smoke._config(config, root / "fake_data_validation_v2" / "output")
                with mock.patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
                    with self.assertRaisesRegex(ValueError, "fallback"):
                        integration_smoke._config(config, root / "output")

    def test_smoke_child_cannot_claim_success_with_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            output = root / "output"
            child = mock.Mock()
            child.wait.return_value = 0

            def start_child(command):
                del command
                output.mkdir()
                (output / "smoke_report.json").write_text(json.dumps({
                    "status": "NO_OPTIMIZER_SMOKE_PASSED", "optimizer_step": True,
                }))
                return child

            with mock.patch.object(runner.subprocess, "Popen", side_effect=start_child), \
                 mock.patch.object(runner.platform, "platform", return_value="fixture-platform"), \
                 mock.patch.object(runner.subprocess, "check_output",
                                   side_effect=["fixture", b"fixture", ""]):
                with self.assertRaisesRegex(ValueError, "no-optimizer"):
                    runner.smoke_attempt(config, output, logs, "fixture")
            manifest = json.loads((logs / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertNotEqual(manifest["returncode"], 0)

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

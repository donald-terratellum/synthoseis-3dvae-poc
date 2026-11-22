"""Audit, preparation and owned TensorBoard services; not a streamed trainer."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


HOST = "127.0.0.1"
PORT = 6006


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict, *, replace: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp") if replace else path
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_attempt(config: Path, output: Path, log_dir: Path, run_id: str) -> int:
    manifest_path = log_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing an existing attempt manifest: {manifest_path}")
    command = [
        sys.executable, "-u", "-m", "scripts.streamed_overlap_audit",
        "--config", str(config), "--output", str(output),
    ]
    manifest = {
        "schema_version": 1,
        "mode": "audit",
        "run_id": run_id,
        "attempt_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(),
        "status": "preflight",
        "command": command,
        "config_path": str(config.resolve()),
        "output_path": str(output.resolve()),
        "log_path": str((log_dir / "training.log").resolve()),
        "production_training_allowed": False,
        "training_device": None,
        "python": sys.version,
        "platform": platform.platform(),
        "runner_sha256": file_hash(Path(__file__)),
        "versions": {},
    }
    save_json(manifest_path, manifest)
    returncode = 1
    try:
        manifest["config_sha256"] = file_hash(config)
        resolved_config = json.loads(config.read_text())
        if not isinstance(resolved_config, dict):
            raise ValueError("Audit config must be a JSON object.")
        manifest["resolved_config"] = resolved_config
        manifest["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        changes = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
        manifest["git_diff_sha256"] = hashlib.sha256(changes).hexdigest()
        manifest["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        manifest["versions"] = {
            package: importlib.metadata.version(package)
            for package in ("numpy", "zarr", "torch", "tensorboard")
        }
        manifest["status"] = "running"
        save_json(manifest_path, manifest, replace=True)
        print(f"Audit command: {shlex.join(command)}", flush=True)
        # A single foreground process group lets terminal interruption reach both
        # the runner and audit child. TensorBoard has a separate service lifetime.
        child = subprocess.Popen(command)
        try:
            returncode = child.wait()
        except (KeyboardInterrupt, SystemExit) as interruption:
            code = interruption.code if isinstance(interruption, SystemExit) else 130
            returncode = code if isinstance(code, int) else 1
            manifest["status"] = "interrupted"
            child.send_signal(signal.SIGTERM if isinstance(interruption, SystemExit) else signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            raise
        if returncode == 0:
            manifest["artifact_sha256"] = {
                name: file_hash(output / name)
                for name in (
                    "source_manifest.json", "target_contract.json",
                    "resolved_audit_config.json", "audit_report.json",
                )
            }
            manifest["status"] = "audit_complete_target_approval_pending"
        else:
            manifest["status"] = "failed"
        manifest["child_returncode"] = returncode
        if returncode < 0:
            returncode = 128 - returncode
        return returncode
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        returncode = 130
        raise
    finally:
        if manifest["status"] in ("preflight", "running"):
            manifest["status"] = "failed"
            if returncode == 0:
                returncode = 1
        manifest["returncode"] = returncode
        manifest["finished_at_utc"] = utc_now()
        save_json(manifest_path, manifest, replace=True)


def smoke_attempt(config: Path, output: Path, log_dir: Path, run_id: str) -> int:
    manifest_path = log_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing an existing attempt manifest: {manifest_path}")
    command = [
        sys.executable, "-u", "-m", "scripts.streamed_overlap_smoke",
        "--config", str(config), "--output", str(output),
    ]
    manifest = {
        "schema_version": 1, "mode": "smoke", "run_id": run_id, "attempt_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(), "status": "preflight", "command": command,
        "config_path": str(config.resolve()), "output_path": str(output.resolve()),
        "log_path": str((log_dir / "training.log").resolve()),
        "production_training_allowed": False, "training_device": "mps",
        "python": sys.version, "platform": platform.platform(), "runner_sha256": file_hash(Path(__file__)),
        "smoke_sha256": file_hash(Path(__file__).with_name("streamed_overlap_smoke.py")),
    }
    save_json(manifest_path, manifest)
    returncode = 1
    try:
        manifest["config_sha256"] = file_hash(config)
        resolved_config = json.loads(config.read_text())
        if not isinstance(resolved_config, dict):
            raise ValueError("Smoke config must be a JSON object.")
        manifest["resolved_config"] = resolved_config
        manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
        manifest["git_diff_sha256"] = hashlib.sha256(changes).hexdigest()
        manifest["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
        manifest["versions"] = {
            package: importlib.metadata.version(package)
            for package in ("numpy", "zarr", "torch", "tensorboard", "lpips")
        }
        manifest["status"] = "running"
        save_json(manifest_path, manifest, replace=True)
        print(f"Smoke command: {shlex.join(command)}", flush=True)
        child = subprocess.Popen(command)
        try:
            returncode = child.wait()
        except (KeyboardInterrupt, SystemExit) as interruption:
            code = interruption.code if isinstance(interruption, SystemExit) else 130
            returncode = code if isinstance(code, int) else 1
            manifest["status"] = "interrupted"
            child.send_signal(signal.SIGTERM if isinstance(interruption, SystemExit) else signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            raise
        if returncode == 0:
            report_path = output / "smoke_report.json"
            report = json.loads(report_path.read_text())
            if report.get("status") != "NO_OPTIMIZER_SMOKE_PASSED" or report.get("optimizer_step") is not False:
                raise ValueError("Smoke child did not produce the required no-optimizer report.")
            manifest["artifact_sha256"] = {"smoke_report.json": file_hash(report_path)}
            manifest["status"] = "smoke_complete_training_not_authorized"
        else:
            manifest["status"] = "failed"
        manifest["child_returncode"] = returncode
        return returncode if returncode >= 0 else 128 - returncode
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        returncode = 130
        raise
    finally:
        if manifest["status"] in ("preflight", "running"):
            manifest["status"] = "failed"
            if returncode == 0:
                returncode = 1
        manifest["returncode"] = returncode
        manifest["finished_at_utc"] = utc_now()
        save_json(manifest_path, manifest, replace=True)


def prepare_attempt(
    config: Path, output: Path, log_dir: Path, run_id: str, *, mps_memory_fraction: float, max_wall_seconds: float,
) -> int:
    manifest_path = log_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing an existing attempt manifest: {manifest_path}")
    command = [
        sys.executable, "-u", "-m", "scripts.streamed_overlap_prepare",
        "--config", str(config), "--output", str(output),
        "--mps-memory-fraction", str(mps_memory_fraction), "--max-wall-seconds", str(max_wall_seconds),
    ]
    manifest = {
        "schema_version": 1, "mode": "prepare", "run_id": run_id, "attempt_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(), "status": "preflight", "command": command,
        "config_path": str(config.resolve()), "output_path": str(output.resolve()),
        "log_path": str((log_dir / "training.log").resolve()),
        "production_training_allowed": False, "training_device": "mps",
        "python": sys.version, "platform": platform.platform(), "runner_sha256": file_hash(Path(__file__)),
        "prepare_sha256": file_hash(Path(__file__).with_name("streamed_overlap_prepare.py")),
    }
    save_json(manifest_path, manifest)
    returncode = 1
    try:
        config_bytes = config.read_bytes()
        manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        resolved_config = json.loads(config_bytes)
        if not isinstance(resolved_config, dict):
            raise ValueError("Suite preparation config must be a JSON object.")
        manifest["resolved_config"] = resolved_config
        manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
        manifest["git_diff_sha256"] = hashlib.sha256(changes).hexdigest()
        manifest["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
        manifest["versions"] = {
            package: importlib.metadata.version(package)
            for package in ("numpy", "zarr", "torch", "tensorboard", "lpips")
        }
        manifest["status"] = "running"
        save_json(manifest_path, manifest, replace=True)
        print(f"Preparation command: {shlex.join(command)}", flush=True)
        child = subprocess.Popen(command)
        try:
            returncode = child.wait()
        except (KeyboardInterrupt, SystemExit) as interruption:
            code = interruption.code if isinstance(interruption, SystemExit) else 130
            returncode = code if isinstance(code, int) else 1
            manifest["status"] = "interrupted"
            child.send_signal(signal.SIGTERM if isinstance(interruption, SystemExit) else signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            raise
        if returncode == 0:
            from scripts.streamed_overlap_audit import content_hash

            if file_hash(config) != manifest["config_sha256"]:
                raise ValueError("Suite preparation config changed during the attempt.")
            if output.is_symlink() or not output.is_file():
                raise ValueError("Suite preparation did not produce a regular input record.")
            output_bytes = output.read_bytes()
            record = json.loads(output_bytes)
            if not isinstance(record, dict) or record.get("schema") != "streamed-suite-inputs-v1":
                raise ValueError("Suite preparation did not produce a suite input record.")
            output_hash = hashlib.sha256(output_bytes).hexdigest()
            if file_hash(output) != output_hash:
                raise ValueError("Suite input record changed during preparation.")
            manifest["suite_config_sha256"] = content_hash(record)
            manifest["artifact_sha256"] = {"suite_input_record": output_hash}
            manifest["status"] = "preparation_child_complete_log_unconfirmed"
        else:
            manifest["status"] = "failed"
        manifest["child_returncode"] = returncode
        returncode = returncode if returncode >= 0 else 128 - returncode
        return returncode
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        returncode = 130
        raise
    finally:
        if manifest["status"] in ("preflight", "running"):
            manifest["status"] = "failed"
            if returncode == 0:
                returncode = 1
        manifest["returncode"] = returncode
        manifest["finished_at_utc"] = utc_now()
        save_json(manifest_path, manifest, replace=True)


def confirm_prepare_log(log_dir: Path) -> None:
    from scripts.streamed_overlap_audit import content_hash

    manifest_path = log_dir / "run_manifest.json"
    log_path = log_dir / "training.log"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, log_path)):
        raise ValueError("Preparation requires a regular attempt manifest and completed log.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("mode") != "prepare"
            or manifest.get("status") != "preparation_child_complete_log_unconfirmed"
            or manifest.get("production_training_allowed") is not False
            or manifest.get("returncode") != 0 or manifest.get("child_returncode") != 0
            or manifest.get("log_path") != str(log_path.resolve())):
        raise ValueError("Preparation attempt is not awaiting log confirmation.")
    config = Path(manifest["config_path"])
    output = Path(manifest["output_path"])
    if (any(path.is_symlink() or not path.is_file() for path in (config, output))
            or file_hash(config) != manifest["config_sha256"]
            or file_hash(output) != manifest["artifact_sha256"]["suite_input_record"]
            or content_hash(json.loads(output.read_text(encoding="utf-8"))) != manifest["suite_config_sha256"]):
        raise ValueError("Preparation config or input record changed before log confirmation.")
    with log_path.open("rb") as stream:
        os.fsync(stream.fileno())
    manifest["status"] = "preparation_complete_training_not_authorized"
    save_json(manifest_path, manifest, replace=True)


def process_identity(pid: int) -> dict | None:
    if pid <= 1:
        raise ValueError("Refusing an unsafe service PID.")
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "uid=", "-o", "lstart=", "-o", "stat=", "-o", "args="],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode:
        raise RuntimeError(f"Cannot verify service PID {pid}: {result.stderr.strip()}")
    fields = result.stdout.strip().split(maxsplit=7)
    if len(fields) != 8:
        raise RuntimeError(f"Cannot parse process identity for PID {pid}.")
    if fields[6].startswith("Z"):
        return None
    return {"uid": int(fields[0]), "started": " ".join(fields[1:6]), "command": fields[7]}


def verify_owner(owner: dict, run_dir: Path) -> bool:
    if owner["run_dir"] != str(run_dir.resolve()) or owner["host"] != HOST:
        raise RuntimeError("TensorBoard ownership record belongs to a different suite/host.")
    identity = process_identity(owner["pid"])
    if identity is None:
        return False
    if identity != owner["identity"] or identity["uid"] != os.getuid():
        raise RuntimeError("TensorBoard PID identity changed; refusing to reuse or signal it.")
    if identity["command"] != " ".join(owner["command"]):
        raise RuntimeError("TensorBoard process command differs from its ownership record.")
    if owner["command"] != tensorboard_command(run_dir.resolve(), owner["port"]):
        raise RuntimeError("Ownership record is not the expected TensorBoard command.")
    return True


def service_event(log_dir: Path, event: str, owner: dict) -> None:
    with (log_dir / "tensorboard_history.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, "utc": utc_now(), "owner": owner}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def tensorboard_command(run_dir: Path, port: int) -> list[str]:
    return [
        sys.executable, "-u", "-m", "tensorboard.main",
        "--logdir", str(run_dir), "--host", HOST, "--port", str(port),
    ]


def ready(owner: dict) -> bool:
    url = f"http://{HOST}:{owner['port']}/data/logdir"
    try:
        # Never route a local health check through environment-configured proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False
    if not isinstance(data, dict) or data.get("logdir") != owner["run_dir"]:
        raise RuntimeError("TensorBoard HTTP endpoint does not serve the recorded suite.")
    return True


def load_owner(log_dir: Path) -> dict:
    owner = json.loads((log_dir / "tensorboard_owner.json").read_text())
    if not isinstance(owner, dict):
        raise ValueError("Invalid TensorBoard ownership record.")
    pid = int((log_dir / "tensorboard.pid").read_text().strip())
    if pid != owner["pid"]:
        raise RuntimeError("TensorBoard PID and ownership records disagree.")
    return owner


@contextmanager
def service_lock(log_dir: Path):
    with (log_dir / ".tensorboard.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another TensorBoard lifecycle action owns this run's lock.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def print_service(owner: dict, log_dir: Path) -> None:
    print(f"TensorBoard: http://{HOST}:{owner['port']} (PID {owner['pid']})", flush=True)
    print(f"TensorBoard log: {log_dir / 'tensorboard.log'}", flush=True)
    print("Verify ownership and stop only this PID with:", flush=True)
    print(
        "bash scripts/streamed_overlap_experiment.sh --mode tensorboard-stop "
        f"--run-id {shlex.quote(log_dir.name)} --run-dir {shlex.quote(owner['run_dir'])}",
        flush=True,
    )


def start_tensorboard(
    run_dir: Path, log_dir: Path, *, port: int = PORT, timeout: float = 30
) -> dict:
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir() or not log_dir.is_dir():
        raise ValueError("TensorBoard requires existing suite and log directories.")
    record_path = log_dir / "tensorboard_owner.json"
    if record_path.exists():
        owner = load_owner(log_dir)
        if verify_owner(owner, run_dir):
            if owner["port"] != port or not ready(owner):
                raise RuntimeError("Run-owned TensorBoard exists but is unhealthy or on a different port.")
            print("Reusing verified run-owned TensorBoard.", flush=True)
            service_event(log_dir, "reused", owner)
            print_service(owner, log_dir)
            return owner
        service_event(log_dir, "previous_process_exited", owner)
    elif (log_dir / "tensorboard.pid").exists():
        raise RuntimeError("TensorBoard PID exists without an ownership record; manual review required.")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, port))
        except OSError as error:
            raise RuntimeError(f"TensorBoard port {HOST}:{port} is occupied; no process was stopped.") from error
    command = tensorboard_command(run_dir, port)
    print(f"Starting TensorBoard: {shlex.join(command)}", flush=True)
    with (log_dir / "tensorboard.log").open("ab") as log:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    owner = {
        "pid": child.pid,
        "run_dir": str(run_dir),
        "host": HOST,
        "port": port,
        "command": command,
        "created_at_utc": utc_now(),
    }
    healthy = False
    try:
        identity = process_identity(child.pid)
        if identity is None:
            raise RuntimeError(f"TensorBoard exited during startup; see {log_dir / 'tensorboard.log'}")
        owner["identity"] = identity
        if identity["command"] != " ".join(command):
            raise RuntimeError("Started TensorBoard command could not be verified.")
        save_json(record_path, owner, replace=True)
        with (log_dir / "tensorboard.pid").open("w") as stream:
            stream.write(f"{child.pid}\n")
            stream.flush()
            os.fsync(stream.fileno())
        service_event(log_dir, "starting", owner)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(
                    f"TensorBoard exited with {child.returncode}; see {log_dir / 'tensorboard.log'}"
                )
            if ready(owner) and verify_owner(owner, run_dir):
                service_event(log_dir, "ready", owner)
                print_service(owner, log_dir)
                healthy = True
                return owner
            time.sleep(0.1)
        raise TimeoutError(f"TensorBoard did not become ready; see {log_dir / 'tensorboard.log'}")
    finally:
        # Only the child created by this invocation is cleaned up on failed startup.
        if not healthy and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def stop_tensorboard(run_dir: Path, log_dir: Path, *, timeout: float = 10) -> None:
    owner = load_owner(log_dir)
    if not verify_owner(owner, run_dir):
        raise RuntimeError(f"Recorded TensorBoard PID {owner['pid']} is no longer running.")
    os.kill(owner["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        identity = process_identity(owner["pid"])
        if identity is None:
            service_event(log_dir, "stopped", owner)
            print(f"Stopped owned TensorBoard PID {owner['pid']}.", flush=True)
            return
        if identity != owner["identity"]:
            raise RuntimeError("PID identity changed while waiting for TensorBoard exit.")
        time.sleep(0.1)
    raise TimeoutError(f"TensorBoard PID {owner['pid']} has not exited; no other process was signaled.")


def main() -> None:
    def terminate(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=(
        "audit", "smoke", "prepare", "confirm-prepare-log", "tensorboard", "tensorboard-status", "tensorboard-stop",
    ))
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--mps-memory-fraction", type=float)
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args()
    if args.action == "confirm-prepare-log":
        confirm_prepare_log(args.log_dir)
        return
    if args.action in {"audit", "smoke", "prepare"}:
        if args.config is None or args.output is None or args.run_id is None:
            parser.error("Audit, smoke and prepare require --config, --output and --run-id.")
        if args.action == "prepare":
            if args.mps_memory_fraction is None or args.max_wall_seconds is None:
                parser.error("Prepare requires --mps-memory-fraction and --max-wall-seconds.")
            raise SystemExit(prepare_attempt(
                args.config, args.output, args.log_dir, args.run_id,
                mps_memory_fraction=args.mps_memory_fraction, max_wall_seconds=args.max_wall_seconds,
            ))
        attempt = audit_attempt if args.action == "audit" else smoke_attempt
        raise SystemExit(attempt(args.config, args.output, args.log_dir, args.run_id))
    if args.run_dir is None:
        parser.error("TensorBoard actions require --run-dir.")
    manifest = json.loads((args.log_dir / "run_manifest.json").read_text())
    if manifest["output_path"] != str(args.run_dir.resolve(strict=True)):
        raise ValueError("Monitoring suite does not match the run manifest.")
    with service_lock(args.log_dir):
        if args.action == "tensorboard":
            start_tensorboard(args.run_dir, args.log_dir)
        elif args.action == "tensorboard-stop":
            stop_tensorboard(args.run_dir, args.log_dir)
        else:
            owner = load_owner(args.log_dir)
            if not verify_owner(owner, args.run_dir) or not ready(owner):
                raise RuntimeError("Recorded TensorBoard is not running and healthy.")
            print_service(owner, args.log_dir)


if __name__ == "__main__":
    main()

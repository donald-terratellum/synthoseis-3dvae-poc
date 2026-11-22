"""Durable, single-writer CSV journal and explicit epoch-boundary reconciliation."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import uuid


IDENTITY_FIELDS = (
    "sequence", "event_id", "attempt_id", "logical_step",
    "completed_pairs", "cumulative_exposure", "replay_of_event",
)

STEP_FIELDS = (
    *IDENTITY_FIELDS, "run_id", "arm", "seed", "timestamp_utc", "epoch", "batch_index",
    "reconstructed_a", "total_loss", "reconstruction_mae", "lpips", "kl",
    "weighted_lpips", "weighted_kl", "overlap_loss", "weighted_overlap",
    "coverage_loss", "weighted_coverage", "valid_coverage", "slice_counts",
    "unavailable_reasons", "learning_rates", "gradient_norm", "data_read_seconds",
    "batch_seconds", "peak_rss_platform_units", "mps_driver_bytes", "provenance",
)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_record(directory: Path, record: dict) -> Path:
    name = uuid.uuid4().hex
    temporary = directory / f".{name}.tmp"
    destination = directory / f"{name}.json"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _sync_directory(directory)
    return destination


def _line(values: list[str]) -> bytes:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerow(values)
    return output.getvalue().encode("utf-8")


@dataclass(frozen=True)
class ResumePosition:
    logical_step: int
    cumulative_exposure: int
    replay_events: dict[int, str]
    recovery_record: Path


class BatchJournal:
    """Keep immutable completed-step rows and separate immutable recovery records.

    A failed optimizer or journal write must terminate the attempt; continuation
    is only via a verified epoch checkpoint. Close the journal to release its lock.
    """

    def __init__(self, path: Path, fields: tuple[str, ...], *, create: bool):
        if len(set(fields)) != len(fields) or any(field not in fields for field in IDENTITY_FIELDS):
            raise ValueError("Journal schema must have unique fields and all identity fields.")
        if any(not field or "\n" in field or "\r" in field for field in fields):
            raise ValueError("Journal field names must be nonempty single-line strings.")
        self.path = path
        self.fields = fields
        self.recovery_dir = path.parent / "recovery"
        self.lock_fd = os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._initialize(create)
        except BaseException:
            os.close(self.lock_fd)
            self.lock_fd = -1
            raise

    def _initialize(self, create: bool) -> None:
        if create:
            self.recovery_dir.mkdir(exist_ok=False)
            _sync_directory(self.path.parent)
            with self.path.open("xb") as handle:
                handle.write(_line(list(self.fields)))
                handle.flush()
                os.fsync(handle.fileno())
            _sync_directory(self.path.parent)
        elif not self.path.is_file() or not self.recovery_dir.is_dir():
            raise FileNotFoundError("Journal and recovery directory must already exist.")
        self.rows: list[dict[str, str]] = []
        self.event_ids: set[str] = set()
        self.pending_delimiter = False
        self.position: ResumePosition | None = None
        self.attempt_id: str | None = None
        self._read()

    def close(self) -> None:
        if self.lock_fd != -1:
            os.close(self.lock_fd)
            self.lock_fd = -1
            self.position = None

    def _records(self) -> list[dict]:
        records = []
        for path in sorted(self.recovery_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") != "streamed-journal-recovery-v1":
                raise ValueError(f"Unknown recovery record: {path}")
            records.append(record)
        return records

    def _read(self) -> bytes:
        data = self.path.read_bytes()
        self.pending_delimiter = False
        spans = []
        for record in self._records():
            torn = record.get("torn_tail")
            if torn is None:
                continue
            start, end = torn["start"], torn["end"]
            if (not isinstance(start, int) or not isinstance(end, int)
                    or start < 0 or end <= start or end > len(data)
                    or hashlib.sha256(data[start:end]).hexdigest() != torn["sha256"]):
                raise ValueError("Recorded torn journal bytes do not match the journal.")
            spans.append((start, end))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Overlapping torn journal recovery records.")
        rows = []
        offset = 0
        tail = b""
        for raw in data.splitlines(keepends=True):
            end = offset + len(raw)
            covered = [(start, stop) for start, stop in spans if start == offset and stop <= end]
            if covered:
                if len(covered) != 1 or raw[covered[0][1] - offset:] not in (b"", b"\n"):
                    raise ValueError("Invalid torn journal delimiter.")
                if not raw.endswith(b"\n"):
                    self.pending_delimiter = True
            elif any(start < end and stop > offset for start, stop in spans):
                raise ValueError("Torn journal record is not aligned with a row.")
            elif not raw.endswith(b"\n"):
                if not rows:
                    raise ValueError("Journal header is incomplete.")
                tail = raw
                break
            else:
                try:
                    parsed = next(csv.reader([raw.decode("utf-8")], strict=True))
                except (UnicodeDecodeError, csv.Error, StopIteration) as error:
                    raise ValueError("Invalid journal CSV row.") from error
                if not rows:
                    if parsed != list(self.fields):
                        raise ValueError("Journal schema differs from the expected schema.")
                    rows.append(None)
                elif len(parsed) != len(self.fields):
                    raise ValueError("Invalid completed journal row.")
                else:
                    rows.append(dict(zip(self.fields, parsed)))
            offset = end
        if not rows:
            raise ValueError("Journal has no valid header.")
        completed = rows[1:]
        seen = set()
        attempts: dict[str, tuple[int, int]] = {}
        identity = None
        for index, row in enumerate(completed, 1):
            if (row is None or row["sequence"] != str(index) or not row["event_id"]
                    or not row["attempt_id"] or row["event_id"] in seen):
                raise ValueError("Invalid journal sequence or duplicate event ID.")
            if any(not row[field].isdigit() or int(row[field]) < 1
                   for field in ("logical_step", "completed_pairs", "cumulative_exposure")):
                raise ValueError("Invalid journal trajectory values.")
            if row["replay_of_event"] and row["replay_of_event"] not in seen:
                raise ValueError("Journal replay refers to an unknown prior event.")
            if {"run_id", "arm", "seed"} <= row.keys():
                current = (row["run_id"], row["arm"], row["seed"])
                if identity is not None and current != identity:
                    raise ValueError("Journal run identity changed.")
                identity = current
            step, exposure = int(row["logical_step"]), int(row["cumulative_exposure"])
            previous = attempts.get(row["attempt_id"])
            if previous is not None and (
                step != previous[0] + 1 or exposure != previous[1] + int(row["completed_pairs"])
            ):
                raise ValueError("Journal attempt trajectory is not monotonic.")
            attempts[row["attempt_id"]] = (step, exposure)
            seen.add(row["event_id"])
        self.rows = completed
        self.event_ids = seen
        return tail

    def append(self, row: dict) -> None:
        if self.lock_fd == -1 or self.position is None or self.pending_delimiter:
            raise ValueError("Journal must be reconciled before appending.")
        if set(row) != set(self.fields):
            raise ValueError("Journal row does not match the declared schema.")
        values = [str(row[field]) if row[field] is not None else "" for field in self.fields]
        if any("\n" in value or "\r" in value for value in values):
            raise ValueError("Journal cells must be single-line values.")
        item = dict(zip(self.fields, values))
        if (item["sequence"] != str(len(self.rows) + 1) or not item["event_id"]
                or not item["attempt_id"] or item["event_id"] in self.event_ids):
            raise ValueError("Invalid next sequence, attempt or event ID.")
        for field in ("logical_step", "completed_pairs", "cumulative_exposure"):
            if not item[field].isdigit() or int(item[field]) < 1:
                raise ValueError(f"Invalid positive journal field: {field}")
        step = int(item["logical_step"])
        if (item["attempt_id"] != self.attempt_id
                or step != self.position.logical_step + 1
                or int(item["cumulative_exposure"]) != self.position.cumulative_exposure + int(item["completed_pairs"])
                or item["replay_of_event"] != self.position.replay_events.get(step, "")):
            raise ValueError("Journal row does not match the reconciled model trajectory.")
        try:
            with self.path.open("ab") as handle:
                handle.write(_line(values))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            self.position = None
            raise
        self.rows.append(item)
        self.event_ids.add(item["event_id"])
        self.position = ResumePosition(
            step, int(item["cumulative_exposure"]), self.position.replay_events,
            self.position.recovery_record,
        )

    def reconcile(
        self, *, watermark: str | None, attempt_id: str,
        expected_step: int, expected_exposure: int,
    ) -> ResumePosition:
        if self.lock_fd == -1:
            raise ValueError("A closed journal cannot be reconciled.")
        if not isinstance(attempt_id, str) or not attempt_id or "\r" in attempt_id or "\n" in attempt_id:
            raise ValueError("Recovery requires a new single-line attempt ID.")
        tail = self._read()
        records = self._records()
        if any(record.get("attempt_id") == attempt_id for record in records):
            raise ValueError("Recovery attempt ID must be new.")
        matches = [i for i, row in enumerate(self.rows) if row["event_id"] == watermark]
        if watermark is not None and len(matches) != 1:
            raise ValueError("Checkpoint event watermark is absent from the journal.")
        index = matches[0] if matches else -1
        committed = self.rows[index] if index >= 0 else None
        step = int(committed["logical_step"]) if committed else 0
        exposure = int(committed["cumulative_exposure"]) if committed else 0
        if (step != expected_step or exposure != expected_exposure
                or any(watermark in record.get("uncommitted_event_ids", []) for record in records)):
            raise ValueError("Checkpoint state or watermark is incompatible with journal recovery.")
        if tail:
            start = self.path.stat().st_size - len(tail)
            _atomic_record(self.recovery_dir, {
                "schema": "streamed-journal-recovery-v1",
                "kind": "torn_tail",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "torn_tail": {
                    "start": start, "end": start + len(tail),
                    "sha256": hashlib.sha256(tail).hexdigest(),
                },
            })
            self.pending_delimiter = True
        if self.pending_delimiter:
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self._read() or self.pending_delimiter:
                raise ValueError("Journal tail remains incomplete after recovery.")
        stale = self.rows[index + 1:]
        replay = {int(row["logical_step"]): row["event_id"] for row in stale}
        record = _atomic_record(self.recovery_dir, {
            "schema": "streamed-journal-recovery-v1",
            "kind": "checkpoint_reconciliation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempt_id": attempt_id,
            "checkpoint_watermark": watermark,
            "checkpoint_logical_step": step,
            "checkpoint_cumulative_exposure": exposure,
            "uncommitted_event_ids": [row["event_id"] for row in stale],
        })
        self.position = ResumePosition(step, exposure, replay, record)
        self.attempt_id = attempt_id
        return self.position

import multiprocessing as mp
import shutil
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import zarr

from src.tokenizer.core.batching import adaptive_batch_map
from src.tokenizer.core.events import JobEvent
from src.tokenizer.core.io_zarr import compute_padding, remove_padding
from src.tokenizer.core.metrics import write_benchmark_report
from src.tokenizer.core.model_adapter import VaeLatentAdapter, cube_to_latent_128
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.search_engine import run_similarity_search_on_padded_volume


def compute_windows_per_axis(axis_len: int, patch_size: int = 32, stride: int = 16) -> int:
    if axis_len <= 0:
        raise ValueError("axis_len must be > 0")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if axis_len < patch_size:
        raise ValueError("axis_len must be >= patch_size")
    remainder = (axis_len - patch_size) % stride
    if remainder != 0:
        raise ValueError("axis_len must align to patch_size/stride grid")
    return 1 + ((axis_len - patch_size) // stride)


def compute_total_windows_from_padded_shape(
    padded_shape: Sequence[int],
    patch_size: int = 32,
    stride: int = 16,
) -> int:
    if len(padded_shape) != 3:
        raise ValueError("padded_shape must have 3 axes")
    wx = compute_windows_per_axis(int(padded_shape[0]), patch_size=patch_size, stride=stride)
    wy = compute_windows_per_axis(int(padded_shape[1]), patch_size=patch_size, stride=stride)
    wz = compute_windows_per_axis(int(padded_shape[2]), patch_size=patch_size, stride=stride)
    return int(wx * wy * wz)


def estimate_total_windows_for_source_shape(
    volume_shape: Sequence[int],
    patch_size: int = 32,
    stride: int = 16,
    base_pad: int = 16,
) -> int:
    padding = compute_padding(volume_shape, patch_size=patch_size, base_pad=base_pad)
    padded_shape = tuple(int(axis + low + high) for axis, (low, high) in zip(volume_shape, padding))
    return compute_total_windows_from_padded_shape(
        padded_shape,
        patch_size=patch_size,
        stride=stride,
    )


@dataclass
class SearchExecutionSpec:
    search_volume: np.ndarray
    token_latent: np.ndarray
    patch_size: int = 32
    stride: int = 16
    batch_size: int = 32
    output_zarr_path: Optional[str] = None
    latent_mode: str = "pooled"
    model_path: Optional[str] = None
    device: str = "auto"
    keep_partial_output: bool = False
    benchmark_json_path: Optional[str] = None


def cleanup_output_artifact(path_str: Optional[str]) -> None:
    if not path_str:
        return
    path = Path(path_str)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _search_worker(
    total_windows: int,
    event_queue: mp.Queue,
    control_queue: mp.Queue,
    execution_spec: Optional[SearchExecutionSpec],
) -> None:
    try:
        event_queue.put(JobEvent(kind="status", message="started"))

        if execution_spec is None:
            started = time.time()
            for idx in range(total_windows):
                try:
                    command = control_queue.get_nowait()
                    if command == "cancel":
                        event_queue.put(JobEvent(kind="status", message="canceled"))
                        event_queue.put(JobEvent(kind="done", message="canceled"))
                        return
                except queue.Empty:
                    pass

                time.sleep(0.002)
                completed = idx + 1
                elapsed = max(1e-6, time.time() - started)
                rate = completed / elapsed
                remaining = max(0, total_windows - completed)
                eta = remaining / max(rate, 1e-6)
                event_queue.put(
                    JobEvent(
                        kind="progress",
                        total_windows=total_windows,
                        completed_windows=completed,
                        eta_seconds=float(eta),
                    )
                )

            event_queue.put(JobEvent(kind="status", message="completed"))
            event_queue.put(JobEvent(kind="done", message="completed"))
            return

        spec = execution_spec
        volume = np.asarray(spec.search_volume, dtype=np.float32)
        padding = compute_padding(volume.shape, patch_size=spec.patch_size, base_pad=16)
        padded = np.pad(volume, pad_width=padding, mode="constant", constant_values=0.0).astype(np.float32)
        started = time.time()
        last_completed = 0
        total_for_metrics = total_windows

        if not spec.keep_partial_output:
            cleanup_output_artifact(spec.output_zarr_path)

        if spec.latent_mode == "vae":
            if not spec.model_path:
                raise ValueError("model_path is required for vae latent_mode")
            adapter = VaeLatentAdapter(checkpoint_path=spec.model_path, device=spec.device)
            latent_batch_fn = lambda cubes: adaptive_batch_map(
                cubes,
                adapter.encode_batch,
                initial_batch_size=spec.batch_size,
            ).astype(np.float32, copy=False)
        else:
            latent_batch_fn = lambda cubes: adaptive_batch_map(
                cubes,
                lambda part: np.stack([cube_to_latent_128(c) for c in part], axis=0).astype(np.float32),
                initial_batch_size=spec.batch_size,
            ).astype(np.float32, copy=False)

        def should_cancel() -> bool:
            try:
                command = control_queue.get_nowait()
                if command == "cancel":
                    return True
            except queue.Empty:
                pass
            return False

        def on_progress(completed: int, total: int, eta: float) -> None:
            nonlocal last_completed, total_for_metrics
            last_completed = completed
            total_for_metrics = total
            event_queue.put(
                JobEvent(
                    kind="progress",
                    total_windows=total,
                    completed_windows=completed,
                    eta_seconds=eta,
                )
            )

        similarity_padded = run_similarity_search_on_padded_volume(
            padded_volume=padded,
            token_latent=np.asarray(spec.token_latent, dtype=np.float32),
            patch_size=spec.patch_size,
            stride=spec.stride,
            preprocess_fn=preprocess_for_token,
            latent_batch_fn=latent_batch_fn,
            batch_size=spec.batch_size,
            progress_callback=on_progress,
            should_cancel=should_cancel,
        )

        if should_cancel():
            elapsed = max(1e-6, time.time() - started)
            windows_per_sec = float(last_completed / elapsed)
            event_queue.put(
                JobEvent(
                    kind="status",
                    message=(
                        f"metrics: completed_windows={last_completed} total_windows={total_for_metrics} "
                        f"elapsed_s={elapsed:.3f} windows_per_sec={windows_per_sec:.3f}"
                    ),
                )
            )
            if not spec.keep_partial_output:
                cleanup_output_artifact(spec.output_zarr_path)
                event_queue.put(JobEvent(kind="status", message="recovery: partial output cleaned"))
            if spec.benchmark_json_path:
                write_benchmark_report(
                    spec.benchmark_json_path,
                    {
                        "status": "canceled",
                        "completed_windows": int(last_completed),
                        "total_windows": int(total_for_metrics),
                        "elapsed_s": float(elapsed),
                        "windows_per_sec": float(windows_per_sec),
                        "patch_size": int(spec.patch_size),
                        "stride": int(spec.stride),
                        "batch_size": int(spec.batch_size),
                        "latent_mode": str(spec.latent_mode),
                        "device": str(spec.device),
                    },
                )
                event_queue.put(
                    JobEvent(kind="artifact", artifact_path=str(spec.benchmark_json_path), message="benchmark_json")
                )
            event_queue.put(JobEvent(kind="status", message="canceled"))
            event_queue.put(JobEvent(kind="done", message="canceled"))
            return

        similarity = remove_padding(similarity_padded, padding)
        if spec.output_zarr_path:
            out_path = Path(spec.output_zarr_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            root = zarr.open(str(out_path), mode="w")
            root.create_array("data", data=similarity, chunks=(16, 16, similarity.shape[2]))
            event_queue.put(JobEvent(kind="artifact", artifact_path=str(out_path), message="output_zarr"))

        elapsed = max(1e-6, time.time() - started)
        windows_per_sec = float(total_for_metrics / elapsed)
        if spec.benchmark_json_path:
            write_benchmark_report(
                spec.benchmark_json_path,
                {
                    "status": "completed",
                    "completed_windows": int(total_for_metrics),
                    "total_windows": int(total_for_metrics),
                    "elapsed_s": float(elapsed),
                    "windows_per_sec": float(windows_per_sec),
                    "patch_size": int(spec.patch_size),
                    "stride": int(spec.stride),
                    "batch_size": int(spec.batch_size),
                    "latent_mode": str(spec.latent_mode),
                    "device": str(spec.device),
                    "output_shape": list(similarity.shape),
                    "output_min": float(similarity.min()),
                    "output_max": float(similarity.max()),
                },
            )
            event_queue.put(
                JobEvent(kind="artifact", artifact_path=str(spec.benchmark_json_path), message="benchmark_json")
            )
        event_queue.put(
            JobEvent(
                kind="status",
                message=(
                    f"metrics: completed_windows={total_for_metrics} total_windows={total_for_metrics} "
                    f"elapsed_s={elapsed:.3f} windows_per_sec={windows_per_sec:.3f}"
                ),
            )
        )

        event_queue.put(JobEvent(kind="status", message="completed"))
        event_queue.put(JobEvent(kind="done", message="completed"))
    except Exception as exc:
        event_queue.put(JobEvent(kind="error", message=str(exc)))


@dataclass
class SearchJobRunner:
    total_windows: Optional[int] = None
    execution_spec: Optional[SearchExecutionSpec] = None

    def __post_init__(self) -> None:
        if self.execution_spec is not None:
            self.total_windows = estimate_total_windows_for_source_shape(
                self.execution_spec.search_volume.shape,
                patch_size=self.execution_spec.patch_size,
                stride=self.execution_spec.stride,
                base_pad=16,
            )
        if self.total_windows is None or self.total_windows <= 0:
            raise ValueError("total_windows must be > 0")
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._event_queue: mp.Queue = ctx.Queue()
        self._control_queue: mp.Queue = ctx.Queue()
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            raise RuntimeError("job is already running")
        self._process = self._ctx.Process(
            target=_search_worker,
            args=(self.total_windows, self._event_queue, self._control_queue, self.execution_spec),
            daemon=True,
        )
        self._process.start()

    def cancel(self) -> None:
        self._control_queue.put("cancel")

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def poll_events(self) -> list[JobEvent]:
        events: list[JobEvent] = []
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            event.validate()
            events.append(event)
        return events

    def join(self, timeout: Optional[float] = None) -> None:
        if self._process is not None:
            self._process.join(timeout=timeout)

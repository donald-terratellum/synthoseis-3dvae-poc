from typing import Optional
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QTimer

from src.tokenizer.core.io_zarr import extract_centered_cube, load_volume_numpy
from src.tokenizer.core.jobs import SearchExecutionSpec, SearchJobRunner
from src.tokenizer.core.model_adapter import cube_to_latent_128
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.token_picker import build_preprocessed_token_cube
from src.tokenizer.ui.state import DisplayState


class TokenizerController:
    def __init__(self, window, patch_size: int = 32):
        self.window = window
        self.patch_size = patch_size
        self.volume: Optional[np.ndarray] = None
        self.output_volume: Optional[np.ndarray] = None
        self._latest_output_path: Optional[str] = None
        self.last_token_cube: Optional[np.ndarray] = None
        self.display_state = DisplayState()
        self._job_runner: Optional[SearchJobRunner] = None
        self._job_poll_timer = QTimer()
        self._job_poll_timer.setInterval(30)
        self._job_poll_timer.timeout.connect(self._poll_job_events)
        self.window.tokenPicked.connect(self.on_token_picked)
        self.window.sourceLoadRequested.connect(self.on_source_load_requested)
        self.window.outputLoadRequested.connect(self.on_output_load_requested)
        self.window.displayStateChanged.connect(self.on_display_state_changed)
        self.window.startSearchRequested.connect(self.on_start_search_requested)
        self.window.cancelSearchRequested.connect(self.on_cancel_search_requested)

    def set_volume(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.window.set_volume(volume)
        self.on_display_state_changed(self.window.get_display_state())

    def on_token_picked(self, x: int, y: int, z: int) -> None:
        if self.volume is None:
            self.window.status_label.setText("No volume loaded")
            return
        self.last_token_cube = build_preprocessed_token_cube(
            self.volume,
            center_xyz=(x, y, z),
            patch_size=self.patch_size,
        )
        non_zero = int(np.count_nonzero(self.last_token_cube))
        self.window.status_label.setText(
            f"Token ready at ({x}, {y}, {z}), non_zero={non_zero}"
        )

    def on_source_load_requested(self, source_path: str) -> None:
        try:
            volume = load_volume_numpy(Path(source_path))
            self.set_volume(volume)
            self.window.source_path.setText(source_path)
            self.window.status_label.setText(f"Source loaded: shape={volume.shape}")
        except Exception as exc:
            self.window.status_label.setText(f"Source load failed: {exc}")

    def on_output_load_requested(self, output_path: str) -> None:
        try:
            output = load_volume_numpy(Path(output_path))
            if self.volume is not None and output.shape != self.volume.shape:
                raise ValueError(
                    f"output shape {output.shape} does not match source shape {self.volume.shape}"
                )
            self.output_volume = output
            self.window.output_path.setText(output_path)
            self.window.set_output_volume(output)
            self.window.status_label.setText(f"Output loaded: shape={output.shape}")
            self._apply_overlay_preview_state()
        except Exception as exc:
            self.window.status_label.setText(f"Output load failed: {exc}")

    def on_display_state_changed(self, snapshot: dict) -> None:
        self.display_state.update_from_snapshot(snapshot)
        self.window.set_slice_view_state(snapshot)
        self._apply_overlay_preview_state()

    def _apply_overlay_preview_state(self) -> None:
        if self.volume is None or self.output_volume is None:
            self.display_state.output_loaded = self.output_volume is not None
            return

        i = int(np.clip(self.display_state.inline_index, 0, self.volume.shape[0] - 1))
        c = int(np.clip(self.display_state.crossline_index, 0, self.volume.shape[1] - 1))
        z = int(np.clip(self.display_state.z_index, 0, self.volume.shape[2] - 1))

        # Blend orthogonal slices to get a compact overlay preview statistic.
        s_in = (
            self.volume[i, :, :].astype(np.float32),
            self.volume[:, c, :].astype(np.float32),
            self.volume[:, :, z].astype(np.float32),
        )
        s_out = (
            self.output_volume[i, :, :].astype(np.float32),
            self.output_volume[:, c, :].astype(np.float32),
            self.output_volume[:, :, z].astype(np.float32),
        )

        alpha = float(np.clip(self.display_state.overlay_alpha, 0.0, 1.0))
        in_clip = max(1e-6, float(self.display_state.input_clip))
        out_clip = max(1e-6, float(self.display_state.output_clip))

        blended_parts = []
        for a, b in zip(s_in, s_out):
            a_lim = max(np.max(np.abs(a)) * in_clip, 1e-6)
            b_lim = max(np.max(np.abs(b)) * out_clip, 1e-6)
            a_c = np.clip(a, -a_lim, a_lim) / a_lim
            b_c = np.clip(b, -b_lim, b_lim) / b_lim
            blended_parts.append(((1.0 - alpha) * a_c) + (alpha * b_c))

        preview = np.concatenate([p.reshape(-1) for p in blended_parts], axis=0)
        self.display_state.output_loaded = True
        self.display_state.overlay_preview_mean = float(preview.mean())
        self.display_state.overlay_preview_std = float(preview.std())
        self.window.status_label.setText(
            "Overlay preview updated: "
            f"mean={self.display_state.overlay_preview_mean:.4f}, "
            f"std={self.display_state.overlay_preview_std:.4f}"
        )

    def on_start_search_requested(self) -> None:
        if self.volume is None:
            self.window.status_label.setText("Cannot start search: no source volume loaded")
            return
        if self._job_runner is not None and self._job_runner.is_running():
            self.window.status_label.setText("Search already running")
            return

        if self.last_token_cube is not None:
            token_cube = self.last_token_cube
        else:
            x = int(np.clip(self.display_state.inline_index, 0, self.volume.shape[0] - 1))
            y = int(np.clip(self.display_state.crossline_index, 0, self.volume.shape[1] - 1))
            z = int(np.clip(self.display_state.z_index, 0, self.volume.shape[2] - 1))
            token_cube = preprocess_for_token(
                extract_centered_cube(self.volume, x=x, y=y, z=z, patch_size=self.patch_size)
            )
        token_latent = cube_to_latent_128(token_cube)

        out_path = Path(tempfile.gettempdir()) / "temp_seismic" / "ui_similarity_output.zarr"
        self._latest_output_path = str(out_path)
        spec = SearchExecutionSpec(
            search_volume=np.asarray(self.volume, dtype=np.float32),
            token_latent=token_latent,
            patch_size=self.patch_size,
            stride=max(1, self.patch_size // 2),
            batch_size=32,
            output_zarr_path=str(out_path),
            latent_mode="pooled",
        )

        self._job_runner = SearchJobRunner(execution_spec=spec)
        total_windows = int(self._job_runner.total_windows or 0)
        self._job_runner.start()
        self.window.set_job_progress_visible(True)
        self.window.update_job_progress(0, total_windows, None)
        self.window.status_label.setText("Background search started")
        self._job_poll_timer.start()

    def on_cancel_search_requested(self) -> None:
        if self._job_runner is None or not self._job_runner.is_running():
            self.window.status_label.setText("No active search job")
            return
        self._job_runner.cancel()
        self.window.status_label.setText("Cancel requested")

    def _poll_job_events(self) -> None:
        if self._job_runner is None:
            self._job_poll_timer.stop()
            return

        for event in self._job_runner.poll_events():
            if event.kind == "progress":
                self.window.update_job_progress(
                    completed=int(event.completed_windows or 0),
                    total=int(event.total_windows or 0),
                    eta_seconds=event.eta_seconds,
                )
            elif event.kind == "status":
                self.window.status_label.setText(f"Search status: {event.message}")
            elif event.kind == "artifact":
                if event.artifact_path:
                    self.on_output_load_requested(event.artifact_path)
            elif event.kind == "error":
                self.window.status_label.setText(f"Search error: {event.message}")
                self.window.set_job_progress_visible(False)
                self._job_poll_timer.stop()

            if event.kind == "done":
                self.window.status_label.setText(f"Search finished: {event.message}")
                self.window.set_job_progress_visible(False)
                self._job_poll_timer.stop()

        if not self._job_runner.is_running():
            self._job_poll_timer.stop()

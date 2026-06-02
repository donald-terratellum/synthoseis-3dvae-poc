from typing import Optional
from pathlib import Path
import json
import tempfile

import numpy as np
from PySide6.QtCore import QTimer

from src.tokenizer.core.io_zarr import extract_centered_cube, load_volume_numpy
from src.tokenizer.core.jobs import SearchExecutionSpec, SearchJobRunner
from src.tokenizer.core.model_adapter import VaeLatentAdapter, cube_to_latent_128
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.token_picker import build_preprocessed_token_cube
from src.tokenizer.config.defaults import RuntimeConfig
from src.tokenizer.ui.state import DisplayState


class TokenizerController:
    def __init__(
        self,
        window,
        patch_size: int = 32,
        latent_mode: str = "pooled",
        model_path: Optional[str] = None,
        device: str = "auto",
        state_file: Optional[str | Path] = None,
    ):
        self.window = window
        self.patch_size = patch_size
        self.latent_mode = latent_mode
        self.device = device
        runtime = RuntimeConfig()
        self.model_path = model_path if model_path is not None else str(runtime.model_path)
        self._vae_adapter: Optional[VaeLatentAdapter] = None
        if self.latent_mode == "vae":
            # Align UI token extraction and search windows with checkpoint patch metadata.
            self.patch_size = self._get_vae_adapter().patch_shape
        self.window.set_patch_shape(self.patch_size)
        self._state_file = Path(state_file) if state_file is not None else (Path.home() / ".synthoseis_tokenizer_ui_state.json")
        self._skip_next_close_save = False
        self._pending_display_snapshot: Optional[dict] = None
        self._pending_selected_point: Optional[tuple[int, int, int]] = None
        self._pending_output_path: Optional[str] = None
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
        self.window.resetUiStateRequested.connect(self.on_reset_ui_state_requested)
        self.window.windowClosing.connect(self.on_window_closing)

    def on_window_closing(self) -> None:
        if self._skip_next_close_save:
            self._skip_next_close_save = False
            return
        self.save_session_state()

    def on_reset_ui_state_requested(self) -> None:
        self._pending_display_snapshot = None
        self._pending_selected_point = None
        self._pending_output_path = None
        try:
            if self._state_file.exists():
                self._state_file.unlink()
            # Keep deleted state deleted if user closes immediately after reset.
            self._skip_next_close_save = True
            self.window.status_label.setText(f"UI state reset: removed {self._state_file}")
        except Exception as exc:
            self.window.status_label.setText(f"UI state reset failed: {exc}")

    def restore_session_state(self, auto_load_source: bool = True) -> None:
        payload = self._read_session_state()
        if payload is None:
            return

        snapshot = payload.get("display_state")
        if isinstance(snapshot, dict):
            # Restore controls that are not shape-dependent immediately.
            self.window.input_clip_slider.setValue(int(round(100.0 * float(snapshot.get("input_clip", 0.5)))))
            self.window.output_clip_slider.setValue(int(round(100.0 * float(snapshot.get("output_clip", 0.5)))))
            self.window.overlay_threshold_slider.setValue(
                int(round(100.0 * float(snapshot.get("overlay_threshold", 0.5))))
            )
            self.window.overlay_alpha_slider.setValue(int(round(100.0 * float(snapshot.get("overlay_alpha", 0.6)))))
            mode = str(snapshot.get("similarity_mode", "cosine"))
            mode_idx = self.window.similarity_mode_combo.findData(mode)
            if mode_idx >= 0:
                self.window.similarity_mode_combo.setCurrentIndex(mode_idx)

            # Restore volume-dependent indices after source data is loaded.
            self._pending_display_snapshot = {
                "inline_index": int(snapshot.get("inline_index", 0)),
                "crossline_index": int(snapshot.get("crossline_index", 0)),
                "z_index": int(snapshot.get("z_index", 0)),
                "similarity_mode": mode,
            }

        point = payload.get("selected_point")
        if isinstance(point, list) and len(point) == 3:
            try:
                self._pending_selected_point = (int(point[0]), int(point[1]), int(point[2]))
            except Exception:
                self._pending_selected_point = None

        source_path = payload.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            self.window.source_path.setText(source_path)

        output_path = payload.get("output_path")
        if isinstance(output_path, str) and output_path.strip():
            self.window.output_path.setText(output_path)
            self._pending_output_path = output_path

        vertical_exaggeration = payload.get("vertical_exaggeration")
        if vertical_exaggeration is not None:
            try:
                self.window.set_vertical_exaggeration(float(vertical_exaggeration))
            except Exception:
                pass

        if auto_load_source and isinstance(source_path, str) and source_path.strip():
            self.on_source_load_requested(source_path)

    def _read_session_state(self) -> Optional[dict]:
        if not self._state_file.exists():
            return None
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_session_state(self) -> None:
        payload = {
            "source_path": str(self.window.source_path.text()).strip(),
            "output_path": str(self.window.output_path.text()).strip(),
            "selected_point": [
                int(self.window.x_spin.value()),
                int(self.window.y_spin.value()),
                int(self.window.z_spin.value()),
            ],
            "vertical_exaggeration": float(self.window.get_vertical_exaggeration()),
            "display_state": self.window.get_display_state(),
        }
        try:
            self._state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # UI persistence should not interrupt core workflow.
            pass

    def _get_vae_adapter(self) -> VaeLatentAdapter:
        if self._vae_adapter is None:
            self._vae_adapter = VaeLatentAdapter(checkpoint_path=self.model_path, device=self.device)
        return self._vae_adapter

    def set_volume(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.window.set_volume(volume)
        if self._pending_display_snapshot is not None:
            nx, ny, nz = volume.shape
            ix = int(np.clip(self._pending_display_snapshot.get("inline_index", 0), 0, max(0, nx - 1)))
            iy = int(np.clip(self._pending_display_snapshot.get("crossline_index", 0), 0, max(0, ny - 1)))
            iz = int(np.clip(self._pending_display_snapshot.get("z_index", 0), 0, max(0, nz - 1)))
            self.window.inline_slider.setValue(ix)
            self.window.crossline_slider.setValue(iy)
            self.window.z_slider.setValue(iz)
            mode = str(self._pending_display_snapshot.get("similarity_mode", "cosine"))
            mode_idx = self.window.similarity_mode_combo.findData(mode)
            if mode_idx >= 0:
                self.window.similarity_mode_combo.setCurrentIndex(mode_idx)
            self._pending_display_snapshot = None

        if self._pending_selected_point is not None:
            px, py, pz = self._pending_selected_point
            nx, ny, nz = volume.shape
            rx = int(np.clip(px, 0, max(0, nx - 1)))
            ry = int(np.clip(py, 0, max(0, ny - 1)))
            rz = int(np.clip(pz, 0, max(0, nz - 1)))
            self.window.x_spin.setValue(rx)
            self.window.y_spin.setValue(ry)
            self.window.z_spin.setValue(rz)
            self.window.slice_viewer.set_selected_point(rx, ry, rz)
            self._pending_selected_point = None

        if self._pending_output_path:
            pending = self._pending_output_path
            self._pending_output_path = None
            self.on_output_load_requested(pending)

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
        self.save_session_state()

    def on_source_load_requested(self, source_path: str) -> None:
        try:
            volume = load_volume_numpy(Path(source_path))
            self.set_volume(volume)
            self.window.source_path.setText(source_path)
            self.window.status_label.setText(f"Source loaded: shape={volume.shape}")
            self.save_session_state()
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
            self.save_session_state()
        except Exception as exc:
            self.window.status_label.setText(f"Output load failed: {exc}")

    def on_display_state_changed(self, snapshot: dict) -> None:
        self.display_state.update_from_snapshot(snapshot)
        self.window.set_slice_view_state(snapshot)
        self._apply_overlay_preview_state()
        self.save_session_state()

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
        overlay_threshold = float(np.clip(self.display_state.overlay_threshold, 0.0, 1.0))

        blended_parts = []
        for a, b in zip(s_in, s_out):
            a_lim = max(np.max(np.abs(a)) * in_clip, 1e-6)
            b_lim = max(np.max(np.abs(b)) * out_clip, 1e-6)
            a_c = np.clip(a, -a_lim, a_lim) / a_lim
            b_c = np.clip(b, -b_lim, b_lim) / b_lim
            b_mask = (np.abs(b_c) >= overlay_threshold).astype(np.float32)
            blended_parts.append(((1.0 - alpha) * a_c) + (alpha * (b_c * b_mask)))

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
        try:
            if self.latent_mode == "vae":
                token_latent = self._get_vae_adapter().encode_cube(token_cube)
            else:
                token_latent = cube_to_latent_128(token_cube)
        except Exception as exc:
            self.window.status_label.setText(f"Token latent build failed: {exc}")
            return

        out_path = Path(tempfile.gettempdir()) / "temp_seismic" / "ui_similarity_output.zarr"
        self._latest_output_path = str(out_path)
        spec = SearchExecutionSpec(
            search_volume=np.asarray(self.volume, dtype=np.float32),
            token_latent=token_latent,
            patch_size=self.patch_size,
            stride=max(1, int(min(self.patch_size) // 2)) if isinstance(self.patch_size, tuple) else max(1, self.patch_size // 2),
            batch_size=32,
            output_zarr_path=str(out_path),
            latent_mode=self.latent_mode,
            similarity_mode=self.display_state.similarity_mode,
            model_path=self.model_path,
            device=self.device,
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

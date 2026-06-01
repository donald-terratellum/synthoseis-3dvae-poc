from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenuBar, QVBoxLayout, QWidget


@dataclass
class SliceViewState:
    inline_index: int = 0
    crossline_index: int = 0
    z_index: int = 0
    input_clip: float = 0.5
    output_clip: float = 0.5
    overlay_alpha: float = 0.6


class SliceViewer(QWidget):
    pointSelectionPreview = Signal(int, int, int)
    pointSelectionCommitted = Signal(int, int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._volume: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None
        self._output_clim: Optional[tuple[float, float]] = None
        self._state = SliceViewState()
        self._vertical_exaggeration = 0.1
        self._camera_initialized = False
        self._force_camera_reset = True
        self._last_source_shape: Optional[tuple[int, int, int]] = None
        self._selected_point_idx: Optional[tuple[int, int, int]] = None
        self._preview_point_idx: Optional[tuple[int, int, int]] = None
        self._point_pick_mode = False
        self._pick_drag_active = False
        self._headless = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
        self.plotter: Optional[Any] = None
        self._pv_module: Optional[Any] = None
        self._cell_picker: Optional[Any] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self.menu_bar = QMenuBar(self)
        selection_menu = self.menu_bar.addMenu("Selection")
        self.point_pick_action = QAction("Point Pick Mode", self)
        self.point_pick_action.setCheckable(True)
        self.point_pick_action.toggled.connect(self.set_point_pick_mode)
        selection_menu.addAction(self.point_pick_action)
        top_row.addWidget(self.menu_bar, stretch=0)
        self.pick_status_label = QLabel("Selection off")
        top_row.addWidget(self.pick_status_label, stretch=1)
        layout.addLayout(top_row)

        if self._headless:
            # Keep tests stable in offscreen environments where VTK teardown can crash.
            placeholder = QLabel("Slice view disabled in headless mode")
            layout.addWidget(placeholder)
        else:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            self._pv_module = pv
            self.plotter = QtInteractor(self)
            self.plotter.set_background("white")
            self.plotter.show_axes()
            layout.addWidget(self.plotter.interactor)
            self.plotter.interactor.installEventFilter(self)

            # Use explicit screen-space cell picking so MB1 pick coordinates match cursor location.
            self._cell_picker = self._pv_module._vtk.vtkCellPicker()
            self._cell_picker.SetTolerance(0.0005)

        self._markers: list[Any] = []

    def set_source_volume(self, volume: np.ndarray) -> None:
        src = np.asarray(volume, dtype=np.float32)
        shape = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        if self._last_source_shape is None or self._last_source_shape != shape:
            self._force_camera_reset = True
        self._last_source_shape = shape
        self._volume = src
        if self._selected_point_idx is None:
            self._selected_point_idx = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        else:
            self._selected_point_idx = self._clamp_indices(*self._selected_point_idx)
        self._render_scene()

    def set_output_volume(self, volume: np.ndarray) -> None:
        self._output = np.asarray(volume, dtype=np.float32)
        self._output_clim = self._compute_output_clim(self._output)
        self._render_scene()

    def _compute_output_clim(self, volume: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(volume[np.isfinite(volume)], dtype=np.float32)
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.percentile(finite, 2.0))
        hi = float(np.percentile(finite, 88.0))
        span = max(abs(lo), abs(hi), 1e-6)
        return (-span, span)

    def _build_display_volume(self, volume: np.ndarray) -> np.ndarray:
        return np.asarray(volume[:, :, ::-1], dtype=np.float32)

    def _build_output_overlay_volume(self) -> Optional[np.ndarray]:
        if self._output is None:
            return None

        display_volume = self._build_display_volume(self._output)
        lo, hi = self._output_clim if self._output_clim is not None else (0.0, 1.0)
        clip_fraction = float(np.clip(self._state.output_clip, 0.0, 1.0))
        span = max(abs(lo), abs(hi), 1e-6)
        threshold = clip_fraction * span
        masked = np.where(np.abs(display_volume) >= threshold, display_volume, np.nan)
        return np.asarray(masked, dtype=np.float32)

    def set_state(self, state: SliceViewState) -> None:
        self._state = state
        self._render_scene()

    def set_vertical_exaggeration(self, value: float) -> None:
        self._vertical_exaggeration = max(0.1, float(value))
        self._render_scene()

    def get_vertical_exaggeration(self) -> float:
        return float(self._vertical_exaggeration)

    def set_point_pick_mode(self, enabled: bool) -> None:
        self._point_pick_mode = bool(enabled)
        self._pick_drag_active = False
        if not self._point_pick_mode:
            self._preview_point_idx = None
            self._render_scene()
        self._update_pick_status_label()

    def set_selected_point(self, x: int, y: int, z: int) -> None:
        if self._volume is None:
            self._selected_point_idx = (int(x), int(y), int(z))
        else:
            self._selected_point_idx = self._clamp_indices(x, y, z)
        self._preview_point_idx = None
        self._update_pick_status_label()
        self._render_scene()

    def _clamp_indices(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        if self._volume is None:
            return int(x), int(y), int(z)
        nx, ny, nz = self._volume.shape
        cx = int(np.clip(int(x), 0, max(0, nx - 1)))
        cy = int(np.clip(int(y), 0, max(0, ny - 1)))
        cz = int(np.clip(int(z), 0, max(0, nz - 1)))
        return (cx, cy, cz)

    def _update_pick_status_label(self) -> None:
        if not self._point_pick_mode:
            self.pick_status_label.setText("Selection off")
            return
        idx = self._preview_point_idx if self._preview_point_idx is not None else self._selected_point_idx
        if idx is None:
            self.pick_status_label.setText("Selection on")
            return
        self.pick_status_label.setText(f"Selection: x={idx[0]} y={idx[1]} z={idx[2]}")

    def _index_to_plot_z(self, z_idx: int) -> float:
        if self._volume is None:
            return 0.0
        nz = int(self._volume.shape[2])
        rz = max(0, min(nz - 1, (nz - 1) - int(z_idx)))
        return float(rz) * float(self._vertical_exaggeration)

    def _world_to_indices(self, world_xyz: tuple[float, float, float]) -> Optional[tuple[int, int, int]]:
        if self._volume is None:
            return None
        nx, ny, nz = self._volume.shape
        xw, yw, zw = world_xyz
        z_scale = max(float(self._vertical_exaggeration), 1e-6)
        xi = int(round(xw))
        yi = int(round(yw))
        rz = int(round(zw / z_scale))
        zi = (nz - 1) - rz
        if xi < 0 or yi < 0 or zi < 0 or xi >= nx or yi >= ny or zi >= nz:
            return None
        return (xi, yi, zi)

    def _pick_world_position_from_event(self, event: QEvent) -> Optional[tuple[float, float, float]]:
        if self.plotter is None or self._cell_picker is None:
            return None

        if not hasattr(event, "position"):
            return None

        pos = event.position()
        x_qt = int(round(float(pos.x())))
        y_qt = int(round(float(pos.y())))

        if self.plotter.interactor is None:
            return None
        window_height = int(self.plotter.interactor.height())
        y_vtk = max(0, window_height - 1 - y_qt)

        try:
            picked = bool(self._cell_picker.Pick(float(x_qt), float(y_vtk), 0.0, self.plotter.renderer))
        except Exception:
            return None
        if not picked:
            return None

        vals = tuple(float(v) for v in self._cell_picker.GetPickPosition())
        if not np.isfinite(np.asarray(vals, dtype=np.float64)).all():
            return None
        return vals

    def _update_pick_from_event(self, event: QEvent, commit: bool) -> None:
        world = self._pick_world_position_from_event(event)
        if world is None:
            return
        idx = self._world_to_indices(world)
        if idx is None:
            return

        if commit:
            self._selected_point_idx = idx
            self._preview_point_idx = None
            self.pointSelectionCommitted.emit(*idx)
        else:
            self._preview_point_idx = idx
            self.pointSelectionPreview.emit(*idx)

        self._update_pick_status_label()
        self._render_scene()

    def eventFilter(self, watched: object, event: object) -> bool:
        if self.plotter is None or watched is not self.plotter.interactor:
            return super().eventFilter(watched, event)
        if not self._point_pick_mode:
            return super().eventFilter(watched, event)
        if not isinstance(event, QEvent):
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and hasattr(event, "button"):
            if event.button() == Qt.MouseButton.LeftButton:
                self._pick_drag_active = True
                self._update_pick_from_event(event, commit=False)
                return True
        elif event_type == QEvent.Type.MouseMove:
            if self._pick_drag_active:
                self._update_pick_from_event(event, commit=False)
                return True
        elif event_type == QEvent.Type.MouseButtonRelease and hasattr(event, "button"):
            if event.button() == Qt.MouseButton.LeftButton and self._pick_drag_active:
                self._pick_drag_active = False
                self._update_pick_from_event(event, commit=True)
                return True

        return super().eventFilter(watched, event)

    def _clear_scene(self) -> None:
        if self.plotter is None:
            return
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.show_axes()
        self._markers.clear()

    def _to_image_data(self, volume: np.ndarray) -> Any:
        if self._pv_module is None:
            raise RuntimeError("PyVista module is not initialized")
        nx, ny, nz = volume.shape
        grid = self._pv_module.ImageData(
            dimensions=(nx, ny, nz),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, float(self._vertical_exaggeration)),
        )
        display_volume = self._build_display_volume(volume)
        grid.point_data["values"] = display_volume.ravel(order="F")
        grid.active_scalars_name = "values"
        return grid

    def _to_image_data_from_scalars(self, scalars: np.ndarray) -> Any:
        if self._pv_module is None:
            raise RuntimeError("PyVista module is not initialized")
        nx, ny, nz = scalars.shape
        grid = self._pv_module.ImageData(
            dimensions=(nx, ny, nz),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, float(self._vertical_exaggeration)),
        )
        grid.point_data["values"] = np.asarray(scalars, dtype=np.float32).ravel(order="F")
        grid.active_scalars_name = "values"
        return grid

    def _render_scene(self) -> None:
        if self.plotter is None:
            return

        saved_camera = None
        if self._camera_initialized and not self._force_camera_reset:
            try:
                saved_camera = self.plotter.camera_position
            except Exception:
                saved_camera = None

        self._clear_scene()
        if self._volume is None:
            self.plotter.add_text("Load a source volume to view slices", font_size=12)
            self.plotter.render()
            return

        src = self._to_image_data(self._volume)
        ox = self._state.inline_index
        oy = self._state.crossline_index
        oz = self._state.z_index
        z_scale = float(self._vertical_exaggeration)
        z_plot = self._index_to_plot_z(oz)

        # Orthogonal slices: x (inline), y (crossline), z (time/depth).
        x_slice = src.slice(normal="x", origin=(ox, 0, 0))
        y_slice = src.slice(normal="y", origin=(0, oy, 0))
        z_slice = src.slice(normal="z", origin=(0, 0, z_plot))

        cmap = "gray"
        self.plotter.add_mesh(
            x_slice,
            cmap=cmap,
            opacity=1.0,
            scalar_bar_args={"title": "Source"},
        )
        self.plotter.add_mesh(y_slice, cmap=cmap, opacity=1.0, show_scalar_bar=False)
        self.plotter.add_mesh(z_slice, cmap=cmap, opacity=1.0, show_scalar_bar=False)

        point_idx = self._preview_point_idx
        if point_idx is None:
            point_idx = self._selected_point_idx if self._selected_point_idx is not None else (ox, oy, oz)
        sx, sy, sz = point_idx
        sz_plot = self._index_to_plot_z(sz)

        # Token marker and cube outline.
        token = np.array([[sx, sy, sz_plot]], dtype=np.float32)
        self.plotter.add_points(token, color="red", point_size=18.0, render_points_as_spheres=True)

        cross_len_xy = 8.0
        cross_len_z = 8.0 * z_scale
        x_line = self._pv_module.Line((sx - cross_len_xy, sy, sz_plot), (sx + cross_len_xy, sy, sz_plot))
        y_line = self._pv_module.Line((sx, sy - cross_len_xy, sz_plot), (sx, sy + cross_len_xy, sz_plot))
        z_line = self._pv_module.Line((sx, sy, sz_plot - cross_len_z), (sx, sy, sz_plot + cross_len_z))
        self.plotter.add_mesh(x_line, color="red", line_width=3.0, show_scalar_bar=False)
        self.plotter.add_mesh(y_line, color="red", line_width=3.0, show_scalar_bar=False)
        self.plotter.add_mesh(z_line, color="red", line_width=3.0, show_scalar_bar=False)

        half = 16
        if self._pv_module is None:
            raise RuntimeError("PyVista module is not initialized")
        cube = self._pv_module.Cube(
            center=(float(sx), float(sy), sz_plot),
            x_length=float(2 * half),
            y_length=float(2 * half),
            z_length=float(2 * half) * z_scale,
        )
        self.plotter.add_mesh(cube, style="wireframe", color="red", line_width=2.0)

        if self._output is not None:
            overlay_volume = self._build_output_overlay_volume()
            if overlay_volume is None:
                overlay_volume = self._build_display_volume(self._output)
            out = self._to_image_data_from_scalars(overlay_volume)
            out_x = out.slice(normal="x", origin=(ox, 0, 0))
            out_y = out.slice(normal="y", origin=(0, oy, 0))
            out_z = out.slice(normal="z", origin=(0, 0, z_plot))
            out_clim = self._output_clim if self._output_clim is not None else (0.0, 1.0)
            self.plotter.add_mesh(
                out_x,
                cmap="bwr",
                clim=out_clim,
                opacity=float(self._state.overlay_alpha) * 0.55,
                nan_opacity=0.0,
                scalar_bar_args={"title": "Similarity (symmetric p2-p88)"},
            )
            self.plotter.add_mesh(
                out_y,
                cmap="bwr",
                clim=out_clim,
                opacity=float(self._state.overlay_alpha) * 0.55,
                nan_opacity=0.0,
                show_scalar_bar=False,
            )
            self.plotter.add_mesh(
                out_z,
                cmap="bwr",
                clim=out_clim,
                opacity=float(self._state.overlay_alpha) * 0.55,
                nan_opacity=0.0,
                show_scalar_bar=False,
            )

        if saved_camera is not None:
            try:
                self.plotter.camera_position = saved_camera
            except Exception:
                self.plotter.reset_camera()
        else:
            self.plotter.reset_camera()

        self._camera_initialized = True
        self._force_camera_reset = False
        self.plotter.render()

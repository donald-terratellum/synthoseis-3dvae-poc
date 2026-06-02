from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.tokenizer.ui.slice_viewer import SliceViewer


class MainWindow(QMainWindow):
    tokenPicked = Signal(int, int, int)
    sourceLoadRequested = Signal(str)
    outputLoadRequested = Signal(str)
    displayStateChanged = Signal(dict)
    windowClosing = Signal()
    resetUiStateRequested = Signal()
    startSearchRequested = Signal()
    cancelSearchRequested = Signal()

    _MIN_VERTICAL_EXAGGERATION = 0.1
    _MAX_VERTICAL_EXAGGERATION = 10.0
    _VERTICAL_EXAGGERATION_STEP = 0.1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Tokenizer")
        self._volume: Optional[np.ndarray] = None
        self._vertical_exaggeration = 0.1

        root = QWidget(self)
        layout = QVBoxLayout(root)

        content_row = QHBoxLayout()
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        content_row.addWidget(controls_panel, stretch=0)

        self.slice_viewer = SliceViewer()
        self.slice_viewer.setMinimumHeight(420)
        self.slice_viewer.pointSelectionPreview.connect(self._on_viewer_point_preview)
        self.slice_viewer.pointSelectionCommitted.connect(self._on_viewer_point_committed)
        content_row.addWidget(self.slice_viewer, stretch=1)

        layout.addLayout(content_row, stretch=1)

        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("Source zarr path")
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_path)
        self.browse_source_button = QPushButton("Browse")
        self.browse_source_button.clicked.connect(self._browse_source)
        source_row.addWidget(self.browse_source_button)
        self.load_source_button = QPushButton("Load")
        self.load_source_button.clicked.connect(self._emit_source_load)
        source_row.addWidget(self.load_source_button)
        controls_layout.addLayout(source_row)

        xyz_row = QHBoxLayout()
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        self.z_spin = QSpinBox()
        for widget, name in ((self.x_spin, "X"), (self.y_spin, "Y"), (self.z_spin, "Z")):
            widget.setMinimum(0)
            widget.setPrefix(f"{name}: ")
            xyz_row.addWidget(widget)
        controls_layout.addLayout(xyz_row)

        input_label = QLabel("Input Display")
        controls_layout.addWidget(input_label)

        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setMinimum(0)
        self.inline_slider.setMaximum(0)
        self.inline_slider.setTracking(True)
        self.inline_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row("Inline", self.inline_value_label, self.inline_slider)
        )
        self._bind_slider_readout(self.inline_slider, self.inline_value_label, lambda value: f"{value:d}")
        self._bind_slider_commit(self.inline_slider, self._on_display_control_changed)

        self.crossline_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossline_slider.setMinimum(0)
        self.crossline_slider.setMaximum(0)
        self.crossline_slider.setTracking(True)
        self.crossline_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row(
                "Crossline", self.crossline_value_label, self.crossline_slider
            )
        )
        self._bind_slider_readout(
            self.crossline_slider, self.crossline_value_label, lambda value: f"{value:d}"
        )
        self._bind_slider_commit(self.crossline_slider, self._on_display_control_changed)

        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setTracking(True)
        self.z_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row("Depth", self.z_value_label, self.z_slider)
        )
        self._bind_slider_readout(self.z_slider, self.z_value_label, lambda value: f"{value:d}")
        self._bind_slider_commit(self.z_slider, self._on_slice_changed)

        self.input_clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.input_clip_slider.setMinimum(0)
        self.input_clip_slider.setMaximum(100)
        self.input_clip_slider.setValue(50)
        self.input_clip_slider.setTracking(True)
        self.input_clip_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row(
                "Input Clip", self.input_clip_value_label, self.input_clip_slider
            )
        )
        self._bind_slider_readout(
            self.input_clip_slider,
            self.input_clip_value_label,
            lambda value: f"{value / 100.0:.2f}",
        )
        self._bind_slider_commit(self.input_clip_slider, self._on_display_control_changed)

        output_label = QLabel("Output Overlay")
        controls_layout.addWidget(output_label)

        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Output similarity zarr path")
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self._browse_output)
        output_row.addWidget(self.browse_output_button)
        self.load_output_button = QPushButton("Load")
        self.load_output_button.clicked.connect(self._emit_output_load)
        output_row.addWidget(self.load_output_button)
        controls_layout.addLayout(output_row)

        self.output_clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.output_clip_slider.setMinimum(0)
        self.output_clip_slider.setMaximum(100)
        self.output_clip_slider.setValue(50)
        self.output_clip_slider.setTracking(True)
        self.output_clip_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row(
                "Output Clip", self.output_clip_value_label, self.output_clip_slider
            )
        )
        self._bind_slider_readout(
            self.output_clip_slider,
            self.output_clip_value_label,
            lambda value: f"{value / 100.0:.2f}",
        )
        self._bind_slider_commit(self.output_clip_slider, self._on_display_control_changed)

        self.overlay_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_threshold_slider.setMinimum(0)
        self.overlay_threshold_slider.setMaximum(100)
        self.overlay_threshold_slider.setValue(50)
        self.overlay_threshold_slider.setTracking(True)
        self.overlay_threshold_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row(
                "Overlay Threshold", self.overlay_threshold_value_label, self.overlay_threshold_slider
            )
        )
        self._bind_slider_readout(
            self.overlay_threshold_slider,
            self.overlay_threshold_value_label,
            lambda value: f"{value / 100.0:.2f}",
        )
        self._bind_slider_commit(self.overlay_threshold_slider, self._on_display_control_changed)

        self.overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_alpha_slider.setMinimum(0)
        self.overlay_alpha_slider.setMaximum(100)
        self.overlay_alpha_slider.setValue(60)
        self.overlay_alpha_slider.setTracking(True)
        self.overlay_alpha_value_label = QLabel()
        controls_layout.addLayout(
            self._create_labeled_slider_row(
                "Overlay Alpha", self.overlay_alpha_value_label, self.overlay_alpha_slider
            )
        )
        self._bind_slider_readout(
            self.overlay_alpha_slider,
            self.overlay_alpha_value_label,
            lambda value: f"{value / 100.0:.2f}",
        )
        self._bind_slider_commit(self.overlay_alpha_slider, self._on_display_control_changed)

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Similarity Metric"))
        self.similarity_mode_combo = QComboBox()
        self.similarity_mode_combo.addItem("Cosine", "cosine")
        self.similarity_mode_combo.addItem("Dot Product", "dot")
        self.similarity_mode_combo.currentIndexChanged.connect(lambda _idx: self._on_display_control_changed(0))
        metric_row.addWidget(self.similarity_mode_combo)
        controls_layout.addLayout(metric_row)

        self.start_search_button = QPushButton("Start Search")
        self.start_search_button.clicked.connect(lambda: self.startSearchRequested.emit())
        controls_layout.addWidget(self.start_search_button)

        self.job_progress_label = QLabel("Search progress")
        controls_layout.addWidget(self.job_progress_label)
        self.job_progress_bar = QProgressBar()
        self.job_progress_bar.setMinimum(0)
        self.job_progress_bar.setMaximum(100)
        self.job_progress_bar.setValue(0)
        controls_layout.addWidget(self.job_progress_bar)
        self.job_cancel_button = QPushButton("Cancel Search")
        self.job_cancel_button.clicked.connect(lambda: self.cancelSearchRequested.emit())
        controls_layout.addWidget(self.job_cancel_button)
        self.set_job_progress_visible(False)

        self.pick_button = QPushButton("Pick Token")
        self.pick_button.clicked.connect(self._emit_token_pick)
        controls_layout.addWidget(self.pick_button)

        self.reset_state_button = QPushButton("Reset UI State")
        self.reset_state_button.clicked.connect(lambda: self.resetUiStateRequested.emit())
        controls_layout.addWidget(self.reset_state_button)

        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)

        self._increase_vertical_exag_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        self._increase_vertical_exag_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._increase_vertical_exag_shortcut.activated.connect(
            lambda: self._adjust_vertical_exaggeration(self._VERTICAL_EXAGGERATION_STEP)
        )

        self._decrease_vertical_exag_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        self._decrease_vertical_exag_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._decrease_vertical_exag_shortcut.activated.connect(
            lambda: self._adjust_vertical_exaggeration(-self._VERTICAL_EXAGGERATION_STEP)
        )

        self.setCentralWidget(root)

    def _create_labeled_slider_row(
        self,
        title: str,
        value_label: QLabel,
        slider: QSlider,
    ) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel(title))
        header.addStretch(1)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(value_label)
        row.addLayout(header)
        row.addWidget(slider)
        return row

    def _bind_slider_readout(self, slider: QSlider, label: QLabel, formatter) -> None:
        label.setText(formatter(int(slider.value())))
        slider.valueChanged.connect(lambda value, fmt=formatter, target=label: target.setText(fmt(int(value))))

    def _bind_slider_commit(self, slider: QSlider, handler) -> None:
        slider.sliderReleased.connect(lambda s=slider, h=handler: h(int(s.value())))
        slider.valueChanged.connect(
            lambda value, s=slider, h=handler: h(int(value)) if not s.isSliderDown() else None
        )

    def _adjust_vertical_exaggeration(self, delta: float) -> None:
        self.set_vertical_exaggeration(self._vertical_exaggeration + float(delta))

    def set_vertical_exaggeration(self, value: float) -> None:
        exag = max(self._MIN_VERTICAL_EXAGGERATION, min(self._MAX_VERTICAL_EXAGGERATION, float(value)))
        if abs(exag - self._vertical_exaggeration) < 1e-9:
            return
        self._vertical_exaggeration = exag
        self.slice_viewer.set_vertical_exaggeration(exag)
        self.status_label.setText(f"Vertical exaggeration: {exag:.1f}x")

    def set_patch_shape(self, patch_size: int | tuple[int, int, int] | list[int]) -> None:
        self.slice_viewer.set_patch_shape(patch_size)

    def get_vertical_exaggeration(self) -> float:
        return float(self._vertical_exaggeration)

    def _browse_source(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Zarr",
            "",
            "Zarr or all files (*)",
        )
        if selected:
            self.source_path.setText(selected)

    def _emit_source_load(self) -> None:
        path = self.source_path.text().strip()
        if not path:
            self.status_label.setText("Source path is empty")
            return
        self.status_label.setText(f"Loading source: {path}")
        self.sourceLoadRequested.emit(path)

    def _browse_output(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Output Zarr",
            "",
            "Zarr or all files (*)",
        )
        if selected:
            self.output_path.setText(selected)

    def _emit_output_load(self) -> None:
        path = self.output_path.text().strip()
        if not path:
            self.status_label.setText("Output path is empty")
            return
        self.status_label.setText(f"Loading output: {path}")
        self.outputLoadRequested.emit(path)

    def set_volume(self, volume: np.ndarray) -> None:
        if volume.ndim != 3:
            raise ValueError("volume must be 3D")
        self._volume = volume
        nx, ny, nz = volume.shape

        self.x_spin.setMaximum(max(0, nx - 1))
        self.y_spin.setMaximum(max(0, ny - 1))
        self.z_spin.setMaximum(max(0, nz - 1))
        self.z_slider.setMaximum(max(0, nz - 1))
        self.inline_slider.setMaximum(max(0, nx - 1))
        self.crossline_slider.setMaximum(max(0, ny - 1))

        self.inline_slider.setValue(nx // 2 if nx > 0 else 0)
        self.crossline_slider.setValue(ny // 2 if ny > 0 else 0)
        self.z_slider.setValue(nz // 2 if nz > 0 else 0)
        self.slice_viewer.set_selected_point(int(self.x_spin.value()), int(self.y_spin.value()), int(self.z_spin.value()))

        self.status_label.setText(f"Volume loaded: shape={volume.shape}")
        self.displayStateChanged.emit(self.get_display_state())
        self.slice_viewer.set_source_volume(volume)

    def _on_slice_changed(self, value: int) -> None:
        self.z_spin.setValue(value)
        self.status_label.setText(f"Slice updated: z={value}")
        self.displayStateChanged.emit(self.get_display_state())

    def _on_display_control_changed(self, _value: int) -> None:
        self.displayStateChanged.emit(self.get_display_state())

    def _emit_token_pick(self) -> None:
        x = int(self.x_spin.value())
        y = int(self.y_spin.value())
        z = int(self.z_spin.value())
        self.slice_viewer.set_selected_point(x, y, z)
        self.z_slider.setValue(z)
        self.status_label.setText(f"Token picked at ({x}, {y}, {z})")
        self.tokenPicked.emit(x, y, z)

    def _on_viewer_point_preview(self, x: int, y: int, z: int) -> None:
        self.status_label.setText(f"Selecting point: ({x}, {y}, {z})")

    def _on_viewer_point_committed(self, x: int, y: int, z: int) -> None:
        self.x_spin.setValue(int(x))
        self.y_spin.setValue(int(y))
        self.z_spin.setValue(int(z))
        self.status_label.setText(f"Selected point: ({x}, {y}, {z})")

    def get_display_state(self) -> dict:
        return {
            "inline_index": int(self.inline_slider.value()),
            "crossline_index": int(self.crossline_slider.value()),
            "z_index": int(self.z_slider.value()),
            "input_clip": float(self.input_clip_slider.value()) / 100.0,
            "output_clip": float(self.output_clip_slider.value()) / 100.0,
            "overlay_threshold": float(self.overlay_threshold_slider.value()) / 100.0,
            "overlay_alpha": float(self.overlay_alpha_slider.value()) / 100.0,
            "similarity_mode": str(self.similarity_mode_combo.currentData() or "cosine"),
        }

    def set_output_volume(self, volume: np.ndarray) -> None:
        self.slice_viewer.set_output_volume(volume)

    def set_slice_view_state(self, snapshot: dict) -> None:
        from src.tokenizer.ui.slice_viewer import SliceViewState

        self.slice_viewer.set_state(
            SliceViewState(
                inline_index=int(snapshot.get("inline_index", 0)),
                crossline_index=int(snapshot.get("crossline_index", 0)),
                z_index=int(snapshot.get("z_index", 0)),
                input_clip=float(snapshot.get("input_clip", 0.5)),
                output_clip=float(snapshot.get("output_clip", 0.5)),
                overlay_threshold=float(snapshot.get("overlay_threshold", 0.5)),
                overlay_alpha=float(snapshot.get("overlay_alpha", 0.6)),
            )
        )

        mode = str(snapshot.get("similarity_mode", "cosine"))
        idx = self.similarity_mode_combo.findData(mode)
        if idx >= 0 and idx != self.similarity_mode_combo.currentIndex():
            self.similarity_mode_combo.setCurrentIndex(idx)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.windowClosing.emit()
        super().closeEvent(event)

    def set_job_progress_visible(self, visible: bool) -> None:
        self.job_progress_label.setVisible(visible)
        self.job_progress_bar.setVisible(visible)
        self.job_cancel_button.setVisible(visible)

    def update_job_progress(self, completed: int, total: int, eta_seconds: Optional[float]) -> None:
        total_safe = max(1, total)
        pct = int(round((100.0 * completed) / total_safe))
        self.job_progress_bar.setValue(max(0, min(100, pct)))
        if eta_seconds is None:
            self.job_progress_label.setText(f"Search progress: {completed}/{total}")
        else:
            self.job_progress_label.setText(
                f"Search progress: {completed}/{total}, ETA {eta_seconds:.1f}s"
            )

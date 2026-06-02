import os
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import zarr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

from src.tokenizer.ui.controller import TokenizerController
from src.tokenizer.ui.main_window import MainWindow


class TokenizerUiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_slider_and_token_pick_signal(self):
        window = MainWindow()
        volume = np.zeros((6, 7, 8), dtype=np.float32)
        window.set_volume(volume)

        received = []
        window.tokenPicked.connect(lambda x, y, z: received.append((x, y, z)))

        window.z_slider.setValue(5)
        self.assertIn("z=5", window.status_label.text())

        window.x_spin.setValue(1)
        window.y_spin.setValue(2)
        window.z_spin.setValue(3)
        window.pick_button.click()

        self.assertTrue(received)
        self.assertEqual(received[-1], (1, 2, 3))
        self.assertEqual(window.get_display_state()["z_index"], 3)

    def test_load_button_emits_source_request_signal(self):
        window = MainWindow()
        received = []
        window.sourceLoadRequested.connect(lambda path: received.append(path))

        window.source_path.setText("/tmp/example.zarr")
        window.load_source_button.click()

        self.assertEqual(received, ["/tmp/example.zarr"])

    def test_load_button_emits_output_request_signal(self):
        window = MainWindow()
        received = []
        window.outputLoadRequested.connect(lambda path: received.append(path))

        window.output_path.setText("/tmp/output.zarr")
        window.load_output_button.click()

        self.assertEqual(received, ["/tmp/output.zarr"])

    def test_controller_loads_volume_from_zarr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "source.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("data", data=np.ones((4, 5, 6), dtype=np.float32))

            window = MainWindow()
            controller = TokenizerController(window)
            controller.on_source_load_requested(str(zarr_path))

            self.assertIsNotNone(controller.volume)
            self.assertEqual(controller.volume.shape, (4, 5, 6))
            self.assertIn("Source loaded", window.status_label.text())

    def test_display_controls_update_controller_state(self):
        window = MainWindow()
        controller = TokenizerController(window)
        volume = np.zeros((10, 11, 12), dtype=np.float32)
        controller.set_volume(volume)

        window.inline_slider.setValue(7)
        window.crossline_slider.setValue(8)
        window.z_slider.setValue(9)
        window.input_clip_slider.setValue(25)
        window.output_clip_slider.setValue(75)
        window.overlay_alpha_slider.setValue(40)

        self.assertEqual(controller.display_state.inline_index, 7)
        self.assertEqual(controller.display_state.crossline_index, 8)
        self.assertEqual(controller.display_state.z_index, 9)
        self.assertAlmostEqual(controller.display_state.input_clip, 0.25)
        self.assertAlmostEqual(controller.display_state.output_clip, 0.75)
        self.assertAlmostEqual(controller.display_state.overlay_alpha, 0.40)
        self.assertEqual(window.inline_value_label.text(), "7")
        self.assertEqual(window.crossline_value_label.text(), "8")
        self.assertEqual(window.z_value_label.text(), "9")
        self.assertEqual(window.input_clip_value_label.text(), "0.25")
        self.assertEqual(window.output_clip_value_label.text(), "0.75")
        self.assertEqual(window.overlay_alpha_value_label.text(), "0.40")

    def test_controller_overlay_preview_updates_from_output_volume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.zarr"
            output_path = tmp / "output.zarr"

            src = zarr.open(str(source_path), mode="w")
            src_data = np.ones((8, 9, 10), dtype=np.float32)
            src.create_array("data", data=src_data)

            out = zarr.open(str(output_path), mode="w")
            out_data = np.zeros((8, 9, 10), dtype=np.float32)
            out_data[4, :, :] = 2.0
            out.create_array("data", data=out_data)

            window = MainWindow()
            controller = TokenizerController(window)
            controller.on_source_load_requested(str(source_path))
            controller.on_output_load_requested(str(output_path))

            self.assertTrue(controller.display_state.output_loaded)
            baseline_std = controller.display_state.overlay_preview_std

            window.overlay_alpha_slider.setValue(100)
            self.assertNotEqual(controller.display_state.overlay_preview_std, baseline_std)
            self.assertIn("Overlay preview updated", window.status_label.text())

    def test_slice_viewer_receives_volume_and_state(self):
        window = MainWindow()
        controller = TokenizerController(window)

        volume = np.zeros((12, 13, 14), dtype=np.float32)
        controller.set_volume(volume)

        self.assertIsNotNone(window.slice_viewer._volume)
        self.assertEqual(tuple(window.slice_viewer._volume.shape), (12, 13, 14))

        window.inline_slider.setValue(4)
        window.crossline_slider.setValue(5)
        window.z_slider.setValue(6)
        snapshot = window.get_display_state()
        self.assertEqual(snapshot["inline_index"], 4)
        self.assertEqual(snapshot["crossline_index"], 5)
        self.assertEqual(snapshot["z_index"], 6)

    def test_similarity_overlay_uses_percentile_color_bounds(self):
        window = MainWindow()

        out = np.linspace(-2.0, 10.0, 200, dtype=np.float32).reshape(5, 5, 8)
        window.set_output_volume(out)

        self.assertIsNotNone(window.slice_viewer._output_clim)
        lo, hi = window.slice_viewer._output_clim
        span = max(abs(float(np.percentile(out, 2.0))), abs(float(np.percentile(out, 88.0))))
        self.assertAlmostEqual(lo, -span, places=5)
        self.assertAlmostEqual(hi, span, places=5)

    def test_similarity_overlay_masks_values_below_output_clip_threshold(self):
        window = MainWindow()

        out = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(4, 4, 4)
        window.set_output_volume(out)
        window.set_slice_view_state(
            {
                "inline_index": 0,
                "crossline_index": 0,
                "z_index": 0,
                "input_clip": 0.5,
                "output_clip": 0.5,
                "overlay_alpha": 0.6,
            }
        )

        masked = window.slice_viewer._build_output_overlay_volume()
        self.assertIsNotNone(masked)
        assert masked is not None
        finite = masked[np.isfinite(masked)]
        self.assertGreater(finite.size, 0)

        lo, hi = window.slice_viewer._output_clim
        threshold = 0.5 * max(abs(lo), abs(hi))
        self.assertGreaterEqual(float(np.abs(finite).min()), float(threshold) - 1e-6)
        self.assertTrue(np.isnan(masked).any())

    def test_arrow_keys_adjust_vertical_exaggeration(self):
        window = MainWindow()
        window.show()
        self._app.processEvents()

        baseline = window.get_vertical_exaggeration()
        QTest.keyClick(window, Qt.Key.Key_Up)
        self._app.processEvents()
        self.assertGreater(window.get_vertical_exaggeration(), baseline)

        QTest.keyClick(window, Qt.Key.Key_Down)
        self._app.processEvents()
        self.assertAlmostEqual(window.get_vertical_exaggeration(), baseline)

        window.set_vertical_exaggeration(0.1)
        QTest.keyClick(window, Qt.Key.Key_Down)
        self._app.processEvents()
        self.assertAlmostEqual(window.get_vertical_exaggeration(), 0.1)

        window.set_vertical_exaggeration(10.0)
        QTest.keyClick(window, Qt.Key.Key_Up)
        self._app.processEvents()
        self.assertAlmostEqual(window.get_vertical_exaggeration(), 10.0)

    def test_point_pick_mode_toggle_updates_viewer_state(self):
        window = MainWindow()

        self.assertFalse(window.slice_viewer.point_pick_action.isChecked())
        self.assertEqual(window.slice_viewer.pick_status_label.text(), "Selection off")

        window.slice_viewer.point_pick_action.setChecked(True)
        self.assertTrue(window.slice_viewer.point_pick_action.isChecked())
        self.assertIn("Selection", window.slice_viewer.pick_status_label.text())

        window.slice_viewer.point_pick_action.setChecked(False)
        self.assertFalse(window.slice_viewer.point_pick_action.isChecked())
        self.assertEqual(window.slice_viewer.pick_status_label.text(), "Selection off")

    def test_point_pick_commit_updates_xyz_selection(self):
        window = MainWindow()
        controller = TokenizerController(window)
        volume = np.zeros((10, 11, 12), dtype=np.float32)
        controller.set_volume(volume)

        window.slice_viewer.pointSelectionCommitted.emit(3, 4, 5)

        self.assertEqual(window.x_spin.value(), 3)
        self.assertEqual(window.y_spin.value(), 4)
        self.assertEqual(window.z_spin.value(), 5)
        self.assertIn("Selected point", window.status_label.text())

    def test_progress_panel_lifecycle_during_background_job(self):
        window = MainWindow()
        controller = TokenizerController(window)
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        controller.set_volume(volume)

        self.assertTrue(window.job_progress_bar.isHidden())
        window.start_search_button.click()
        self.assertFalse(window.job_progress_bar.isHidden())

        deadline = __import__("time").time() + 5.0
        while __import__("time").time() < deadline and not window.job_progress_bar.isHidden():
            self._app.processEvents()

        self.assertTrue(window.job_progress_bar.isHidden())
        self.assertIn("Search", window.status_label.text())

    def test_controller_state_payload_save_restore_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "ui_state.json"

            window_a = MainWindow()
            controller_a = TokenizerController(window_a, state_file=state_path)
            volume = np.zeros((12, 13, 14), dtype=np.float32)
            controller_a.set_volume(volume)

            window_a.source_path.setText("/tmp/source.zarr")
            window_a.inline_slider.setValue(7)
            window_a.crossline_slider.setValue(8)
            window_a.z_slider.setValue(9)
            window_a.input_clip_slider.setValue(33)
            window_a.output_clip_slider.setValue(44)
            window_a.overlay_threshold_slider.setValue(55)
            window_a.overlay_alpha_slider.setValue(66)
            window_a.x_spin.setValue(3)
            window_a.y_spin.setValue(4)
            window_a.z_spin.setValue(5)
            controller_a.save_session_state()

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("display_state", payload)
            self.assertEqual(payload["display_state"]["overlay_threshold"], 0.55)

            window_b = MainWindow()
            controller_b = TokenizerController(window_b, state_file=state_path)
            controller_b.restore_session_state(auto_load_source=False)
            controller_b.set_volume(np.zeros((12, 13, 14), dtype=np.float32))

            self.assertEqual(window_b.source_path.text(), "/tmp/source.zarr")
            self.assertEqual(window_b.inline_slider.value(), 7)
            self.assertEqual(window_b.crossline_slider.value(), 8)
            self.assertEqual(window_b.z_slider.value(), 9)
            self.assertEqual(window_b.overlay_threshold_slider.value(), 55)
            self.assertEqual(window_b.overlay_alpha_slider.value(), 66)

    def test_reset_ui_state_button_removes_state_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "ui_state.json"

            window = MainWindow()
            controller = TokenizerController(window, state_file=state_path)
            controller.save_session_state()
            self.assertTrue(state_path.exists())

            window.reset_state_button.click()

            self.assertFalse(state_path.exists())
            self.assertIn("UI state reset", window.status_label.text())


if __name__ == "__main__":
    unittest.main()

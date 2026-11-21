import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import zarr

from scripts.streamed_overlap_audit import (
    DEFAULT_SEISMIC_KEY,
    CONTEXT_KEYS,
    FLUID_KEYS,
    LABEL_KEYS,
    PENDING,
    ReadBudget,
    Source,
    bounded_crop,
    canonical_overlap,
    content_hash,
    crop_statistics,
    discover_sources,
    inspect_source,
    lithology_masks,
    parameter_metadata,
    run_audit,
)


def make_source(root, name="seismic__fixture", complete=True):
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "model_data.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.create_array(DEFAULT_SEISMIC_KEY, data=np.ones((40, 40, 150), dtype="f4"),
                       chunks=(16, 16, 150))
    if complete:
        age = np.broadcast_to(np.arange(151, dtype="f4") / 8, (40, 40, 151))
        group.create_array(LABEL_KEYS[0], data=age, chunks=(16, 16, 151))
        group.create_array(LABEL_KEYS[1], data=np.ones(age.shape, dtype="f4"),
                           chunks=(16, 16, 151))
        for key in (*FLUID_KEYS, LABEL_KEYS[-1], *CONTEXT_KEYS):
            mask = np.zeros(age.shape, dtype="u1")
            if key == FLUID_KEYS[0]:
                mask[:, :, 60:65] = 1
            group.create_array(key, data=mask, chunks=(16, 16, 151))
    (directory / "model_parameters_fixture.txt").write_text(
        "bandpass_bandlimits: 4.0, 35.0\nSHA: fixture-only\n"
    )
    (directory / "model_config_fixture.json").write_text(json.dumps({
        "digi": 4, "bandwidth_ord": 4, "cube_shape": [40, 40, 151],
        "seeds": {"model_params_seed": 123},
    }))
    return path


class StreamedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_overlap_geometry_and_invalid_support(self):
        origin = (0, 0, 0)
        self.assertEqual(canonical_overlap(origin, origin, (32, 32, 64), True), 1)
        self.assertEqual(canonical_overlap(origin, (0, 0, 32), (32, 32, 64), True), .5)
        self.assertEqual(canonical_overlap(origin, (8, 8, 16), (32, 32, 64), True), .75**3)
        self.assertEqual(canonical_overlap(origin, (0, 0, 64), (32, 32, 64), True), 0)
        self.assertEqual(canonical_overlap(origin, origin, (32, 32, 64), False), 0)
        with self.assertRaises(ValueError):
            canonical_overlap(origin, origin, (32, 0, 64), True)
        with self.assertRaises(ValueError):
            canonical_overlap((-1, 0, 0), origin, (32, 32, 64), False)

    def test_channel_override_threshold_and_raw_water_disagreement(self):
        lithology = np.array([0, -1, 0, 0.5, 1, -1], dtype="f4").reshape(1, 1, 6)
        channels = np.array([1, 2, 0.5, 0, 0, 0], dtype="f4").reshape(1, 1, 6)
        masks = lithology_masks(lithology, channels)
        np.testing.assert_array_equal(masks["sand"].ravel(), [True, True, False, True, True, False])
        np.testing.assert_array_equal(masks["water"].ravel(), [False, False, False, False, False, True])
        crop = {
            DEFAULT_SEISMIC_KEY: np.ones_like(lithology),
            LABEL_KEYS[0]: np.ones_like(lithology),
            LABEL_KEYS[1]: lithology,
            LABEL_KEYS[-1]: channels,
            **{key: np.zeros_like(lithology) for key in FLUID_KEYS},
        }
        stats = crop_statistics(crop, DEFAULT_SEISMIC_KEY)
        self.assertEqual(stats["channel_added_sand_voxels"], 2)
        self.assertEqual(stats["channel_over_raw_water_voxels"], 1)
        self.assertEqual(stats["sand_fraction_over_rock_at_diagnostic_threshold"], 4 / 5)
        self.assertEqual(stats["sand_without_any_closure_mask_voxels"], 4)
        # A closure on shale cannot create sand; overlap with channels is reported,
        # not silently interpreted as the final seismic fluid.
        crop[FLUID_KEYS[0]][0, 0, :3] = 1
        stats = crop_statistics(crop, DEFAULT_SEISMIC_KEY)
        self.assertEqual(stats["sand_fraction_over_rock_at_diagnostic_threshold"], 4 / 5)
        self.assertEqual(stats["closure_positive_outside_corrected_sand_voxels"], 1)
        self.assertEqual(stats["channel_hydrocarbon_closure_overlap_voxels"], 2)
        self.assertIsNone(stats["semantic_target"])

    def test_lithology_mask_rejects_misaligned_or_nonfinite_channels(self):
        with self.assertRaisesRegex(ValueError, "matching 3D"):
            lithology_masks(np.zeros((1, 1, 2)), np.zeros((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "finite"):
            lithology_masks(np.zeros((1, 1, 1)), np.full((1, 1, 1), np.nan))

    def test_split_exclusions_prune_stores_and_skip_incomplete_sources(self):
        train = make_source(self.root)
        validation = make_source(self.root / "validation")
        make_source(self.root / "test")
        make_source(self.root / "test_synthoseis")
        active = make_source(self.root, "seismic__active")
        (active.parent.parent / "temp_folder__active").mkdir()
        found = discover_sources(self.root)
        self.assertEqual({source.path for source in found}, {train.resolve(), validation.resolve()})
        self.assertEqual({source.split for source in found}, {"train", "validation"})

    def test_symlinked_holdout_alias_fails_closed(self):
        make_source(self.root)
        validation = make_source(self.root / "validation")
        (self.root / "hidden_holdout_alias").symlink_to(validation.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            discover_sources(self.root)
        with self.assertRaisesRegex(ValueError, "parent training-source"):
            discover_sources(self.root / "validation")

    def test_validation_root_cannot_alias_final_test(self):
        make_source(self.root)
        make_source(self.root / "test_synthoseis")
        (self.root / "validation").symlink_to(self.root / "test_synthoseis",
                                             target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinked split"):
            discover_sources(self.root)

    def test_parameter_metadata_requires_exact_band_and_matching_config(self):
        path = make_source(self.root)
        directory = path.parent
        info = parameter_metadata(directory)
        self.assertEqual(info["bandlimits_hz"], [4, 35])
        self.assertEqual(info["sample_spacing_ms_from_digi"], 4)
        parameter = directory / "model_parameters_fixture.txt"
        for text in (
            "SHA: fixture-only\n",
            "bandpass_bandlimits: 4, 35\nbandpass_bandlimits: 4, 35\nSHA: fixture-only\n",
            "bandpass_bandlimits: 4, 126\nSHA: fixture-only\n",
            "bandpass_bandlimits: nan, 35\nSHA: fixture-only\n",
        ):
            with self.subTest(text=text):
                parameter.write_text(text)
                with self.assertRaises(ValueError):
                    parameter_metadata(directory)
        parameter.write_text("bandpass_bandlimits: 4, 35\nSHA: fixture-only\n")
        (directory / "model_config_fixture.json").unlink()
        with self.assertRaises(FileNotFoundError):
            parameter_metadata(directory)

    def test_missing_labels_are_inventory_issues_not_zero_arrays(self):
        path = make_source(self.root, complete=False)
        info = inspect_source(Source(path, "validation"), DEFAULT_SEISMIC_KEY)
        self.assertFalse(info["semantic_arrays_complete"])
        for key in LABEL_KEYS:
            self.assertIn(f"Missing array: {key}", info["issues"])
        group = zarr.open_group(str(path), mode="r")
        with self.assertRaises(KeyError):
            bounded_crop(group, (0, 0, 0), ReadBudget(), DEFAULT_SEISMIC_KEY, 1)

    def test_bounded_reads_offset_finite_and_decode_budget(self):
        path = make_source(self.root)
        group = zarr.open_group(str(path), mode="r")
        budget = ReadBudget()
        crop = bounded_crop(group, (0, 0, 0), budget, DEFAULT_SEISMIC_KEY, 1)
        self.assertEqual(float(crop[LABEL_KEYS[0]][0, 0, 0]), 1 / 8)
        array = group[DEFAULT_SEISMIC_KEY]
        with self.assertRaisesRegex(ValueError, "bounds"):
            budget.read(array, (39, 0, 0), (2, 1, 1))
        with self.assertRaisesRegex(ValueError, "region budget"):
            budget.read(array, (0, 0, 0), (40, 40, 150))
        with self.assertRaisesRegex(ValueError, "decoded-chunk"):
            ReadBudget(max_decoded_bytes_per_read=1).read(array, (0, 0, 0), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "total requested"):
            ReadBudget(max_total_requested_voxels=1).read(array, (0, 0, 0), (1, 1, 2))
        writable = zarr.open_group(str(path), mode="a")
        writable[DEFAULT_SEISMIC_KEY][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            budget.read(array, (0, 0, 0), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "bounds"):
            bounded_crop(group, (0, 0, 0), ReadBudget(), DEFAULT_SEISMIC_KEY, -1)

    def test_fixture_end_to_end_has_durable_unapproved_contract(self):
        make_source(self.root / "sources")
        make_source(self.root / "sources" / "validation")
        make_source(self.root / "sources" / "validation", "seismic__missing", complete=False)
        output = self.root / "output"
        config = {
            "source_root": str(self.root / "sources"),
            "seismic_key": DEFAULT_SEISMIC_KEY,
            "sample_sources_per_split": 1,
            "label_z_offset_hypothesis": 1,
            "seed": 20260926,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_audit(config, output)
        self.assertEqual(result["status"], PENDING)
        self.assertFalse(result["production_training_allowed"])
        self.assertEqual(result["inventory"]["validation"]["complete_semantic_sources"], 1)
        self.assertEqual(len(result["examples"]), 2)
        self.assertTrue(result["label_policy"]["channel_identity_preserved_under_fluid_fill"])
        self.assertFalse(result["label_policy"]["closures_define_lithology"])
        self.assertGreater(result["read_budget"]["reads"], 0)
        manifest = json.loads((output / "source_manifest.json").read_text())
        self.assertEqual(content_hash(manifest), result["manifest_sha256"])
        contract = json.loads((output / "target_contract.json").read_text())
        self.assertFalse(contract["approved"])
        self.assertTrue((output / "audit_examples.html").is_file())
        with self.assertRaisesRegex(ValueError, "read-only source tree"):
            run_audit(config, self.root / "sources" / "forbidden-output")
        for example in result["examples"]:
            self.assertFalse(example["alignment_verified"])
            self.assertEqual(example["physical_overlap_A_to_shift"], .75)
            self.assertEqual(example["physical_overlap_A_to_disjoint"], 0)
            self.assertIsNone(example["views"]["A"]["statistics"]["semantic_target"])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            run_audit(config, output)


class AuditRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        original = Path(__file__).resolve().parents[1] / "scripts/streamed_overlap_experiment.sh"
        self.script = self.root / "scripts/streamed_overlap_experiment.sh"
        shutil.copy2(original, self.script)
        self.config = self.root / "config.json"
        self.config.write_text("{}")
        self.output = self.root / "external"

    def invoke(self, *args, env=None):
        return subprocess.run(
            ["bash", str(self.script), *args], capture_output=True, text=True, env=env,
        )

    def test_production_modes_and_traversal_are_rejected_without_side_effects(self):
        for mode in ("train", "resume", "smoke", "evaluate"):
            result = self.invoke("--mode", mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(PENDING, result.stderr)
        result = self.invoke("--mode", "audit", "--run-id", "../escape",
                             "--config", str(self.config))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "logs").exists())

    def test_print_mode_has_no_side_effects_and_existing_output_is_rejected(self):
        args = ("--mode", "audit", "--config", str(self.config), "--run-id", "fixture",
                "--output-root", str(self.output))
        result = self.invoke(*args, "--print-commands")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts.streamed_overlap_runner", result.stdout)
        self.assertFalse((self.root / "logs").exists())
        (self.output / "fixture").mkdir(parents=True)
        result = self.invoke(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing existing", result.stderr)

    def test_failed_preflight_is_logged_and_exit_status_is_preserved(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-q",
                        "--allow-empty", "-m", "Fixture"], check=True)
        fake_python = self.root / "fake-python"
        fake_python.write_text("#!/bin/sh\nprintf 'fixture stdout\\n'\nprintf 'fixture stderr\\n' >&2\nexit 7\n")
        fake_python.chmod(0o700)
        result = self.invoke(
            "--mode", "audit", "--config", str(self.config), "--run-id", "failure",
            "--output-root", str(self.output),
            env={**os.environ, "PYTHON": str(fake_python)},
        )
        self.assertEqual(result.returncode, 7)
        log = (self.root / "logs/failure/training.log").read_text()
        self.assertIn("fixture stdout", log)
        self.assertIn("fixture stderr", log)
        self.assertIn("AUDIT FAILED: exit 7", log)
        self.assertFalse(self.output.exists())

    def test_missing_config_is_logged_from_preflight(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-q",
                        "--allow-empty", "-m", "Fixture"], check=True)
        result = self.invoke(
            "--mode", "audit", "--config", str(self.root / "missing.json"),
            "--run-id", "missing-config", "--output-root", str(self.output),
            env={**os.environ, "PYTHON": "/usr/bin/true"},
        )
        self.assertEqual(result.returncode, 2)
        log = (self.root / "logs/missing-config/training.log").read_text()
        self.assertIn("Missing audit config", log)
        self.assertIn("AUDIT FAILED", log)

    def test_tee_failure_is_not_reported_as_success(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-q",
                        "--allow-empty", "-m", "Fixture"], check=True)
        binaries = self.root / "bin"
        binaries.mkdir()
        tee = binaries / "tee"
        tee.write_text("#!/bin/sh\ncat > /dev/null\nprintf 'fixture log write failure\\n' >&2\nexit 9\n")
        tee.chmod(0o700)
        result = self.invoke(
            "--mode", "audit", "--config", str(self.config), "--run-id", "log-failure",
            "--output-root", str(self.output),
            env={**os.environ, "PYTHON": "/usr/bin/true", "PATH": f"{binaries}:{os.environ['PATH']}"},
        )
        self.assertEqual(result.returncode, 9)
        self.assertIn("fixture log write failure", result.stderr)

    def test_monitoring_print_mode_starts_nothing(self):
        result = self.invoke(
            "--mode", "tensorboard", "--run-dir", str(self.output),
            "--run-id", "monitor", "--print-commands",
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--action tensorboard", result.stdout)
        self.assertFalse((self.root / "logs").exists())


if __name__ == "__main__":
    unittest.main()

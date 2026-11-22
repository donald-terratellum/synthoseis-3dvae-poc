import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_initialization import _publish_copy, prepare_arm_initial_checkpoints
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_matching import verify_arm_initial_checkpoint
from scripts.streamed_overlap_model import build_optimizer, fresh_e1_model
from scripts.train import EarlyStoppingState


class FixtureSampler:
    def __init__(self, seed, source_hash):
        streams = np.random.SeedSequence(seed).spawn(2)
        self.state = {
            "schema": "streamed-pair-sampler-v1", "split": "train", "source_manifest_hash": source_hash,
            "batches_drawn": 0, "pair_rng": np.random.default_rng(streams[0]).bit_generator.state,
            "augmentation_rng": np.random.default_rng(streams[1]).bit_generator.state,
        }

    def state_dict(self):
        return copy.deepcopy(self.state)


class ArmInitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.source = root / "source"
        self.source.mkdir()
        self.run_dir = root / "arm"
        self.run_dir.mkdir()
        self.journal = BatchJournal(self.run_dir / "batch_metrics.csv", STEP_FIELDS, create=True)
        self.addCleanup(self.journal.close)
        self.journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
        self.config = {
            "schema": "streamed-matching-config-v1", "run_id": "synthetic", "experiment_seed": 1,
            "model_seed": 2, "head_seed": 3, "sampler_seed": 4, "monitor_seed": 5, "validation_seed": 6,
            "final_epoch": 2, "batches_per_epoch": 3, "sparse_probability": 0.5,
        }
        self.hashes = {
            "config": "a" * 64, "source": "b" * 64, "target": "c" * 64,
            "validation_manifest": "d" * 64, "matching": content_hash(self.config),
        }
        self.model = fresh_e1_model("S3", model_seed=2, head_seed=3)
        self.optimizer = build_optimizer(self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        self.sampler = FixtureSampler(4, self.hashes["source"])
        self.stopping = EarlyStoppingState(float("inf"), 0)

    def prepare(self, **overrides):
        arguments = {
            "model": self.model, "optimizer": self.optimizer, "scheduler": self.scheduler,
            "sampler": self.sampler, "early_stopping": self.stopping, "journal": self.journal,
            "hashes": self.hashes, "matching_config": self.config, "source_root": self.source,
        }
        return prepare_arm_initial_checkpoints(**{**arguments, **overrides})

    def test_publishes_separate_identical_fresh_checkpoints_and_reuses_them(self):
        result = self.prepare()
        initial, last = result["initial_checkpoint"], result["checkpoint_path"]
        self.assertNotEqual(initial.stat().st_ino, last.stat().st_ino)
        self.assertEqual(initial.read_bytes(), last.read_bytes())
        self.assertEqual(result["initial_checkpoint_sha256"], verify_arm_initial_checkpoint(
            initial, last, arm="S3", matching_config=self.config, expected_hashes=self.hashes,
        ))
        self.assertEqual(self.prepare(), result)
        assert self.journal.position is not None
        self.assertEqual(self.journal.position.logical_step, 0)
        self.assertEqual(self.journal.rows, [])

    def test_interrupted_publication_recovers_without_replacing_initial(self):
        def interrupted(source, destination, expected):
            if destination.name == "last.pt":
                raise OSError("interrupted")
            return _publish_copy(source, destination, expected)

        with patch("scripts.streamed_overlap_initialization._publish_copy", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.prepare()
        initial = self.run_dir / "initial.pt"
        self.assertTrue(initial.is_file())
        original = initial.read_bytes()
        self.assertFalse((self.run_dir / "last.pt").exists())
        result = self.prepare()
        self.assertEqual(initial.read_bytes(), original)
        self.assertEqual(result["checkpoint_path"].read_bytes(), original)

    def test_failed_copy_removes_only_the_new_partial_file(self):
        source = self.run_dir / "fixture-source"
        source.write_bytes(b"source")
        destination = self.run_dir / "fixture-destination"
        with patch("scripts.streamed_overlap_initialization._digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "changed during publication"):
                _publish_copy(source, destination, "1" * 64)
        self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), b"source")

    def test_failed_copy_does_not_remove_replaced_destination(self):
        source = self.run_dir / "fixture-source"
        source.write_bytes(b"source")
        destination = self.run_dir / "fixture-destination"
        replacement = self.run_dir / "replacement"
        replacement.write_bytes(b"unrelated")

        def replace_and_fail(_):
            replacement.replace(destination)
            return "0" * 64

        with patch("scripts.streamed_overlap_initialization._digest", side_effect=replace_and_fail):
            with self.assertRaisesRegex(ValueError, "changed during publication"):
                _publish_copy(source, destination, hashlib.sha256(b"source").hexdigest())
        self.assertEqual(destination.read_bytes(), b"unrelated")

    def test_copy_refuses_existing_destination(self):
        source = self.run_dir / "fixture-source"
        destination = self.run_dir / "fixture-destination"
        source.write_bytes(b"source")
        destination.write_bytes(b"unrelated")
        with self.assertRaises(FileExistsError):
            _publish_copy(source, destination, hashlib.sha256(b"source").hexdigest())
        self.assertEqual(destination.read_bytes(), b"unrelated")

    def test_partial_checkpoint_is_not_replaced(self):
        initial = self.run_dir / "initial.pt"
        initial.write_bytes(b"")
        with self.assertRaises(EOFError):
            self.prepare()
        self.assertEqual(initial.read_bytes(), b"")
        self.assertFalse((self.run_dir / "last.pt").exists())

    def test_partial_last_checkpoint_is_not_replaced(self):
        result = self.prepare()
        initial, last = result["initial_checkpoint"], result["checkpoint_path"]
        before = initial.read_bytes()
        last.write_bytes(b"")
        with self.assertRaises(EOFError):
            self.prepare()
        self.assertEqual(initial.read_bytes(), before)
        self.assertEqual(last.read_bytes(), b"")

    def test_mismatched_fresh_model_sampler_or_optimizer_fails_before_publication(self):
        with torch.no_grad():
            self.model.vae.encoder.fc_mu.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "fresh E1 initialization"):
            self.prepare()
        self.model = fresh_e1_model("S3", model_seed=2, head_seed=3)
        self.optimizer = build_optimizer(self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        self.sampler = FixtureSampler(8, self.hashes["source"])
        with self.assertRaisesRegex(ValueError, "sampler.*specified stream seed"):
            self.prepare()
        self.sampler = FixtureSampler(4, self.hashes["source"])
        self.optimizer.param_groups[0]["lr"] = 0
        with self.assertRaisesRegex(ValueError, "approved parameter groups"):
            self.prepare()
        self.assertFalse((self.run_dir / "initial.pt").exists())
        self.assertFalse((self.run_dir / "last.pt").exists())

    def test_existing_unrelated_checkpoint_is_not_overwritten(self):
        last = self.run_dir / "last.pt"
        last.write_bytes(b"unrelated")
        with self.assertRaisesRegex(FileExistsError, "without its initial snapshot"):
            self.prepare()
        self.assertEqual(last.read_bytes(), b"unrelated")
        self.assertFalse((self.run_dir / "initial.pt").exists())

    def test_changed_last_checkpoint_is_not_replaced(self):
        result = self.prepare()
        last = result["checkpoint_path"]
        state = torch.load(last, weights_only=True)
        state["completed_epoch"] = 1
        torch.save(state, last)
        changed = last.read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match the retained epoch-zero"):
            self.prepare()
        self.assertEqual(last.read_bytes(), changed)

    def test_rejects_source_directory_and_validation_v2_without_writing(self):
        with self.assertRaisesRegex(ValueError, "separate from source data"):
            self.prepare(source_root=self.run_dir)
        with self.assertRaisesRegex(ValueError, "permitted regular run directory"):
            self.prepare(source_root=self.source / "fake_data_validation_v2")
        source_file = self.source / "not-a-directory"
        source_file.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "separate from source data"):
            self.prepare(source_root=source_file)
        self.assertFalse((self.run_dir / "initial.pt").exists())
        self.assertFalse((self.run_dir / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts.streamed_overlap_augmentation import ViewTransform
from scripts.streamed_overlap_checkpoint import restore_epoch_state, save_epoch_state
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer
from scripts.streamed_overlap_sampler import PairBatch, PairSpec
from scripts.streamed_overlap_targets import Window
from scripts.train import EarlyStoppingState, train_streamed_pair_step
from src.model import VAE3D


class StreamedStepTests(unittest.TestCase):
    def batch(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(12, 1, 8, 8, 8)).astype(np.float32)
        b = rng.normal(size=a.shape).astype(np.float32)
        window = Window("train/a", (0, 0, 0), (8, 8, 8))
        categories = ["same_overlap"] * 6 + ["same_disjoint"] * 3 + ["cross_source"] * 3
        specs = tuple(PairSpec(category, window, window, i, i + 12)
                      for i, category in enumerate(categories))
        transform = ViewTransform(0, (1, 1, 1), False, False, False)
        stats = tuple(({"transform": transform, "noise": None, "retained_fraction": 0.3},) * 2
                      for _ in range(12))
        return PairBatch(
            a, b, a.copy(), np.full(12, 0.5, dtype=np.float32),
            np.full(12, 0.8, dtype=np.float32), np.ones(12, dtype=bool),
            (None,) * 12, specs, stats,
        )

    def test_each_arm_updates_shared_encoder_and_enabled_heads_in_one_step(self):
        for arm in ("S1", "S2", "S3"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7)
                model = StreamedPairVAE(
                    VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"),
                    arm,
                )
                optimizer = build_optimizer(model)
                journal = BatchJournal(Path(directory) / "batch_metrics.csv", STEP_FIELDS, create=True)
                self.addCleanup(journal.close)
                journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
                tracked = [model.vae.encoder.fc_mu.weight]
                if model.overlap_head is not None:
                    tracked.append(model.overlap_head.raw_scale)
                if model.coverage_head is not None:
                    tracked.append(model.coverage_head.network[2].weight)
                before = [parameter.detach().clone() for parameter in tracked]
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    row = train_streamed_pair_step(
                        model, optimizer, self.batch(), torch.device("cpu"),
                        lambda x, y: F.mse_loss(x, y), journal,
                        run_id="test", seed=1, epoch=1, batch_index=1, data_read_seconds=0.1,
                    )
                self.assertEqual(step.call_count, 1)
                for old, parameter in zip(before, tracked):
                    self.assertFalse(torch.equal(old, parameter))
                self.assertEqual(len(journal.rows), 1)
                self.assertEqual(row["completed_pairs"], 12)
                self.assertTrue((Path(directory) / row["provenance"]).is_file())
                self.assertEqual(row["overlap_loss"] is None, arm == "S1")
                self.assertEqual(row["coverage_loss"] is None, arm != "S3")

    def test_nonfinite_gradients_prevent_optimizer_step_and_completion_row(self):
        with tempfile.TemporaryDirectory() as directory:
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"),
                "S1",
            )
            optimizer = build_optimizer(model)
            journal = BatchJournal(Path(directory) / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            parameter = model.vae.encoder.fc_mu.weight
            handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
            self.addCleanup(handle.remove)
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    train_streamed_pair_step(
                        model, optimizer, self.batch(), torch.device("cpu"),
                        lambda x, y: F.mse_loss(x, y), journal,
                        run_id="test", seed=1, epoch=1, batch_index=1, data_read_seconds=0,
                    )
            self.assertEqual(step.call_count, 0)
            self.assertEqual(journal.rows, [])

    def test_joint_step_is_reproduced_after_epoch_checkpoint(self):
        class Sampler:
            def state_dict(self):
                return {"schema": "test", "split": "train", "source_manifest_hash": "b" * 64}

            def load_state_dict(self, state):
                if state != self.state_dict():
                    raise ValueError("Sampler mismatch")

        def components():
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S3",
            )
            optimizer = build_optimizer(model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, min_lr=1e-5)
            return model, optimizer, scheduler

        hashes = {"config": "a" * 64, "source": "b" * 64, "target": "c" * 64, "validation_manifest": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer, scheduler = components()
            journal = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            sampler = Sampler()
            stop = EarlyStoppingState(float("inf"), 0)

            def step(model, optimizer, journal, epoch):
                return train_streamed_pair_step(
                    model, optimizer, self.batch(), torch.device("cpu"),
                    lambda x, y: F.mse_loss(x, y), journal,
                    run_id="test", seed=1, epoch=epoch, batch_index=1, data_read_seconds=0,
                )

            step(model, optimizer, journal, 1)
            save_epoch_state(
                root / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler, early_stopping=stop, journal=journal, hashes=hashes,
                completed_epoch=1, batches_per_epoch=1, selection_state={},
            )
            stale = step(model, optimizer, journal, 2)
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            journal.close()
            restored, restored_optimizer, restored_scheduler = components()
            resumed = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=False)
            self.addCleanup(resumed.close)
            restore_epoch_state(
                root / "last.pt", model=restored, optimizer=restored_optimizer,
                scheduler=restored_scheduler, sampler=sampler, early_stopping=stop,
                journal=resumed, expected_hashes=hashes, expected_batches_per_epoch=1, attempt_id="second",
            )
            replay = step(restored, restored_optimizer, resumed, 2)
            self.assertEqual(stale["total_loss"], replay["total_loss"])
            self.assertEqual(replay["replay_of_event"], stale["event_id"])
            self.assertEqual(replay["cumulative_exposure"], 24)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_journal_failure_after_step_poisoned_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S1",
            )
            optimizer = build_optimizer(model)
            journal = BatchJournal(Path(directory) / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            with patch.object(journal, "append", side_effect=OSError("disk failure")):
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(OSError, "disk failure"):
                        train_streamed_pair_step(
                            model, optimizer, self.batch(), torch.device("cpu"),
                            lambda x, y: F.mse_loss(x, y), journal,
                            run_id="test", seed=1, epoch=1, batch_index=1, data_read_seconds=0,
                        )
            self.assertEqual(step.call_count, 1)
            self.assertIsNone(journal.position)
            self.assertEqual(journal.rows, [])

    def test_zero_encoder_learning_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S1",
            )
            optimizer = build_optimizer(model)
            optimizer.param_groups[0]["lr"] = 0
            journal = BatchJournal(Path(directory) / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            with self.assertRaisesRegex(ValueError, "learning rates"):
                train_streamed_pair_step(
                    model, optimizer, self.batch(), torch.device("cpu"),
                    lambda x, y: F.mse_loss(x, y), journal,
                    run_id="test", seed=1, epoch=1, batch_index=1, data_read_seconds=0,
                )
            self.assertEqual(journal.rows, [])


if __name__ == "__main__":
    unittest.main()

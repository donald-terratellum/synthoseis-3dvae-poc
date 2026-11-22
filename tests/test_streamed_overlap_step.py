import tempfile
import unittest
import random
import csv
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
from scripts.train import (
    EarlyStoppingState, _validate_streamed_epoch_metrics, initialize_streamed_epoch_metrics, monitor_streamed_pairs,
    streamed_monitor_hash, train_streamed_epoch, train_streamed_pair_step,
)
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
        stats = tuple(({"transform": transform, "noise": None, "retained_fraction": 0.3,
                        "input_mode": "extrema"},) * 2
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

    def test_training_step_rejects_holdout_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S1",
            )
            optimizer = build_optimizer(model)
            journal = BatchJournal(Path(directory) / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            batch = self.batch()
            invalid = replace(batch, provenance=(
                replace(batch.provenance[0], a=replace(batch.provenance[0].a, source_id="validation/a")),
                *batch.provenance[1:],
            ))
            with self.assertRaisesRegex(ValueError, "holdout"):
                train_streamed_pair_step(
                    model, optimizer, invalid, torch.device("cpu"),
                    lambda x, y: F.mse_loss(x, y), journal,
                    run_id="test", seed=1, epoch=1, batch_index=1, data_read_seconds=0,
                )
            self.assertEqual(journal.rows, [])

    def test_fixed_monitor_is_deterministic_and_restores_rng_and_module_modes(self):
        class Writer:
            def __init__(self):
                self.events = []

            def add_scalar(self, tag, value, step):
                self.events.append((tag, value, step))

            def flush(self):
                pass

        model = StreamedPairVAE(
            VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S3",
        )
        model.train()
        model.vae.decoder.eval()
        flags = [module.training for module in model.modules()]
        batch = self.batch()
        validation = replace(batch, provenance=tuple(
            replace(spec, a=replace(spec.a, source_id="validation/a"),
                    b=replace(spec.b, source_id="validation/a")) for spec in batch.provenance
        ))
        batches = {"train": batch, "validation": validation}
        torch.manual_seed(19)
        np.random.seed(19)
        random.seed(19)
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        writer = Writer()

        def loss(a, b):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            return F.mse_loss(a, b)

        first, _ = monitor_streamed_pairs(model, loss, "cpu", batches, writer, seed=7, step=100)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], after_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
        self.assertEqual(numpy_state[2:], after_numpy[2:])
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(flags, [module.training for module in model.modules()])
        torch.rand(3)
        random.random()
        np.random.random()
        second, _ = monitor_streamed_pairs(model, loss, "cpu", batches, writer, seed=7, step=101)
        self.assertEqual(first, second)
        self.assertTrue(any(event[0] == "monitor/validation/coverage" for event in writer.events))

    def test_monitor_failure_restores_training_rng_and_modes(self):
        class Writer:
            def add_scalar(self, *args):
                raise AssertionError("Unexpected logging")

            def flush(self):
                raise AssertionError("Unexpected flushing")

        model = StreamedPairVAE(
            VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S1",
        )
        batch = self.batch()
        validation = replace(batch, provenance=tuple(
            replace(spec, a=replace(spec.a, source_id="validation/a"),
                    b=replace(spec.b, source_id="validation/a")) for spec in batch.provenance
        ))
        rng = torch.get_rng_state().clone()

        def fail(_a, _b):
            raise RuntimeError("monitor failure")

        with self.assertRaisesRegex(RuntimeError, "monitor failure"):
            monitor_streamed_pairs(model, fail, "cpu", {"train": batch, "validation": validation},
                                   Writer(), seed=7, step=1)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def _epoch_validation_logging_checkpoint_and_failed_epoch_replay(self, device):
        class Writer:
            def __init__(self):
                self.events = []

            def add_scalar(self, tag, value, step):
                self.events.append((tag, value, step))

            def flush(self):
                pass

        class Sampler:
            split = "train"
            source_manifest_hash = "b" * 64

            def __init__(self, batch):
                self.batch = batch
                self.batches_drawn = 0

            def next_specs(self):
                self.batches_drawn += 1
                return self.batch.provenance

            def load_batch(self, specs, *, sparse_probability):
                if specs != self.batch.provenance or sparse_probability != 0.5:
                    raise ValueError("Unexpected synthetic sampling configuration")
                return self.batch

            def state_dict(self):
                return {"schema": "test", "split": self.split,
                        "source_manifest_hash": self.source_manifest_hash,
                        "batches_drawn": self.batches_drawn}

            def load_state_dict(self, state):
                self.batches_drawn = state["batches_drawn"]

        def components():
            model = StreamedPairVAE(
                VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S3",
            ).to(device)
            optimizer = build_optimizer(model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, min_lr=1e-5)
            return model, optimizer, scheduler

        hashes = {"config": "a" * 64, "source": "b" * 64, "target": "c" * 64, "validation_manifest": "d" * 64}
        batch = self.batch()
        validation = replace(batch, provenance=tuple(
            replace(spec, a=replace(spec.a, source_id="validation/a"),
                    b=replace(spec.b, source_id="validation/a")) for spec in batch.provenance
        ))
        hashes["monitor"] = streamed_monitor_hash({"train": batch, "validation": validation})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer, scheduler = components()
            journal = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            sampler, stop = Sampler(batch), EarlyStoppingState(float("inf"), 0)
            writer = Writer()
            initialize_streamed_epoch_metrics(root / "training_metrics.csv")
            save_epoch_state(
                root / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler, early_stopping=stop, journal=journal, hashes=hashes,
                completed_epoch=0, batches_per_epoch=2, selection_state={},
            )

            def validate(model):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                return {"split": "validation", "manifest_sha256": hashes["validation_manifest"],
                        "epoch_validation_sha256": "e" * 64,
                        "reconstruction_mae": 1.0, "lpips_loss": 2.0, "kl": 3.0,
                        "metrics": {"mu_cosine_ndcg10": 0.4}}

            class Validator:
                config_sha256 = "e" * 64

                def __init__(self, loss_fn):
                    self.lpips_loss_fn = loss_fn

                def __call__(self, model):
                    return validate(model)

            lpips_fn = lambda x, y: F.mse_loss(x, y)
            validator = Validator(lpips_fn)

            def run(model, optimizer, scheduler, sampler, journal, validator, selection_state=None):
                return train_streamed_epoch(
                    model, optimizer, scheduler, sampler, journal,
                    lpips_fn, device, writer,
                    run_id="test", seed=1, epoch=1, batches_per_epoch=2, sparse_probability=0.5,
                    monitor_batches={"train": batch, "validation": validation},
                    monitor_seed=7, validation_seed=8, validate_epoch=validator,
                    epoch_csv=root / "training_metrics.csv", checkpoint_path=root / "last.pt",
                    hashes=hashes, early_stopping=stop,
                    selection_state={} if selection_state is None else selection_state,
                )

            previous = batch.input_a[0, 0, 0, 0, 0]
            batch.input_a[0, 0, 0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "monitor hash"):
                run(model, optimizer, scheduler, sampler, journal, validate)
            self.assertEqual(sampler.batches_drawn, 0)
            self.assertEqual(journal.rows, [])
            batch.input_a[0, 0, 0, 0, 0] = previous
            with self.assertRaisesRegex(ValueError, "validation configuration"):
                run(model, optimizer, scheduler, sampler, journal, validator)
            self.assertEqual(sampler.batches_drawn, 0)
            hashes["epoch_validation"] = validator.config_sha256
            with self.assertRaisesRegex(ValueError, "Stored epoch validation"):
                run(model, optimizer, scheduler, sampler, journal, validator, {
                    "_epoch_validation": {"epoch_validation_sha256": "f" * 64},
                })
            self.assertEqual(sampler.batches_drawn, 0)
            save_epoch_state(
                root / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler, early_stopping=stop, journal=journal, hashes=hashes,
                completed_epoch=0, batches_per_epoch=2, selection_state={},
            )
            with patch("scripts.streamed_overlap_checkpoint.save_epoch_state",
                       side_effect=OSError("checkpoint interrupted")):
                with self.assertRaisesRegex(OSError, "checkpoint interrupted"):
                    run(model, optimizer, scheduler, sampler, journal, validator)
            stale = list(journal.rows)
            self.assertIsNone(journal.position)
            self.assertEqual(torch.load(root / "last.pt", weights_only=True)["completed_epoch"], 0)
            with (root / "training_metrics.csv").open(newline="") as handle:
                stale_epochs = list(csv.DictReader(handle))
            self.assertEqual(stale_epochs[0]["batch_event_id"], stale[-1]["event_id"])
            journal.close()
            model, optimizer, scheduler = components()
            sampler = Sampler(batch)
            resumed = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=False)
            self.addCleanup(resumed.close)
            restore_epoch_state(
                root / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler, early_stopping=stop, journal=resumed, expected_hashes=hashes,
                expected_batches_per_epoch=2, attempt_id="second",
            )
            with (root / "training_metrics.csv").open("ab") as handle:
                handle.write(b"interrupted-epoch-row")
            with patch("scripts.train.monitor_streamed_pairs", wraps=monitor_streamed_pairs) as monitor:
                result = run(model, optimizer, scheduler, sampler, resumed, validator)
            self.assertEqual(result["checkpoint"]["attempt_id"], "second")
            self.assertEqual(result["checkpoint"]["checkpoint_sha256"],
                             hashlib.sha256((root / "last.pt").read_bytes()).hexdigest())
            saved = torch.load(root / "last.pt", weights_only=True)
            self.assertEqual(saved["selection_state"]["_epoch_validation"]["metrics"]["mu_cosine_ndcg10"], 0.4)
            self.assertEqual(saved["selection_state"]["_epoch_validation"]["epoch_validation_sha256"], "e" * 64)
            self.assertEqual(scheduler.best, 1.0 + 0.1 * 2.0 + 0.001 * 3.0)
            self.assertEqual(monitor.call_count, 1)
            self.assertEqual([row["replay_of_event"] for row in resumed.rows[-2:]],
                             [row["event_id"] for row in stale])
            self.assertEqual([row["total_loss"] for row in resumed.rows[-2:]],
                             [row["total_loss"] for row in stale])
            self.assertEqual(resumed.rows[-1]["cumulative_exposure"], "24")
            csv_lines = (root / "training_metrics.csv").read_text().splitlines()
            self.assertEqual(csv_lines[2], "interrupted-epoch-row")
            self.assertEqual(len(list((root / "epoch_recovery").glob("*.json"))), 1)
            rows = list(csv.DictReader([csv_lines[0], csv_lines[1], csv_lines[-1]]))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["batch_event_id"], stale[-1]["event_id"])
            self.assertAlmostEqual(float(rows[-1]["val_loss"]), 1.203)
            self.assertEqual(rows[-1]["batch_event_id"], resumed.rows[-1]["event_id"])
            expected_train_loss = sum(
                float(row["reconstruction_mae"]) + float(row["weighted_lpips"]) + float(row["weighted_kl"])
                for row in resumed.rows[-2:]
            ) / 2
            self.assertAlmostEqual(float(rows[-1]["train_loss"]), expected_train_loss)
            self.assertEqual(rows[-1]["real_test_mae"], "")
            self.assertEqual(torch.load(root / "last.pt", weights_only=True)["completed_epoch"], 1)
            self.assertTrue(any(event[0] == "validation/mu_cosine_ndcg10" for event in writer.events))

    def test_epoch_validation_logging_checkpoint_and_failed_epoch_replay(self):
        self._epoch_validation_logging_checkpoint_and_failed_epoch_replay("cpu")

    @unittest.skipUnless(os.environ.get("STREAMED_MPS_SMOKE") == "1" and torch.backends.mps.is_available(),
                         "explicit MPS smoke opt-in and MPS are required")
    def test_synthetic_epoch_recovery_on_mps(self):
        self._epoch_validation_logging_checkpoint_and_failed_epoch_replay("mps")

    def test_monitor_runs_every_100_steps_and_once_at_epoch_boundary(self):
        class Writer:
            def add_scalar(self, *args):
                pass

            def flush(self):
                pass

        class Sampler:
            split = "train"
            source_manifest_hash = "b" * 64

            def __init__(self, batch):
                self.batch = batch
                self.batches_drawn = 0

            def state_dict(self):
                return {"batches_drawn": self.batches_drawn}

            def next_specs(self):
                self.batches_drawn += 1
                return self.batch.provenance

            def load_batch(self, _specs, *, sparse_probability):
                return self.batch

        batch = self.batch()
        validation = replace(batch, provenance=tuple(
            replace(spec, a=replace(spec.a, source_id="validation/a"),
                    b=replace(spec.b, source_id="validation/a")) for spec in batch.provenance
        ))
        hashes = {"config": "a" * 64, "source": "b" * 64, "target": "c" * 64, "validation_manifest": "d" * 64}
        hashes["monitor"] = streamed_monitor_hash({"train": batch, "validation": validation})
        for budget, expected in ((100, [100]), (105, [100, 105])):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                initialize_streamed_epoch_metrics(root / "training_metrics.csv")
                model = StreamedPairVAE(
                    VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), "S1",
                )
                optimizer = build_optimizer(model)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, min_lr=1e-5)
                journal = SimpleNamespace(
                    position=SimpleNamespace(logical_step=0, cumulative_exposure=0),
                    rows=[], attempt_id="first", fields=STEP_FIELDS,
                )

                def fake_step(_model, _optimizer, _batch, _device, _lpips, journal, **kwargs):
                    step = journal.position.logical_step + 1
                    journal.position = SimpleNamespace(logical_step=step, cumulative_exposure=12 * step)
                    row = {name: None for name in STEP_FIELDS}
                    row.update({
                        "event_id": f"e{step}", "attempt_id": "first", "epoch": str(kwargs["epoch"]),
                        "logical_step": step, "reconstruction_mae": 1.0, "weighted_lpips": 0.2,
                        "weighted_kl": 0.003, "lpips": 2.0, "batch_seconds": 1.0,
                    })
                    journal.rows.append(row)
                    return row

                def validate(_model):
                    return {"split": "validation", "manifest_sha256": hashes["validation_manifest"],
                            "reconstruction_mae": 1.0, "lpips_loss": 2.0, "kl": 3.0,
                            "metrics": {"mu_cosine_ndcg10": 0.4}}

                with patch("scripts.train.train_streamed_pair_step", side_effect=fake_step):
                    with patch("scripts.train.monitor_streamed_pairs") as monitor:
                        with patch("scripts.streamed_overlap_checkpoint.save_epoch_state"):
                            train_streamed_epoch(
                                model, optimizer, scheduler, Sampler(batch), journal,
                                lambda x, y: F.mse_loss(x, y), "cpu", Writer(),
                                run_id="test", seed=1, epoch=1, batches_per_epoch=budget,
                                sparse_probability=0.5,
                                monitor_batches={"train": batch, "validation": validation},
                                monitor_seed=7, validation_seed=8, validate_epoch=validate,
                                epoch_csv=root / "training_metrics.csv", checkpoint_path=root / "last.pt",
                                hashes=hashes, early_stopping=EarlyStoppingState(float("inf"), 0),
                                selection_state={},
                            )
                self.assertEqual([call.kwargs["step"] for call in monitor.call_args_list], expected)

    def test_epoch_tail_recovery_survives_interrupted_delimiter_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training_metrics.csv"
            initialize_streamed_epoch_metrics(path)
            journal = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
            with path.open("ab") as handle:
                handle.write(b"partial-epoch")
            real_open = Path.open

            def fail_delimiter(current, mode="r", *args, **kwargs):
                if current == path and mode == "ab":
                    raise OSError("interrupted")
                return real_open(current, mode, *args, **kwargs)

            with patch.object(Path, "open", fail_delimiter):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    _validate_streamed_epoch_metrics(path, journal)
            self.assertEqual(len(list((root / "epoch_recovery").glob("*.json"))), 1)
            _validate_streamed_epoch_metrics(path, journal)
            _validate_streamed_epoch_metrics(path, journal)
            self.assertTrue(path.read_bytes().endswith(b"partial-epoch\n"))
            self.assertEqual(len(list((root / "epoch_recovery").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()

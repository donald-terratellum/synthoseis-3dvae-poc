from copy import deepcopy
from dataclasses import dataclass
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.streamed_overlap_checkpoint import restore_epoch_state, save_epoch_state
from scripts.streamed_overlap_journal import BatchJournal, IDENTITY_FIELDS


HASHES = {
    "config": "a" * 64, "source": "b" * 64, "target": "c" * 64,
    "validation_manifest": "d" * 64,
}


@dataclass
class StopState:
    best_val_loss: float = 1.0
    epochs_without_improvement: int = 0


class Sampler:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.count = 0

    def state_dict(self):
        return {
            "schema": "test", "split": "train", "source_manifest_hash": HASHES["source"],
            "rng": deepcopy(self.rng.bit_generator.state), "count": self.count,
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.count = state["count"]


class Model(torch.nn.Linear):
    arm = "S3"


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "last.pt"
        self.fields = (*IDENTITY_FIELDS, "epoch", "batch_index", "total_loss")
        self.journal = BatchJournal(self.root / "batch_metrics.csv", self.fields, create=True)
        self.addCleanup(self.journal.close)
        self.journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
        self.model, self.optimizer, self.scheduler, self.sampler, self.stop = self.components()

    def components(self):
        model = Model(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, min_lr=1e-5)
        return model, optimizer, scheduler, Sampler(), StopState()

    def step(self, model, optimizer, scheduler, sampler, journal, event):
        x = torch.randn(4, 2) + float(sampler.rng.random()) + random.random() + float(np.random.random())
        sampler.count += 1
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        position = journal.position
        journal.append({
            "sequence": len(journal.rows) + 1, "event_id": event,
            "attempt_id": journal.attempt_id, "logical_step": position.logical_step + 1,
            "completed_pairs": 12, "cumulative_exposure": position.cumulative_exposure + 12,
            "replay_of_event": position.replay_events.get(position.logical_step + 1, ""),
            "epoch": position.logical_step + 1, "batch_index": 1, "total_loss": loss.item(),
        })
        return loss.item()

    def save(self):
        save_epoch_state(
            self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            sampler=self.sampler, early_stopping=self.stop, journal=self.journal,
            hashes=HASHES, completed_epoch=1, batches_per_epoch=1, selection_state={"best": None},
        )

    def test_epoch_resume_restores_rng_optimizer_sampler_and_replay(self):
        self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        self.save()
        expected_loss = self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e2")
        expected_model = deepcopy(self.model.state_dict())
        expected_optimizer = deepcopy(self.optimizer.state_dict())
        expected_scheduler = deepcopy(self.scheduler.state_dict())
        model, optimizer, scheduler, sampler, stop = self.components()
        self.journal.close()
        resumed = BatchJournal(self.root / "batch_metrics.csv", self.fields, create=False)
        self.addCleanup(resumed.close)
        info = restore_epoch_state(
            self.path, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
            early_stopping=stop, journal=resumed, expected_hashes=HASHES,
            expected_batches_per_epoch=1, attempt_id="second",
        )
        self.assertEqual(info["completed_epoch"], 1)
        actual_loss = self.step(model, optimizer, scheduler, sampler, resumed, "e3")
        self.assertEqual(expected_loss, actual_loss)
        for key in expected_model:
            torch.testing.assert_close(model.state_dict()[key], expected_model[key], rtol=0, atol=0)
        for key, value in expected_optimizer["state"].items():
            for name, expected in value.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][key][name], expected, rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), expected_scheduler)
        self.assertEqual(sampler.count, 2)
        self.assertEqual(resumed.rows[-1]["replay_of_event"], "e2")
        self.assertEqual(resumed.rows[-1]["cumulative_exposure"], "24")

    def test_incompatible_hash_rejected_before_journal_reconciliation(self):
        self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        self.save()
        self.journal.close()
        resumed = BatchJournal(self.root / "batch_metrics.csv", self.fields, create=False)
        self.addCleanup(resumed.close)
        before = set((self.root / "recovery").iterdir())
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_epoch_state(
                self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                sampler=self.sampler, early_stopping=self.stop, journal=resumed,
                expected_hashes={**HASHES, "target": "e" * 64}, expected_batches_per_epoch=1,
                attempt_id="second",
            )
        self.assertEqual(before, set((self.root / "recovery").iterdir()))

    def test_interrupted_replace_preserves_previous_checkpoint(self):
        self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        self.save()
        original = self.path.read_bytes()
        with patch("scripts.streamed_overlap_checkpoint.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.save()
        self.assertEqual(self.path.read_bytes(), original)

    def test_partial_epoch_cannot_be_saved_as_complete(self):
        self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        with self.assertRaisesRegex(ValueError, "completed"):
            save_epoch_state(
                self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                sampler=self.sampler, early_stopping=self.stop, journal=self.journal,
                hashes=HASHES, completed_epoch=1, batches_per_epoch=2, selection_state={},
            )
        self.assertFalse(self.path.exists())

    def test_initial_checkpoint_replays_first_epoch_after_interruption(self):
        save_epoch_state(
            self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            sampler=self.sampler, early_stopping=self.stop, journal=self.journal,
            hashes=HASHES, completed_epoch=0, batches_per_epoch=1, selection_state={},
        )
        expected = self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        self.journal.close()
        model, optimizer, scheduler, sampler, stop = self.components()
        resumed = BatchJournal(self.root / "batch_metrics.csv", self.fields, create=False)
        self.addCleanup(resumed.close)
        state = restore_epoch_state(
            self.path, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
            early_stopping=stop, journal=resumed, expected_hashes=HASHES,
            expected_batches_per_epoch=1, attempt_id="second",
        )
        self.assertEqual(state["completed_epoch"], 0)
        self.assertEqual(expected, self.step(model, optimizer, scheduler, sampler, resumed, "e2"))
        self.assertEqual(resumed.rows[-1]["replay_of_event"], "e1")
        self.assertEqual(resumed.rows[-1]["cumulative_exposure"], "12")

    def test_wrong_device_or_scheduler_rejected_before_reconciliation(self):
        self.step(self.model, self.optimizer, self.scheduler, self.sampler, self.journal, "e1")
        self.save()
        self.journal.close()
        resumed = BatchJournal(self.root / "batch_metrics.csv", self.fields, create=False)
        self.addCleanup(resumed.close)
        records = set(resumed.recovery_dir.iterdir())
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        state["device"] = "mps"
        torch.save(state, self.path)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_epoch_state(
                self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                sampler=self.sampler, early_stopping=self.stop, journal=resumed,
                expected_hashes=HASHES, expected_batches_per_epoch=1, attempt_id="second",
            )
        state["device"] = "cpu"
        torch.save(state, self.path)
        self.scheduler.patience = 1
        with self.assertRaisesRegex(ValueError, "scheduler"):
            restore_epoch_state(
                self.path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                sampler=self.sampler, early_stopping=self.stop, journal=resumed,
                expected_hashes=HASHES, expected_batches_per_epoch=1, attempt_id="second",
            )
        self.assertEqual(records, set(resumed.recovery_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()

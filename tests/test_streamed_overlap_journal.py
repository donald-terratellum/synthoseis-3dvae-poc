import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.streamed_overlap_journal import BatchJournal, IDENTITY_FIELDS


FIELDS = (*IDENTITY_FIELDS, "total_loss")


class BatchJournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "batch_metrics.csv"
        self.journal = BatchJournal(self.path, FIELDS, create=True)
        self.addCleanup(self.journal.close)
        self.journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)

    def reopen(self):
        self.journal.close()
        journal = BatchJournal(self.path, FIELDS, create=False)
        self.addCleanup(journal.close)
        return journal

    def append(self, journal, sequence, event, attempt, step, exposure, replay=""):
        journal.append({
            "sequence": sequence, "event_id": event, "attempt_id": attempt,
            "logical_step": step, "completed_pairs": 12, "cumulative_exposure": exposure,
            "replay_of_event": replay, "total_loss": 0.5,
        })

    def test_completed_rows_are_fsynced_and_replay_does_not_double_count(self):
        with patch("scripts.streamed_overlap_journal.os.fsync", wraps=os.fsync) as sync:
            self.append(self.journal, 1, "e1", "first", 1, 12)
            self.assertEqual(sync.call_count, 1)
        self.append(self.journal, 2, "e2", "first", 2, 24)
        self.append(self.journal, 3, "e3", "first", 3, 36)
        resumed = self.reopen()
        position = resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=12)
        self.assertEqual(position.replay_events, {2: "e2", 3: "e3"})
        self.assertEqual(json.loads(position.recovery_record.read_text())["uncommitted_event_ids"], ["e2", "e3"])
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.append(resumed, 4, "e4", "second", 2, 48, "e2")
        self.append(resumed, 4, "e4", "second", 2, 24, "e2")
        self.append(resumed, 5, "e5", "second", 3, 36, "e3")
        self.assertEqual(resumed.rows[-1]["cumulative_exposure"], "36")
        self.assertEqual([row["event_id"] for row in resumed.rows], ["e1", "e2", "e3", "e4", "e5"])

    def test_torn_tail_is_preserved_and_following_rows_remain_readable(self):
        self.append(self.journal, 1, "e1", "first", 1, 12)
        original = self.path.read_bytes()
        with self.path.open("ab") as handle:
            handle.write(b"2,partial")
        resumed = self.reopen()
        resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=12)
        self.append(resumed, 2, "e2", "second", 2, 24)
        self.assertTrue(self.path.read_bytes().startswith(original + b"2,partial\n"))
        resumed.close()
        again = BatchJournal(self.path, FIELDS, create=False)
        self.addCleanup(again.close)
        self.assertEqual([row["event_id"] for row in again.rows], ["e1", "e2"])

    def test_incompatible_checkpoint_does_not_modify_torn_journal(self):
        self.append(self.journal, 1, "e1", "first", 1, 12)
        with self.path.open("ab") as handle:
            handle.write(b"partial")
        resumed = self.reopen()
        before = self.path.read_bytes()
        records = set(resumed.recovery_dir.iterdir())
        with self.assertRaisesRegex(ValueError, "incompatible"):
            resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=99)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(resumed.recovery_dir.iterdir()), records)

    def test_interruption_after_torn_record_before_delimiter_recovers(self):
        self.append(self.journal, 1, "e1", "first", 1, 12)
        with self.path.open("ab") as handle:
            handle.write(b"partial")
        resumed = self.reopen()
        real_open = Path.open

        def fail_delimiter(path, mode="r", *args, **kwargs):
            if path == self.path and mode == "ab":
                raise OSError("simulated interruption")
            return real_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", fail_delimiter), self.assertRaisesRegex(OSError, "interruption"):
            resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=12)
        resumed.close()
        again = BatchJournal(self.path, FIELDS, create=False)
        self.addCleanup(again.close)
        again.reconcile(watermark="e1", attempt_id="third", expected_step=1, expected_exposure=12)
        self.append(again, 2, "e2", "third", 2, 24)
        self.assertIn(b"partial\n", self.path.read_bytes())

    def test_checkpoint_mismatch_and_stale_watermark_fail(self):
        self.append(self.journal, 1, "e1", "first", 1, 12)
        self.append(self.journal, 2, "e2", "first", 2, 24)
        resumed = self.reopen()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=99)
        resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=12)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            resumed.reconcile(watermark="e2", attempt_id="third", expected_step=2, expected_exposure=24)

    def test_failed_fsync_poisoned_attempt_must_reconcile(self):
        with patch("scripts.streamed_overlap_journal.os.fsync", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.append(self.journal, 1, "e1", "first", 1, 12)
        with self.assertRaisesRegex(ValueError, "reconciled"):
            self.append(self.journal, 1, "e1", "first", 1, 12)
        resumed = self.reopen()
        state = resumed.reconcile(watermark=None, attempt_id="second", expected_step=0, expected_exposure=0)
        self.assertEqual(state.replay_events, {1: "e1"})
        self.append(resumed, 2, "e2", "second", 1, 12, "e1")

    def test_changed_preserved_tail_fails_closed(self):
        self.append(self.journal, 1, "e1", "first", 1, 12)
        with self.path.open("ab") as handle:
            handle.write(b"partial")
        resumed = self.reopen()
        resumed.reconcile(watermark="e1", attempt_id="second", expected_step=1, expected_exposure=12)
        data = self.path.read_bytes().replace(b"partial", b"changed")
        self.path.write_bytes(data)
        resumed.close()
        with self.assertRaisesRegex(ValueError, "torn journal bytes"):
            BatchJournal(self.path, FIELDS, create=False)

    def test_second_writer_cannot_open_locked_journal(self):
        with self.assertRaises(BlockingIOError):
            BatchJournal(self.path, FIELDS, create=False)
        self.journal.close()
        with self.assertRaisesRegex(ValueError, "closed journal"):
            self.journal.reconcile(watermark=None, attempt_id="second", expected_step=0, expected_exposure=0)


if __name__ == "__main__":
    unittest.main()

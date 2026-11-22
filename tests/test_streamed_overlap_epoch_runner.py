import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import complete_epoch, run_arm_epochs


class EpochCompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir()
        self.checkpoint = self.root / "last.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint")
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.search_manifest = {"searches": []}
        self.search_manifest["search_manifest_sha256"] = content_hash(self.search_manifest)
        self.diagnostic_manifest = {"windows": {}}
        self.diagnostic_manifest["diagnostic_manifest_sha256"] = content_hash(self.diagnostic_manifest)
        self.config = {
            "final_epoch": 6, "device": "mps", "inference_seed": 1, "encode_batch_size": 4,
            "max_wall_seconds": 30, "random_seed": 2, "random_repeats": 3,
            "bootstrap_seed": 4, "bootstrap_repeats": 5, "diagnostic_max_wall_seconds": 30,
            "search_manifest_sha256": self.search_manifest["search_manifest_sha256"],
            "diagnostic_manifest_sha256": self.diagnostic_manifest["diagnostic_manifest_sha256"],
        }
        self.manifest = {"manifest_sha256": "c" * 64}
        self.epoch, self.arm = 5, "S3"
        self.state = {"hashes": {"formal_evaluation": content_hash(self.config)}}
        self.patches = [
            patch("scripts.streamed_overlap_epoch_runner.load_manifest", return_value=self.manifest),
            patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                  side_effect=lambda _: (self.state, self.digest)),
            patch("scripts.streamed_overlap_epoch_runner._identity", side_effect=lambda *_: self.identity()),
            patch("scripts.streamed_overlap_epoch_runner.open_validation_sources", return_value={}),
            patch("scripts.streamed_overlap_epoch_runner.evaluate_checkpoint", side_effect=self.evaluate),
            patch("scripts.streamed_overlap_epoch_runner.run_bounded_searches", side_effect=self.search),
            patch("scripts.streamed_overlap_epoch_runner.diagnostic_report", return_value={"diagnostic": "ok"}),
            patch("scripts.streamed_overlap_epoch_runner.select_checkpoints", return_value={"selection": "ok"}),
        ]
        self.mocks = [item.start() for item in self.patches]
        for item in reversed(self.patches):
            self.addCleanup(item.stop)

    def identity(self):
        return {"epoch": self.epoch, "arm": self.arm, "checkpoint_sha256": self.digest}

    def evaluate(self, *_args, **kwargs):
        self.assertEqual(kwargs["score_pair_head"], self.arm == "S3")
        self.assertEqual(kwargs["score_overlap"], self.arm in {"S2", "S3"})
        return {
            "checkpoint": self.identity(),
            "scorers": {"pair_head": {"event_coverage_diagnostics": {"pairs": [
                {"query_id": "q", "candidate_id": 0, "prediction": 0.4},
            ]}}},
        }

    def search(self, *args):
        output = args[-1]
        output.mkdir()
        (output / "tokenizer.pt").write_bytes(b"fixture tokenizer")
        return {
            "checkpoint": {"checkpoint": self.identity(),
                           "tokenizer_checkpoint_sha256": hashlib.sha256(b"fixture tokenizer").hexdigest()},
            "searches": [],
        }

    def run_completion(self, config=None):
        return complete_epoch(
            self.checkpoint, object(), self.root / "manifest", self.source,
            self.search_manifest, self.diagnostic_manifest, self.config if config is None else config,
            evaluation_dir=self.evaluation,
        )

    def test_interrupted_formal_evaluation_retries_before_completion(self):
        search_mock, selection_mock = self.mocks[5], self.mocks[7]
        def interrupted_once(*args):
            if search_mock.call_count == 1:
                raise OSError("interrupted")
            return self.search(*args)

        search_mock.side_effect = interrupted_once
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.run_completion()
        self.assertEqual(list((self.evaluation / "completed").glob("*.json")), [])
        selection_mock.assert_not_called()
        result = self.run_completion()
        self.assertTrue(result["formal_evaluation_due"])
        self.assertEqual(set(result["artifacts"]), {"retrieval", "tokenizer_search", "diagnostics", "selection"})
        self.assertEqual(len(list(self.evaluation.glob("epoch-*"))), 2)
        self.assertIsNotNone(selection_mock.call_args.kwargs["retrieval_report"])
        with patch("scripts.streamed_overlap_epoch_runner.evaluate_checkpoint",
                   side_effect=AssertionError("must not rescore")):
            self.assertEqual(self.run_completion(), result)
        search_root = (self.evaluation / result["artifacts"]["tokenizer_search"]["path"]).parent / "tokenizer_search"
        tokenizer = search_root / "tokenizer.pt"
        tokenizer.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "tokenizer checkpoint"):
            self.run_completion()
        tokenizer.write_bytes(b"fixture tokenizer")
        (self.evaluation / result["artifacts"]["selection"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.run_completion()

    def test_unscheduled_epoch_selects_without_formal_scoring(self):
        self.epoch = 4
        result = self.run_completion()
        self.assertFalse(result["formal_evaluation_due"])
        self.assertEqual(set(result["artifacts"]), {"selection"})
        self.mocks[4].assert_not_called()
        self.mocks[5].assert_not_called()

    def test_s1_never_passes_cosine_report_as_pair_head_selection(self):
        self.epoch, self.arm = 6, "S1"
        result = self.run_completion()
        self.assertEqual(set(result["artifacts"]), {"retrieval", "tokenizer_search", "selection"})
        self.assertIsNone(self.mocks[7].call_args.kwargs["retrieval_report"])
        self.mocks[6].assert_not_called()

    def test_changed_frozen_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "configuration disagree"):
            self.run_completion({**self.config, "random_seed": 99})
        self.assertEqual(list(self.evaluation.iterdir()), [])

    def test_changed_manifest_is_rejected_even_when_completion_exists(self):
        self.run_completion()
        self.diagnostic_manifest["windows"]["changed"] = {}
        with self.assertRaisesRegex(ValueError, "frozen MPS evaluation"):
            self.run_completion()

    def test_resume_finishes_pending_evaluation_before_next_optimizer_epoch(self):
        order = []

        def finish(*_args, **_kwargs):
            order.append("evaluate")
            return {"done": len(order)}

        def train(*_args, **kwargs):
            order.append(f"train-{kwargs['epoch']}")
            return {"epoch_validation_sha256": "e" * 64}

        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 4, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", side_effect=finish), \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch", side_effect=train):
            result = run_arm_epochs(
                *([object()] * 7), run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                hashes={"formal_evaluation": content_hash(self.config)}, early_stopping=object(),
                manifest_path=self.root / "manifest", source_root=self.source,
                search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                formal_config=self.config, evaluation_dir=self.evaluation,
            )
        self.assertEqual(order, ["evaluate", "train-5", "evaluate", "train-6", "evaluate"])
        self.assertEqual(result["done"], 5)

    def test_pending_evaluation_failure_prevents_next_epoch(self):
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 4, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", side_effect=OSError("evaluation failed")), \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(OSError, "evaluation failed"):
                run_arm_epochs(
                    *([object()] * 7), run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                    sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                    validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                    hashes={"formal_evaluation": content_hash(self.config)}, early_stopping=object(),
                    manifest_path=self.root / "manifest", source_root=self.source,
                    search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                    formal_config=self.config, evaluation_dir=self.evaluation,
                )
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()

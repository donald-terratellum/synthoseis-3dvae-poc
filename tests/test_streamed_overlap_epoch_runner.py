import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_augmentation import ViewTransform
from scripts.streamed_overlap_evaluation import StreamedEpochValidator
from scripts.streamed_overlap_journal import BatchJournal
from scripts.streamed_overlap_epoch_runner import (
    HistoricalReconstruction, SelectedReconstruction, SuiteInitialization, _completed, _validate_epoch_inputs,
    complete_epoch,
    complete_selected_checkpoint, prepare_epoch_inputs, run_arm_epochs, run_suite, run_suite_from_record,
    suite_storage_preflight, suite_storage_preflight_from_record, suite_workload,
)
from scripts.streamed_overlap_prepare import prepare
from scripts.streamed_overlap_sampler import PairBatch, PairSpec
from scripts.streamed_overlap_targets import Window
from scripts.train import streamed_monitor_hash


class EpochInputBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.lpips = lambda x, y: (x - y).square().mean()
        self.validator = StreamedEpochValidator(
            self.manifest_path, self.root, manifest_sha256="b" * 64, reconstruction_query_ids=("query",),
            reconstruction_seed=3, encode_batch_size=4, reconstruction_batch_size=2,
            max_wall_seconds=60, lpips_loss_fn=self.lpips,
        )
        self.monitor = {}
        for split in ("train", "validation"):
            window = Window(f"{split}/sample", (0, 0, 0), (1, 1, 1))
            pairs = tuple(PairSpec("same_overlap", window, window, i, i + 12) for i in range(12))
            transform = ViewTransform(0, (1, 1, 1), False, False, False)
            view = {"transform": transform, "noise": None, "retained_fraction": 0.3, "input_mode": "extrema"}
            values = np.zeros((12, 1, 1, 1, 1), dtype=np.float32)
            self.monitor[split] = PairBatch(
                values, values.copy(), values.copy(), np.zeros(12), np.zeros(12), np.ones(12, dtype=bool),
                (None,) * 12, pairs, tuple((view, view) for _ in range(12)),
            )
        self.hashes = {
            "monitor": streamed_monitor_hash(self.monitor), "epoch_validation": self.validator.config_sha256,
            "validation_manifest": "b" * 64,
        }
        self.guardrail = {"protocol": {"lpips_sha256": "c" * 64}}
        self.fingerprint = patch("scripts.streamed_overlap_epoch_runner.lpips_fingerprint", return_value="c" * 64)
        self.fingerprint.start()
        self.addCleanup(self.fingerprint.stop)

    def check(self, **overrides):
        arguments = {
            "monitor_batches": self.monitor, "validate_epoch": self.validator, "lpips_loss_fn": self.lpips,
            "hashes": self.hashes, "manifest_path": self.manifest_path, "source_root": self.root,
            "guardrail_manifest": self.guardrail,
        }
        _validate_epoch_inputs(**{**arguments, **overrides})

    def test_rejects_changed_monitors_validator_and_lpips(self):
        self.check()
        self.monitor["train"].input_a[0, 0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "monitor inputs"):
            self.check()
        self.monitor["train"].input_a[0, 0, 0, 0, 0] = 0
        self.validator.reconstruction_seed += 1
        with self.assertRaisesRegex(ValueError, "validation inputs or configuration"):
            self.check()
        self.validator.reconstruction_seed -= 1
        with self.assertRaisesRegex(ValueError, "validation inputs or configuration"):
            self.check(lpips_loss_fn=lambda x, y: x - y)
        with self.assertRaisesRegex(ValueError, "validation inputs or configuration"):
            self.check(manifest_path=self.root / "other.json")
        with self.assertRaisesRegex(ValueError, "frozen epoch validator"):
            self.check(validate_epoch=object())
        with patch("scripts.streamed_overlap_epoch_runner.lpips_fingerprint", return_value="d" * 64):
            with self.assertRaisesRegex(ValueError, "LPIPS does not match"):
                self.check()

    def test_rejects_mutated_validator_before_evaluation(self):
        self.validator.encode_batch_size += 1
        with self.assertRaisesRegex(ValueError, "changed after it was frozen"):
            self.validator(object())

    def test_prepares_repeatable_epoch_inputs_from_separate_monitor_samplers(self):
        self.manifest_path.touch()
        source = self.root / "source"
        source.mkdir()
        hashes = {"source": "a" * 64, "validation_manifest": "b" * 64, "target": "d" * 64}
        manifest = {"manifest_sha256": hashes["validation_manifest"], "target_implementation_sha256": hashes["target"]}
        matching = {
            "schema": "streamed-matching-config-v1", "run_id": "test", "experiment_seed": 1,
            "model_seed": 2, "head_seed": 3, "sampler_seed": 4, "monitor_seed": 5, "validation_seed": 6,
            "final_epoch": 2, "batches_per_epoch": 3, "sparse_probability": 0.5,
        }
        settings = {
            "reconstruction_query_ids": ["query"], "encode_batch_size": 4,
            "reconstruction_batch_size": 2, "max_wall_seconds": 60,
        }
        lpips = Mock()
        lpips.to.return_value.eval.return_value = lpips
        samplers = []

        def sampler_factory(root, split, seed):
            self.assertEqual((root, seed), (source.resolve(), matching["monitor_seed"]))
            sampler = Mock(source_manifest_hash=hashes["source"])
            sampler.next_specs.return_value = f"{split}-specs"
            sampler.load_batch.return_value = self.monitor[split]
            samplers.append(sampler)
            return sampler

        arguments = {
            "manifest_path": self.manifest_path, "source_root": source, "hashes": hashes,
            "guardrail_manifest": self.guardrail, "matching_config": matching, "epoch_validation": settings,
        }
        with patch("scripts.streamed_overlap_epoch_runner.load_manifest", return_value=manifest), \
             patch("scripts.streamed_overlap_epoch_runner._eligible_queries", return_value=[{"id": "query"}]), \
             patch("scripts.streamed_overlap_epoch_runner.torch.backends.mps.is_available", return_value=True), \
             patch("scripts.streamed_overlap_epoch_runner.SliceLPIPSLoss",
                   side_effect=lambda: (torch.rand(1), lpips)[1]), \
             patch("scripts.streamed_overlap_epoch_runner.StreamedPairSampler", side_effect=sampler_factory):
            rng = torch.get_rng_state()
            batches, validator, loss, observed = prepare_epoch_inputs(**arguments)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertIs(loss, lpips)
            self.assertEqual(batches, self.monitor)
            self.assertEqual(validator.reconstruction_seed, matching["validation_seed"])
            self.assertEqual(observed, {
                "monitor": streamed_monitor_hash(self.monitor), "epoch_validation": validator.config_sha256,
            })
            _, _, _, repeated = prepare_epoch_inputs(**{**arguments, "hashes": {**hashes, **observed}})
            self.assertEqual(repeated, observed)
            for sampler in samplers:
                sampler.next_specs.assert_called_once_with()
                sampler.load_batch.assert_called_once_with(
                    sampler.next_specs.return_value, sparse_probability=matching["sparse_probability"],
                )
            with self.assertRaisesRegex(ValueError, "Prepared monitor differs"):
                prepare_epoch_inputs(**{**arguments, "hashes": {**hashes, "monitor": "e" * 64}})
            with self.assertRaisesRegex(ValueError, "Prepared epoch_validation differs"):
                prepare_epoch_inputs(**{
                    **arguments, "hashes": {**hashes, **observed},
                    "epoch_validation": {**settings, "reconstruction_batch_size": 3},
                })
            with self.assertRaisesRegex(ValueError, "eligible manifest"):
                prepare_epoch_inputs(**{**arguments, "epoch_validation": {
                    **settings, "reconstruction_query_ids": ["not-eligible"],
                }})

    def test_suite_connects_explicit_epoch_settings_before_bootstrap(self):
        prepared = (self.monitor, self.validator, self.lpips, {})
        arguments = {
            "resume": False, "attempt_id": "test", "manifest_path": self.manifest_path,
            "source_root": self.root, "search_manifest": {}, "diagnostic_manifest": {},
            "formal_config": {}, "guardrail_manifest": self.guardrail, "guardrail_config": {},
            "historical": object(), "matching_config": {}, "hashes": self.hashes,
            "epoch_validation": {"settings": "fixture"},
        }
        with patch("scripts.streamed_overlap_epoch_runner.prepare_epoch_inputs", return_value=prepared) as prepare, \
             patch("scripts.streamed_overlap_epoch_runner._validate_suite_inputs",
                   side_effect=ValueError("stop before bootstrap")) as validate:
            with self.assertRaisesRegex(ValueError, "stop before bootstrap"):
                run_suite(self.root / "suite", **arguments)
            prepare.assert_called_once()
            self.assertIs(validate.call_args.kwargs["monitor_batches"], self.monitor)
            self.assertIs(validate.call_args.kwargs["validate_epoch"], self.validator)
            self.assertIs(validate.call_args.kwargs["lpips_loss_fn"], self.lpips)
            self.assertFalse((self.root / "suite").exists())
            with self.assertRaisesRegex(ValueError, "either epoch validation"):
                run_suite(self.root / "suite", **{**arguments, "monitor_batches": self.monitor})

    def test_prepares_record_for_runner_without_training(self):
        source = self.root / "source"
        source.mkdir()
        self.manifest_path.touch()
        matching = {
            "schema": "streamed-matching-config-v1", "run_id": "test", "experiment_seed": 1,
            "model_seed": 2, "head_seed": 3, "sampler_seed": 4, "monitor_seed": 5, "validation_seed": 6,
            "final_epoch": 2, "batches_per_epoch": 3, "sparse_probability": 0.5,
        }
        settings = {"reconstruction_query_ids": ["query"], "encode_batch_size": 4,
                    "reconstruction_batch_size": 2, "max_wall_seconds": 60}
        validator = StreamedEpochValidator(
            self.manifest_path, source, manifest_sha256="b" * 64, reconstruction_query_ids=("query",),
            reconstruction_seed=6, encode_batch_size=4, reconstruction_batch_size=2,
            max_wall_seconds=60, lpips_loss_fn=self.lpips,
        )
        search = {"schema": "streamed-bounded-tokenizer-search-v1", "searches": [{}],
                  "validation_manifest_sha256": "b" * 64}
        search["search_manifest_sha256"] = content_hash(search)
        diagnostic = {"validation_manifest_sha256": "b" * 64, "windows": {}}
        diagnostic["diagnostic_manifest_sha256"] = content_hash(diagnostic)
        guardrail = {
            "schema": "streamed-reconstruction-guardrail-v1", "usage": "post_selection_guardrail",
            "protocol": {"input_mode": "tokenizer", "latent_mode": "sampled",
                         "sampling_method": "normal_reparameterization", "seed": 1,
                         "patch_shape": [32, 32, 64], "lpips_sha256": "c" * 64, "device": "mps"},
            "datasets": {domain: [{"id": domain, "input_sha256": "a" * 64, "target_sha256": "d" * 64}]
                         for domain in ("synthetic", "real")},
        }
        guardrail["manifest_sha256"] = content_hash(guardrail)
        checkpoint = self.root / "checkpoint"
        checkpoint.write_bytes(b"fixture historical checkpoint")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report = {
            "manifest_sha256": guardrail["manifest_sha256"], "protocol": guardrail["protocol"],
            "checkpoint": {"arm": "P0b", "checkpoint_sha256": checkpoint_hash, "checkpoint_label": "reference"},
            "datasets": {
                domain: [
                    {**item, "mae": 1.0, **({"lpips": 0.1, "kl": 0.1} if domain == "synthetic" else {})}
                    for item in records
                ] for domain, records in guardrail["datasets"].items()
            },
        }
        report_path = self.root / "report"
        report_path.write_text(json.dumps(report))
        sidecars = {
            "search_manifest": search, "diagnostic_manifest": diagnostic, "guardrail_manifest": guardrail,
        }
        paths = {}
        for key, value in sidecars.items():
            path = self.root / f"{key}.json"
            path.write_text(json.dumps(value))
            paths[key] = str(path)
        config = {
            "suite_dir": str(self.root / "new-suite"), "source_root": str(source),
            "validation_manifest": str(self.manifest_path), "source_sha256": "a" * 64,
            "target_sha256": "d" * 64, "validation_manifest_sha256": "b" * 64,
            "matching_config": matching, "epoch_validation": settings,
            "formal_config": {
                "final_epoch": 2, "device": "mps", "inference_seed": 1, "encode_batch_size": 4,
                "max_wall_seconds": 30, "random_seed": 2, "random_repeats": 3,
                "bootstrap_seed": 4, "bootstrap_repeats": 5, "diagnostic_max_wall_seconds": 30,
                "search_manifest_sha256": search["search_manifest_sha256"],
                "diagnostic_manifest_sha256": diagnostic["diagnostic_manifest_sha256"],
            },
            "guardrail_config": {"manifest_sha256": guardrail["manifest_sha256"],
                                 "device": "mps", "max_wall_seconds": 60}, **paths,
            "historical": {"report_path": str(report_path),
                           "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                           "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                           "checkpoint_label": "reference"},
        }
        observed = {"monitor": streamed_monitor_hash(self.monitor), "epoch_validation": validator.config_sha256}
        output = self.root / "suite-inputs.json"
        with patch("scripts.streamed_overlap_prepare.torch.backends.mps.is_available", return_value=True), \
             patch("scripts.streamed_overlap_prepare.torch.mps.set_per_process_memory_fraction") as set_limit, \
             patch("scripts.streamed_overlap_prepare.prepare_epoch_inputs",
                   return_value=(self.monitor, validator, self.lpips, observed)), \
             patch("scripts.streamed_overlap_epoch_runner.run_suite", return_value={"fixture": {}}) as run:
            result = prepare(config, output, mps_memory_fraction=0.5, max_wall_seconds=30)
            run.assert_not_called()
            set_limit.assert_called_once_with(0.5)
            self.assertEqual(result["status"], "SUITE_INPUTS_PREPARED_NOT_AUTHORIZED")
            record = json.loads(output.read_text())
            self.assertEqual(result["config_sha256"], content_hash(record))
            self.assertEqual(record["input_hashes"]["monitor"], observed["monitor"])
            self.assertFalse((self.root / "new-suite").exists())
            storage_options = {
                "expected_config_sha256": result["config_sha256"],
                "checkpoint_upper_bytes": {"S1": 1, "S2": 1, "S3": 1},
                "other_output_bytes": 1, "free_reserve_bytes": 1,
            }
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer must not step")):
                storage = suite_storage_preflight_from_record(output, self.root / "new-suite", **storage_options)
            self.assertEqual(storage["suite_config_sha256"], result["config_sha256"])
            self.assertEqual(storage["checkpoint_counts"], {"S1": 5, "S2": 5, "S3": 6})
            run.assert_not_called()
            with self.assertRaisesRegex(ValueError, "pinned configuration hash"):
                suite_storage_preflight_from_record(
                    output, self.root / "new-suite", **{**storage_options, "expected_config_sha256": "0" * 64},
                )
            with self.assertRaisesRegex(ValueError, "pinned destination"):
                suite_storage_preflight_from_record(output, self.root / "wrong-suite", **storage_options)
            changed_record = copy.deepcopy(record)
            changed_record["matching_config"]["final_epoch"] = 3
            output.write_text(json.dumps(changed_record))
            with self.assertRaisesRegex(ValueError, "inconsistent matching"):
                suite_storage_preflight_from_record(
                    output, self.root / "new-suite",
                    **{**storage_options, "expected_config_sha256": content_hash(changed_record)},
                )
            output.write_text(json.dumps(record))
            command = [
                sys.executable, "-m", "scripts.streamed_overlap_storage",
                "--suite-record", str(output), "--suite-dir", str(self.root / "new-suite"),
                "--config-sha256", result["config_sha256"], "--s1-checkpoint-upper-bytes", "1",
                "--s2-checkpoint-upper-bytes", "1", "--s3-checkpoint-upper-bytes", "1",
                "--other-output-bytes", "1", "--free-reserve-bytes", "1",
            ]
            review = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(review.stdout)["suite_config_sha256"], result["config_sha256"])
            command[-1] = str(10**30)
            failed_review = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed_review.returncode, 0)
            self.assertIn("Insufficient estimated suite storage", failed_review.stderr)
            self.assertFalse((self.root / "new-suite").exists())
            run_suite_from_record(
                output, self.root / "new-suite", expected_config_sha256=result["config_sha256"],
                resume=False, attempt_id="fixture",
            )
            self.assertEqual(run.call_args.kwargs["epoch_validation"], settings)
            changed_report = json.loads(report_path.read_text())
            changed_report["datasets"]["real"] = []
            report_path.write_text(json.dumps(changed_report))
            changed_config = {**config, "historical": {
                **config["historical"], "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }}
            rejected_output = self.root / "rejected-suite-inputs.json"
            with self.assertRaisesRegex(ValueError, "full real"):
                prepare(changed_config, rejected_output, mps_memory_fraction=0.5, max_wall_seconds=30)
            self.assertFalse(rejected_output.exists())
            with patch("scripts.streamed_overlap_epoch_runner.run_suite", wraps=run_suite), \
                 patch("scripts.streamed_overlap_epoch_runner.prepare_epoch_inputs",
                       return_value=(self.monitor, validator, self.lpips, observed)), \
                 patch("scripts.streamed_overlap_epoch_runner.prepare_suite_initial_checkpoints") as bootstrap, \
                 patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
                with self.assertRaisesRegex(ValueError, "Historical reconstruction.*missing or changed"):
                    run_suite_from_record(
                        output, self.root / "new-suite", expected_config_sha256=result["config_sha256"],
                        resume=False, attempt_id="fixture",
                    )
                bootstrap.assert_not_called()
                train.assert_not_called()
                self.assertFalse((self.root / "new-suite").exists())
            report_path.write_text(json.dumps(report))
            with self.assertRaises(FileExistsError):
                prepare(config, output, mps_memory_fraction=0.5, max_wall_seconds=30)
            with self.assertRaisesRegex(ValueError, "must be separate"):
                prepare(config, source / "output.json", mps_memory_fraction=0.5, max_wall_seconds=30)
            with self.assertRaisesRegex(ValueError, "memory limits"):
                prepare(config, self.root / "other.json", mps_memory_fraction=0.8, max_wall_seconds=30)
            with self.assertRaisesRegex(ValueError, "matching hashes"):
                prepare({**config, "validation_manifest_sha256": "0" * 64}, self.root / "other.json",
                        mps_memory_fraction=0.5, max_wall_seconds=30)
            with patch("scripts.streamed_overlap_prepare.time.monotonic", side_effect=[0, 31]):
                with self.assertRaisesRegex(TimeoutError, "no input record"):
                    prepare(config, self.root / "other.json", mps_memory_fraction=0.5, max_wall_seconds=30)
            self.assertFalse((self.root / "other.json").exists())


class SuiteWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.matching = {
            "schema": "streamed-matching-config-v1", "run_id": "test", "experiment_seed": 1,
            "model_seed": 2, "head_seed": 3, "sampler_seed": 4, "monitor_seed": 5, "validation_seed": 6,
            "final_epoch": 6, "batches_per_epoch": 21, "sparse_probability": 0.5,
        }
        self.search: dict = {"schema": "streamed-bounded-tokenizer-search-v1", "searches": [{}, {}]}
        self.search["search_manifest_sha256"] = content_hash(self.search)

    def test_counts_exposure_monitors_and_final_selected_evaluations(self):
        result = suite_workload(self.matching, self.search)
        self.assertEqual(result["per_arm"], {
            "epochs": 6, "optimizer_steps": 126, "training_pairs": 1512, "training_encoder_views": 3024,
            "training_reconstructed_a": 1512, "epoch_validations": 6, "monitor_runs": 7,
            "epoch_formal_evaluations": 2,
        })
        self.assertEqual(result["suite"], {
            "optimizer_steps": 378, "training_pairs": 4536, "training_encoder_views": 9072,
            "training_reconstructed_a": 4536, "epoch_validations": 18, "monitor_runs": 21,
            "epoch_formal_evaluations": 6, "selected_formal_evaluations": 4,
            "s3_diagnostic_evaluations": 4, "tokenizer_searches": 20,
        })

    def test_epoch_five_and_step_100_are_not_double_counted(self):
        result = suite_workload({**self.matching, "final_epoch": 5, "batches_per_epoch": 20}, self.search)
        self.assertEqual(result["per_arm"]["monitor_runs"], 5)
        self.assertEqual(result["per_arm"]["epoch_formal_evaluations"], 1)
        self.assertEqual(result["suite"]["tokenizer_searches"], 14)

    def test_rejects_missing_budget_and_changed_search_manifest(self):
        with self.assertRaisesRegex(ValueError, "matching configuration"):
            suite_workload({**self.matching, "final_epoch": 0}, self.search)
        with self.assertRaisesRegex(ValueError, "frozen bounded-search"):
            suite_workload(self.matching, {**self.search, "searches": [{}]})
        empty: dict = {"schema": self.search["schema"], "searches": []}
        empty["search_manifest_sha256"] = content_hash(empty)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            suite_workload(self.matching, empty)

    def test_storage_preflight_counts_snapshots_and_shared_transient_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            suite = directory / "suite"
            options = {
                "checkpoint_upper_bytes": {"S1": 100, "S2": 200, "S3": 300},
                "other_output_bytes": 1000, "free_reserve_bytes": 200,
            }
            with patch("scripts.streamed_overlap_epoch_runner.os.statvfs",
                       return_value=SimpleNamespace(f_frsize=10, f_bavail=720)):
                result = suite_storage_preflight(self.matching, self.search, suite, **options)
            self.assertEqual(result["checkpoint_counts"], {"S1": 9, "S2": 9, "S3": 10})
            self.assertEqual(result["checkpoint_bytes"], 5700)
            self.assertEqual(result["transient_checkpoint_bytes"], 300)
            self.assertEqual(result["required_bytes"], 7200)
            self.assertEqual(result["available_bytes"], 7200)
            self.assertEqual(list(directory.iterdir()), [])
            with patch("scripts.streamed_overlap_epoch_runner.os.statvfs",
                       return_value=SimpleNamespace(f_frsize=10, f_bavail=719)):
                with self.assertRaisesRegex(OSError, "need 7200 bytes, available 7190 bytes"):
                    suite_storage_preflight(self.matching, self.search, suite, **options)

    def test_storage_preflight_rejects_invalid_bounds_and_filesystems(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            suite = directory / "suite"
            options = {
                "checkpoint_upper_bytes": {"S1": 100, "S2": 200, "S3": 300},
                "other_output_bytes": 1000, "free_reserve_bytes": 200,
            }
            for overrides in (
                {"checkpoint_upper_bytes": {"S1": 100, "S2": 200}},
                {"checkpoint_upper_bytes": {"S1": True, "S2": 200, "S3": 300}},
                {"other_output_bytes": 0}, {"free_reserve_bytes": -1},
            ):
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "positive"):
                    suite_storage_preflight(self.matching, self.search, suite, **{**options, **overrides})
            with self.assertRaisesRegex(ValueError, "existing permitted parent"):
                suite_storage_preflight(self.matching, self.search, directory / "missing" / "suite", **options)
            with self.assertRaisesRegex(ValueError, "new absolute suite"):
                suite_storage_preflight(self.matching, self.search, directory, **options)
            alias = directory / "alias"
            alias.symlink_to(directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "new absolute suite"):
                suite_storage_preflight(self.matching, self.search, alias / "suite", **options)
            with self.assertRaisesRegex(ValueError, "frozen bounded-search"):
                suite_storage_preflight(self.matching, {**self.search, "searches": [{}]}, suite, **options)
            with patch("scripts.streamed_overlap_epoch_runner.os.statvfs",
                       return_value=SimpleNamespace(f_frsize=0, f_bavail=10)):
                with self.assertRaisesRegex(RuntimeError, "available-space"):
                    suite_storage_preflight(self.matching, self.search, suite, **options)
            measured = suite_storage_preflight(
                self.matching, self.search, suite,
                checkpoint_upper_bytes={"S1": 1, "S2": 1, "S3": 1},
                other_output_bytes=1, free_reserve_bytes=1,
            )
            self.assertGreaterEqual(measured["available_bytes"], measured["required_bytes"])


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
        self.guardrail_manifest = {
            "schema": "streamed-reconstruction-guardrail-v1", "usage": "post_selection_guardrail",
            "protocol": {
                "input_mode": "tokenizer", "latent_mode": "sampled",
                "sampling_method": "normal_reparameterization", "seed": 1,
                "patch_shape": [32, 32, 64], "lpips_sha256": "a" * 64, "device": "mps",
            },
            "datasets": {
                domain: [{"id": domain, "input_sha256": "b" * 64, "target_sha256": "c" * 64}]
                for domain in ("synthetic", "real")
            },
        }
        self.guardrail_manifest["manifest_sha256"] = content_hash(self.guardrail_manifest)
        self.guardrail_config = {
            "manifest_sha256": self.guardrail_manifest["manifest_sha256"],
            "device": "mps", "max_wall_seconds": 60,
        }
        historical_checkpoint = self.root / "p0b.pt"
        historical_checkpoint.write_bytes(b"historical checkpoint")
        historical_checkpoint_hash = hashlib.sha256(historical_checkpoint.read_bytes()).hexdigest()
        historical_report = {
            "manifest_sha256": self.guardrail_manifest["manifest_sha256"],
            "protocol": self.guardrail_manifest["protocol"],
            "checkpoint": {"arm": "P0b", "checkpoint_sha256": historical_checkpoint_hash,
                           "checkpoint_label": "reconstruction-best"},
            "datasets": {
                domain: [{**row, "mae": 1.0, **({"lpips": 0.1, "kl": 0.1} if domain == "synthetic" else {})}
                         for row in rows] for domain, rows in self.guardrail_manifest["datasets"].items()
            },
        }
        historical_report_path = self.root / "p0b.json"
        historical_report_path.write_text(json.dumps(historical_report))
        self.historical = HistoricalReconstruction(
            historical_report_path, hashlib.sha256(historical_report_path.read_bytes()).hexdigest(),
            historical_checkpoint, historical_checkpoint_hash, "reconstruction-best",
        )
        s1_journal = Mock(spec=BatchJournal)
        s1_journal.path = self.root / "batch_metrics.csv"
        self.s1 = SelectedReconstruction(
            self.checkpoint, s1_journal, self.root / "manifest", self.source,
            self.search_manifest, self.diagnostic_manifest, self.config, self.evaluation, "reconstruction",
        )
        self.arm_args = (
            object(), object(), object(), SimpleNamespace(path=self.root / "batch_metrics.csv"), object(), object(),
        )
        self.matching_config = {
            "schema": "streamed-matching-config-v1", "run_id": "run", "experiment_seed": 1,
            "model_seed": 1, "head_seed": 2, "sampler_seed": 3, "monitor_seed": 1, "validation_seed": 1,
            "final_epoch": 6, "batches_per_epoch": 2, "sparse_probability": 0.5,
        }
        self.suite_initialization = SuiteInitialization(
            self.root / "initialization_evidence.json", "f" * 64,
            {arm: self.root / "initial.pt" for arm in ("S1", "S2", "S3")},
        )
        self.hashes = {
            "formal_evaluation": content_hash(self.config),
            "matching": content_hash(self.matching_config),
            "reconstruction_guardrail": content_hash(self.guardrail_config),
            "reconstruction_reference": content_hash({
                "report_sha256": self.historical.report_sha256,
                "checkpoint_sha256": self.historical.checkpoint_sha256,
                "checkpoint_label": "reconstruction-best",
            }),
        }
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
            patch("scripts.streamed_overlap_epoch_runner.verify_arm_initial_checkpoint", return_value="a" * 64),
            patch("scripts.streamed_overlap_epoch_runner.verify_saved_suite_initialization", return_value={
                "initialization": {"initial_checkpoint_sha256": {
                    arm: "a" * 64 for arm in ("S1", "S2", "S3")
                }},
            }),
            patch("scripts.streamed_overlap_epoch_runner.verify_suite_run_metadata",
                  return_value={"initialization_sha256": "f" * 64}),
            patch("scripts.streamed_overlap_epoch_runner._validate_epoch_inputs"),
            patch("scripts.streamed_overlap_epoch_runner.complete_matched_arm",
                  return_value={"record": {"status": "MATCHED_RECORDED_STREAMS_ONLY"}}),
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

    def test_fresh_suite_rejects_unbound_config_before_creating_output(self):
        destination = self.root / "suite"
        with patch("scripts.streamed_overlap_epoch_runner.StreamedPairSampler") as sampler:
            with self.assertRaisesRegex(ValueError, "frozen formal evaluation"):
                run_suite(
                    destination, resume=False, attempt_id="fixture", monitor_batches={},
                    validate_epoch=object(), lpips_loss_fn=object(), manifest_path=self.root / "manifest",
                    source_root=self.source, search_manifest=self.search_manifest,
                    diagnostic_manifest=self.diagnostic_manifest, formal_config={**self.config, "random_seed": 99},
                    guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                    historical=self.historical, matching_config=self.matching_config, hashes={
                        **self.hashes, "config": "a" * 64, "source": "b" * 64,
                        "target": "c" * 64, "validation_manifest": "d" * 64,
                    },
                )
            sampler.assert_not_called()
        self.assertFalse(destination.exists())

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
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch", side_effect=train), \
             patch("scripts.streamed_overlap_epoch_runner.complete_matched_arm",
                   side_effect=lambda *args, **kwargs: order.append("match") or {"record": {"status": "matched"}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   side_effect=lambda *args, **kwargs: order.append(kwargs["selection_kind"]) or {"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_reconstruction_comparison",
                   side_effect=lambda candidate, historical, **kwargs: (
                       order.append(f"compare-{candidate.selection_kind}") or {
                           "comparison": {"status": "FAIL" if candidate.selection_kind == "retrieval"
                                          else "PASS_NUMERIC_ONLY"},
                       }
                   )) as compare:
            result = run_arm_epochs(
                SimpleNamespace(arm="S3"), *self.arm_args,
                run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                hashes=self.hashes, early_stopping=object(),
                manifest_path=self.root / "manifest", source_root=self.source,
                search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                formal_config=self.config, evaluation_dir=self.evaluation,
                guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                historical=self.historical, s1=self.s1, matching_config=self.matching_config,
                initial_checkpoint=self.root / "initial.pt",
                suite_initialization=self.suite_initialization,
            )
        self.assertEqual(order, ["evaluate", "train-5", "evaluate", "train-6", "evaluate", "match",
                                 "reconstruction", "retrieval", "compare-reconstruction", "compare-retrieval"])
        self.assertEqual(result["last_epoch"]["done"], 5)
        self.assertEqual(result["matched_streams"]["record"]["status"], "matched")
        self.assertEqual(set(result["selected_evaluations"]), {"reconstruction", "retrieval"})
        self.assertEqual(result["reconstruction_comparisons"]["retrieval"]["comparison"]["status"], "FAIL")
        self.assertEqual(compare.call_count, 2)
        self.assertTrue(all(call.kwargs["s1"] is self.s1 for call in compare.call_args_list))

    def test_evaluation_only_rejects_incomplete_restore_and_never_trains(self):
        hashes = {**self.hashes, "validation_manifest": "c" * 64}
        self.state = {
            "arm": "S3", "device": "mps", "completed_epoch": 6,
            "batches_per_epoch": 2, "hashes": hashes,
        }
        arguments = {
            "run_id": "run", "seed": 1, "attempt_id": "evaluation", "batches_per_epoch": 2,
            "sparse_probability": 0.5, "monitor_batches": {}, "monitor_seed": 1, "validation_seed": 1,
            "validate_epoch": object(), "epoch_csv": self.root / "epoch.csv", "checkpoint_path": self.checkpoint,
            "hashes": hashes, "early_stopping": object(), "manifest_path": self.root / "manifest",
            "source_root": self.source, "search_manifest": self.search_manifest,
            "diagnostic_manifest": self.diagnostic_manifest, "formal_config": self.config,
            "evaluation_dir": self.evaluation, "guardrail_manifest": self.guardrail_manifest,
            "guardrail_config": self.guardrail_config, "historical": self.historical, "s1": self.s1,
            "matching_config": self.matching_config, "initial_checkpoint": self.root / "initial.pt",
            "suite_initialization": self.suite_initialization, "evaluate_only": True,
        }
        with patch("scripts.streamed_overlap_epoch_runner._identity",
                   return_value={"run_id": "run", "experiment_seed": 1}), \
             patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 5, "selection_state": {}}) as restore, \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", return_value={"done": True}) as finish, \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint", return_value={"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_reconstruction_comparison",
                   return_value={"comparison": {"status": "PASS_NUMERIC_ONLY"}}), \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            self.state["completed_epoch"] = 5
            with self.assertRaisesRegex(ValueError, "completed, configuration-matched"):
                run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **arguments)
            restore.assert_not_called()
            self.state["completed_epoch"] = 6
            with self.assertRaisesRegex(ValueError, "cannot advance an incomplete arm"):
                run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **arguments)
            finish.assert_not_called()
            restore.return_value = {"completed_epoch": 6, "selection_state": {}}
            result = run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **arguments)
            self.assertEqual(set(result["selected_evaluations"]), {"reconstruction", "retrieval"})
            finish.assert_called_once()
            train.assert_not_called()

    def test_pending_evaluation_failure_prevents_next_epoch(self):
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 4, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", side_effect=OSError("evaluation failed")), \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(OSError, "evaluation failed"):
                run_arm_epochs(
                    SimpleNamespace(arm="S3"), *self.arm_args, run_id="run", seed=1,
                    attempt_id="resume", batches_per_epoch=2,
                    sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                    validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                    hashes=self.hashes, early_stopping=object(),
                    manifest_path=self.root / "manifest", source_root=self.source,
                    search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                    formal_config=self.config, evaluation_dir=self.evaluation,
                    guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                    historical=self.historical, s1=self.s1, matching_config=self.matching_config,
                    initial_checkpoint=self.root / "initial.pt",
                    suite_initialization=self.suite_initialization,
                )
            train.assert_not_called()

    def test_changed_historical_checkpoint_stops_before_restore_or_training(self):
        self.historical.checkpoint_path.write_bytes(b"changed")
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state") as restore, \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(ValueError, "Historical reconstruction.*missing or changed"):
                run_arm_epochs(
                    SimpleNamespace(arm="S3"), *self.arm_args, run_id="run", seed=1,
                    attempt_id="resume", batches_per_epoch=2,
                    sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                    validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                    hashes=self.hashes, early_stopping=object(), manifest_path=self.root / "manifest",
                    source_root=self.source, search_manifest=self.search_manifest,
                    diagnostic_manifest=self.diagnostic_manifest, formal_config=self.config,
                    evaluation_dir=self.evaluation, guardrail_manifest=self.guardrail_manifest,
                    guardrail_config=self.guardrail_config, historical=self.historical, s1=self.s1,
                    matching_config=self.matching_config, initial_checkpoint=self.root / "initial.pt",
                    suite_initialization=self.suite_initialization,
                )
            restore.assert_not_called()
            train.assert_not_called()

    def test_guardrails_are_required_before_restoring_or_training(self):
        def run(arm="S3", *, hashes=None, s1=None):
            return run_arm_epochs(
                SimpleNamespace(arm=arm), *self.arm_args,
                run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                hashes=self.hashes if hashes is None else hashes, early_stopping=object(),
                manifest_path=self.root / "manifest", source_root=self.source,
                search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                formal_config=self.config, evaluation_dir=self.evaluation,
                guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                historical=self.historical, s1=s1, matching_config=self.matching_config,
                initial_checkpoint=self.root / "initial.pt",
                suite_initialization=self.suite_initialization,
            )

        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state") as restore:
            with self.assertRaisesRegex(ValueError, "bind the frozen"):
                run(hashes={**self.hashes, "reconstruction_reference": "f" * 64}, s1=self.s1)
            with self.assertRaisesRegex(ValueError, "bind the frozen"):
                run(hashes={**self.hashes, "reconstruction_guardrail": "f" * 64}, s1=self.s1)
            with self.assertRaisesRegex(ValueError, "require the selected S1"):
                run()
            with self.assertRaisesRegex(ValueError, "S1 needs P0b only"):
                run("S1", s1=self.s1)
            self.mocks[-2].side_effect = ValueError("invalid epoch inputs")
            with self.assertRaisesRegex(ValueError, "invalid epoch inputs"):
                run(s1=self.s1)
            restore.assert_not_called()

    def test_matching_configuration_is_required_before_restoring_or_training(self):
        kwargs = {
            "run_id": "run", "seed": 1, "attempt_id": "resume", "batches_per_epoch": 2,
            "sparse_probability": 0.5, "monitor_batches": {}, "monitor_seed": 1, "validation_seed": 1,
            "validate_epoch": object(), "epoch_csv": self.root / "epoch.csv", "checkpoint_path": self.checkpoint,
            "hashes": self.hashes, "early_stopping": object(), "manifest_path": self.root / "manifest",
            "source_root": self.source, "search_manifest": self.search_manifest,
            "diagnostic_manifest": self.diagnostic_manifest, "formal_config": self.config,
            "evaluation_dir": self.evaluation, "guardrail_manifest": self.guardrail_manifest,
            "guardrail_config": self.guardrail_config, "historical": self.historical, "s1": self.s1,
            "matching_config": self.matching_config, "initial_checkpoint": self.root / "initial.pt",
            "suite_initialization": self.suite_initialization,
        }
        cases = {
            "run_id": "other", "seed": 2, "batches_per_epoch": 3, "sparse_probability": 0.1,
            "monitor_seed": 2, "validation_seed": 2,
            "matching_config": {**self.matching_config, "model_seed": 9},
        }
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state") as restore, \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            for key, value in cases.items():
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen matching configuration"):
                    run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **{**kwargs, key: value})
            changed_formal = {**self.config, "final_epoch": 7}
            with self.assertRaisesRegex(ValueError, "frozen matching configuration"):
                run_arm_epochs(
                    SimpleNamespace(arm="S3"), *self.arm_args,
                    **{**kwargs, "formal_config": changed_formal,
                       "hashes": {**self.hashes, "formal_evaluation": content_hash(changed_formal)}},
                )
            for key, changed in (
                ("search_manifest", {**self.search_manifest, "searches": ["changed"]}),
                ("diagnostic_manifest", {**self.diagnostic_manifest, "windows": {"changed": {}}}),
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen MPS evaluation"):
                    run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **{**kwargs, key: changed})
            for key, value in (
                ("encode_batch_size", 0), ("inference_seed", -1), ("random_repeats", 1),
                ("bootstrap_repeats", 1), ("max_wall_seconds", 0), ("diagnostic_max_wall_seconds", 0),
            ):
                invalid_formal = {**self.config, key: value}
                invalid_hashes = {**self.hashes, "formal_evaluation": content_hash(invalid_formal)}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    run_arm_epochs(
                        SimpleNamespace(arm="S3"), *self.arm_args,
                        **{**kwargs, "formal_config": invalid_formal, "hashes": invalid_hashes},
                    )
            self.mocks[8].side_effect = ValueError("initial checkpoint invalid")
            with self.assertRaisesRegex(ValueError, "initial checkpoint invalid"):
                run_arm_epochs(SimpleNamespace(arm="S3"), *self.arm_args, **kwargs)
            restore.assert_not_called()
            train.assert_not_called()

    def test_s1_guardrail_retries_after_post_selection_failure_without_training(self):
        attempts = []

        def compare(candidate, historical, **kwargs):
            attempts.append((candidate.selection_kind, historical, kwargs["s1"]))
            if len(attempts) == 1:
                raise OSError("guardrail interrupted")
            return {"comparison": {"status": "PASS_NUMERIC_ONLY"}}

        def run():
            return run_arm_epochs(
                SimpleNamespace(arm="S1"), *self.arm_args,
                run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                hashes=self.hashes, early_stopping=object(),
                manifest_path=self.root / "manifest", source_root=self.source,
                search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                formal_config=self.config, evaluation_dir=self.evaluation,
                guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                historical=self.historical, s1=None, matching_config=self.matching_config,
                initial_checkpoint=self.root / "initial.pt",
                suite_initialization=self.suite_initialization,
            )

        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 6, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", return_value={"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_reconstruction_comparison", side_effect=compare), \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(OSError, "guardrail interrupted"):
                run()
            result = run()
            train.assert_not_called()
        self.assertEqual(attempts, [("reconstruction", self.historical, None)] * 2)
        self.assertEqual(set(result["reconstruction_comparisons"]), {"reconstruction"})
        self.mocks[-1].assert_not_called()

    def test_s2_compares_only_reconstruction_with_s1_reference(self):
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 6, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", return_value={"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"done": True}) as selected, \
             patch("scripts.streamed_overlap_epoch_runner.complete_reconstruction_comparison",
                   return_value={"comparison": {"status": "PASS_NUMERIC_ONLY"}}) as compare, \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            result = run_arm_epochs(
                SimpleNamespace(arm="S2"), *self.arm_args,
                run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                hashes=self.hashes, early_stopping=object(),
                manifest_path=self.root / "manifest", source_root=self.source,
                search_manifest=self.search_manifest, diagnostic_manifest=self.diagnostic_manifest,
                formal_config=self.config, evaluation_dir=self.evaluation,
                guardrail_manifest=self.guardrail_manifest, guardrail_config=self.guardrail_config,
                historical=self.historical, s1=self.s1, matching_config=self.matching_config,
                initial_checkpoint=self.root / "initial.pt",
                suite_initialization=self.suite_initialization,
            )
        train.assert_not_called()
        selected.assert_called_once()
        self.assertEqual(selected.call_args.kwargs["selection_kind"], "reconstruction")
        self.assertEqual(compare.call_args.args[0].selection_kind, "reconstruction")
        self.assertIs(compare.call_args.kwargs["s1"], self.s1)
        self.assertEqual(set(result["reconstruction_comparisons"]), {"reconstruction"})
        self.assertEqual(result["matched_streams"]["record"]["status"], "MATCHED_RECORDED_STREAMS_ONLY")
        self.mocks[-1].assert_called_once()

    def test_unmatched_candidate_stops_before_selected_evaluation(self):
        self.mocks[-1].side_effect = ValueError("unmatched streams")
        with patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   return_value={"completed_epoch": 6, "selection_state": {}}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_epoch", return_value={"done": True}), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint") as selected, \
             patch("scripts.streamed_overlap_epoch_runner.complete_reconstruction_comparison") as compare, \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(ValueError, "unmatched streams"):
                run_arm_epochs(
                    SimpleNamespace(arm="S2"), *self.arm_args,
                    run_id="run", seed=1, attempt_id="resume", batches_per_epoch=2,
                    sparse_probability=0.5, monitor_batches={}, monitor_seed=1, validation_seed=1,
                    validate_epoch=object(), epoch_csv=self.root / "epoch.csv", checkpoint_path=self.checkpoint,
                    hashes=self.hashes, early_stopping=object(), manifest_path=self.root / "manifest",
                    source_root=self.source, search_manifest=self.search_manifest,
                    diagnostic_manifest=self.diagnostic_manifest, formal_config=self.config,
                    evaluation_dir=self.evaluation, guardrail_manifest=self.guardrail_manifest,
                    guardrail_config=self.guardrail_config, historical=self.historical, s1=self.s1,
                    matching_config=self.matching_config, initial_checkpoint=self.root / "initial.pt",
                    suite_initialization=self.suite_initialization,
                )
            train.assert_not_called()
            selected.assert_not_called()
            compare.assert_not_called()

    def test_selected_earlier_checkpoint_is_evaluated_and_recovered(self):
        self.epoch = 6
        self.state["hashes"]["epoch_validation"] = "e" * 64
        selection = self.root / "selection"
        events, snapshots = selection / "events", selection / "snapshots"
        events.mkdir(parents=True)
        snapshots.mkdir()
        selected_bytes = b"earlier checkpoint"
        selected_hash = hashlib.sha256(selected_bytes).hexdigest()
        snapshot = snapshots / f"{selected_hash}.pt"
        snapshot.write_bytes(selected_bytes)
        alias = self.root / "best_reconstruction.pt"
        alias.write_bytes(selected_bytes)
        last_identity = {**self.identity(), "step": 6}
        selected_identity = {**last_identity, "epoch": 1, "step": 1, "checkpoint_sha256": selected_hash}
        record = {
            "checkpoint": {**selected_identity, "epoch_validation_sha256": "e" * 64},
            "reconstruction": 0.1, "retrieval": None,
        }
        journal = SimpleNamespace(position=SimpleNamespace(logical_step=6, cumulative_exposure=72))

        def state_for(path):
            return self.state, self.digest if path == self.checkpoint else selected_hash

        def identity_for(_state, digest, *_args):
            return last_identity if digest == self.digest else selected_identity

        def evaluate(path, *_args, **_kwargs):
            self.assertEqual(path, snapshot)
            return {
                "checkpoint": selected_identity,
                "scorers": {"pair_head": {"event_coverage_diagnostics": {"pairs": [
                    {"query_id": "q", "candidate_id": 0, "prediction": 0.4},
                ]}}},
            }

        def search(path, *_args):
            self.assertEqual(path, snapshot)
            output = _args[-1]
            output.mkdir()
            (output / "tokenizer.pt").write_bytes(b"fixture tokenizer")
            return {
                "checkpoint": {"checkpoint": selected_identity,
                               "tokenizer_checkpoint_sha256": hashlib.sha256(b"fixture tokenizer").hexdigest()},
                "searches": [],
            }

        def run():
            return complete_selected_checkpoint(
                self.checkpoint, journal, self.root / "manifest", self.source,
                self.search_manifest, self.diagnostic_manifest, self.config,
                selection_kind="reconstruction", evaluation_dir=self.evaluation,
            )

        def completed(directory, *args):
            return {"done": True} if directory == self.evaluation else _completed(directory, *args)

        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state", side_effect=state_for), \
             patch("scripts.streamed_overlap_epoch_runner._identity", side_effect=identity_for), \
             patch("scripts.streamed_overlap_epoch_runner._completed", side_effect=completed), \
             patch("scripts.streamed_overlap_epoch_runner._records", return_value=[record]), \
             patch("scripts.streamed_overlap_epoch_runner.evaluate_checkpoint", side_effect=evaluate), \
             patch("scripts.streamed_overlap_epoch_runner.run_bounded_searches",
                   side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                run()
        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state", side_effect=state_for), \
             patch("scripts.streamed_overlap_epoch_runner._identity", side_effect=identity_for), \
             patch("scripts.streamed_overlap_epoch_runner._completed", side_effect=completed), \
             patch("scripts.streamed_overlap_epoch_runner._records", return_value=[record]), \
             patch("scripts.streamed_overlap_epoch_runner.evaluate_checkpoint", side_effect=evaluate) as score, \
             patch("scripts.streamed_overlap_epoch_runner.run_bounded_searches", side_effect=search):
            result = run()
            self.assertEqual(result["checkpoint"], selected_identity)
            self.assertEqual(run(), result)
            self.assertEqual(score.call_count, 1)
            self.assertEqual(len(list((self.evaluation / "selected-reconstruction").glob("checkpoint-*"))), 2)
            with self.assertRaisesRegex(ValueError, "selection is unavailable"):
                complete_selected_checkpoint(
                    self.checkpoint, journal, self.root / "manifest", self.source,
                    self.search_manifest, self.diagnostic_manifest, self.config,
                    selection_kind="retrieval", evaluation_dir=self.evaluation,
                )
            alias.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "snapshot or alias"):
                run()

    def test_selected_evaluation_requires_reconciled_final_epoch(self):
        self.state["hashes"]["epoch_validation"] = "e" * 64
        journal = SimpleNamespace(position=SimpleNamespace(logical_step=5, cumulative_exposure=60))
        with patch("scripts.streamed_overlap_epoch_runner._identity",
                   return_value={**self.identity(), "step": 5}), \
             patch("scripts.streamed_overlap_epoch_runner._records") as records:
            with self.assertRaisesRegex(ValueError, "reconciled final epoch"):
                complete_selected_checkpoint(
                    self.checkpoint, journal, self.root / "manifest", self.source,
                    self.search_manifest, self.diagnostic_manifest, self.config,
                    selection_kind="reconstruction", evaluation_dir=self.evaluation,
                )
            records.assert_not_called()

    def test_selected_evaluation_requires_completed_final_epoch_evaluation(self):
        self.epoch = 6
        self.state["hashes"]["epoch_validation"] = "e" * 64
        journal = SimpleNamespace(position=SimpleNamespace(logical_step=6, cumulative_exposure=72))
        with patch("scripts.streamed_overlap_epoch_runner._identity",
                   return_value={**self.identity(), "step": 6}), \
             patch("scripts.streamed_overlap_epoch_runner._records") as records:
            with self.assertRaisesRegex(ValueError, "Final epoch evaluation must be complete"):
                complete_selected_checkpoint(
                    self.checkpoint, journal, self.root / "manifest", self.source,
                    self.search_manifest, self.diagnostic_manifest, self.config,
                    selection_kind="reconstruction", evaluation_dir=self.evaluation,
                )
            records.assert_not_called()


if __name__ == "__main__":
    unittest.main()

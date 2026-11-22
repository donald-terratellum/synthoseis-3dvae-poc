import copy
import hashlib
import json
import os
import random
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import torch

from scripts.p7_control import capture_training_state
from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import (
    ArmRuntime, HistoricalReconstruction, SelectedReconstruction, SuiteInitialization, run_arm_epochs, run_suite_epochs,
)
from scripts.streamed_overlap_initialization import (
    _publish_copy, prepare_arm_initial_checkpoints, prepare_suite_initial_checkpoints,
)
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_matching import (
    ArmEvidence, _rngs_equal, save_suite_initialization, save_suite_run_metadata, verify_arm_initial_checkpoint,
    verify_saved_suite_initialization, verify_suite_initialization, verify_suite_run_metadata,
)
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer, fresh_e1_model
from scripts.streamed_overlap_sampler import StreamedPairSampler
from scripts.train import EarlyStoppingState
from torch.utils.tensorboard import SummaryWriter


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
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        if torch.backends.mps.is_available():
            self.addCleanup(torch.mps.set_rng_state, torch.mps.get_rng_state())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.source = root / "source"
        self.source.mkdir()
        self.suite_dir = root / "suite"
        self.suite_dir.mkdir()
        self.run_dir = self.suite_dir / "arm"
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

    def prepare_suite(self, device="cpu"):
        evidence = {}
        self.model = self.model.to(device)
        self.optimizer = build_optimizer(self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        result = self.prepare()
        evidence["S3"] = ArmEvidence(result["initial_checkpoint"], result["checkpoint_path"], self.journal)
        for arm in ("S1", "S2"):
            directory = self.run_dir.parent / arm
            directory.mkdir()
            journal = BatchJournal(directory / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id=arm, expected_step=0, expected_exposure=0)
            model = fresh_e1_model(arm, model_seed=2, head_seed=3).to(device)
            optimizer = build_optimizer(model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=6, min_lr=1e-5,
            )
            result = self.prepare(model=model, optimizer=optimizer, scheduler=scheduler, journal=journal)
            evidence[arm] = ArmEvidence(result["initial_checkpoint"], result["checkpoint_path"], journal)
        return evidence

    def empty_suite_journals(self):
        journals = {"S3": self.journal}
        for arm in ("S1", "S2"):
            directory = self.suite_dir / arm
            directory.mkdir()
            journal = BatchJournal(directory / "batch_metrics.csv", STEP_FIELDS, create=True)
            self.addCleanup(journal.close)
            journal.reconcile(watermark=None, attempt_id=arm, expected_step=0, expected_exposure=0)
            journals[arm] = journal
        return journals

    def test_suite_bootstrap_recovers_partial_arm_and_pins_evidence(self):
        journals = self.empty_suite_journals()
        def bootstrap():
            return prepare_suite_initial_checkpoints(
                self.suite_dir, journals, sampler=self.sampler, device="cpu",
                hashes=self.hashes, matching_config=self.config, source_root=self.source,
            )

        def interrupt(source, destination, expected):
            if destination == journals["S2"].path.parent / "last.pt":
                raise OSError("interrupted S2")
            return _publish_copy(source, destination, expected)

        with patch("scripts.streamed_overlap_initialization._publish_copy", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interrupted S2"):
                bootstrap()
        s1 = journals["S1"].path.parent / "initial.pt"
        s2 = journals["S2"].path.parent / "initial.pt"
        original_s1, original_s2 = s1.read_bytes(), s2.read_bytes()
        self.assertFalse((self.suite_dir / "initialization_evidence.json").exists())
        self.assertFalse((self.suite_dir / "suite_metadata.json").exists())
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer must not step")), \
             patch("scripts.streamed_overlap_initialization.save_suite_run_metadata",
                   side_effect=OSError("interrupted metadata")):
            with self.assertRaisesRegex(OSError, "interrupted metadata"):
                bootstrap()
        evidence = (self.suite_dir / "initialization_evidence.json").read_bytes()
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer must not step")):
            result = bootstrap()
        self.assertEqual(result["evidence_path"].read_bytes(), evidence)
        self.assertEqual(s1.read_bytes(), original_s1)
        self.assertEqual(s2.read_bytes(), original_s2)
        self.assertEqual(bootstrap(), result)
        metadata = verify_suite_run_metadata(
            self.suite_dir, result["initial_checkpoints"], matching_config=self.config, expected_hashes=self.hashes,
        )
        self.assertEqual(metadata["initialization_sha256"], result["evidence_sha256"])
        self.assertEqual(metadata["sha256"], result["metadata_sha256"])
        for arm, initial in result["initial_checkpoints"].items():
            self.assertEqual(initial.read_bytes(), initial.with_name("last.pt").read_bytes(), arm)
            self.assertEqual(journals[arm].rows, [])

    def test_suite_bootstrap_rejects_unlocked_or_advanced_suite_before_writes(self):
        journals = self.empty_suite_journals()
        def bootstrap():
            return prepare_suite_initial_checkpoints(
                self.suite_dir, journals, sampler=self.sampler, device="cpu",
                hashes=self.hashes, matching_config=self.config, source_root=self.source,
            )

        journals["S3"].close()
        with self.assertRaisesRegex(ValueError, "empty, reconciled, locked"):
            bootstrap()
        self.assertFalse((journals["S1"].path.parent / "initial.pt").exists())
        journal = BatchJournal(self.run_dir / "batch_metrics.csv", STEP_FIELDS, create=False)
        self.addCleanup(journal.close)
        journal.reconcile(watermark=None, attempt_id="reopened", expected_step=0, expected_exposure=0)
        journals["S3"] = journal
        result = bootstrap()
        initial = result["initial_checkpoints"]["S3"]
        last = initial.with_name("last.pt")
        state = torch.load(last, weights_only=True)
        state["completed_epoch"] = 1
        torch.save(state, last)
        before = result["metadata_path"].read_bytes()
        with self.assertRaisesRegex(ValueError, "advanced suite must use resume"):
            bootstrap()
        self.assertEqual(result["metadata_path"].read_bytes(), before)
        last.unlink()
        with self.assertRaisesRegex(ValueError, "requires all retained"):
            bootstrap()
        self.assertFalse(last.exists())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_suite_bootstrap_prepares_mps_without_optimizer_step(self):
        journals = self.empty_suite_journals()
        before = torch.mps.get_rng_state()
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer must not step")):
            result = prepare_suite_initial_checkpoints(
                self.suite_dir, journals, sampler=self.sampler, device="mps",
                hashes=self.hashes, matching_config=self.config, source_root=self.source,
            )
        self.assertTrue(torch.equal(before, torch.mps.get_rng_state()))
        evidence = verify_saved_suite_initialization(
            result["evidence_path"], result["initial_checkpoints"], matching_config=self.config,
            expected_hashes=self.hashes, expected_evidence_sha256=result["evidence_sha256"],
        )
        self.assertEqual(evidence["initialization"]["device"], "mps")

    def test_suite_preflight_verifies_all_fresh_arms_without_mutating_rng(self):
        evidence = self.prepare_suite()
        before = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        report = verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)
        after = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        self.assertTrue(_rngs_equal(before, after, "cpu"))
        self.assertEqual(report["status"], "MATCHED_INITIALIZATION_ONLY")
        self.assertEqual(set(report["initial_checkpoint_sha256"]), {"S1", "S2", "S3"})
        self.assertEqual(report["sha256"], content_hash({
            key: value for key, value in report.items() if key != "sha256"
        }))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_suite_preflight_verifies_mps_rng_without_mutating_it(self):
        evidence = self.prepare_suite("mps")
        before = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        report = verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)
        after = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        self.assertEqual(report["device"], "mps")
        self.assertTrue(_rngs_equal(before, after, "mps"))

    def test_suite_preflight_rejects_changed_rng_and_optimizer(self):
        evidence = self.prepare_suite()
        candidate = evidence["S2"]
        original = candidate.initial_checkpoint.read_bytes()
        for key in ("rng", "optimizer"):
            with self.subTest(key=key):
                state = torch.load(candidate.initial_checkpoint, weights_only=True)
                if key == "rng":
                    state["training"]["torch_rng"][0] ^= 1
                else:
                    state["training"]["optimizer"]["param_groups"][0]["lr"] = 0.0
                torch.save(state, candidate.initial_checkpoint)
                candidate.final_checkpoint.write_bytes(candidate.initial_checkpoint.read_bytes())
                with self.assertRaisesRegex(ValueError, "optimizer, scheduler, stopping or seeded RNG"):
                    verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)
                candidate.initial_checkpoint.write_bytes(original)
                candidate.final_checkpoint.write_bytes(original)

    def test_suite_preflight_rejects_missing_or_advanced_arm_and_unlocked_journal(self):
        evidence = self.prepare_suite()
        with self.assertRaisesRegex(ValueError, "requires S1, S2 and S3"):
            verify_suite_initialization({"S1": evidence["S1"]}, matching_config=self.config,
                                        expected_hashes=self.hashes)
        with self.assertRaisesRegex(ValueError, "own regular journal"):
            verify_suite_initialization({**evidence, "S2": evidence["S1"]},
                                        matching_config=self.config, expected_hashes=self.hashes)
        last = evidence["S2"].final_checkpoint
        original = last.read_bytes()
        state = torch.load(last, weights_only=True)
        state["completed_epoch"] = 1
        torch.save(state, last)
        with self.assertRaisesRegex(ValueError, "identical epoch-zero"):
            verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)
        last.write_bytes(original)
        evidence["S2"].journal.close()
        with self.assertRaisesRegex(ValueError, "empty, reconciled, locked"):
            verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)

    def test_suite_preflight_rejects_symlinked_recovery_before_reading(self):
        evidence = self.prepare_suite()
        journal = evidence["S1"].journal
        recovery = journal.recovery_dir
        recovery.rename(recovery.with_name("original-recovery"))
        recovery.symlink_to(self.run_dir.parent / "fake_data_validation_v2", target_is_directory=True)
        with patch.object(journal, "_read", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "regular run-owned directory"):
                verify_suite_initialization(evidence, matching_config=self.config, expected_hashes=self.hashes)

    def test_persisted_suite_evidence_is_reused_and_verified_after_advancing(self):
        arms = self.prepare_suite()
        result = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        original = result["path"].read_bytes()
        self.assertEqual(save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        ), result)
        self.assertEqual(result["path"].read_bytes(), original)
        state = torch.load(arms["S1"].final_checkpoint, weights_only=True)
        state["completed_epoch"] = 1
        torch.save(state, arms["S1"].final_checkpoint)
        arms["S1"].journal.close()
        record = verify_saved_suite_initialization(
            result["path"], {arm: item.initial_checkpoint for arm, item in arms.items()},
            matching_config=self.config, expected_hashes=self.hashes,
            expected_evidence_sha256=result["evidence_sha256"],
        )
        self.assertEqual(record["matching_config"], self.config)
        self.assertEqual(record["initialization"]["status"], "MATCHED_INITIALIZATION_ONLY")

    def test_suite_metadata_pins_initialization_and_reuses_after_advancing(self):
        arms = self.prepare_suite()
        evidence = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        checkpoints = {arm: item.initial_checkpoint for arm, item in arms.items()}

        def save(pin=evidence["evidence_sha256"]):
            return save_suite_run_metadata(
                self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
                initialization_sha256=pin,
            )

        with self.assertRaisesRegex(ValueError, "pinned configuration"):
            save("0" * 64)
        self.assertFalse((self.suite_dir / "suite_metadata.json").exists())
        result = save()
        original = result["path"].read_bytes()
        self.assertEqual(save(), result)
        self.assertEqual(result["path"].read_bytes(), original)
        state = torch.load(arms["S1"].final_checkpoint, weights_only=True)
        state["completed_epoch"] = 1
        torch.save(state, arms["S1"].final_checkpoint)
        record = verify_suite_run_metadata(
            self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
        )
        self.assertEqual(record["initialization_sha256"], evidence["evidence_sha256"])
        self.assertEqual(record["sha256"], result["metadata_sha256"])
        with self.assertRaisesRegex(ValueError, "requested configuration"):
            verify_suite_run_metadata(
                self.suite_dir, checkpoints, matching_config={**self.config, "experiment_seed": 7},
                expected_hashes=self.hashes,
            )
        initial = checkpoints["S2"]
        initial_bytes = initial.read_bytes()
        state = torch.load(initial, weights_only=True)
        state["training"]["torch_rng"][0] ^= 1
        torch.save(state, initial)
        with self.assertRaisesRegex(ValueError, "does not match saved suite evidence"):
            verify_suite_run_metadata(
                self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
            )
        initial.write_bytes(initial_bytes)
        changed = json.loads(original)
        changed["initialization_sha256"] = "0" * 64
        changed["sha256"] = content_hash({key: value for key, value in changed.items() if key != "sha256"})
        result["path"].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "pinned configuration"):
            verify_suite_run_metadata(
                self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
            )

    def test_partial_suite_metadata_is_not_overwritten(self):
        arms = self.prepare_suite()
        evidence = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        path = self.suite_dir / "suite_metadata.json"
        path.write_bytes(b"partial")
        with self.assertRaises(json.JSONDecodeError):
            save_suite_run_metadata(
                self.suite_dir, {arm: item.initial_checkpoint for arm, item in arms.items()},
                matching_config=self.config, expected_hashes=self.hashes,
                initialization_sha256=evidence["evidence_sha256"],
            )
        self.assertEqual(path.read_bytes(), b"partial")

    def test_suite_metadata_rejects_symlinked_suite_before_reading_evidence(self):
        link = self.suite_dir.parent / "suite-link"
        link.symlink_to(self.suite_dir, target_is_directory=True)
        with patch("scripts.streamed_overlap_matching.verify_saved_suite_initialization") as verify:
            with self.assertRaisesRegex(ValueError, "regular directory"):
                save_suite_run_metadata(
                    link, {}, matching_config=self.config, expected_hashes=self.hashes,
                    initialization_sha256="0" * 64,
                )
            verify.assert_not_called()
        self.assertFalse((self.suite_dir / "suite_metadata.json").exists())

    def test_arm_runner_rechecks_pinned_suite_before_restore(self):
        formal = {"final_epoch": self.config["final_epoch"]}
        guardrail = {}
        self.hashes.update(
            formal_evaluation=content_hash(formal), reconstruction_guardrail=content_hash(guardrail),
            reconstruction_reference="f" * 64,
        )
        arms = self.prepare_suite()
        saved = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        suite = SuiteInitialization(
            saved["path"], saved["evidence_sha256"],
            {arm: item.initial_checkpoint for arm, item in arms.items()},
        )
        s1 = SelectedReconstruction(
            arms["S1"].final_checkpoint, arms["S1"].journal, self.suite_dir / "manifest",
            self.source, {}, {}, formal, self.suite_dir, "reconstruction",
        )
        arguments = {
            "run_id": self.config["run_id"], "seed": self.config["experiment_seed"], "attempt_id": "resume",
            "batches_per_epoch": self.config["batches_per_epoch"], "sparse_probability": 0.5,
            "monitor_batches": {}, "monitor_seed": self.config["monitor_seed"],
            "validation_seed": self.config["validation_seed"], "validate_epoch": object(),
            "epoch_csv": self.run_dir / "epochs.csv", "checkpoint_path": arms["S3"].final_checkpoint,
            "hashes": self.hashes, "early_stopping": self.stopping, "manifest_path": self.suite_dir / "manifest",
            "source_root": self.source, "search_manifest": {}, "diagnostic_manifest": {},
            "formal_config": formal, "evaluation_dir": self.run_dir, "guardrail_manifest": {},
            "guardrail_config": guardrail, "historical": object(), "s1": s1,
            "matching_config": self.config, "initial_checkpoint": arms["S3"].initial_checkpoint,
            "suite_initialization": suite,
        }

        def run(**overrides):
            return run_arm_epochs(
                self.model, self.optimizer, self.scheduler, self.sampler, self.journal, object(), object(),
                **{**arguments, **overrides},
            )

        with patch("scripts.streamed_overlap_epoch_runner._validate_formal_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_reconstruction_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_epoch_inputs"), \
             patch("scripts.streamed_overlap_epoch_runner._reference_hash", return_value="f" * 64), \
             patch("scripts.streamed_overlap_epoch_runner.restore_epoch_state",
                   side_effect=RuntimeError("restore reached")) as restore, \
             patch("scripts.streamed_overlap_epoch_runner.train_streamed_epoch") as train:
            with self.assertRaisesRegex(ValueError, "retained suite initialization checkpoint"):
                run(suite_initialization=replace(suite, evidence_path=self.suite_dir / "missing.json"))
            with self.assertRaisesRegex(ValueError, "regular file"):
                run()
            metadata = save_suite_run_metadata(
                self.suite_dir, suite.initial_checkpoints, matching_config=self.config, expected_hashes=self.hashes,
                initialization_sha256=suite.evidence_sha256,
            )
            with self.assertRaisesRegex(ValueError, "pin recorded in suite metadata"):
                run(suite_initialization=replace(suite, evidence_sha256="0" * 64))
            original_metadata = metadata["path"].read_bytes()
            changed_metadata = json.loads(original_metadata)
            changed_metadata["matching_config"]["experiment_seed"] += 1
            changed_metadata["sha256"] = content_hash({
                key: value for key, value in changed_metadata.items() if key != "sha256"
            })
            metadata["path"].write_text(json.dumps(changed_metadata))
            with self.assertRaisesRegex(ValueError, "Suite metadata is invalid"):
                run()
            metadata["path"].write_bytes(original_metadata)
            with self.assertRaisesRegex(ValueError, "retained suite initialization checkpoint"):
                run(initial_checkpoint=arms["S1"].initial_checkpoint)
            with self.assertRaisesRegex(ValueError, "retained suite initialization checkpoint"):
                run(checkpoint_path=arms["S1"].final_checkpoint)
            with patch.object(self.journal, "path", arms["S1"].journal.path):
                with self.assertRaisesRegex(ValueError, "retained suite initialization checkpoint"):
                    run()
            with self.assertRaisesRegex(ValueError, "S1 reconstruction reference must belong"):
                run(s1=replace(s1, checkpoint_path=arms["S3"].final_checkpoint))
            initial = arms["S2"].initial_checkpoint
            original = initial.read_bytes()
            state = torch.load(initial, weights_only=True)
            state["training"]["torch_rng"][0] ^= 1
            torch.save(state, initial)
            with self.assertRaisesRegex(ValueError, "does not match saved suite evidence"):
                run()
            initial.write_bytes(original)
            restore.assert_not_called()
            train.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "restore reached"):
                run()
            restore.assert_called_once()
            train.assert_not_called()
            restore.reset_mock()
            last = arms["S3"].final_checkpoint
            state = torch.load(last, weights_only=True)
            state["completed_epoch"] = 1
            torch.save(state, last)
            metadata["path"].unlink()
            with self.assertRaisesRegex(ValueError, "regular file"):
                run()
            restore.assert_not_called()
            metadata["path"].write_bytes(original_metadata)
            with self.assertRaisesRegex(RuntimeError, "restore reached"):
                run()
            restore.assert_called_once()
            train.assert_not_called()

    def test_suite_runs_arms_sequentially_and_stops_on_failure(self):
        arms = self.prepare_suite()
        evidence = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        checkpoints = {arm: item.initial_checkpoint for arm, item in arms.items()}
        suite = SuiteInitialization(evidence["path"], evidence["evidence_sha256"], checkpoints)
        metadata = save_suite_run_metadata(
            self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
            initialization_sha256=evidence["evidence_sha256"],
        )
        for item in arms.values():
            (item.initial_checkpoint.parent / "evaluation").mkdir()
        calls = []

        @contextmanager
        def runtime_factory(arm):
            calls.append(("open", arm))
            try:
                model = create_autospec(StreamedPairVAE, instance=True)
                model.arm = arm
                yield ArmRuntime(
                    model, create_autospec(torch.optim.Optimizer, instance=True),
                    create_autospec(torch.optim.lr_scheduler.ReduceLROnPlateau, instance=True),
                    create_autospec(StreamedPairSampler, instance=True),
                    create_autospec(EarlyStoppingState, instance=True), create_autospec(SummaryWriter, instance=True),
                )
            finally:
                calls.append(("close", arm))

        def arm_runner(model, *args, **kwargs):
            arm = model.arm
            calls.append(("run", arm))
            self.assertIs(args[3], arms[arm].journal)
            self.assertEqual(kwargs["checkpoint_path"], arms[arm].final_checkpoint)
            self.assertEqual(kwargs["initial_checkpoint"], arms[arm].initial_checkpoint)
            self.assertEqual(kwargs["attempt_id"], "attempt")
            if arm == "S1":
                self.assertIsNone(kwargs["s1"])
            else:
                self.assertEqual(kwargs["s1"].checkpoint_path, arms["S1"].final_checkpoint)
                self.assertIs(kwargs["s1"].journal, arms["S1"].journal)
                self.assertEqual(kwargs["s1"].selection_kind, "reconstruction")
            if arm == "S2" and fail_s2:
                raise RuntimeError("interrupted S2")
            return {"arm": arm}

        arguments = {
            "attempt_id": "attempt", "monitor_batches": {}, "validate_epoch": object(), "lpips_loss_fn": object(),
            "manifest_path": self.suite_dir / "manifest", "source_root": self.source,
            "search_manifest": {}, "diagnostic_manifest": {}, "formal_config": {}, "guardrail_manifest": {},
            "guardrail_config": {}, "historical": object(), "matching_config": self.config, "hashes": self.hashes,
            "suite_initialization": suite,
        }
        journals = {arm: item.journal for arm, item in arms.items()}
        with patch("scripts.streamed_overlap_epoch_runner._validate_formal_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_reconstruction_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_epoch_inputs") as epoch_gate, \
             patch("scripts.streamed_overlap_epoch_runner.run_arm_epochs", side_effect=arm_runner) as run_arm:
            original = metadata["path"].read_bytes()
            metadata["path"].unlink()
            with self.assertRaisesRegex(ValueError, "regular file"):
                run_suite_epochs(journals, runtime_factory, **arguments)
            run_arm.assert_not_called()
            self.assertEqual(calls, [])
            metadata["path"].write_bytes(original)
            epoch_gate.side_effect = ValueError("invalid epoch inputs")
            with self.assertRaisesRegex(ValueError, "invalid epoch inputs"):
                run_suite_epochs(journals, runtime_factory, **arguments)
            run_arm.assert_not_called()
            self.assertEqual(calls, [])
            epoch_gate.side_effect = None
            fail_s2 = True
            with self.assertRaisesRegex(RuntimeError, "interrupted S2"):
                run_suite_epochs(journals, runtime_factory, **arguments)
            self.assertEqual(calls, [("open", "S1"), ("run", "S1"), ("close", "S1"),
                                     ("open", "S2"), ("run", "S2"), ("close", "S2")])
            calls.clear()
            @contextmanager
            def suppressing_factory(arm):
                with runtime_factory(arm) as runtime:
                    try:
                        yield runtime
                    except RuntimeError:
                        pass

            with self.assertRaisesRegex(RuntimeError, "S2 runtime suppressed"):
                run_suite_epochs(journals, suppressing_factory, **arguments)
            self.assertNotIn(("open", "S3"), calls)
            calls.clear()
            fail_s2 = False
            result = run_suite_epochs(journals, runtime_factory, **arguments)
            self.assertEqual(result, {arm: {"arm": arm} for arm in ("S1", "S2", "S3")})
            self.assertEqual(calls, [(action, arm) for arm in ("S1", "S2", "S3")
                                     for action in ("open", "run", "close")])

    def test_suite_rejects_unlocked_journal_before_opening_runtime(self):
        arms = self.prepare_suite()
        evidence = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        checkpoints = {arm: item.initial_checkpoint for arm, item in arms.items()}
        save_suite_run_metadata(
            self.suite_dir, checkpoints, matching_config=self.config, expected_hashes=self.hashes,
            initialization_sha256=evidence["evidence_sha256"],
        )
        for item in arms.values():
            (item.initial_checkpoint.parent / "evaluation").mkdir()
        arms["S3"].journal.close()
        factory = Mock()
        with patch("scripts.streamed_overlap_epoch_runner._validate_formal_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_reconstruction_config"), \
             patch("scripts.streamed_overlap_epoch_runner._validate_epoch_inputs"), \
             patch("scripts.streamed_overlap_epoch_runner.run_arm_epochs") as run_arm:
            with self.assertRaisesRegex(ValueError, "locked journal"):
                run_suite_epochs(
                    {arm: item.journal for arm, item in arms.items()}, factory,
                    attempt_id="attempt", monitor_batches={}, validate_epoch=object(), lpips_loss_fn=object(),
                    manifest_path=self.suite_dir / "manifest", source_root=self.source,
                    search_manifest={}, diagnostic_manifest={}, formal_config={}, guardrail_manifest={},
                    guardrail_config={}, historical=HistoricalReconstruction(
                        self.suite_dir / "report", "a" * 64, self.suite_dir / "checkpoint", "b" * 64, "fixture",
                    ), matching_config=self.config, hashes=self.hashes,
                    suite_initialization=SuiteInitialization(evidence["path"], evidence["evidence_sha256"], checkpoints),
                )
            run_arm.assert_not_called()
            factory.assert_not_called()

    def test_persisted_suite_evidence_detects_tampering_and_wrong_pin(self):
        arms = self.prepare_suite()
        result = save_suite_initialization(
            self.suite_dir, arms, matching_config=self.config, expected_hashes=self.hashes, source_root=self.source,
        )
        paths = {arm: item.initial_checkpoint for arm, item in arms.items()}
        def verify(pin=result["evidence_sha256"]):
            return verify_saved_suite_initialization(
                result["path"], paths, matching_config=self.config, expected_hashes=self.hashes,
                expected_evidence_sha256=pin,
            )

        with self.assertRaisesRegex(ValueError, "pinned configuration"):
            verify("0" * 64)
        initial = paths["S2"]
        original = initial.read_bytes()
        state = torch.load(initial, weights_only=True)
        state["training"]["torch_rng"][0] ^= 1
        torch.save(state, initial)
        with self.assertRaisesRegex(ValueError, "does not match saved suite evidence"):
            verify()
        initial.write_bytes(original)
        record = json.loads(result["path"].read_text())
        record["matching_config"]["experiment_seed"] += 1
        result["path"].write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "pinned configuration"):
            verify()

    def test_existing_partial_evidence_is_not_overwritten(self):
        arms = self.prepare_suite()
        path = self.suite_dir / "initialization_evidence.json"
        path.write_bytes(b"partial")
        with self.assertRaises(json.JSONDecodeError):
            save_suite_initialization(
                self.suite_dir, arms, matching_config=self.config,
                expected_hashes=self.hashes, source_root=self.source,
            )
        self.assertEqual(path.read_bytes(), b"partial")

    def test_failed_evidence_publication_cleans_owned_file(self):
        arms = self.prepare_suite()
        path = self.suite_dir / "initialization_evidence.json"
        with patch("scripts.streamed_overlap_journal._sync_directory",
                   side_effect=[OSError("interrupted"), None]):
            with self.assertRaisesRegex(OSError, "interrupted"):
                save_suite_initialization(
                    self.suite_dir, arms, matching_config=self.config,
                    expected_hashes=self.hashes, source_root=self.source,
                )
        self.assertFalse(path.exists())

    def test_suite_evidence_rejects_source_directory_and_validation_v2(self):
        arms = self.prepare_suite()
        for source in (self.suite_dir, self.source / "fake_data_validation_v2"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                save_suite_initialization(
                    self.suite_dir, arms, matching_config=self.config,
                    expected_hashes=self.hashes, source_root=source,
                )
        self.assertFalse((self.suite_dir / "initialization_evidence.json").exists())

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

    def check_matched_initial_rng(self, device):
        self.model = self.model.to(device)
        self.optimizer = build_optimizer(self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        self.model.train()
        random.seed(100)
        np.random.seed(101)
        torch.default_generator.manual_seed(102)
        if device == "mps":
            torch.mps.manual_seed(103)
        before = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        initial = self.prepare()["initial_checkpoint"]
        after = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        self.assertTrue(_rngs_equal(before, after, device))
        self.assertTrue(self.model.training)

        control_dir = self.run_dir.parent / "control"
        control_dir.mkdir()
        journal = BatchJournal(control_dir / "batch_metrics.csv", STEP_FIELDS, create=True)
        self.addCleanup(journal.close)
        journal.reconcile(watermark=None, attempt_id="control", expected_step=0, expected_exposure=0)
        model = fresh_e1_model("S1", model_seed=2, head_seed=3).to(device)
        optimizer = build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        random.seed(200)
        np.random.seed(201)
        torch.default_generator.manual_seed(202)
        if device == "mps":
            torch.mps.manual_seed(203)
        before_control = capture_training_state(optimizer, scheduler, None, self.stopping)
        control = self.prepare(
            model=model, optimizer=optimizer, scheduler=scheduler, journal=journal,
        )["initial_checkpoint"]
        after_control = capture_training_state(optimizer, scheduler, None, self.stopping)
        self.assertTrue(_rngs_equal(before_control, after_control, device))
        left = torch.load(initial, map_location="cpu", weights_only=True)
        right = torch.load(control, map_location="cpu", weights_only=True)
        self.assertTrue(_rngs_equal(left["training"], right["training"], device))

    def test_initial_rng_is_matched_and_caller_rng_is_preserved(self):
        self.check_matched_initial_rng("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_initial_mps_rng_is_matched_and_caller_rng_is_preserved(self):
        self.check_matched_initial_rng("mps")

    def test_rng_is_restored_when_checkpoint_save_fails(self):
        before = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        with patch("scripts.streamed_overlap_initialization.save_epoch_state", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.prepare()
        after = capture_training_state(self.optimizer, self.scheduler, None, self.stopping)
        self.assertTrue(_rngs_equal(before, after, "cpu"))
        self.assertTrue(self.model.training)

    def test_retained_initial_with_wrong_rng_is_rejected(self):
        result = self.prepare()
        initial, last = result["initial_checkpoint"], result["checkpoint_path"]
        state = torch.load(initial, weights_only=True)
        state["training"]["torch_rng"][0] ^= 1
        torch.save(state, initial)
        last.write_bytes(initial.read_bytes())
        before = initial.read_bytes()
        with self.assertRaisesRegex(ValueError, "RNG state does not match"):
            self.prepare()
        self.assertEqual(initial.read_bytes(), before)
        self.assertEqual(last.read_bytes(), before)

    def test_failed_copy_removes_only_the_new_partial_file(self):
        source = self.run_dir / "fixture-source"
        source.write_bytes(b"source")
        destination = self.run_dir / "fixture-destination"
        with patch("scripts.streamed_overlap_initialization._digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "changed during publication"):
                _publish_copy(source, destination, "1" * 64)
        self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), b"source")

    def test_failed_copy_handles_identity_changing_after_write(self):
        source = self.run_dir / "fixture-source"
        destination = self.run_dir / "fixture-destination"
        source.write_bytes(b"source")
        real_fstat = os.fstat

        def changing_identity(fd):
            stat = real_fstat(fd)
            if os.lseek(fd, 0, os.SEEK_CUR) == 0:
                values = list(stat)
                values[1] = stat.st_ino ^ 1
                return os.stat_result(values)
            return stat

        with patch("scripts.streamed_overlap_journal.os.fstat", side_effect=changing_identity):
            with self.assertRaisesRegex(ValueError, "changed during publication"):
                _publish_copy(source, destination, "1" * 64)
        self.assertFalse(destination.exists())

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

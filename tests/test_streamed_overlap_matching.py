import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _atomic_record
from scripts.streamed_overlap_matching import ArmEvidence, verify_matched_arm_records
from scripts.streamed_overlap_matching import _recorded_stream
from scripts.streamed_overlap_model import fresh_e1_model


HASHES = {
    "config": "a" * 64, "source": "b" * 64, "target": "c" * 64, "validation_manifest": "d" * 64,
}


def fixture_model(arm, model_seed, head_seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        vae = torch.nn.Linear(2, 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed)
        model = torch.nn.Module()
        model.add_module("vae", vae)
        if arm in {"S2", "S3"}:
            model.add_module("overlap_head", torch.nn.Linear(2, 1))
        if arm == "S3":
            model.add_module("coverage_head", torch.nn.Linear(2, 1))
    return model


def sampler_state(seed):
    streams = np.random.SeedSequence(seed).spawn(2)
    return {
        "schema": "streamed-pair-sampler-v1", "split": "train", "source_manifest_hash": HASHES["source"],
        "batches_drawn": 0, "pair_rng": np.random.default_rng(streams[0]).bit_generator.state,
        "augmentation_rng": np.random.default_rng(streams[1]).bit_generator.state,
    }


class MatchedArmEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.factory = patch("scripts.streamed_overlap_matching.fresh_e1_model", side_effect=fixture_model)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.reference = self.arm("S1")
        self.candidate = self.arm("S3", replay=True)

    def arm(self, arm, replay=False):
        root = self.root / arm
        root.mkdir()
        (root / "provenance").mkdir()
        journal = BatchJournal(root / "batch_metrics.csv", STEP_FIELDS, create=True)
        self.addCleanup(journal.close)
        journal.reconcile(watermark=None, attempt_id="first", expected_step=0, expected_exposure=0)
        training = {
            "optimizer": {"state": {}}, "torch_rng": torch.tensor([1, 2], dtype=torch.uint8),
            "python_rng": (3, (1, 2), None), "numpy_rng": ["MT19937", [1, 2], 0, 0, 0.0],
        }
        state = {
            "schema": "streamed-epoch-state-v1", "arm": arm, "device": "cpu",
            "completed_epoch": 0, "batches_per_epoch": 2, "hashes": HASHES,
            "model": fixture_model(arm, 10, 11).state_dict(), "training": training,
            "sampler": sampler_state(12), "selection_state": {},
            "journal": {"event_id": None, "logical_step": 0, "cumulative_exposure": 0},
        }
        initial = root / "initial.pt"
        torch.save(state, initial)

        def append(step, event, marker):
            window = {"source_id": "train/a", "origin": [0, 0, 0], "shape": [32, 32, 64]}
            categories = ["same_overlap"] * 6 + ["same_disjoint"] * 3 + ["cross_source"] * 3
            view = {
                "input_mode": "extrema", "noise": None, "retained_fraction": 0.3,
                "transform": {"angle_degrees": marker, "scale": [1.0, 1.0, 1.0],
                              "flip_x": False, "flip_y": False, "swap_xy": False},
            }
            record = {
                "schema": "streamed-step-provenance-v1", "event_id": event,
                "pairs": [
                    {"category": category, "a": window, "b": window,
                     "augmentation_seed_a": index + step * 20, "augmentation_seed_b": index + step * 20 + 1}
                    for index, category in enumerate(categories)
                ],
                "augmentation": [[view, view] for _ in range(12)],
            }
            path = _atomic_record(root / "provenance", record)
            row = {field: "" for field in STEP_FIELDS}
            row.update({
                "sequence": len(journal.rows) + 1, "event_id": event, "attempt_id": journal.attempt_id,
                "logical_step": step, "completed_pairs": 12, "cumulative_exposure": step * 12,
                "replay_of_event": journal.position.replay_events.get(step, ""),
                "run_id": "suite", "arm": arm, "seed": 7, "epoch": 1, "batch_index": step,
                "reconstructed_a": 12, "provenance": str(path.relative_to(root)),
            })
            journal.append(row)

        if replay:
            append(1, f"{arm}-stale", 90.0)
            journal.reconcile(watermark=None, attempt_id="replay", expected_step=0, expected_exposure=0)
        append(1, f"{arm}-1", 0.0)
        append(2, f"{arm}-2", 0.0)
        state = copy.deepcopy(state)
        state["completed_epoch"] = 1
        state["sampler"]["batches_drawn"] = 2
        state["journal"] = {"event_id": f"{arm}-2", "logical_step": 2, "cumulative_exposure": 24}
        final = root / "last.pt"
        torch.save(state, final)
        return ArmEvidence(initial, final, journal)

    def verify(self):
        return verify_matched_arm_records(
            self.reference, self.candidate, model_seed=10, head_seed=11, sampler_seed=12,
        )

    def mutate_checkpoint(self, path, mutate):
        state = torch.load(path, weights_only=True)
        mutate(state)
        torch.save(state, path)

    def test_compares_fresh_initialization_and_live_records_after_replay(self):
        result = self.verify()
        self.assertEqual(result["status"], "MATCHED_RECORDED_STREAMS_ONLY")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["reference"]["arm"], "S1")
        self.assertEqual(result["candidate"]["arm"], "S3")
        self.assertEqual(result["sha256"], content_hash({
            key: value for key, value in result.items() if key != "sha256"
        }))

    def test_verifies_real_fresh_e1_checkpoint_tensors(self):
        for item, arm in ((self.reference, "S1"), (self.candidate, "S3")):
            model_state = fresh_e1_model(arm, 10, 11).state_dict()
            for path in (item.initial_checkpoint, item.final_checkpoint):
                self.mutate_checkpoint(path, lambda state: state.update(model=model_state))
        with patch("scripts.streamed_overlap_matching.fresh_e1_model", side_effect=fresh_e1_model):
            self.assertEqual(self.verify()["status"], "MATCHED_RECORDED_STREAMS_ONLY")

    def test_s2_is_checked_against_s1(self):
        self.candidate = self.arm("S2")
        self.assertEqual(self.verify()["candidate"]["arm"], "S2")

    def test_changed_fresh_weights_or_sampler_seed_are_rejected(self):
        self.mutate_checkpoint(
            self.candidate.initial_checkpoint, lambda state: state["model"]["vae.weight"].add_(1),
        )
        with self.assertRaisesRegex(ValueError, "fresh E1 initialization"):
            self.verify()
        self.mutate_checkpoint(
            self.candidate.initial_checkpoint, lambda state: state["model"]["vae.weight"].sub_(1),
        )
        with self.assertRaisesRegex(ValueError, "sampler.*specified stream seed"):
            verify_matched_arm_records(
                self.reference, self.candidate, model_seed=10, head_seed=11, sampler_seed=13,
            )

    def test_mismatched_live_augmentation_or_final_rng_is_rejected(self):
        row = self.candidate.journal.rows[-1]
        path = self.candidate.journal.path.parent / row["provenance"]
        record = json.loads(path.read_text())
        record["augmentation"][0][0]["transform"]["angle_degrees"] = 1.0
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "different committed pair or augmentation"):
            self.verify()
        record["augmentation"][0][0]["transform"]["angle_degrees"] = 0.0
        path.write_text(json.dumps(record))
        self.mutate_checkpoint(
            self.candidate.final_checkpoint, lambda state: state["training"]["torch_rng"].add_(1),
        )
        with self.assertRaisesRegex(ValueError, "different configuration, exposure or RNG"):
            self.verify()

    def test_missing_or_unbound_provenance_is_rejected(self):
        row = self.candidate.journal.rows[-1]
        path = self.candidate.journal.path.parent / row["provenance"]
        record = json.loads(path.read_text())
        record["event_id"] = "unrelated"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance record is invalid"):
            self.verify()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.verify()

    def test_invalid_or_holdout_pair_records_are_rejected(self):
        path = self.candidate.journal.path.parent / self.candidate.journal.rows[-1]["provenance"]
        record = json.loads(path.read_text())
        record["pairs"][0]["a"]["source_id"] = "train/test/crop"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance record is invalid"):
            self.verify()

    def test_matched_noise_metadata_is_validated(self):
        paths = [item.journal.path.parent / item.journal.rows[-1]["provenance"]
                 for item in (self.reference, self.candidate)]
        for path in paths:
            record = json.loads(path.read_text())
            record["augmentation"][0][0]["noise"] = {
                "snr_db": 12.0, "pre_noise_rms": 1.0, "added_noise_rms": 0.25, "filter_irlen": 3,
            }
            path.write_text(json.dumps(record))
        self.assertEqual(self.verify()["status"], "MATCHED_RECORDED_STREAMS_ONLY")
        for path in paths:
            record = json.loads(path.read_text())
            record["augmentation"][0][0]["noise"]["snr_db"] = 11.0
            path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance record is invalid"):
            self.verify()

    def test_validation_v2_is_rejected_before_checkpoint_loading(self):
        forbidden = replace(
            self.reference, initial_checkpoint=self.root / "fake_data_validation_v2" / "initial.pt",
        )
        with patch("scripts.streamed_overlap_matching._checkpoint_state") as load:
            with self.assertRaisesRegex(ValueError, "outside validation-v2"):
                verify_matched_arm_records(
                    forbidden, self.candidate, model_seed=10, head_seed=11, sampler_seed=12,
                )
            load.assert_not_called()

    def test_recovery_records_cannot_change_during_verification(self):
        def mutate(item, identity, batches):
            result = _recorded_stream(item, identity, batches)
            if item is self.reference:
                _atomic_record(item.journal.recovery_dir, {
                    "schema": "streamed-journal-recovery-v1", "uncommitted_event_ids": [],
                })
            return result

        with patch("scripts.streamed_overlap_matching._recorded_stream", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "recovery records changed"):
                self.verify()

    def test_incomplete_or_unmatched_trajectories_are_rejected(self):
        self.mutate_checkpoint(
            self.candidate.final_checkpoint, lambda state: state["sampler"].update(batches_drawn=1),
        )
        with self.assertRaisesRegex(ValueError, "sampler identity"):
            self.verify()

    def test_config_and_initial_optimizer_state_must_match(self):
        self.mutate_checkpoint(
            self.candidate.initial_checkpoint,
            lambda state: state["training"]["optimizer"]["state"].update({0: {"step": 1}}),
        )
        with self.assertRaisesRegex(ValueError, "initial optimizer must have no updates"):
            self.verify()
        self.mutate_checkpoint(
            self.candidate.initial_checkpoint, lambda state: state["training"]["optimizer"]["state"].clear(),
        )
        self.mutate_checkpoint(
            self.candidate.final_checkpoint, lambda state: state["hashes"].update(config="f" * 64),
        )
        with self.assertRaisesRegex(ValueError, "not bound to its final trajectory"):
            self.verify()


if __name__ == "__main__":
    unittest.main()

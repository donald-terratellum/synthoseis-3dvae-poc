import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_evaluation import evaluate_checkpoint
from scripts.streamed_overlap_integrity import input_fingerprints, verify_inputs
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer
from scripts.streamed_overlap_yield import SEISMIC_KEY
from src.model import VAE3D
from tests.test_streamed_overlap_metrics import frozen_fixture


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        values = inputs.flatten(1)
        mu = self.linear(torch.stack((values.mean(1), values.square().mean(1)), dim=1))
        return mu, torch.zeros_like(mu)


class ToyVAE(VAE3D):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.patch_shape = (2, 2, 5)
        self.latent_dim = 2
        self.encoder = Encoder()
        self.decoder = torch.nn.Linear(2, 2)


def toy_factory(arm, model_seed, head_seed):
    del model_seed, head_seed
    with torch.random.fork_rng():
        torch.manual_seed(10)
        return StreamedPairVAE(ToyVAE(), arm)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = frozen_fixture()
        ids = {query["window"]["source_id"] for query in self.manifest["queries"]} | {"validation/other"}
        self.groups = {}
        for index, source_id in enumerate(sorted(ids)):
            array = np.arange(20, dtype=np.float32).reshape(2, 2, 5) + index
            self.groups[source_id] = {
                SEISMIC_KEY: array,
                "geologic_age_faulted": array.copy(),
                "oil_closures": np.zeros_like(array),
                "gas_closures": np.zeros_like(array),
            }
        for query in self.manifest["queries"]:
            for window in (query["window"], *(candidate["window"] for candidate in query["candidates"])):
                window.update({"origin": [0, 0, 0], "shape": [2, 2, 5]})
        plans = [{
            "id": q["id"], "cohort": q["cohort"], "window": q["window"],
            "candidates": [{"category": c["category"], "window": c["window"]} for c in q["candidates"]],
        } for q in self.manifest["queries"]]
        self.manifest["input_integrity"]["fingerprints"] = input_fingerprints(plans, self.groups)
        self.manifest.pop("manifest_sha256")
        self.manifest["manifest_sha256"] = content_hash(self.manifest)
        self.journal = BatchJournal(self.root / "batch.csv", STEP_FIELDS, create=True)
        self.addCleanup(self.journal.close)
        self.journal.reconcile(watermark=None, attempt_id="attempt-1", expected_step=0, expected_exposure=0)
        self.model = toy_factory("S3", 0, 0)
        self.optimizer = build_optimizer(self.model)
        a = torch.arange(40, dtype=torch.float32).reshape(2, 1, 2, 2, 5) / 40
        b = torch.flip(a, dims=[-1])
        mu_a, _ = self.model.vae.encoder(a)
        mu_b, _ = self.model.vae.encoder(b)
        head = self.model.coverage_head
        assert head is not None
        loss = (head(mu_a, mu_b) - torch.tensor([0.2, 0.8])).square().mean()
        loss.backward()
        self.optimizer.step()
        row: dict[str, object] = dict.fromkeys(STEP_FIELDS, "")
        row.update({
            "sequence": 1, "event_id": "event-1", "attempt_id": "attempt-1",
            "logical_step": 1, "completed_pairs": 12, "cumulative_exposure": 12,
            "run_id": "run-1", "arm": "S3", "seed": 3, "epoch": 1,
            "batch_index": 1, "valid_coverage": 2, "coverage_loss": loss.item(),
        })
        self.journal.append(row)
        self.path = self.root / "checkpoint.pt"
        self.save()

    def save(self):
        torch.save({
            "schema": "streamed-epoch-state-v1", "arm": "S3", "completed_epoch": 1,
            "batches_per_epoch": 1, "model": self.model.state_dict(),
            "training": {"optimizer": self.optimizer.state_dict()},
            "sampler": {"split": "train", "batches_drawn": 1, "source_manifest_hash": "b" * 64},
            "hashes": {
                "validation_manifest": self.manifest["manifest_sha256"],
                "source": "b" * 64, "config": "c" * 64, "target": "d" * 64,
            },
            "journal": {"event_id": "event-1", "logical_step": 1, "cumulative_exposure": 12},
        }, self.path)

    def evaluate(self, *, pair_head=True):
        with patch("scripts.streamed_overlap_evaluation.load_manifest", return_value=self.manifest), \
             patch("scripts.streamed_overlap_evaluation.open_validation_sources", return_value=self.groups), \
             patch("scripts.streamed_overlap_evaluation.fresh_e1_model", side_effect=toy_factory):
            return evaluate_checkpoint(
                self.path, self.journal, self.root / "manifest.json", self.root,
                device="cpu", inference_seed=7, encode_batch_size=11, max_wall_seconds=60,
                random_seed=3, random_repeats=3, bootstrap_seed=4, bootstrap_repeats=10,
                score_pair_head=pair_head,
            )

    def test_scoring_verifies_inputs_and_records_checkpoint_and_scorers(self):
        rng = torch.get_rng_state().clone()
        result = self.evaluate()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(result["checkpoint"]["attempt_id"], "attempt-1")
        self.assertEqual(result["checkpoint"]["checkpoint_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["input_verification"]["status"], "VALIDATION_INPUTS_VERIFIED")
        self.assertEqual(set(result["scorers"]), {"mu_cosine", "pair_head"})
        self.assertEqual(len(result["scorers"]["pair_head"]["rows"]), 200)

    def test_untrained_head_rejected_before_source_verification(self):
        self.optimizer = build_optimizer(self.model)
        self.save()
        with self.assertRaisesRegex(ValueError, "not received"):
            self.evaluate()
        result = self.evaluate(pair_head=False)
        self.assertEqual(set(result["scorers"]), {"mu_cosine"})

    def test_zero_only_coverage_gradients_do_not_count_as_training(self):
        head = self.model.coverage_head
        assert head is not None
        for parameter in head.parameters():
            self.optimizer.state[parameter]["exp_avg_sq"].zero_()
        self.save()
        with self.assertRaisesRegex(ValueError, "no recorded nonzero gradients"):
            self.evaluate()

    def test_changed_validation_input_and_discarded_checkpoint_fail(self):
        self.groups["validation/source-0"][SEISMIC_KEY][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.evaluate()
        self.groups["validation/source-0"][SEISMIC_KEY][0, 0, 0] -= 1
        self.journal.reconcile(watermark=None, attempt_id="attempt-2", expected_step=0, expected_exposure=0)
        with self.assertRaisesRegex(ValueError, "discarded"):
            self.evaluate()

    def test_change_after_verification_is_detected_before_encoding(self):
        def verify_then_change(manifest, groups):
            verification = verify_inputs(manifest, groups)
            groups["validation/source-0"][SEISMIC_KEY][0, 0, 0] += 1
            return verification

        with patch("scripts.streamed_overlap_evaluation.verify_inputs", side_effect=verify_then_change):
            with self.assertRaisesRegex(ValueError, "changed after fingerprint"):
                self.evaluate()


if __name__ == "__main__":
    unittest.main()

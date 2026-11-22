import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_guardrails import compare_reconstruction_guardrails
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_model import StreamedPairVAE
from scripts.streamed_overlap_reconstruction import (
    freeze_reconstruction_manifest, measure_reconstruction,
)
from scripts.train import SliceLPIPSLoss, build_checkpoint_payload
from src.model import VAE3D


def fixture_manifest():
    manifest = {
        "schema": "streamed-reconstruction-guardrail-v1",
        "usage": "post_selection_guardrail",
        "protocol": {
            "input_mode": "tokenizer", "latent_mode": "sampled",
            "sampling_method": "normal_reparameterization", "seed": 12,
            "patch_shape": [32, 32, 64], "lpips_sha256": "a" * 64,
        },
        "datasets": {
            domain: [
                {"id": f"{domain}-{index}", "input_sha256": "b" * 64, "target_sha256": "c" * 64}
                for index in range(2)
            ]
            for domain in ("synthetic", "real")
        },
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    return manifest


def measurement(manifest, arm, checkpoint_char, scale):
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol": copy.deepcopy(manifest["protocol"]),
        "checkpoint": {
            "arm": arm, "checkpoint_sha256": checkpoint_char * 64,
            **({"checkpoint_label": "reconstruction-best"} if arm == "P0b" else {
                "run_id": "run", "attempt_id": "attempt", "step": 1, "experiment_seed": 2,
            }),
        },
        "datasets": {
            domain: [
                {**item, "mae": scale, **({"lpips": 0.0, "kl": 0.0} if domain == "synthetic" else {})}
                for item in items
            ]
            for domain, items in manifest["datasets"].items()
        },
    }


class GuardrailTests(unittest.TestCase):
    def test_exact_two_percent_boundary_and_independent_domain_failure(self):
        manifest = fixture_manifest()
        candidate = measurement(manifest, "S3", "d", 1.02)
        s1 = measurement(manifest, "S1", "e", 1.0)
        p0b = measurement(manifest, "P0b", "f", 1.0)
        result = compare_reconstruction_guardrails(manifest, candidate, {"S1": s1, "P0b": p0b})
        self.assertEqual(result["status"], "PASS_NUMERIC_ONLY")
        candidate["datasets"]["real"][0]["mae"] = 1.021
        result = compare_reconstruction_guardrails(manifest, candidate, {"S1": s1, "P0b": p0b})
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["comparisons"]["S1"]["synthetic_within_2_percent"])
        self.assertFalse(result["comparisons"]["S1"]["real_within_2_percent"])

    def test_recipe_includes_weighted_lpips_and_kl(self):
        manifest = fixture_manifest()
        candidate = measurement(manifest, "S2", "d", 1.0)
        control = measurement(manifest, "S1", "e", 1.0)
        for item in candidate["datasets"]["synthetic"]:
            item["lpips"] = 0.19
            item["kl"] = 2.0
        result = compare_reconstruction_guardrails(manifest, candidate, {"S1": control})
        self.assertAlmostEqual(result["candidate_synthetic_recipe_total"], 1.021)
        self.assertEqual(result["status"], "FAIL")

    def test_partial_or_mismatched_evidence_is_rejected(self):
        manifest = fixture_manifest()
        candidate = measurement(manifest, "S3", "d", 1.0)
        control = measurement(manifest, "S1", "e", 1.0)
        altered = copy.deepcopy(control)
        altered["datasets"]["real"].pop()
        with self.assertRaisesRegex(ValueError, "full real"):
            compare_reconstruction_guardrails(manifest, candidate, {"S1": altered})
        altered = copy.deepcopy(control)
        altered["datasets"]["synthetic"][0]["target_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "targets differ"):
            compare_reconstruction_guardrails(manifest, candidate, {"S1": altered})
        altered = copy.deepcopy(control)
        altered["protocol"]["seed"] = 13
        with self.assertRaisesRegex(ValueError, "protocol"):
            compare_reconstruction_guardrails(manifest, candidate, {"S1": altered})
        altered = copy.deepcopy(control)
        altered["checkpoint"]["checkpoint_sha256"] = candidate["checkpoint"]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_reconstruction_guardrails(manifest, candidate, {"S1": altered})
        altered = copy.deepcopy(control)
        altered["checkpoint"]["experiment_seed"] = 3
        with self.assertRaisesRegex(ValueError, "matched experiment seed"):
            compare_reconstruction_guardrails(manifest, candidate, {"S1": altered})


class FixtureLPIPS(SliceLPIPSLoss):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.min_spatial_size = 64
        self.eps = 1e-6

    def forward(self, recon, target):
        return (recon - target).square().mean() * self.weight[0]


class FixtureEncoder(torch.nn.Module):
    def forward(self, values):
        means = values.flatten(1).mean(1)
        mu = torch.stack((means, means * 0.5), dim=1)
        return mu, torch.zeros_like(mu)


class FixtureDecoder(torch.nn.Module):
    def forward(self, latent):
        return latent[:, :1, None, None, None].expand(-1, 1, 32, 32, 64), None


class FixtureVAE(VAE3D):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.patch_shape = (32, 32, 64)
        self.latent_dim = 2
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.encoder = FixtureEncoder()
        self.decoder = FixtureDecoder()


class ReconstructionProducerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stores = {}
        for domain in ("synthetic", "real"):
            path = self.root / f"{domain}.zarr"
            group = zarr.open_group(str(path), mode="w")
            group.attrs["scaling_mode"] = "divide_by_std"
            data = np.broadcast_to(np.linspace(-1, 1, 64, dtype=np.float32), (2, 32, 32, 64)).copy()
            data[1] *= 2
            group.create_array("patches", data=data, chunks=(1, 32, 32, 64))
            self.stores[domain] = path
        self.checkpoint = self.root / "p0b.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint")
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.lpips = FixtureLPIPS().eval()
        self.loss_patch = patch("scripts.streamed_overlap_reconstruction.SliceLPIPSLoss", FixtureLPIPS)
        self.loss_patch.start()
        self.addCleanup(self.loss_patch.stop)

    def freeze(self):
        return freeze_reconstruction_manifest(
            self.stores["synthetic"], self.stores["real"], seed=7, device="cpu",
            lpips_loss_fn=self.lpips, max_wall_seconds=60,
        )

    def measure(self, manifest):
        def initialize(adapter, checkpoint_path, device):
            del checkpoint_path, device
            adapter.model = FixtureVAE().eval()

        with patch("scripts.streamed_overlap_reconstruction.VaeLatentAdapter.__init__", initialize):
            return measure_reconstruction(
                manifest, self.checkpoint, arm="P0b", device="cpu", lpips_loss_fn=self.lpips,
                max_wall_seconds=60, historical_checkpoint_sha256=self.digest,
                historical_checkpoint_label="reconstruction-best",
            )

    def test_full_store_measurement_is_repeatable_and_numeric_gate_compatible(self):
        manifest = self.freeze()
        before = torch.get_rng_state().clone()
        first = self.measure(manifest)
        second = self.measure(manifest)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first["datasets"], second["datasets"])
        self.assertEqual(len(first["datasets"]["synthetic"]), 2)
        self.assertEqual(len(first["datasets"]["real"]), 2)
        self.assertEqual(first["checkpoint"]["checkpoint_sha256"], self.digest)
        self.assertTrue(all(record["mae"] >= 0 and record["lpips"] >= 0 and record["kl"] >= 0
                            for record in first["datasets"]["synthetic"]))
        candidate = copy.deepcopy(first)
        candidate["checkpoint"] = {
            "arm": "S3", "checkpoint_sha256": "f" * 64, "run_id": "fixture",
            "attempt_id": "fixture", "step": 1, "experiment_seed": 1,
        }
        self.assertEqual(compare_reconstruction_guardrails(manifest, candidate, {"P0b": first})["status"],
                         "PASS_NUMERIC_ONLY")

    def test_changed_store_lpips_and_checkpoint_fail_closed(self):
        manifest = self.freeze()
        altered = copy.deepcopy(manifest)
        altered["protocol"]["runtime"]["versions"]["torch"] = "different"
        altered["manifest_sha256"] = content_hash({
            key: value for key, value in altered.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            self.measure(altered)
        group = zarr.open_group(str(self.stores["real"]), mode="r+")
        patches = group["patches"]
        assert isinstance(patches, zarr.Array)
        patches[1, 0, 0, 0] = 123
        with self.assertRaisesRegex(ValueError, "differs from its frozen"):
            self.measure(manifest)
        patches[1, 0, 0, 0] = -2
        with torch.no_grad():
            self.lpips.weight[0] = 2
        with self.assertRaisesRegex(ValueError, "LPIPS state differs"):
            self.measure(manifest)
        with torch.no_grad():
            self.lpips.weight[0] = 1
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "explicit checkpoint hash"):
            self.measure(manifest)

    def test_streamed_checkpoint_is_bound_to_completed_journal(self):
        manifest = self.freeze()
        model = StreamedPairVAE(FixtureVAE(), "S1")
        checkpoint = self.root / "last.pt"
        torch.save({
            "schema": "streamed-epoch-state-v1", "arm": "S1", "completed_epoch": 1,
            "batches_per_epoch": 1, "model": model.state_dict(),
            "sampler": {"split": "train", "batches_drawn": 1, "source_manifest_hash": "b" * 64},
            "hashes": {
                "validation_manifest": "a" * 64, "source": "b" * 64,
                "config": "c" * 64, "target": "d" * 64,
            },
            "journal": {"event_id": "event-1", "logical_step": 1, "cumulative_exposure": 12},
        }, checkpoint)
        journal = BatchJournal(self.root / "batch.csv", STEP_FIELDS, create=True)
        self.addCleanup(journal.close)
        journal.reconcile(watermark=None, attempt_id="attempt-1", expected_step=0, expected_exposure=0)
        row: dict[str, object] = dict.fromkeys(STEP_FIELDS, "")
        row.update({
            "sequence": 1, "event_id": "event-1", "attempt_id": "attempt-1",
            "logical_step": 1, "completed_pairs": 12, "cumulative_exposure": 12,
            "run_id": "run-1", "arm": "S1", "seed": 3, "epoch": 1, "batch_index": 1,
        })
        journal.append(row)
        def factory(arm, model_seed, head_seed):
            del model_seed, head_seed
            return StreamedPairVAE(FixtureVAE(), arm)

        with patch("scripts.streamed_overlap_reconstruction.load_manifest",
                   return_value={"manifest_sha256": "a" * 64}), \
             patch("scripts.streamed_overlap_reconstruction.fresh_e1_model", side_effect=factory):
            report = measure_reconstruction(
                manifest, checkpoint, arm="S1", device="cpu", lpips_loss_fn=self.lpips,
                max_wall_seconds=60, journal=journal, validation_manifest_path=self.root / "validation.json",
            )
            self.assertEqual(report["checkpoint"]["attempt_id"], "attempt-1")
            self.assertEqual(report["checkpoint"]["journal_event_id"], "event-1")
            journal.reconcile(watermark=None, attempt_id="attempt-2", expected_step=0, expected_exposure=0)
            with self.assertRaisesRegex(ValueError, "discarded"):
                measure_reconstruction(
                    manifest, checkpoint, arm="S1", device="cpu", lpips_loss_fn=self.lpips,
                    max_wall_seconds=60, journal=journal, validation_manifest_path=self.root / "validation.json",
                )

    def test_synthetic_smoke_loads_a_real_tokenizer_adapter(self):
        manifest = self.freeze()
        model = VAE3D(base_ch=2, latent_dim=2, patch_shape=(32, 32, 64))
        torch.save(build_checkpoint_payload(model, epoch=1), self.checkpoint)
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        before = torch.get_rng_state().clone()
        result = measure_reconstruction(
            manifest, self.checkpoint, arm="P0b", device="cpu", lpips_loss_fn=self.lpips,
            max_wall_seconds=60, historical_checkpoint_sha256=digest,
            historical_checkpoint_label="synthetic-fixture-only",
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(result["datasets"]["synthetic"]), 2)
        self.assertEqual(len(result["datasets"]["real"]), 2)


if __name__ == "__main__":
    unittest.main()

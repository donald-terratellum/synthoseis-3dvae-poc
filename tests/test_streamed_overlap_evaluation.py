import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_evaluation import (
    StreamedEpochValidator, evaluate_checkpoint, export_tokenizer_checkpoint, physical_overlap_diagnostics,
)
from scripts.streamed_overlap_integrity import input_fingerprints, verify_inputs
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer
from scripts.streamed_overlap_selection import select_checkpoints
from scripts.streamed_overlap_yield import SEISMIC_KEY
from src.model import VAE3D
from scripts.train import compute_vae_losses, streamed_inference_context
from src.tokenizer.core.preprocess import normalize_cube_by_std, preprocess_for_token
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from tests.test_streamed_overlap_metrics import frozen_fixture


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        values = inputs.flatten(1)
        mu = self.linear(torch.stack((values.mean(1), values.square().mean(1)), dim=1))
        return mu, torch.zeros_like(mu)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 20)

    def forward(self, z):
        return self.linear(z).reshape(-1, 1, 2, 2, 5), None


class ToyVAE(VAE3D):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.patch_shape = (2, 2, 5)
        self.latent_dim = 2
        self.encoder = Encoder()
        self.decoder = Decoder()


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
            for candidate in query["candidates"]:
                candidate["target"]["overlap"] = 0.5 if candidate["category"] == "same_overlap" else 0.0
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
        overlap_head = self.model.overlap_head
        assert overlap_head is not None
        overlap_loss = (overlap_head(mu_a, mu_b) - torch.tensor([0.2, 0.8])).square().mean()
        loss = (head(mu_a, mu_b) - torch.tensor([0.2, 0.8])).square().mean() + overlap_loss
        loss.backward()
        self.optimizer.step()
        row: dict[str, object] = dict.fromkeys(STEP_FIELDS, "")
        row.update({
            "sequence": 1, "event_id": "event-1", "attempt_id": "attempt-1",
            "logical_step": 1, "completed_pairs": 12, "cumulative_exposure": 12,
            "run_id": "run-1", "arm": "S3", "seed": 3, "epoch": 1,
            "batch_index": 1, "valid_coverage": 2, "coverage_loss": loss.item(),
            "overlap_loss": overlap_loss.item(),
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

    def evaluate(self, *, pair_head=True, overlap=False):
        with patch("scripts.streamed_overlap_evaluation.load_manifest", return_value=self.manifest), \
             patch("scripts.streamed_overlap_evaluation.open_validation_sources", return_value=self.groups), \
             patch("scripts.streamed_overlap_evaluation.fresh_e1_model", side_effect=toy_factory):
            return evaluate_checkpoint(
                self.path, self.journal, self.root / "manifest.json", self.root,
                device="cpu", inference_seed=7, encode_batch_size=11, max_wall_seconds=60,
                random_seed=3, random_repeats=3, bootstrap_seed=4, bootstrap_repeats=10,
                score_pair_head=pair_head, score_overlap=overlap,
            )

    def test_overlap_scoring_is_bound_to_trained_head_and_frozen_windows(self):
        result = self.evaluate(overlap=True)
        diagnostics = result["physical_overlap_diagnostics"]
        self.assertEqual(len(diagnostics["pairs"]), 2000)
        self.assertEqual(diagnostics["categories"]["same_overlap"]["count"], 600)
        self.assertEqual(diagnostics["constant_baseline"], 0)
        self.assertIn("Pearson correlation", diagnostics["waveform_protocol"])
        self.assertAlmostEqual(diagnostics["pairs"][0]["waveform"], 1.0)
        head = self.model.overlap_head
        assert head is not None
        for parameter in head.parameters():
            self.optimizer.state[parameter]["exp_avg_sq"].zero_()
        self.save()
        with self.assertRaisesRegex(ValueError, "no recorded nonzero gradients"):
            self.evaluate(overlap=True)

    def test_overlap_scoring_rejects_untrained_head_and_constant_waveform(self):
        self.optimizer = build_optimizer(self.model)
        self.save()
        with self.assertRaisesRegex(ValueError, "not received"):
            self.evaluate(pair_head=False, overlap=True)
        head = self.model.overlap_head
        assert head is not None
        inputs = torch.arange(40, dtype=torch.float32).reshape(2, 1, 2, 2, 5) / 40
        a, _ = self.model.vae.encoder(inputs)
        b, _ = self.model.vae.encoder(torch.flip(inputs, dims=[-1]))
        (head(a, b) - torch.tensor([0.2, 0.8])).square().mean().backward()
        self.optimizer.step()
        self.save()
        with patch("scripts.streamed_overlap_evaluation._read_original",
                   return_value=np.zeros((2, 2, 5), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "nonconstant original windows"):
                self.evaluate(pair_head=False, overlap=True)

    def test_overlap_diagnostic_time_is_included_in_evaluation_deadline(self):
        clock = {"expired": False}

        def summarize(*args, **kwargs):
            result = physical_overlap_diagnostics(*args, **kwargs)
            clock["expired"] = True
            return result

        with patch("scripts.streamed_overlap_evaluation.time",
                   SimpleNamespace(monotonic=lambda: 61.0 if clock["expired"] else 0.0)), \
             patch("scripts.streamed_overlap_evaluation.physical_overlap_diagnostics", side_effect=summarize):
            with self.assertRaisesRegex(TimeoutError, "wall-time limit"):
                self.evaluate(overlap=True)

    def test_scoring_verifies_inputs_and_records_checkpoint_and_scorers(self):
        rng = torch.get_rng_state().clone()
        result = self.evaluate()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(result["checkpoint"]["attempt_id"], "attempt-1")
        self.assertEqual(result["checkpoint"]["checkpoint_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["input_verification"]["status"], "VALIDATION_INPUTS_VERIFIED")
        self.assertEqual(set(result["scorers"]), {"mu_cosine", "pair_head"})
        self.assertEqual(len(result["scorers"]["pair_head"]["rows"]), 200)
        diagnostics = result["scorers"]["pair_head"]["event_coverage_diagnostics"]
        self.assertEqual(len(diagnostics["pairs"]), 2000)
        self.assertEqual(diagnostics["categories"]["same_overlap"]["count"], 600)

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

    def test_epoch_validation_is_deterministic_and_restores_training_state(self):
        ids = tuple(query["id"] for query in self.manifest["queries"][:3])
        lpips = lambda x, y: (x - y).square().mean()
        validator = StreamedEpochValidator(
            self.root / "manifest.json", self.root, manifest_sha256=self.manifest["manifest_sha256"],
            reconstruction_query_ids=ids, reconstruction_seed=12, encode_batch_size=11,
            reconstruction_batch_size=2, max_wall_seconds=60, lpips_loss_fn=lpips,
        )
        self.model.train()
        before = torch.get_rng_state().clone()
        with patch("scripts.streamed_overlap_evaluation.load_manifest", return_value=self.manifest), \
             patch("scripts.streamed_overlap_evaluation.open_validation_sources", return_value=self.groups):
            with streamed_inference_context(self.model, "cpu", seed=4):
                result = validator(self.model)
            with streamed_inference_context(self.model, "cpu", seed=5):
                repeated = validator(self.model)
        self.assertEqual(result, repeated)
        self.assertEqual(result["epoch_validation_sha256"], validator.config_sha256)
        self.assertAlmostEqual(result["metrics"]["mu_cosine_ndcg10"],
                               self.evaluate(pair_head=False)["scorers"]["mu_cosine"]["summary"]["ndcg_at_10"])
        originals = [self.groups[query["window"]["source_id"]][SEISMIC_KEY]
                     for query in self.manifest["queries"][:3]]
        inputs = torch.from_numpy(np.stack([preprocess_for_token(cube) for cube in originals])[:, None])
        targets = torch.from_numpy(np.stack([normalize_cube_by_std(cube) for cube in originals])[:, None])
        with streamed_inference_context(self.model, "cpu", seed=4):
            mu, logvar = self.model.vae.encoder(inputs)
            noise = torch.randn((3, 2), generator=torch.Generator().manual_seed(12))
            prediction, _ = self.model.vae.decoder(mu + (0.5 * logvar).exp() * noise)
            _, mae, kl, perceptual, _ = compute_vae_losses(
                prediction, targets, mu, logvar, 0.001, rec_loss_fn=torch.nn.functional.l1_loss,
                lpips_loss_fn=lpips, lpips_weight=0.1,
            )
        self.assertAlmostEqual(result["reconstruction_mae"], mae.item(), places=6)
        self.assertAlmostEqual(result["lpips_loss"], perceptual.item(), places=6)
        self.assertAlmostEqual(result["kl"], kl.item(), places=6)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "eval mode"):
            validator(self.model)

    def test_tokenizer_export_preserves_mu_and_does_not_overwrite(self):
        def factory(arm, model_seed, head_seed):
            del model_seed, head_seed
            with torch.random.fork_rng():
                torch.manual_seed(10)
                return StreamedPairVAE(
                    VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"), arm,
                )

        self.model = factory("S3", 0, 0)
        self.optimizer = build_optimizer(self.model)
        self.save()
        destination = self.root / "tokenizer.pt"
        with patch("scripts.streamed_overlap_evaluation.load_manifest", return_value=self.manifest), \
             patch("scripts.streamed_overlap_evaluation.fresh_e1_model", side_effect=factory):
            result = export_tokenizer_checkpoint(
                self.path, self.journal, self.root / "manifest.json", destination,
            )
            with self.assertRaises(FileExistsError):
                export_tokenizer_checkpoint(self.path, self.journal, self.root / "manifest.json", destination)
        self.assertEqual(result["tokenizer_checkpoint_sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())
        adapter = VaeLatentAdapter(destination, device="cpu")
        self.model.eval()
        cubes = np.random.default_rng(42).normal(size=(2, 8, 8, 8)).astype(np.float32)
        with torch.inference_mode():
            expected, _ = self.model.vae.encoder(torch.from_numpy(cubes[:, None]))
        np.testing.assert_allclose(adapter.encode_batch(cubes), expected.numpy(), rtol=1e-6, atol=1e-6)

    def selection_checkpoint(self, epoch, mae):
        state = torch.load(self.path, weights_only=True)
        state["completed_epoch"] = epoch
        state["journal"] = {
            "event_id": f"event-{epoch}", "logical_step": epoch, "cumulative_exposure": 12 * epoch,
        }
        state["sampler"]["batches_drawn"] = epoch
        state["hashes"]["epoch_validation"] = "e" * 64
        state["selection_state"] = {
            "_epoch_validation": {
                "split": "validation", "manifest_sha256": self.manifest["manifest_sha256"],
                "epoch_validation_sha256": "e" * 64,
                "reconstruction_mae": mae, "lpips_loss": 0.0, "kl": 0.0,
            },
        }
        temporary = self.root / "replacement.pt"
        torch.save(state, temporary)
        temporary.replace(self.path)

    def select(self, report):
        with patch("scripts.streamed_overlap_selection.load_manifest", return_value=self.manifest), \
             patch("scripts.streamed_overlap_selection.fresh_e1_model", side_effect=toy_factory):
            return select_checkpoints(
                self.path, self.journal, self.root / "manifest.json", retrieval_report=report,
            )

    def test_selection_keeps_distinct_bests_and_reconciles_stale_events(self):
        last = self.root / "last.pt"
        self.path.replace(last)
        self.path = last
        self.selection_checkpoint(1, 1.0)
        report = self.evaluate()
        first = self.select(report)
        first_hash = first["last"]["checkpoint_sha256"]
        self.assertEqual(first["best_retrieval"]["checkpoint"]["checkpoint_sha256"], first_hash)
        self.assertEqual(hashlib.sha256((self.root / "best_reconstruction.pt").read_bytes()).hexdigest(), first_hash)
        row: dict[str, object] = dict(self.journal.rows[-1])
        row.update({
            "sequence": 2, "event_id": "event-2", "logical_step": 2, "epoch": 2,
            "cumulative_exposure": 24,
        })
        self.journal.append(row)
        with self.assertRaisesRegex(ValueError, "currently reconciled"):
            self.select(None)
        self.selection_checkpoint(2, 0.5)
        second = self.select(None)
        self.assertEqual(second["best_reconstruction"]["checkpoint"]["step"], 2)
        self.assertEqual(second["best_retrieval"]["checkpoint"]["step"], 1)
        self.assertNotEqual(second["last"]["checkpoint_sha256"], first_hash)
        self.journal.reconcile(watermark="event-1", attempt_id="attempt-2",
                               expected_step=1, expected_exposure=12)
        replacement = self.root / "replacement.pt"
        replacement.write_bytes((self.root / "best_retrieval.pt").read_bytes())
        replacement.replace(last)
        restored = self.select(None)
        self.assertEqual(restored["best_reconstruction"]["checkpoint"]["step"], 1)
        self.assertEqual(restored["best_retrieval"]["checkpoint"]["step"], 1)
        self.assertEqual(hashlib.sha256((self.root / "best_reconstruction.pt").read_bytes()).hexdigest(), first_hash)
        self.assertEqual(len(list((self.root / "selection" / "events").glob("*.json"))), 3)

    def test_selection_rejects_mismatched_results_and_recovers_alias_failure(self):
        last = self.root / "last.pt"
        self.path.replace(last)
        self.path = last
        self.selection_checkpoint(1, 1.0)
        report = self.evaluate()
        report["scorers"]["pair_head"]["rows"][0]["ndcg"] = 0.0
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            self.select(report)
        self.assertFalse((self.root / "best_retrieval.pt").exists())
        report = self.evaluate()
        with patch("scripts.streamed_overlap_selection._publish_alias", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.select(report)
        result = self.select(report)
        self.assertEqual(result["best_retrieval"]["checkpoint"]["step"], 1)
        self.assertTrue((self.root / "best_retrieval.pt").is_file())
        self.selection_checkpoint(1, 0.9)
        with self.assertRaisesRegex(ValueError, "different checkpoint contents"):
            self.select(None)

    def test_selection_does_not_replace_unrelated_checkpoint(self):
        last = self.root / "last.pt"
        self.path.replace(last)
        self.path = last
        self.selection_checkpoint(1, 1.0)
        unrelated = self.root / "best_reconstruction.pt"
        unrelated.write_bytes(b"unrelated")
        with self.assertRaisesRegex(ValueError, "existing checkpoint aliases"):
            self.select(None)
        self.assertEqual(unrelated.read_bytes(), b"unrelated")

    def test_selection_refuses_modified_owned_alias(self):
        last = self.root / "last.pt"
        self.path.replace(last)
        self.path = last
        self.selection_checkpoint(1, 1.0)
        self.select(None)
        alias = self.root / "best_reconstruction.pt"
        alias.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "unrelated selection alias"):
            self.select(None)
        self.assertEqual(alias.read_bytes(), b"modified")


if __name__ == "__main__":
    unittest.main()

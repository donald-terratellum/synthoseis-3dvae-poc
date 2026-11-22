import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import (
    HistoricalReconstruction, SelectedReconstruction, _verify_historical_reconstruction, _verify_reconstruction_inputs,
    complete_reconstruction_comparison, complete_selected_reconstruction,
)
from scripts.streamed_overlap_evaluation import _identity
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


class SelectedReconstructionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir()
        self.checkpoint = self.root / "last.pt"
        self.checkpoint.write_bytes(b"selected checkpoint")
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        snapshots = self.root / "selection" / "snapshots"
        snapshots.mkdir(parents=True)
        self.snapshot = snapshots / f"{digest}.pt"
        self.snapshot.write_bytes(self.checkpoint.read_bytes())
        (self.root / "best_retrieval.pt").write_bytes(self.checkpoint.read_bytes())
        self.identity = {
            "arm": "S3", "checkpoint_sha256": digest, "run_id": "run",
            "attempt_id": "attempt", "step": 1, "epoch": 1, "experiment_seed": 2,
        }
        self.manifest = fixture_manifest()
        self.manifest["protocol"].update({"device": "cpu", "runtime": {"fixture": True}})
        self.manifest["manifest_sha256"] = content_hash({
            key: value for key, value in self.manifest.items() if key != "manifest_sha256"
        })
        self.config = {
            "manifest_sha256": self.manifest["manifest_sha256"],
            "device": "cpu", "max_wall_seconds": 60,
        }
        self.state = {"hashes": {"reconstruction_guardrail": content_hash(self.config)}}
        self.report = measurement(self.manifest, "S3", "d", 1.0)
        self.report["checkpoint"] = self.identity
        self.lpips = torch.nn.Linear(1, 1)

    def run_measurement(self, config=None):
        return complete_selected_reconstruction(
            self.checkpoint, object(), self.root / "validation.json", self.root,
            {}, {}, {"device": "cpu"}, selection_kind="retrieval",
            evaluation_dir=self.evaluation, guardrail_manifest=self.manifest,
            guardrail_config=self.config if config is None else config, lpips_loss_fn=self.lpips,
        )

    def test_selected_measurement_is_recoverable_and_reverified(self):
        def produce(manifest, path, **kwargs):
            self.assertEqual(path, self.snapshot)
            self.assertEqual(kwargs["arm"], "S3")
            self.assertEqual(manifest, self.manifest)
            return self.report

        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                   return_value=(self.state, self.identity["checkpoint_sha256"])), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"checkpoint": self.identity}), \
             patch("scripts.streamed_overlap_epoch_runner.measure_reconstruction",
                   side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.run_measurement()
        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                   return_value=(self.state, self.identity["checkpoint_sha256"])), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"checkpoint": self.identity}), \
             patch("scripts.streamed_overlap_epoch_runner.measure_reconstruction", side_effect=produce) as measure, \
             patch("scripts.streamed_overlap_epoch_runner._reconstruction_runtime", return_value={"fixture": True}), \
             patch("scripts.streamed_overlap_epoch_runner.lpips_fingerprint", return_value="a" * 64), \
             patch("scripts.streamed_overlap_epoch_runner.choose_torch_device",
                   return_value=SimpleNamespace(type="cpu")), \
             patch("scripts.streamed_overlap_epoch_runner._verify_stores") as verify:
            result = self.run_measurement()
            self.assertEqual(self.run_measurement(), result)
            self.assertEqual(measure.call_count, 1)
            verify.assert_called_once()
            artifact = self.evaluation / "guardrail-retrieval" / result["measurement"]["path"]
            artifact.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.run_measurement()

    def test_changed_configuration_is_rejected_before_selected_evaluation(self):
        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                   return_value=(self.state, self.identity["checkpoint_sha256"])), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint") as selected:
            with self.assertRaisesRegex(ValueError, "Final checkpoint and frozen"):
                self.run_measurement({**self.config, "max_wall_seconds": 61})
            selected.assert_not_called()

    def test_changed_alias_during_measurement_cannot_publish_completion(self):
        def produce(*_args, **_kwargs):
            (self.root / "best_retrieval.pt").write_bytes(b"changed")
            return self.report

        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                   return_value=(self.state, self.identity["checkpoint_sha256"])), \
             patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"checkpoint": self.identity}), \
             patch("scripts.streamed_overlap_epoch_runner.measure_reconstruction", side_effect=produce):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                self.run_measurement()
        self.assertEqual(list((self.evaluation / "guardrail-retrieval" / "completed").glob("*.json")), [])


class SelectedComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = fixture_manifest()
        p0b_checkpoint = self.root / "p0b.pt"
        p0b_checkpoint.write_bytes(b"historical checkpoint")
        checkpoint_hash = hashlib.sha256(p0b_checkpoint.read_bytes()).hexdigest()
        report = measurement(self.manifest, "P0b", "f", 1.0)
        report["checkpoint"]["checkpoint_sha256"] = checkpoint_hash
        p0b_report = self.root / "p0b.json"
        p0b_report.write_text(json.dumps(report))
        self.historical = HistoricalReconstruction(
            p0b_report, hashlib.sha256(p0b_report.read_bytes()).hexdigest(),
            p0b_checkpoint, checkpoint_hash, "reconstruction-best",
        )
        self.reference_hash = content_hash({
            "report_sha256": self.historical.report_sha256,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_label": "reconstruction-best",
        })
        self.contexts, self.identities, self.states, self.records = {}, {}, {}, {}
        self.make_selection("S3", "retrieval", 1.02)
        self.make_selection("S1", "reconstruction", 1.0)
        self.patches = [
            patch("scripts.streamed_overlap_epoch_runner._checkpoint_state",
                  side_effect=lambda path: self.states[path]),
            patch("scripts.streamed_overlap_epoch_runner.complete_selected_reconstruction",
                  side_effect=lambda path, *_args, **_kwargs: self.records[path]),
            patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                  side_effect=lambda path, *_args, **_kwargs: {"checkpoint": self.identities[path]}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def make_selection(self, arm, kind, scale):
        root = self.root / arm
        root.mkdir()
        checkpoint = root / "last.pt"
        checkpoint.write_bytes(arm.encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        snapshot = root / "selection" / "snapshots" / f"{digest}.pt"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(checkpoint.read_bytes())
        evaluation = root / "evaluation"
        directory = evaluation / f"guardrail-{kind}" / "attempt"
        directory.mkdir(parents=True)
        identity = {
            "arm": arm, "checkpoint_sha256": digest, "run_id": "run",
            "attempt_id": "attempt", "step": 1, "epoch": 1, "experiment_seed": 2,
            "source_sha256": "1" * 64, "target_sha256": "2" * 64,
            "validation_manifest_sha256": "3" * 64,
        }
        report = measurement(self.manifest, arm, "a", scale)
        report["checkpoint"] = identity
        path = directory / "measurement.json"
        path.write_text(json.dumps(report))
        context = SelectedReconstruction(
            checkpoint, object(), root / "validation.json", root, {}, {}, {}, evaluation, kind,
        )
        state = {"arm": arm, "hashes": {"reconstruction_reference": self.reference_hash}}
        self.states[checkpoint] = (state, digest)
        self.states[snapshot] = (state, digest)
        self.contexts[arm] = context
        self.identities[checkpoint] = identity
        self.records[checkpoint] = {
            "checkpoint": identity,
            "measurement": {"path": "attempt/measurement.json",
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        }

    def compare(self, *, candidate="S3", s1=True):
        return complete_reconstruction_comparison(
            self.contexts[candidate], self.historical, s1=self.contexts["S1"] if s1 else None,
            guardrail_manifest=self.manifest, guardrail_config={}, lpips_loss_fn=object(),
        )

    def test_historical_preflight_checks_full_pinned_guardrail_measurement(self):
        original = json.loads(self.historical.report_path.read_text())
        self.assertEqual(_verify_historical_reconstruction(self.historical, self.manifest), original)
        alterations = (
            ("protocol", lambda report: report["protocol"].update(seed=13), "protocol"),
            ("coverage", lambda report: report["datasets"]["real"].pop(), "full real"),
            ("metric", lambda report: report["datasets"]["synthetic"][0].update(lpips=-1), "finite"),
            ("label", lambda report: report["checkpoint"].update(checkpoint_label="other"), "frozen reference"),
        )
        for name, alter, message in alterations:
            changed = copy.deepcopy(original)
            alter(changed)
            self.historical.report_path.write_text(json.dumps(changed))
            reference = replace(
                self.historical, report_sha256=hashlib.sha256(self.historical.report_path.read_bytes()).hexdigest(),
            )
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                _verify_historical_reconstruction(reference, self.manifest)
        self.historical.report_path.write_text(json.dumps(original))
        self.historical.checkpoint_path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            _verify_historical_reconstruction(self.historical, self.manifest)

    def test_comparison_uses_both_references_and_recovers(self):
        result = self.compare()
        self.assertEqual(result["comparison"]["status"], "PASS_NUMERIC_ONLY")
        self.assertEqual(set(result["comparison"]["comparisons"]), {"S1", "P0b"})
        self.assertEqual(result["reference_checkpoints"]["P0b"]["checkpoint_label"], "reconstruction-best")
        self.assertEqual(result["reference_checkpoints"]["S1"], self.identities[self.contexts["S1"].checkpoint_path])
        self.assertEqual(self.compare(), result)
        directory = self.contexts["S3"].evaluation_dir / "guardrail-retrieval" / "comparisons"
        self.assertEqual(len(list(directory.glob("*.json"))), 1)

    def test_requires_matching_s1_and_unchanged_historical_reference(self):
        with self.assertRaisesRegex(ValueError, "require both"):
            self.compare(s1=False)
        self.identities[self.contexts["S1"].checkpoint_path]["source_sha256"] = "4" * 64
        report_path = self.contexts["S1"].evaluation_dir / "guardrail-reconstruction" / "attempt" / "measurement.json"
        report = json.loads(report_path.read_text())
        report["checkpoint"]["source_sha256"] = "4" * 64
        report_path.write_text(json.dumps(report))
        self.records[self.contexts["S1"].checkpoint_path]["measurement"]["sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.compare()
        self.historical.checkpoint_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.compare()

    def test_s1_uses_only_p0b_and_failed_guardrail_is_not_success(self):
        result = self.compare(candidate="S1", s1=False)
        self.assertEqual(set(result["comparison"]["comparisons"]), {"P0b"})
        path = self.contexts["S3"].evaluation_dir / "guardrail-retrieval" / "attempt" / "measurement.json"
        report = json.loads(path.read_text())
        for item in report["datasets"]["real"]:
            item["mae"] = 1.021
        path.write_text(json.dumps(report))
        self.records[self.contexts["S3"].checkpoint_path]["measurement"]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self.assertEqual(self.compare()["comparison"]["status"], "FAIL")

    def test_selected_snapshot_must_bind_the_reference(self):
        context = self.contexts["S3"]
        digest = self.identities[context.checkpoint_path]["checkpoint_sha256"]
        snapshot = context.checkpoint_path.parent / "selection" / "snapshots" / f"{digest}.pt"
        self.states[snapshot] = ({"arm": "S3", "hashes": {"reconstruction_reference": "f" * 64}}, digest)
        with self.assertRaisesRegex(ValueError, "Selected checkpoint does not bind"):
            self.compare()

    def test_comparison_rejects_regenerating_validation_paths_before_loading(self):
        context = replace(
            self.contexts["S3"], checkpoint_path=self.root / "fake_data_validation_v2" / "last.pt",
        )
        with patch("scripts.streamed_overlap_epoch_runner._checkpoint_state") as load:
            with self.assertRaisesRegex(ValueError, "validation-v2"):
                complete_reconstruction_comparison(
                    context, self.historical, s1=self.contexts["S1"], guardrail_manifest=self.manifest,
                    guardrail_config={}, lpips_loss_fn=object(),
                )
            load.assert_not_called()

    def test_comparison_rejects_changed_completed_record(self):
        self.compare()
        directory = self.contexts["S3"].evaluation_dir / "guardrail-retrieval" / "comparisons"
        next(directory.glob("*.json")).write_text("{}")
        with self.assertRaisesRegex(ValueError, "comparison record is invalid"):
            self.compare()


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

    def test_guardrail_input_preflight_verifies_runtime_and_full_store(self):
        manifest = self.freeze()
        config = {"device": "cpu", "max_wall_seconds": 60}
        _verify_reconstruction_inputs(manifest, config, self.lpips)
        changed = copy.deepcopy(manifest)
        changed["protocol"]["runtime"]["versions"]["torch"] = "different"
        with self.assertRaisesRegex(ValueError, "runtime or LPIPS"):
            _verify_reconstruction_inputs(changed, config, self.lpips)
        with torch.no_grad():
            self.lpips.weight[0] = 2
        with self.assertRaisesRegex(ValueError, "runtime or LPIPS"):
            _verify_reconstruction_inputs(manifest, config, self.lpips)
        with torch.no_grad():
            self.lpips.weight[0] = 1
        group = zarr.open_group(str(self.stores["real"]), mode="r+")
        patches = group["patches"]
        assert isinstance(patches, zarr.Array)
        patches[1, 0, 0, 0] = 123
        with self.assertRaisesRegex(ValueError, "differs from its frozen"):
            _verify_reconstruction_inputs(manifest, config, self.lpips)
        with patch("scripts.streamed_overlap_epoch_runner._verify_stores"), \
             patch("scripts.streamed_overlap_epoch_runner.time.monotonic", side_effect=[0, 61]):
            with self.assertRaisesRegex(TimeoutError, "wall-time limit"):
                _verify_reconstruction_inputs(manifest, config, self.lpips)

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

    def test_selected_measurement_runs_frozen_fixture_producer_and_recovers(self):
        manifest = self.freeze()
        config = {"manifest_sha256": manifest["manifest_sha256"], "device": "cpu", "max_wall_seconds": 60}
        model = StreamedPairVAE(FixtureVAE(), "S1")
        checkpoint = self.root / "last.pt"
        state = {
            "schema": "streamed-epoch-state-v1", "arm": "S1", "completed_epoch": 1,
            "batches_per_epoch": 1, "model": model.state_dict(),
            "sampler": {"split": "train", "batches_drawn": 1, "source_manifest_hash": "b" * 64},
            "hashes": {
                "validation_manifest": "a" * 64, "source": "b" * 64,
                "config": "c" * 64, "target": "d" * 64,
                "reconstruction_guardrail": content_hash(config),
            },
            "journal": {"event_id": "event-1", "logical_step": 1, "cumulative_exposure": 12},
        }
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        snapshots = self.root / "selection" / "snapshots"
        snapshots.mkdir(parents=True)
        (snapshots / f"{digest}.pt").write_bytes(checkpoint.read_bytes())
        (self.root / "best_reconstruction.pt").write_bytes(checkpoint.read_bytes())
        evaluation = self.root / "evaluation"
        evaluation.mkdir()
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
        identity = _identity(state, digest, journal, "a" * 64)

        def factory(arm, model_seed, head_seed):
            del model_seed, head_seed
            return StreamedPairVAE(FixtureVAE(), arm)

        with patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"checkpoint": identity}), \
             patch("scripts.streamed_overlap_reconstruction.load_manifest",
                   return_value={"manifest_sha256": "a" * 64}), \
             patch("scripts.streamed_overlap_reconstruction.fresh_e1_model", side_effect=factory):
            def measure():
                return complete_selected_reconstruction(
                    checkpoint, journal, self.root / "validation.json", self.root,
                    {}, {}, {"device": "cpu"}, selection_kind="reconstruction",
                    evaluation_dir=evaluation, guardrail_manifest=manifest,
                    guardrail_config=config, lpips_loss_fn=self.lpips,
                )
            result = measure()
            self.assertEqual(result["checkpoint"], identity)
            self.assertEqual(measure(), result)
            self.assertTrue((evaluation / "guardrail-reconstruction" / result["measurement"]["path"]).is_file())
            patches = zarr.open_group(str(self.stores["real"]), mode="r+")["patches"]
            patches[0, 0, 0, 0] = 123
            with self.assertRaisesRegex(ValueError, "differs from its frozen"):
                measure()

    def test_selected_s1_comparison_uses_real_fixture_measurements(self):
        manifest = self.freeze()
        p0b = self.measure(manifest)
        p0b_report = self.root / "p0b.json"
        p0b_report.write_text(json.dumps(p0b))
        historical = HistoricalReconstruction(
            p0b_report, hashlib.sha256(p0b_report.read_bytes()).hexdigest(),
            self.checkpoint, self.digest, "reconstruction-best",
        )
        reference_hash = content_hash({
            "report_sha256": historical.report_sha256, "checkpoint_sha256": self.digest,
            "checkpoint_label": historical.checkpoint_label,
        })
        config = {"manifest_sha256": manifest["manifest_sha256"], "device": "cpu", "max_wall_seconds": 60}
        model = StreamedPairVAE(FixtureVAE(), "S1")
        checkpoint = self.root / "last.pt"
        state = {
            "schema": "streamed-epoch-state-v1", "arm": "S1", "completed_epoch": 1,
            "batches_per_epoch": 1, "model": model.state_dict(),
            "sampler": {"split": "train", "batches_drawn": 1, "source_manifest_hash": "b" * 64},
            "hashes": {
                "validation_manifest": "a" * 64, "source": "b" * 64, "config": "c" * 64,
                "target": "d" * 64, "reconstruction_guardrail": content_hash(config),
                "reconstruction_reference": reference_hash,
            },
            "journal": {"event_id": "event-1", "logical_step": 1, "cumulative_exposure": 12},
        }
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        snapshots = self.root / "selection" / "snapshots"
        snapshots.mkdir(parents=True)
        (snapshots / f"{digest}.pt").write_bytes(checkpoint.read_bytes())
        (self.root / "best_reconstruction.pt").write_bytes(checkpoint.read_bytes())
        evaluation = self.root / "evaluation"
        evaluation.mkdir()
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
        identity = _identity(state, digest, journal, "a" * 64)
        selection = SelectedReconstruction(
            checkpoint, journal, self.root / "validation.json", self.root,
            {}, {}, {"device": "cpu"}, evaluation, "reconstruction",
        )

        def factory(arm, model_seed, head_seed):
            del model_seed, head_seed
            return StreamedPairVAE(FixtureVAE(), arm)

        with patch("scripts.streamed_overlap_epoch_runner.complete_selected_checkpoint",
                   return_value={"checkpoint": identity}), \
             patch("scripts.streamed_overlap_reconstruction.load_manifest",
                   return_value={"manifest_sha256": "a" * 64}), \
             patch("scripts.streamed_overlap_reconstruction.fresh_e1_model", side_effect=factory):
            result = complete_reconstruction_comparison(
                selection, historical, s1=None, guardrail_manifest=manifest,
                guardrail_config=config, lpips_loss_fn=self.lpips,
            )
        self.assertEqual(result["comparison"]["status"], "PASS_NUMERIC_ONLY")
        self.assertEqual(result["comparison"]["comparisons"]["P0b"]["checkpoint_sha256"], self.digest)

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

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import zarr

from scripts import evaluate_geology_benchmark as bench
from scripts import train as train_script
from src.geology_classifier import (
    average_precision,
    best_f1_threshold,
    binary_auroc,
    confusion_matrix,
    evaluate_classifier_predictions,
    f1_at_threshold,
)
from src.model import GEOLOGY_PRESENCE_CLASSES, VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter

PATCH = (8, 8, 8)
META_KEYS = ["meta_fault_fraction", "meta_channel_fraction", "meta_flat_spot_fraction"]


class TestMetrics(unittest.TestCase):
    def test_auroc(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(binary_auroc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(binary_auroc(y, [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(binary_auroc(y, [0.5, 0.5, 0.5, 0.5]), 0.5)
        # One positive below one negative: 3 of 4 pairs ordered correctly.
        self.assertEqual(binary_auroc(y, [0.1, 0.6, 0.5, 0.9]), 0.75)
        self.assertIsNone(binary_auroc([1, 1], [0.2, 0.3]))

    def test_average_precision(self):
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), 0.5 + 0.5 * 2 / 3)
        self.assertEqual(average_precision([1, 1, 0], [0.9, 0.8, 0.1]), 1.0)
        self.assertIsNone(average_precision([0, 0], [0.1, 0.2]))

    def test_threshold_and_f1(self):
        y = [1, 1, 0, 0]
        s = [0.9, 0.6, 0.4, 0.1]
        t = best_f1_threshold(y, s)
        self.assertEqual(t, 0.6)
        self.assertEqual(f1_at_threshold(y, s, t), 1.0)
        self.assertAlmostEqual(f1_at_threshold(y, s, 0.95), 0.0)
        self.assertEqual(best_f1_threshold([0, 0], [0.3, 0.4]), 0.5)

    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
        np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    def _targets(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        presence = rng.integers(0, 2, size=(n, 7))
        dip_mean = np.r_[np.zeros(n // 2), rng.integers(0, 6, n - n // 2)]
        return {"presence": presence, "dip_mean": dip_mean, "dip_range": rng.integers(0, 6, n).astype(float)}

    def test_perfect_predictions_pass_gate(self):
        t = self._targets()
        probs = {
            "presence": t["presence"].astype(float),
            "dip_mean": np.eye(6)[t["dip_mean"].astype(int)],
            "dip_range": np.eye(6)[t["dip_range"].astype(int)],
        }
        report = evaluate_classifier_predictions(probs, t)
        self.assertEqual(report["macro_auroc"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["dip_mean"]["accuracy"], 1.0)
        self.assertEqual(int(np.trace(report["dip_mean"]["confusion_matrix"])), 60)
        self.assertTrue(report["sanity_gate"]["passed"])

    def test_uninformative_predictions_fail_gate(self):
        t = self._targets()
        probs = {"presence": np.full((60, 7), 0.5), "dip_mean": np.full((60, 6), 1 / 6), "dip_range": np.full((60, 6), 1 / 6)}
        report = evaluate_classifier_predictions(probs, t)
        self.assertEqual(report["macro_auroc"], 0.5)
        self.assertFalse(report["sanity_gate"]["passed"])
        self.assertAlmostEqual(report["dip_mean"]["majority_rate"], report["dip_mean"]["class_counts"][0] / 60)

    def test_absent_class_excluded_from_macro(self):
        t = self._targets()
        t["presence"][:, 1] = 0
        probs = {"presence": t["presence"].astype(float), "dip_mean": np.eye(6)[t["dip_mean"].astype(int)], "dip_range": np.eye(6)[t["dip_range"].astype(int)]}
        report = evaluate_classifier_predictions(probs, t)
        self.assertIsNone(report["presence"]["fault_x"]["auroc"])
        self.assertIsNone(report["presence"]["fault_x"]["f1"])
        self.assertEqual(report["macro_auroc"], 1.0)


def _save_checkpoint(path, classifier=True):
    torch.manual_seed(0)
    model = VAE3D(base_ch=8, latent_dim=32, patch_shape=PATCH, geology_projection=True,
                  geology_proj_hidden=16, geology_proj_dim=8, geology_classifier=classifier, geology_classifier_hidden=32)
    torch.save(train_script.build_checkpoint_payload(model), path)
    return model


def _write_dataset(path, n=40, sampling_mode="uniform", seed=0):
    rng = np.random.default_rng(seed)
    g = zarr.open_group(str(path), mode="w")
    g.create_array("patches", data=rng.normal(size=(n,) + PATCH).astype(np.float32))
    for key in META_KEYS:
        vals = np.where(rng.random(n) < 0.5, rng.random(n), 0.0).astype(np.float32)
        g.create_array(key, data=vals)
    for c in GEOLOGY_PRESENCE_CLASSES:
        g.create_array(f"label_presence_{c}", data=rng.integers(0, 2, n).astype(np.uint8))
    g.create_array("meta_dip_mean_class", data=rng.integers(0, 6, n).astype(np.float32))
    g.create_array("meta_dip_range_class", data=rng.integers(0, 6, n).astype(np.float32))
    g.attrs["sampling_mode"] = sampling_mode


class TestAdapterClassifier(unittest.TestCase):
    def test_classify_batch_matches_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            model = _save_checkpoint(path)
            adapter = VaeLatentAdapter(path, device="cpu", load_classifier=True)
            cubes = np.random.default_rng(0).normal(size=(3,) + PATCH).astype(np.float32)
            out = adapter.classify_batch(cubes)
            model.eval()
            with torch.no_grad():
                mu, _ = model.encoder(torch.from_numpy(cubes[:, None]))
                ref = torch.sigmoid(model.classify(mu)["presence"]).numpy()
            np.testing.assert_allclose(out["presence"], ref, rtol=1e-5, atol=1e-6)
            self.assertEqual(out["dip_mean"].shape, (3, 6))
            np.testing.assert_allclose(out["dip_range"].sum(axis=1), 1.0, atol=1e-5)
            with self.assertRaises(RuntimeError):
                VaeLatentAdapter(path, device="cpu").classify_batch(cubes)

    def test_load_classifier_requires_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.pt"
            _save_checkpoint(path, classifier=False)
            with self.assertRaisesRegex(ValueError, "no geology classifier"):
                VaeLatentAdapter(path, device="cpu", load_classifier=True)


class TestBenchmarkIntegration(unittest.TestCase):
    def _run(self, tmp, extra):
        out = Path(tmp) / f"report_{len(extra)}.json"
        argv = [
            "evaluate_geology_benchmark.py", "--data", str(Path(tmp) / "val.zarr"), "--checkpoint", str(Path(tmp) / "c.pt"),
            "--out_json", str(out), "--manifest", str(Path(tmp) / "manifest.json"), "--benchmark_size", "30",
            "--device", "cpu", "--use_geo_embedding", "--metadata_keys", *META_KEYS, "--background_keys", *META_KEYS,
            "--bootstrap_samples", "10",
        ] + extra
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            bench.main()
        return json.loads(out.read_text())

    def test_classifier_metrics_added_without_changing_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_checkpoint(Path(tmp) / "c.pt")
            _write_dataset(Path(tmp) / "val.zarr")
            _write_dataset(Path(tmp) / "train.zarr", sampling_mode="class_anchored", seed=1)
            base = self._run(tmp, [])
            with_cls = self._run(tmp, [
                "--classifier_data", str(Path(tmp) / "val.zarr"),
                "--classifier_threshold_data", str(Path(tmp) / "train.zarr"),
            ])
            self.assertNotIn("classifier_metrics", base)
            cm = with_cls.pop("classifier_metrics")
            self.assertEqual(base, with_cls)
            self.assertEqual(cm["n"], 40)
            self.assertEqual(cm["sampling_mode"], "uniform")
            self.assertTrue(cm["threshold_source"].endswith("train.zarr"))
            self.assertEqual(set(cm["presence"]), set(GEOLOGY_PRESENCE_CLASSES))
            self.assertIn("passed", cm["sanity_gate"])
            self.assertEqual(cm["preprocess"], "tokenizer")
            extrema = self._run(tmp, ["--classifier_data", str(Path(tmp) / "val.zarr"), "--classifier_preprocess", "extrema"])
            self.assertEqual(extrema["classifier_metrics"]["preprocess"], "extrema")
            self.assertEqual(extrema["classifier_metrics"]["threshold_source"], "fixed_0.5")

    def test_rejects_class_anchored_eval_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_checkpoint(Path(tmp) / "c.pt")
            _write_dataset(Path(tmp) / "val.zarr", sampling_mode="class_anchored")
            with self.assertRaisesRegex(ValueError, "natural-prevalence"):
                self._run(tmp, ["--classifier_data", str(Path(tmp) / "val.zarr")])

    def test_rejects_eval_data_without_positive_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_checkpoint(Path(tmp) / "c.pt")
            _write_dataset(Path(tmp) / "val.zarr")
            g = zarr.open_group(str(Path(tmp) / "val.zarr"), mode="a")
            for c in GEOLOGY_PRESENCE_CLASSES:
                g[f"label_presence_{c}"][:] = 0
            with self.assertRaisesRegex(ValueError, "no positive presence labels"):
                self._run(tmp, ["--classifier_data", str(Path(tmp) / "val.zarr")])


if __name__ == "__main__":
    unittest.main()

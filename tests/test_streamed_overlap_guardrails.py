import copy
import unittest

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_guardrails import compare_reconstruction_guardrails


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


if __name__ == "__main__":
    unittest.main()

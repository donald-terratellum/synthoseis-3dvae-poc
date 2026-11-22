"""Numerical reconstruction guardrails on fully matched, frozen measurements.

This module does not read datasets or run models. Input and checkpoint hashes in
measurement reports must be verified by their producer before adoption decisions.
"""

import math
from numbers import Real

from scripts.streamed_overlap_audit import content_hash


def _sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _manifest(manifest: dict) -> dict:
    if (manifest.get("schema") != "streamed-reconstruction-guardrail-v1"
            or manifest.get("usage") != "post_selection_guardrail"
            or manifest.get("manifest_sha256") != content_hash({
                key: value for key, value in manifest.items() if key != "manifest_sha256"
            })):
        raise ValueError("A hash-bound post-selection reconstruction manifest is required.")
    protocol = manifest.get("protocol")
    if (not isinstance(protocol, dict) or protocol.get("input_mode") != "tokenizer"
            or protocol.get("latent_mode") != "sampled"
            or protocol.get("sampling_method") != "normal_reparameterization"
            or protocol.get("patch_shape") != [32, 32, 64]
            or isinstance(protocol.get("seed"), bool) or not isinstance(protocol.get("seed"), int)
            or not 0 <= protocol["seed"] < 2**32 or not _sha256(protocol.get("lpips_sha256"))):
        raise ValueError("A fixed tokenizer and sampled-latent reconstruction protocol is required.")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"synthetic", "real"}:
        raise ValueError("Frozen synthetic and real guardrail datasets are required.")
    result = {}
    for domain, records in datasets.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"{domain} guardrail manifest must contain inputs.")
        expected = {}
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                    or not record["id"] or record["id"] in expected
                    or not _sha256(record.get("input_sha256"))
                    or not _sha256(record.get("target_sha256"))):
                raise ValueError(f"Invalid or duplicate {domain} guardrail input identity.")
            expected[record["id"]] = record
        result[domain] = expected
    return result


def _measurement(report: dict, manifest: dict, expected: dict, name: str) -> dict:
    checkpoint = report.get("checkpoint")
    if (report.get("manifest_sha256") != manifest["manifest_sha256"]
            or report.get("protocol") != manifest["protocol"]
            or not isinstance(checkpoint, dict) or not _sha256(checkpoint.get("checkpoint_sha256"))
            or checkpoint.get("arm") != name):
        raise ValueError(f"{name} reconstruction identity or protocol does not match the manifest.")
    if name == "P0b":
        if not isinstance(checkpoint.get("checkpoint_label"), str) or not checkpoint["checkpoint_label"]:
            raise ValueError("Historical reference must identify its reconstruction checkpoint.")
    elif (any(not isinstance(checkpoint.get(key), str) or not checkpoint[key] for key in ("run_id", "attempt_id"))
          or any(isinstance(checkpoint.get(key), bool) or not isinstance(checkpoint.get(key), int)
                 or checkpoint[key] < minimum for key, minimum in (("step", 1), ("experiment_seed", 0)))):
        raise ValueError(f"{name} reconstruction requires arm, run, attempt, step and seed identity.")
    datasets = report.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"synthetic", "real"}:
        raise ValueError(f"{name} requires complete synthetic and real measurements.")
    values = {}
    for domain in ("synthetic", "real"):
        records = datasets[domain]
        if not isinstance(records, list) or len(records) != len(expected[domain]):
            raise ValueError(f"{name} does not cover the full {domain} manifest.")
        seen = set()
        totals = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid {name} {domain} measurement.")
            item_id = record.get("id")
            original = expected[domain].get(item_id)
            if (not isinstance(item_id, str) or item_id in seen or original is None
                    or record.get("input_sha256") != original["input_sha256"]
                    or record.get("target_sha256") != original["target_sha256"]):
                raise ValueError(f"{name} {domain} inputs or targets differ from the frozen manifest.")
            names = ("mae", "lpips", "kl") if domain == "synthetic" else ("mae",)
            if any(isinstance(record.get(key), bool) or not isinstance(record.get(key), Real)
                   or not math.isfinite(record[key]) or record[key] < 0 for key in names):
                raise ValueError(f"{name} {domain} reconstruction metrics must be finite and nonnegative.")
            seen.add(item_id)
            totals.append(float(record["mae"]) + (
                0.1 * float(record["lpips"]) + 0.001 * float(record["kl"]) if domain == "synthetic" else 0
            ))
        if seen != set(expected[domain]):
            raise ValueError(f"{name} does not cover the full {domain} manifest.")
        values[domain] = sum(totals) / len(totals)
    return values


def compare_reconstruction_guardrails(
    manifest: dict, candidate: dict, references: dict[str, dict],
) -> dict:
    """Apply the +2% matched synthetic recipe and real MAE criteria to numeric reports."""
    expected = _manifest(manifest)
    checkpoint = candidate.get("checkpoint")
    candidate_name = checkpoint.get("arm") if isinstance(checkpoint, dict) else None
    if candidate_name not in {"S1", "S2", "S3"}:
        raise ValueError("A streamed experiment candidate checkpoint is required.")
    if not references or not set(references) <= {"S1", "P0b"} or candidate_name in references:
        raise ValueError("Explicit distinct S1 and/or P0b reference checkpoints are required.")
    candidate_metrics = _measurement(candidate, manifest, expected, candidate_name)
    comparisons = {}
    candidate_hash = candidate["checkpoint"]["checkpoint_sha256"]
    seen = {candidate_hash}
    for name, report in references.items():
        reference = _measurement(report, manifest, expected, name)
        if name == "S1" and report["checkpoint"]["experiment_seed"] != candidate["checkpoint"]["experiment_seed"]:
            raise ValueError("S1 reconstruction reference must use the matched experiment seed.")
        reference_hash = report["checkpoint"]["checkpoint_sha256"]
        if reference_hash in seen:
            raise ValueError("Candidate and reference checkpoint identities must be distinct.")
        seen.add(reference_hash)
        comparisons[name] = {
            "checkpoint_sha256": reference_hash,
            "synthetic_recipe_total": reference["synthetic"], "real_mae": reference["real"],
            "synthetic_within_2_percent": candidate_metrics["synthetic"] <= 1.02 * reference["synthetic"],
            "real_within_2_percent": candidate_metrics["real"] <= 1.02 * reference["real"],
        }
    passed = all(value["synthetic_within_2_percent"] and value["real_within_2_percent"]
                 for value in comparisons.values())
    return {
        "status": "PASS_NUMERIC_ONLY" if passed else "FAIL",
        "candidate_checkpoint_sha256": candidate_hash, "manifest_sha256": manifest["manifest_sha256"],
        "candidate_synthetic_recipe_total": candidate_metrics["synthetic"],
        "candidate_real_mae": candidate_metrics["real"], "comparisons": comparisons,
        "limitation": "Numeric comparison only: input, target, checkpoint and measurement provenance must be verified independently.",
    }

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import sys

SCRIPT_DIR = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import zarr

from scripts.train import (
    DEFAULT_BACKGROUND_METADATA_KEYS,
    DEFAULT_DERIVED_METADATA_KEYS,
    compute_latent_geology_diagnostics,
    fit_geology_metadata_calibration,
    prepare_geology_metadata_vectors,
    resolve_background_key_indices,
)
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from src.tokenizer.core.preprocess import preprocess_for_token


def _load_or_create_manifest(
    manifest_path: Path,
    dataset_size: int,
    benchmark_size: int,
    seed: int,
) -> dict:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "indices" not in payload:
            raise ValueError(f"Invalid manifest at {manifest_path}: missing 'indices'.")
        return payload

    rng = np.random.default_rng(int(seed))
    select_count = min(int(benchmark_size), int(dataset_size))
    indices = rng.choice(np.arange(dataset_size, dtype=np.int64), size=select_count, replace=False)
    indices = np.sort(indices)

    payload = {
        "schema_version": 1,
        "dataset_size": int(dataset_size),
        "benchmark_size": int(select_count),
        "seed": int(seed),
        "indices": indices.tolist(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    x_norm = x / norms
    sim = x_norm @ x_norm.T
    return np.clip(sim, -1.0, 1.0)


def _dcg(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, relevance.size + 2, dtype=np.float64))
    return float(np.sum((2.0 ** relevance - 1.0) / denom))


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(low=0, high=arr.size, size=arr.size)
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.percentile(means_arr, 2.5)), float(np.percentile(means_arr, 97.5))


def _compute_ranking_metrics(
    latent_similarity: np.ndarray,
    metadata_similarity: np.ndarray,
    has_selected: np.ndarray,
    ks: Sequence[int],
    positive_threshold: float,
    negative_threshold: float,
    query_mask: np.ndarray,
) -> dict:
    sample_count = int(latent_similarity.shape[0])
    diag_mask = np.eye(sample_count, dtype=bool)

    eligible_query = np.asarray(query_mask, dtype=bool) & np.asarray(has_selected, dtype=bool)
    query_indices = np.where(eligible_query)[0]
    if query_indices.size == 0:
        out = {"query_count": 0}
        for k in ks:
            out[f"recall_at_{int(k)}"] = 0.0
            out[f"precision_at_{int(k)}"] = 0.0
            out[f"ndcg_at_{int(k)}"] = 0.0
            out[f"hard_negative_rate_at_{int(k)}"] = 0.0
        out["query_metrics"] = {}
        return out

    metrics_per_k: dict[int, dict[str, list[float]]] = {
        int(k): {"recall": [], "precision": [], "ndcg": [], "hard_negative_rate": []}
        for k in ks
    }

    for query_idx in query_indices.tolist():
        valid = (~diag_mask[query_idx]) & has_selected
        pos_mask = valid & (metadata_similarity[query_idx] >= float(positive_threshold))
        neg_mask = valid & (metadata_similarity[query_idx] <= float(negative_threshold))
        pos_count = int(np.count_nonzero(pos_mask))
        if pos_count <= 0:
            continue

        ranked = np.argsort(-latent_similarity[query_idx])
        ranked = ranked[ranked != query_idx]
        ranked = ranked[has_selected[ranked]]

        for k in ks:
            eff_k = min(int(k), int(ranked.shape[0]))
            if eff_k <= 0:
                continue
            topk = ranked[:eff_k]
            topk_pos = int(np.count_nonzero(pos_mask[topk]))
            topk_neg = int(np.count_nonzero(neg_mask[topk]))

            recall = float(topk_pos) / float(pos_count)
            precision = float(topk_pos) / float(eff_k)
            relevance = np.clip(metadata_similarity[query_idx, topk], 0.0, 1.0).astype(np.float64)
            ideal = np.sort(np.clip(metadata_similarity[query_idx, pos_mask], 0.0, 1.0).astype(np.float64))[::-1][:eff_k]
            ndcg = _dcg(relevance) / max(_dcg(ideal), 1e-12)
            hard_negative_rate = float(topk_neg) / float(eff_k)

            metrics_per_k[int(k)]["recall"].append(recall)
            metrics_per_k[int(k)]["precision"].append(precision)
            metrics_per_k[int(k)]["ndcg"].append(ndcg)
            metrics_per_k[int(k)]["hard_negative_rate"].append(hard_negative_rate)

    out = {"query_count": int(query_indices.size), "query_metrics": {}}
    for k in ks:
        bucket = metrics_per_k[int(k)]
        recall_arr = np.asarray(bucket["recall"], dtype=np.float64)
        precision_arr = np.asarray(bucket["precision"], dtype=np.float64)
        ndcg_arr = np.asarray(bucket["ndcg"], dtype=np.float64)
        hn_arr = np.asarray(bucket["hard_negative_rate"], dtype=np.float64)

        out[f"recall_at_{int(k)}"] = float(np.mean(recall_arr)) if recall_arr.size else 0.0
        out[f"precision_at_{int(k)}"] = float(np.mean(precision_arr)) if precision_arr.size else 0.0
        out[f"ndcg_at_{int(k)}"] = float(np.mean(ndcg_arr)) if ndcg_arr.size else 0.0
        out[f"hard_negative_rate_at_{int(k)}"] = float(np.mean(hn_arr)) if hn_arr.size else 0.0
        out["query_metrics"][f"k_{int(k)}"] = {
            "recall": recall_arr.tolist(),
            "precision": precision_arr.tolist(),
            "ndcg": ndcg_arr.tolist(),
            "hard_negative_rate": hn_arr.tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer-aligned geology retrieval benchmark (frozen manifest).")
    parser.add_argument("--data", type=Path, required=True, help="Path to sampled zarr dataset containing patches and metadata arrays.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VAE checkpoint used by VaeLatentAdapter.")
    parser.add_argument("--out_json", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmarks/frozen_validation_manifest.json"), help="Frozen benchmark manifest path.")
    parser.add_argument("--benchmark_size", type=int, default=512, help="Number of frozen validation examples to evaluate.")
    parser.add_argument("--seed", type=int, default=20260826, help="Seed used when creating a new manifest.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for tokenizer latent extraction.")
    parser.add_argument("--use_geo_embedding", action="store_true", help="Score the geology projection embedding (z_geo) instead of raw mu. Requires a checkpoint trained with --geology_projection.")
    parser.add_argument("--metadata_keys", nargs="+", default=list(DEFAULT_DERIVED_METADATA_KEYS))
    parser.add_argument("--background_keys", nargs="+", default=list(DEFAULT_BACKGROUND_METADATA_KEYS))
    parser.add_argument("--background_threshold", type=float, default=1e-6)
    parser.add_argument("--diagnostic_topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--positive_threshold", type=float, default=0.70)
    parser.add_argument("--negative_threshold", type=float, default=0.20)
    parser.add_argument("--calibration_path", type=Path, default=None, help="Optional saved calibration artifact. If omitted, uses checkpoint calibration or fit-on-dataset fallback.")
    parser.add_argument("--bootstrap_samples", type=int, default=300)
    args = parser.parse_args()

    root = zarr.open(str(args.data), mode="r")
    if "patches" not in root:
        raise KeyError(f"{args.data} missing 'patches' array.")
    patches = np.asarray(root["patches"])
    if patches.ndim != 4:
        raise ValueError("patches array must be shape [N, X, Y, Z].")

    dataset_size = int(patches.shape[0])
    manifest = _load_or_create_manifest(args.manifest, dataset_size, args.benchmark_size, args.seed)
    indices = np.asarray(manifest["indices"], dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < dataset_size)]
    if indices.size <= 2:
        raise ValueError("Benchmark manifest has too few valid indices.")

    metadata_keys = tuple(str(v) for v in args.metadata_keys)
    metadata_dict = {}
    for key in metadata_keys:
        if key not in root:
            raise KeyError(f"Dataset missing metadata key '{key}'.")
        metadata_dict[key] = np.asarray(root[key])[indices]

    calibration = None
    if args.calibration_path is not None and args.calibration_path.exists():
        calibration = torch.load(args.calibration_path, map_location="cpu")
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict):
            maybe_cal = checkpoint.get("geology_metadata_calibration", None)
            if isinstance(maybe_cal, dict):
                calibration = maybe_cal

    if calibration is None:
        class _CalDs:
            def __init__(self, arrays):
                self._metadata_arrays = arrays

        full_arrays = {key: np.asarray(root[key]) for key in metadata_keys}
        calibration = fit_geology_metadata_calibration(
            _CalDs(full_arrays),
            metadata_keys,
            strategy="robust_log1p",
            eps=1e-6,
            clip=6.0,
            background_keys=args.background_keys,
        )

    background_key_indices = resolve_background_key_indices(metadata_keys, args.background_keys)
    metadata_vectors, has_selected = prepare_geology_metadata_vectors(
        metadata_dict,
        metadata_keys,
        device="cpu",
        dtype=torch.float32,
        geology_metadata_calibration=calibration,
        background_threshold=float(args.background_threshold),
        background_key_indices=background_key_indices,
    )

    adapter = VaeLatentAdapter(args.checkpoint, device=args.device)
    selected_patches = patches[indices]

    use_geo_embedding = bool(args.use_geo_embedding)
    if use_geo_embedding and not getattr(adapter, "geology_projection", False):
        raise ValueError(
            "--use_geo_embedding requires a checkpoint trained with --geology_projection "
            "(no projection head found in the checkpoint)."
        )

    latent_parts = []
    for start in range(0, selected_patches.shape[0], int(args.batch_size)):
        stop = min(start + int(args.batch_size), selected_patches.shape[0])
        batch = selected_patches[start:stop]
        prepped = np.stack([preprocess_for_token(batch[i]) for i in range(batch.shape[0])], axis=0).astype(np.float32)
        if use_geo_embedding:
            latents = adapter.encode_geo_batch(prepped)
        else:
            latents = adapter.encode_batch(prepped)
        latent_parts.append(latents)
    latent_matrix = np.concatenate(latent_parts, axis=0).astype(np.float32, copy=False)

    diagnostics = compute_latent_geology_diagnostics(
        latent_vectors=torch.from_numpy(latent_matrix),
        metadata_vectors=metadata_vectors,
        neighbor_k=min(int(args.diagnostic_topk[0]), 20),
        neighbor_ks=[int(v) for v in args.diagnostic_topk],
        has_selected_geology=has_selected,
    )

    latent_similarity = _cosine_matrix(latent_matrix)
    metadata_similarity = _cosine_matrix(metadata_vectors.detach().cpu().numpy())
    has_selected_np = has_selected.detach().cpu().numpy().astype(bool)

    global_metrics = _compute_ranking_metrics(
        latent_similarity,
        metadata_similarity,
        has_selected_np,
        ks=[int(v) for v in args.diagnostic_topk],
        positive_threshold=float(args.positive_threshold),
        negative_threshold=float(args.negative_threshold),
        query_mask=np.ones((indices.shape[0],), dtype=bool),
    )

    cohort_metrics = {}
    raw_matrix = np.stack([np.asarray(metadata_dict[key], dtype=np.float64) for key in metadata_keys], axis=1)
    for key_idx, key_name in enumerate(metadata_keys):
        present_mask = raw_matrix[:, key_idx] > float(args.background_threshold)
        cohort_metrics[f"feature_present::{key_name}"] = _compute_ranking_metrics(
            latent_similarity,
            metadata_similarity,
            has_selected_np,
            ks=[int(v) for v in args.diagnostic_topk],
            positive_threshold=float(args.positive_threshold),
            negative_threshold=float(args.negative_threshold),
            query_mask=present_mask,
        )

    cohort_metrics["background_only"] = _compute_ranking_metrics(
        latent_similarity,
        metadata_similarity,
        has_selected_np,
        ks=[int(v) for v in args.diagnostic_topk],
        positive_threshold=float(args.positive_threshold),
        negative_threshold=float(args.negative_threshold),
        query_mask=~has_selected_np,
    )

    bootstrap = {}
    query_metrics = global_metrics.get("query_metrics", {})
    for k in [int(v) for v in args.diagnostic_topk]:
        bucket = query_metrics.get(f"k_{int(k)}", {})
        for metric_name in ("recall", "precision", "ndcg", "hard_negative_rate"):
            values = np.asarray(bucket.get(metric_name, []), dtype=np.float64)
            low, high = _bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed) + int(k))
            bootstrap[f"{metric_name}_at_{int(k)}"] = {
                "mean": float(np.mean(values)) if values.size else 0.0,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "n": int(values.size),
            }

    report = {
        "schema_version": 1,
        "dataset": str(args.data),
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_seed": int(manifest.get("seed", args.seed)),
        "benchmark_size": int(indices.shape[0]),
        "embedding_source": "z_geo" if use_geo_embedding else "mu",
        "metadata_keys": list(metadata_keys),
        "background_keys": list(args.background_keys),
        "background_threshold": float(args.background_threshold),
        "positive_threshold": float(args.positive_threshold),
        "negative_threshold": float(args.negative_threshold),
        "diagnostics": diagnostics,
        "global_metrics": {k: v for k, v in global_metrics.items() if k != "query_metrics"},
        "bootstrap": bootstrap,
        "cohort_metrics": {name: {k: v for k, v in payload.items() if k != "query_metrics"} for name, payload in cohort_metrics.items()},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote benchmark report: {args.out_json}")


if __name__ == "__main__":
    main()

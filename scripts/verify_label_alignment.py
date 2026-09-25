#!/usr/bin/env python3

"""Measure the depth offset between label arrays and the seismic cube (WP0).

Convention: seismic[..., z] corresponds to label[..., z + offset].

Reflection strength is taken as the envelope of the depth derivative of the cumsum
seismic, and boundary strength as |d/dz| of the label array. The offset with peak
Pearson correlation is picked per volume and across volumes.

Usage:
    python scripts/verify_label_alignment.py --source /Volumes/CrucialX9/fake_data \
        --n_volumes 8 --out_json data/wp0_label_alignment.json
"""

from pathlib import Path
import sys

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import argparse
import json
import random

import numpy as np
import zarr


def analytic_envelope(x, axis=-1):
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = x.shape[-1]
    spec = np.fft.fft(x, axis=-1)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    env = np.abs(np.fft.ifft(spec * h, axis=-1))
    return np.moveaxis(env, -1, axis)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def offset_correlation_curves(seismic_traces, label_traces, offsets):
    """Correlation per offset for (N, S) seismic and (N, L) label traces.

    Returns dict with 'envelope' (envelope(d seismic) vs |d label|) and
    'signed_diff' (|corr(d seismic, d label)|) curves.
    """
    seis = np.asarray(seismic_traces, dtype=np.float64)
    lab = np.asarray(label_traces, dtype=np.float64)
    ds = np.diff(seis, axis=-1)
    dl = np.diff(lab, axis=-1)
    env = analytic_envelope(ds, axis=-1)
    bnd = np.abs(dl)
    n_s, n_l = ds.shape[-1], dl.shape[-1]
    offsets = list(offsets)
    z0 = max(0, -min(offsets))
    z1 = min(n_s, n_l - max(offsets))
    if z1 - z0 < 8:
        raise ValueError("Offset range leaves too few overlapping depth samples")
    env_c = env[:, z0:z1].ravel()
    ds_c = ds[:, z0:z1].ravel()
    curves = {"envelope": [], "signed_diff": []}
    for off in offsets:
        curves["envelope"].append(_pearson(env_c, bnd[:, z0 + off : z1 + off].ravel()))
        curves["signed_diff"].append(abs(_pearson(ds_c, dl[:, z0 + off : z1 + off].ravel())))
    return {k: np.asarray(v) for k, v in curves.items()}


def refine_peak(curve, offsets):
    """Integer argmax plus parabolic sub-sample refinement."""
    curve = np.asarray(curve, dtype=np.float64)
    i = int(np.argmax(curve))
    best = int(offsets[i])
    if 0 < i < len(curve) - 1:
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return best, best + float(np.clip(frac, -0.5, 0.5))
    return best, float(best)


def read_traces(zvol, seismic_key, label_key, n_blocks, trace_stride, rng):
    seis_arr = zvol[seismic_key]
    lab_arr = zvol[label_key]
    nx, ny = seis_arr.shape[0], seis_arr.shape[1]
    cx = min(seis_arr.chunks[0], nx)
    cy = min(seis_arr.chunks[1], ny)
    origins = [(x, y) for x in range(0, nx - cx + 1, cx) for y in range(0, ny - cy + 1, cy)]
    rng.shuffle(origins)
    seis_list, lab_list = [], []
    for x0, y0 in origins[:n_blocks]:
        sl = (slice(x0, x0 + cx, trace_stride), slice(y0, y0 + cy, trace_stride), slice(None))
        s = np.asarray(seis_arr[sl], dtype=np.float32)
        l = np.asarray(lab_arr[sl], dtype=np.float32)
        seis_list.append(s.reshape(-1, s.shape[-1]))
        lab_list.append(l.reshape(-1, l.shape[-1]))
    return np.concatenate(seis_list), np.concatenate(lab_list)


def measure_volume(zvol, seismic_key, label_key, offsets, n_blocks=3, trace_stride=5, rng=None):
    rng = rng or random.Random(0)
    seis, lab = read_traces(zvol, seismic_key, label_key, n_blocks, trace_stride, rng)
    curves = offset_correlation_curves(seis, lab, offsets)
    result = {
        "seismic_depth": int(seis.shape[-1]),
        "label_depth": int(lab.shape[-1]),
        "n_traces": int(seis.shape[0]),
    }
    for name, curve in curves.items():
        best, refined = refine_peak(curve, offsets)
        result[name] = {
            "best_offset": best,
            "refined_offset": refined,
            "peak_corr": float(curve.max()),
            "curve": [float(v) for v in curve],
        }
    return result


def summarize(per_volume, offsets, tolerance=1):
    summary = {}
    for name in ("envelope", "signed_diff"):
        bests = [v[name]["best_offset"] for v in per_volume]
        mean_curve = np.mean([v[name]["curve"] for v in per_volume], axis=0)
        best, refined = refine_peak(mean_curve, offsets)
        agree = sum(abs(b - best) <= tolerance for b in bests)
        summary[name] = {
            "per_volume_best": bests,
            "mean_curve_best_offset": best,
            "mean_curve_refined_offset": refined,
            "n_agree_within_tolerance": int(agree),
            "mean_curve": [float(v) for v in mean_curve],
        }
    env_best = summary["envelope"]["mean_curve_best_offset"]
    sd_best = summary["signed_diff"]["mean_curve_best_offset"]
    n = len(per_volume)
    summary["recommended_label_z_offset"] = int(env_best)
    summary["tolerance"] = tolerance
    summary["verified"] = bool(
        n >= 5
        and abs(env_best - sd_best) <= tolerance
        and summary["envelope"]["n_agree_within_tolerance"] >= int(np.ceil(0.8 * n))
    )
    return summary


def list_volumes(source):
    src = Path(source)
    if (src / "zarr.json").exists() or (src / ".zgroup").exists():
        return [src]
    return sorted(src.rglob("model_data.zarr"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="directory containing model_data.zarr stores")
    p.add_argument("--n_volumes", type=int, default=8)
    p.add_argument("--seismic_key", default="seismicCubes_cumsum_fullstack")
    p.add_argument("--label_key", default="faulted_lithology")
    p.add_argument("--offset_range", type=int, default=15)
    p.add_argument("--n_blocks", type=int, default=3, help="chunk-aligned XY blocks read per volume")
    p.add_argument("--trace_stride", type=int, default=5, help="XY stride inside each block")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    offsets = list(range(-args.offset_range, args.offset_range + 1))
    rng = random.Random(args.seed)
    vols = list_volumes(args.source)
    if not vols:
        raise SystemExit(f"No model_data.zarr found under {args.source}")
    rng.shuffle(vols)

    per_volume = []
    for vol in vols:
        if len(per_volume) >= args.n_volumes:
            break
        zvol = zarr.open_group(str(vol), mode="r")
        if args.seismic_key not in zvol or args.label_key not in zvol:
            print(f"skip (missing keys): {vol}")
            continue
        res = measure_volume(zvol, args.seismic_key, args.label_key, offsets, args.n_blocks, args.trace_stride, rng)
        res["volume"] = str(vol)
        per_volume.append(res)
        print(
            f"{vol.parent.name}: depth seis={res['seismic_depth']} label={res['label_depth']} "
            f"envelope best={res['envelope']['best_offset']} ({res['envelope']['refined_offset']:+.2f}, "
            f"r={res['envelope']['peak_corr']:.3f}) signed_diff best={res['signed_diff']['best_offset']} "
            f"(r={res['signed_diff']['peak_corr']:.3f})"
        )

    if not per_volume:
        raise SystemExit("No usable volumes")
    summary = summarize(per_volume, offsets)
    print(
        f"recommended label_z_offset={summary['recommended_label_z_offset']} "
        f"(envelope refined {summary['envelope']['mean_curve_refined_offset']:+.2f}, "
        f"signed_diff refined {summary['signed_diff']['mean_curve_refined_offset']:+.2f}, "
        f"agree {summary['envelope']['n_agree_within_tolerance']}/{len(per_volume)}) verified={summary['verified']}"
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "convention": "seismic[..., z] <-> label[..., z + offset]",
            "seismic_key": args.seismic_key,
            "label_key": args.label_key,
            "offsets": offsets,
            "seed": args.seed,
            "summary": summary,
            "per_volume": per_volume,
        }
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Reproduce historical fluid mixing on tiny fabricated arrays, without source data.

Only the fingerprinted historical property method is executed. Rock-property
endmembers, depth shifts and postprocessing are manufactured to isolate mixing;
this is not a reconstruction of stored seismic or a generator modification.
"""

import argparse
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from scripts.streamed_fluid_labels import VERIFIED_SEISMIC_FILES
from scripts.streamed_overlap_audit import write_json


class FixtureLazy:
    def __init__(self, values: np.ndarray):
        self.values = values

    def __add__(self, value: float):
        return FixtureLazy(self.values + value)

    def astype(self, dtype):
        return FixtureLazy(self.values.astype(dtype))

    def compute(self, **_kwargs) -> np.ndarray:
        return self.values


class FixtureStorage:
    def __init__(self, values: dict[str, np.ndarray]):
        self.store = values

    def get_dataset(self, key: str, **kwargs):
        array = self.store[key]
        return FixtureLazy(array) if kwargs.get("use_dask") else array

    def create_dataset(self, key: str, array: np.ndarray) -> None:
        self.store[key] = np.asarray(array)


class FixtureRPM:
    def properties(self, value: float, *depths: np.ndarray) -> SimpleNamespace:
        array = np.full_like(depths[0], value, dtype="f4")
        return SimpleNamespace(rho=array, vp=array * 1000, vs=array * 500)

    def calc_shale_properties(self, *depths: np.ndarray) -> SimpleNamespace:
        return self.properties(2.5, *depths)

    def calc_brine_sand_properties(self, *depths: np.ndarray) -> SimpleNamespace:
        return self.properties(2.0, *depths)

    def calc_oil_sand_properties(self, *depths: np.ndarray) -> SimpleNamespace:
        return self.properties(1.5, *depths)

    def calc_gas_sand_properties(self, *depths: np.ndarray) -> SimpleNamespace:
        return self.properties(1.0, *depths)


def reproduce_method(source: str, sha: str) -> dict:
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    if VERIFIED_SEISMIC_FILES.get(sha) != fingerprint:
        raise ValueError(f"Refusing execution: unexpected historical source fingerprint for {sha}")
    functions = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_property_models_randomised_depth"
    ]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one historical property method in {sha}")
    method = functions[0]
    namespace = {
        "np": np, "perf_counter": perf_counter,
        "ElasticProperties3D": SimpleNamespace, "_DASK_NUM_WORKERS": 1,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 f"{sha}:datagenerator/Seismic.py", "exec"), namespace)
    shape = (6, 1, 16)
    lithology = np.ones(shape, dtype="f4")
    depth = np.ones(shape, dtype="f4")
    age = np.ones(shape, dtype="f4")
    age[:, :, -1] = 2
    oil = np.zeros(shape, dtype="u1")
    gas = np.zeros(shape, dtype="u1")
    oil[[1, 2, 4, 5]] = 1
    gas[[3, 5]] = 1
    channels = np.zeros(shape, dtype="f4")
    channels[[2, 3]] = 1
    channels[4] = 2
    storage = FixtureStorage({
        "oil_closures": oil, "gas_closures": gas, "geologic_age_faulted": age,
        "faults/faulted_channel_labels": channels,
        "faults/faulted_channel_ng": np.full(shape, .4, dtype="f4"),
    })
    cfg = SimpleNamespace(
        rpm_scaling_factors={"layershiftsamples": 0, "RPshiftsamples": 0},
        noise_seed=None, include_channels=True, include_salt=False, verbose=False,
        qc_plots=False, model_qc_volumes=False, cube_shape=shape, storage=storage,
    )
    instance = SimpleNamespace(
        cfg=cfg,
        faults=SimpleNamespace(
            faulted_lithology=lithology, faulted_net_to_gross=np.ones(shape, dtype="f4"),
            faulted_depth=depth,
        ),
        water_properties=lambda _lith, properties: properties,
        get_delta_z_layer=lambda *_args: 0,
        get_delta_z_properties=lambda *_args: (0, 0, 0),
        fix_zero_values_at_base=lambda properties: properties,
        apply_scaling_factors=lambda properties, *_args: properties,
    )
    for key in ("rho", "vp", "vs", "rho_ff", "vp_ff", "vs_ff"):
        setattr(instance, key, np.zeros(shape, dtype="f4"))
    with contextlib.redirect_stdout(io.StringIO()):
        namespace[method.name](instance, FixtureRPM(), mixing_method="inv_vel")
    np.testing.assert_allclose(
        instance.rho[:, 0, 5], [2, 1.5, 1.95, 1.9, 1.7, 2.5], atol=1e-6,
    )
    cases = (
        "unclosed_sand", "oil_sand", "oil_channel",
        "gas_channel", "oil_levee", "conflicting_oil_gas",
    )
    return {
        "generator_sha": sha,
        "seismic_file_sha256": fingerprint,
        "method_lines": [method.lineno, method.end_lineno],
        "artificial_rho_results": dict(zip(cases, instance.rho[:, 0, 5].tolist())),
        "channel_fill_is_not_clean_oil_or_gas": True,
        "late_channel_mixing_retains_prior_property_contribution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-repo", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing existing evidence: {args.output}")
    manifest = json.loads(args.source_manifest.read_text())
    counts = Counter(
        record["parameters"]["generator_sha"]
        for record in manifest["sources"] if record["parameters"] is not None
    )
    unknown = set(counts) - set(VERIFIED_SEISMIC_FILES)
    if unknown:
        raise ValueError(f"Historical generator versions require inspection before execution: {sorted(unknown)}")
    evidence = []
    for sha in sorted(counts):
        source = subprocess.check_output([
            "git", "-C", str(args.generator_repo),
            "show", f"{sha}:datagenerator/Seismic.py",
        ], text=True)
        result = reproduce_method(source, sha)
        result["source_count"] = counts[sha]
        evidence.append(result)
        print(f"VERIFIED historical method {sha}: {result['artificial_rho_results']}", flush=True)
    report = {
        "status": "HISTORICAL_RULES_REPRODUCED_TARGET_APPROVAL_PENDING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "generator_repo": str(args.generator_repo.resolve()),
        "source_cohorts": evidence,
        "limitations": [
            "Manufactured RPM endmembers and identity postprocessing isolate the historical mixing method.",
            "No real source crops, waveform reconstruction, training or generator/data changes.",
            "Recorded source SHA does not establish a clean generator worktree at the time of generation.",
            "Stored segmentation sources do not retain all elastic properties or channel net-to-gross.",
            "No target policy or launch approval is implied.",
        ],
    }
    write_json(args.output, report)
    print(f"Historical fluid evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()

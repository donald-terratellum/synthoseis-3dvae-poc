"""Provisional voxel fluid diagnostics, not approved common-layer targets."""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np

from scripts.sample_patches import DEFAULT_SAND_THRESHOLD


CHANNEL_KEY = "faults/faulted_channel_labels"
FLUID_KEYS = ("gas_closures", "oil_closures", "brine_closures")

# Fingerprints of the exact historical files inspected and exercised locally.
VERIFIED_SEISMIC_FILES = {
    "9e37d59e537d96492302c68433d99810626da426":
        "dcbe9a64aeceaddf0a3b824dfc115a2e527e620d5c46acfcd6b63581d14a8e75",
    "bd51313ebfb620800bb2973c2cea28f4b9325a15":
        "3b1b19d361cb54287889bdcf2473a2fb279df37fe3cdbc77d8d5bb42a7472668",
    "5623f9fadabe576635b3596cd9818ed32e0c32f5":
        "dcbe9a64aeceaddf0a3b824dfc115a2e527e620d5c46acfcd6b63581d14a8e75",
}


class FluidCode(IntEnum):
    NOT_SAND = 0
    UNRESOLVED = 1
    BRINE = 2
    OIL = 3
    GAS = 4


class FluidReason(IntFlag):
    MALFORMED_CLOSURE = 1
    CONFLICTING_CLOSURES = 2
    CLOSURE_OUTSIDE_SAND = 4
    CHANNEL_HYDROCARBON_SYNTHESIS = 8
    UNVERIFIED_GENERATOR = 16
    UNPROVEN_MODELED_DOMAIN = 32


def lithology_masks(
    lithology: np.ndarray, channel_labels: np.ndarray
) -> dict[str, np.ndarray]:
    if lithology.shape != channel_labels.shape or lithology.ndim != 3:
        raise ValueError("Lithology and channel labels must have matching 3D shapes.")
    if not np.isfinite(lithology).all() or not np.isfinite(channel_labels).all():
        raise ValueError("Lithology and channel labels must be finite.")
    channel = channel_labels > 0.5
    raw_sand = lithology >= DEFAULT_SAND_THRESHOLD
    sand = raw_sand | channel
    rock = (lithology >= 0) | channel
    return {
        "channel": channel,
        "raw_sand": raw_sand,
        "sand": sand,
        "rock": rock,
        "shale": rock & ~sand,
        "water": ~rock,
    }


@dataclass(frozen=True)
class FluidInspection:
    intended_code: np.ndarray
    reason_bits: np.ndarray
    provisional_valid: np.ndarray

    def summary(self) -> dict:
        return {
            "status": "PROVISIONAL_FLUID_DIAGNOSTICS_NOT_TARGET_APPROVAL",
            "intended_voxels_by_fluid": {
                code.name.lower(): int(np.count_nonzero(self.intended_code == code))
                for code in FluidCode
            },
            "provisionally_consistent_sand_voxels": int(np.count_nonzero(self.provisional_valid)),
            "reason_voxel_counts": {
                reason.name.lower(): int(np.count_nonzero(self.reason_bits & reason))
                for reason in FluidReason
            },
            "reason_bits_may_overlap": True,
            "does_not_drop_layers_or_change_denominator": True,
        }


def inspect_fluid_fill(
    crop: dict[str, np.ndarray],
    generator_sha: str | None,
    observed_max_integer_age: int | None,
) -> FluidInspection:
    """Keep intended fill separate from historical synthesis-consistency flags.

    A bounded probe's maximum integer age is a lower bound on the source-global
    maximum used by the generator. Ages strictly below it are inside that upper
    boundary; a crop-local maximum alone would not establish this.
    """
    lithology = crop["faulted_lithology"]
    facies = lithology_masks(lithology, crop[CHANNEL_KEY])
    age = crop["geologic_age_faulted"]
    arrays = [crop[key] for key in FLUID_KEYS]
    if any(array.shape != lithology.shape for array in (age, *arrays)):
        raise ValueError("Age, closure and lithology arrays must have matching shapes.")
    if not np.isfinite(age).all():
        raise ValueError("Geologic age must be finite.")
    if observed_max_integer_age is not None and (
        isinstance(observed_max_integer_age, bool)
        or not isinstance(observed_max_integer_age, int)
        or observed_max_integer_age < 1
    ):
        raise ValueError("Observed maximum integer age must be a positive integer.")
    masks = np.stack([array == 1 for array in arrays])
    malformed = np.stack([~np.isin(array, (0, 1)) for array in arrays]).any(axis=0)
    conflicts = masks.sum(axis=0) > 1
    sand = facies["sand"]
    codes = np.full(lithology.shape, FluidCode.NOT_SAND, dtype=np.uint8)
    codes[sand] = FluidCode.UNRESOLVED
    reasons = np.zeros(lithology.shape, dtype=np.uint8)
    verified = generator_sha in VERIFIED_SEISMIC_FILES
    if verified:
        codes[sand & ~masks.any(axis=0)] = FluidCode.BRINE
    else:
        reasons[sand] |= np.uint8(int(FluidReason.UNVERIFIED_GENERATOR))
    for mask, code in zip(masks, (FluidCode.GAS, FluidCode.OIL, FluidCode.BRINE)):
        codes[sand & mask] = code
    reasons[malformed] |= np.uint8(int(FluidReason.MALFORMED_CLOSURE))
    reasons[conflicts] |= np.uint8(int(FluidReason.CONFLICTING_CLOSURES))
    reasons[masks.any(axis=0) & ~sand] |= np.uint8(int(FluidReason.CLOSURE_OUTSIDE_SAND))
    codes[sand & (malformed | conflicts)] = FluidCode.UNRESOLVED
    if verified:
        inconsistent = facies["channel"] & (masks[0] | masks[1])
        reasons[inconsistent] |= np.uint8(int(FluidReason.CHANNEL_HYDROCARBON_SYNTHESIS))
    if observed_max_integer_age is None:
        reasons[sand] |= np.uint8(int(FluidReason.UNPROVEN_MODELED_DOMAIN))
    else:
        integer_age = (age + 0.00001).astype(np.int64)
        outside = (integer_age < 1) | (integer_age >= observed_max_integer_age)
        reasons[sand & outside] |= np.uint8(int(FluidReason.UNPROVEN_MODELED_DOMAIN))
    valid = sand & (codes != FluidCode.UNRESOLVED) & (reasons == 0)
    return FluidInspection(codes, reasons, valid)

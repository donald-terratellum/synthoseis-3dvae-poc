import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import secrets
import sys
from typing import Any, cast, Optional, Sequence
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import itertools
import importlib
import math
import csv
import time
import re
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import zarr
import numpy as np
from src.augmentations import apply_input_trace_dropout
from src.augmentations import apply_input_extrema_mixup
from src.augmentations import apply_input_decimate_trilinear
from src.augmentations import apply_input_random_sparse_keep
from src.augmentations import apply_pair_augmentations
from src.augmentations import keep_trace_extrema_only
from src.augmentations import sample_mixup_corpus_index
from src.deep_supervision import DeepSupervisionLoss
from src.geology_label_augment import (
    AZIMUTH_KEY,
    AZIMUTH_VARIANCE_KEY,
    DIP_CLASS_KEYS,
    DIP_DEGREE_KEYS,
    DIP_DEPENDENT_KEYS,
    DIP_IGNORE_CLASS,
    DIP_LABEL_POLICIES,
    DIP_MEAN_CLASS_EDGES_DEG,
    DIP_RANGE_CLASS_EDGES_DEG,
    DIP_SAMPLE_KEYS,
    adjust_azimuth_deg,
    adjusted_dip_values_for_stretch,
    vertical_warp_stretch_at_source,
)
from src.geology_classifier import (
    ALL_CLASSIFIER_TARGETS,
    PRESENCE_TARGET_KEYS,
    classifier_target_keys,
    compute_geology_classifier_loss,
    compute_pos_weight,
)
from src.geology_sampler import (
    GeologyAwareBatchSampler,
    build_multilabel_strata,
    build_presence_strata,
    presence_rarity_weights,
)
from src.model import GEOLOGY_PRESENCE_CLASSES, VAE3D
from src.tokenizer.core.preprocess import preprocess_for_token

try:
    lpips_lib = importlib.import_module('lpips')
except ImportError:
    lpips_lib = None


DEFAULT_DERIVED_METADATA_KEYS = (
    'meta_dip_mean_deg',
    'meta_dip_std_deg',
    'meta_azimuth_mean_deg',
    'meta_azimuth_circular_variance',
    'meta_fault_fraction',
    'meta_fault_intersection_fraction',
    'meta_geologic_score_mean',
    'meta_sand_fraction',
    'meta_shale_fraction',
    'meta_flat_spot_fraction',
    'meta_onlap_fraction',
    'meta_onlap_variability',
    'meta_channel_fraction',
    'meta_channel_core_fraction',
    'meta_structural_complexity',
)


DEFAULT_BACKGROUND_METADATA_KEYS = (
    'meta_fault_fraction',
    'meta_fault_intersection_fraction',
    'meta_flat_spot_fraction',
    'meta_onlap_fraction',
    'meta_channel_fraction',
    'meta_channel_core_fraction',
)


def normalize_patch_size(values):
    if len(values) == 1:
        v = int(values[0])
        dims = (v, v, v)
    elif len(values) == 3:
        dims = tuple(int(v) for v in values)
    else:
        raise ValueError("--patch_size expects either 1 value or 3 values: X Y Z")
    if any(v <= 0 for v in dims):
        raise ValueError("patch_size values must be positive")
    if any(v % 8 != 0 for v in dims):
        raise ValueError("patch_size values must be divisible by 8 for VAE3D")
    return dims


def resolve_patch_size_xyz(requested_values, dataset_patch_shape):
    dataset_shape = tuple(int(v) for v in dataset_patch_shape)
    if requested_values is None:
        return dataset_shape
    requested_shape = normalize_patch_size(requested_values)
    if requested_shape != dataset_shape:
        raise ValueError(
            f"training patch shape {dataset_shape} does not match --patch_size {requested_shape}"
        )
    return requested_shape


class ZarrPatchDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        scaling='none',
        scaling_mean=0.0,
        scaling_std=1.0,
        augment=False,
        swap_xy_prob=0.5,
        flip_x_prob=0.5,
        flip_y_prob=0.5,
        vertical_warp_prob=0.5,
        phase_rotation_prob=0.0,
        phase_range=(-60.0, 0.0, 40.0),
        stretch_prob=0.0,
        stretch_xy=(1.0, 1.25),
        stretch_z=(1.0, 1.5),
        zero_cluster_min=8,
        zero_cluster_max=12,
        extrema_only: Optional[bool] = None,
        input_extrema_prob=1.0,
        input_sparse_keep_prob=0.0,
        input_decimate_trilinear_prob=0.0,
        sparse_keep_fraction_min=0.10,
        sparse_keep_fraction_max=0.30,
        sparse_poisson_radius_scale=0.85,
        mixup_augment_prob=0.10,
        include_metadata: bool = False,
        geology_metadata_keys: Optional[tuple[str, ...]] = None,
        label_target_keys: Optional[tuple[str, ...]] = None,
        dip_label_policy: str = 'adjust',
    ):
        z = cast(Any, zarr.open(str(zarr_path), mode='r'))
        self.data = cast(Any, z['patches'])
        if len(self.data.shape) != 4:
            raise ValueError("zarr patches array must have shape [N, X, Y, Z]")
        self.patch_shape = tuple(int(v) for v in self.data.shape[1:4])
        self.num_examples = int(self.data.shape[0])
        self.scaling = scaling
        self.scaling_mean = float(scaling_mean)
        self.scaling_std = float(scaling_std)
        self.augment = bool(augment)
        self.swap_xy_prob = float(swap_xy_prob)
        self.flip_x_prob = float(flip_x_prob)
        self.flip_y_prob = float(flip_y_prob)
        self.vertical_warp_prob = float(vertical_warp_prob)
        self.phase_rotation_prob = float(phase_rotation_prob)
        self.phase_range = tuple(float(v) for v in phase_range)
        self.stretch_prob = float(stretch_prob)
        self.stretch_xy = tuple(float(v) for v in stretch_xy)
        self.stretch_z = tuple(float(v) for v in stretch_z)
        self.zero_cluster_min = int(zero_cluster_min)
        self.zero_cluster_max = int(zero_cluster_max)
        self.extrema_only = None if extrema_only is None else bool(extrema_only)
        self.input_extrema_prob = float(input_extrema_prob)
        self.input_sparse_keep_prob = float(input_sparse_keep_prob)
        self.input_decimate_trilinear_prob = float(input_decimate_trilinear_prob)
        self.sparse_keep_fraction_min = float(sparse_keep_fraction_min)
        self.sparse_keep_fraction_max = float(sparse_keep_fraction_max)
        self.sparse_poisson_radius_scale = float(sparse_poisson_radius_scale)
        self.mixup_augment_prob = float(mixup_augment_prob)
        self.include_metadata = bool(include_metadata)
        self.geology_metadata_keys = tuple(str(key) for key in (geology_metadata_keys or ()))
        self._metadata_arrays = {}

        if self.include_metadata:
            for key in self.geology_metadata_keys:
                if key not in z:
                    available_keys = tuple(str(value) for value in z.attrs.get('derived_metadata_keys', ()))
                    raise KeyError(
                        f"Required geology metadata key '{key}' was not found in dataset '{zarr_path}'. "
                        f"Available derived metadata keys: {available_keys}. Regenerate this dataset with "
                        "scripts/sample_patches.py after updating the derived metadata schema."
                    )
                self._metadata_arrays[key] = z[key]

        self.label_target_keys = tuple(str(key) for key in (label_target_keys or ()))
        self._label_arrays = {}
        for key in self.label_target_keys:
            if key not in z:
                raise KeyError(
                    f"Label target '{key}' was not found in dataset '{zarr_path}'. "
                    "Regenerate it with scripts/sample_patches.py (label_presence_* arrays are written by the "
                    "WP2 sampler in every --sampling_mode)."
                )
            self._label_arrays[key] = np.asarray(z[key][:])
        if self._label_arrays:
            self.include_metadata = True

        self.dip_label_policy = str(dip_label_policy)
        if self.dip_label_policy not in DIP_LABEL_POLICIES:
            raise ValueError(f"--dip_label_policy must be one of {DIP_LABEL_POLICIES}.")
        loaded_keys = set(self._metadata_arrays) | set(self._label_arrays)
        self._dip_keys_loaded = tuple(key for key in DIP_DEPENDENT_KEYS if key in loaded_keys)
        self._adjust_azimuth = bool(
            self.augment
            and AZIMUTH_KEY in loaded_keys
            and max(self.swap_xy_prob, self.flip_x_prob, self.flip_y_prob) > 0.0
        )
        self._azimuth_variance = (
            np.asarray(z[AZIMUTH_VARIANCE_KEY][:]) if self._adjust_azimuth and AZIMUTH_VARIANCE_KEY in z else None
        )
        self._dip_source = {}
        self._dip_mean_edges = tuple(float(v) for v in z.attrs.get('dip_mean_class_edges_deg', DIP_MEAN_CLASS_EDGES_DEG))
        self._dip_range_edges = tuple(float(v) for v in z.attrs.get('dip_range_class_edges_deg', DIP_RANGE_CLASS_EDGES_DEG))
        if (
            self.dip_label_policy == 'adjust'
            and self.augment
            and max(float(vertical_warp_prob), float(stretch_prob)) > 0.0
            and self._dip_keys_loaded
        ):
            missing = [key for key in DIP_SAMPLE_KEYS + DIP_DEGREE_KEYS if key not in z]
            if missing:
                raise KeyError(
                    f"Dataset '{zarr_path}' lacks {missing}, needed to adjust {self._dip_keys_loaded} for depth "
                    "warp/stretch. Run scripts/add_dip_samples.py on it, or pass --dip_label_policy mask (drop "
                    "dip-class targets for transformed samples) or ignore (old behavior)."
                )
            source_keys = DIP_SAMPLE_KEYS + DIP_DEGREE_KEYS
            if 'meta_structural_complexity' in self._dip_keys_loaded:
                source_keys += ('meta_structural_complexity',)
            self._dip_source = {key: np.asarray(z[key][:]) for key in source_keys}

        if self.scaling not in {'none', 'divide_by_std', 'zscore'}:
            raise ValueError("--input_scaling must be one of: none, divide_by_std, zscore")
        if self.scaling != 'none' and abs(self.scaling_std) <= 0.0:
            raise ValueError('--input_std must be non-zero when input scaling is enabled.')
        if self.zero_cluster_min < 0 or self.zero_cluster_max < 0:
            raise ValueError('--zero_cluster_min and --zero_cluster_max must be non-negative.')
        if self.zero_cluster_min > self.zero_cluster_max:
            raise ValueError('--zero_cluster_min must be <= --zero_cluster_max.')
        if not 0.0 <= self.vertical_warp_prob <= 1.0:
            raise ValueError('--vertical_warp_prob must be in [0, 1].')
        if not 0.0 <= self.phase_rotation_prob <= 1.0:
            raise ValueError('--phase_rotation_prob must be in [0, 1].')
        phase_min, phase_mode, phase_max = self.phase_range
        if not (phase_min <= phase_mode <= phase_max):
            raise ValueError('--phase_range must be ordered as (min, mode, max) with min <= mode <= max.')
        if not 0.0 <= self.stretch_prob <= 1.0:
            raise ValueError('--stretch_prob must be in [0, 1].')
        if not (1.0 <= self.stretch_xy[0] <= self.stretch_xy[1]):
            raise ValueError('--stretch_xy must be ordered MIN MAX with both values >= 1.')
        if not (1.0 <= self.stretch_z[0] <= self.stretch_z[1]):
            raise ValueError('--stretch_z must be ordered MIN MAX with both values >= 1.')
        if self.sparse_keep_fraction_min < 0.01 or self.sparse_keep_fraction_max > 1.0:
            raise ValueError('--sparse_keep_fraction_min/max must be in [0.01, 1.0].')
        if self.sparse_keep_fraction_min > self.sparse_keep_fraction_max:
            raise ValueError('--sparse_keep_fraction_min must be <= --sparse_keep_fraction_max.')
        if self.sparse_poisson_radius_scale < 0.1 or self.sparse_poisson_radius_scale > 2.0:
            raise ValueError('--sparse_poisson_radius_scale must be in [0.1, 2.0].')
        if not 0.0 <= self.input_extrema_prob <= 1.0:
            raise ValueError('--input_extrema_prob must be in [0, 1].')
        if not 0.0 <= self.input_sparse_keep_prob <= 1.0:
            raise ValueError('--input_sparse_keep_prob must be in [0, 1].')
        if not 0.0 <= self.input_decimate_trilinear_prob <= 1.0:
            raise ValueError('--input_decimate_trilinear_prob must be in [0, 1].')
        if self.extrema_only is None:
            prob_sum = self.input_extrema_prob + self.input_sparse_keep_prob + self.input_decimate_trilinear_prob
            if prob_sum <= 0.0:
                raise ValueError('At least one input transform probability must be > 0.')
        else:
            default_prob_tuple = (1.0, 0.0, 0.0)
            actual_prob_tuple = (
                self.input_extrema_prob,
                self.input_sparse_keep_prob,
                self.input_decimate_trilinear_prob,
            )
            if self.extrema_only and actual_prob_tuple != default_prob_tuple:
                raise ValueError(
                    'extrema_only=True cannot be combined with non-default input transform probabilities. '
                    'Use probability controls only: --input_extrema_prob, --input_sparse_keep_prob, '
                    '--input_decimate_trilinear_prob.'
                )
        if not 0.0 <= self.mixup_augment_prob <= 1.0:
            raise ValueError('--mixup_augment_prob must be in [0, 1].')

    def _apply_one_of_three_input_transform(self, x):
        probs = np.array(
            [
                self.input_extrema_prob,
                self.input_sparse_keep_prob,
                self.input_decimate_trilinear_prob,
            ],
            dtype=np.float64,
        )
        positive_mask = probs > 0.0
        transform_choices = np.where(positive_mask)[0]
        choice_weights = probs[positive_mask]
        choice_weights = choice_weights / float(choice_weights.sum())
        selected_idx = int(np.random.choice(transform_choices, p=choice_weights))

        if selected_idx == 0:
            return keep_trace_extrema_only(x)
        if selected_idx == 1:
            return apply_input_random_sparse_keep(
                x,
                fraction_min=self.sparse_keep_fraction_min,
                fraction_max=self.sparse_keep_fraction_max,
                method='random',
                poisson_radius_scale=self.sparse_poisson_radius_scale,
            )
        return apply_input_decimate_trilinear(x)

    def _apply_scaling(self, arr):
        if self.scaling == 'divide_by_std':
            return arr / self.scaling_std
        if self.scaling == 'zscore':
            return (arr - self.scaling_mean) / self.scaling_std
        return arr

    def _load_scaled_example(self, idx):
        arr = self.data[idx]
        arr = np.asarray(arr, dtype='f4')
        return self._apply_scaling(arr)

    def __len__(self):
        return self.num_examples

    def _read_metadata_for_index(self, idx):
        if not self.include_metadata:
            return {}

        metadata = {}
        for key, arr in self._metadata_arrays.items():
            value = np.asarray(arr)
            if value.ndim == 0:
                metadata[key] = float(value)
            elif value.shape[0] == self.num_examples:
                metadata[key] = value[int(idx)]
            elif value.shape[0] == 1:
                metadata[key] = value[0]
            else:
                metadata[key] = value
        for key, arr in self._label_arrays.items():
            metadata[key] = arr[int(idx)]
        return metadata

    def _adjust_geometric_labels(self, metadata, idx, params):
        """Update azimuth for x/y flips and swaps, and dip-dependent labels for vertical warp, in place."""
        def _like(original, value):
            return np.asarray(value, dtype=np.asarray(original).dtype)[()]

        reflected = params['swap_xy'] or params['flip_x'] or params['flip_y']
        if self._adjust_azimuth and reflected and AZIMUTH_KEY in metadata:
            # Circular variance 1 means no valid gradient; the stored 0 deg azimuth is a placeholder.
            if self._azimuth_variance is None or float(self._azimuth_variance[idx]) < 1.0 - 1e-6:
                metadata[AZIMUTH_KEY] = _like(
                    metadata[AZIMUTH_KEY],
                    adjust_azimuth_deg(metadata[AZIMUTH_KEY], params['swap_xy'], params['flip_x'], params['flip_y']),
                )

        warp = params['vertical_warp_indices']
        stretch_factors = params['stretch_factors']
        if (warp is None and stretch_factors is None) or not self._dip_keys_loaded or self.dip_label_policy == 'ignore':
            return
        if self.dip_label_policy == 'mask':
            for key in DIP_CLASS_KEYS:
                if key in metadata:
                    metadata[key] = _like(metadata[key], DIP_IGNORE_CLASS)
            return
        stored = {key: self._dip_source[key][idx] for key in self._dip_source if key not in DIP_SAMPLE_KEYS}
        dip_stretch = np.ones_like(self._dip_source['dip_samples_deg'][idx], dtype=np.float64)
        density_weights = np.ones_like(dip_stretch)
        if stretch_factors is not None:
            scale_xy, _, scale_z = stretch_factors
            dip_stretch *= float(scale_z) / float(scale_xy)
        if warp is not None:
            warp_stretch = vertical_warp_stretch_at_source(warp, self._dip_source['dip_samples_z'][idx])
            dip_stretch *= warp_stretch
            density_weights *= warp_stretch
        new_values = adjusted_dip_values_for_stretch(
            stored,
            self._dip_source['dip_samples_deg'][idx],
            dip_stretch,
            density_weights=density_weights,
            mean_edges=self._dip_mean_edges,
            range_edges=self._dip_range_edges,
        )
        for key in self._dip_keys_loaded:
            if key in metadata and key in new_values:
                metadata[key] = _like(metadata[key], new_values[key])

    def __getitem__(self, idx):
        arr = self._load_scaled_example(int(idx))

        # For denoising-style augmentation, label stays clean while input is perturbed.
        x = arr.copy()
        y = arr.copy()
        aug_params = None
        if self.augment:
            x, y, aug_params = cast(
                tuple[Any, Any, dict[str, Any]],
                apply_pair_augmentations(
                    x,
                    y,
                    self.swap_xy_prob,
                    self.flip_x_prob,
                    self.flip_y_prob,
                    self.vertical_warp_prob,
                    phase_rotation_prob=self.phase_rotation_prob,
                    phase_range=self.phase_range,
                    stretch_prob=self.stretch_prob,
                    stretch_xy=self.stretch_xy,
                    stretch_z=self.stretch_z,
                    return_params=True,
                ),
            )
            x = apply_input_trace_dropout(x, self.zero_cluster_min, self.zero_cluster_max)
        if self.extrema_only is None:
            x = self._apply_one_of_three_input_transform(x)
        elif self.extrema_only:
            x = keep_trace_extrema_only(x)

        if self.augment and np.random.random() < self.mixup_augment_prob:
            mixup_idx = sample_mixup_corpus_index(int(idx), self.num_examples)
            mixup_arr = self._load_scaled_example(mixup_idx)
            x = apply_input_extrema_mixup(x, mixup_arr)

        x = x.astype('f4', copy=False)

        x = np.ascontiguousarray(x[np.newaxis, ...])
        y = np.ascontiguousarray(y[np.newaxis, ...])
        sample = (torch.from_numpy(x), torch.from_numpy(y))
        if self.include_metadata:
            metadata = self._read_metadata_for_index(int(idx))
            if aug_params is not None:
                self._adjust_geometric_labels(metadata, int(idx), aug_params)
            return sample[0], sample[1], metadata
        return sample[0], sample[1]


class CubeDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(base_ch * 4, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.head(h)


def resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()
    if requested_device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if requested_device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        raise RuntimeError('CUDA requested but is not available on this machine.')

    if requested_device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        raise RuntimeError('MPS requested but is not available in this PyTorch build or on this machine.')

    if requested_device == 'cpu':
        return torch.device('cpu')

    raise ValueError("Unsupported device. Use one of: 'auto', 'cuda', 'mps', 'cpu'.")


class CombinedReconLoss(nn.Module):
    """Weighted combination of MSE and percent-MSE reconstruction losses.

    combined = mse_weight * MSE + (1 - mse_weight) * PMSE
    where PMSE = mean((pred-label)^2) / mean(label^2).
    """

    def __init__(self, mse_weight: float = 0.6, eps: float = 0.01):
        super().__init__()
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError('mse_weight must be in [0, 1].')
        self.mse_weight = float(mse_weight)
        self.pmse_weight = 1.0 - self.mse_weight
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(pred, target)
        if self.pmse_weight == 0.0:
            return mse
        label_energy = (target ** 2).mean()
        # Use a minimum of 0.01 so near-zero or all-zero patches (e.g., unwritten
        # zarr slots) don't cause PMSE to blow up to millions.
        pmse = mse / torch.clamp(label_energy, min=self.eps)
        return self.mse_weight * mse + self.pmse_weight * pmse


class MultiComponentReconLoss(nn.Module):
    """MAE-dominant reconstruction loss with optional TV and gradient matching."""

    def __init__(self, mae_weight=1.0, tv_weight=0.0, gdl_weight=0.0):
        super().__init__()
        self.mae_weight = float(mae_weight)
        self.tv_weight = float(tv_weight)
        self.gdl_weight = float(gdl_weight)
        if min(self.mae_weight, self.tv_weight, self.gdl_weight) < 0.0:
            raise ValueError('multi-component reconstruction weights must be non-negative.')

    @staticmethod
    def _differences(value):
        return (
            value[:, :, 1:, :, :] - value[:, :, :-1, :, :],
            value[:, :, :, 1:, :] - value[:, :, :, :-1, :],
            value[:, :, :, :, 1:] - value[:, :, :, :, :-1],
        )

    def forward(self, pred, target):
        total = self.mae_weight * torch.nn.functional.l1_loss(pred, target)
        if self.tv_weight:
            tv = torch.stack([diff.abs().mean() for diff in self._differences(pred)]).mean()
            total = total + self.tv_weight * tv
        if self.gdl_weight:
            pred_diff = self._differences(pred)
            target_diff = self._differences(target)
            gdl = torch.stack([
                (pred_value.abs() - target_value.abs()).abs().mean()
                for pred_value, target_value in zip(pred_diff, target_diff)
            ]).mean()
            total = total + self.gdl_weight * gdl
        return total


def build_reconstruction_loss(
    loss_type: str,
    mse_weight: float = 0.6,
    recon_mae_weight: float = 1.0,
    recon_tv_weight: float = 0.0,
    recon_gdl_weight: float = 0.0,
) -> nn.Module:
    if loss_type == 'mse_pmse':
        return CombinedReconLoss(mse_weight=mse_weight)
    if loss_type == 'mae':
        return nn.L1Loss()
    if loss_type == 'multi_component':
        return MultiComponentReconLoss(recon_mae_weight, recon_tv_weight, recon_gdl_weight)
    raise ValueError(f"Unsupported reconstruction loss: {loss_type!r}")


def _get_primary_prediction(recon):
    if isinstance(recon, (list, tuple)):
        return recon[0]
    return recon


def _get_autocast_disabled_context(device_type: str):
    if device_type in {'cpu', 'cuda'}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class SliceLPIPSLoss(nn.Module):
    def __init__(self, net: str = 'alex', min_spatial_size: int = 64, eps: float = 1e-6):
        super().__init__()
        if lpips_lib is None:
            raise RuntimeError(
                'lpips is required when --lpips_weight > 0. Install dependencies from requirements.txt or pyproject.toml.'
            )
        if min_spatial_size <= 0:
            raise ValueError('min_spatial_size must be positive.')
        self.min_spatial_size = int(min_spatial_size)
        self.eps = float(eps)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r".*The parameter 'pretrained' is deprecated since 0\.13.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message=r".*Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
                category=UserWarning,
            )
            try:
                self.network = lpips_lib.LPIPS(net=net, verbose=False)
            except TypeError:
                self.network = lpips_lib.LPIPS(net=net)
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad_(False)

    def _extract_mid_slices(self, volume: torch.Tensor):
        mid_x = int(volume.shape[2] // 2)
        mid_y = int(volume.shape[3] // 2)
        inline = volume[:, :, :, mid_y, :]
        crossline = volume[:, :, mid_x, :, :]
        return inline, crossline

    def _resize_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        target_h = max(height, self.min_spatial_size)
        target_w = max(width, self.min_spatial_size)
        if target_h == height and target_w == width:
            return image
        return F.interpolate(image, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _normalize_pair(self, pred: torch.Tensor, target: torch.Tensor):
        scale = torch.maximum(
            pred.detach().abs().amax(dim=(-2, -1), keepdim=True),
            target.detach().abs().amax(dim=(-2, -1), keepdim=True),
        )
        scale = torch.clamp(scale, min=self.eps)
        pred_norm = torch.clamp(pred / scale, min=-1.0, max=1.0)
        target_norm = torch.clamp(target / scale, min=-1.0, max=1.0)
        return pred_norm, target_norm.detach()

    def _prepare_slices(self, pred: torch.Tensor, target: torch.Tensor):
        pred_norm, target_norm = self._normalize_pair(pred.float(), target.float())
        pred_rgb = self._resize_if_needed(pred_norm.repeat(1, 3, 1, 1))
        target_rgb = self._resize_if_needed(target_norm.repeat(1, 3, 1, 1))
        return pred_rgb, target_rgb

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = _get_primary_prediction(recon)
        inline_pred, crossline_pred = self._extract_mid_slices(pred)
        inline_target, crossline_target = self._extract_mid_slices(target)

        losses = []
        with _get_autocast_disabled_context(pred.device.type):
            for pred_slice, target_slice in (
                (inline_pred, inline_target),
                (crossline_pred, crossline_target),
            ):
                pred_rgb, target_rgb = self._prepare_slices(pred_slice, target_slice)
                losses.append(self.network(pred_rgb, target_rgb).mean())
        return torch.stack(losses).mean()


def _normalize_metadata_vector(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / norm


def _metadata_batch_to_matrix(metadata_batch, metadata_keys):
    key_tuple = tuple(str(k) for k in metadata_keys)
    if not key_tuple:
        raise ValueError('metadata_keys must not be empty when geology metadata is enabled.')

    if isinstance(metadata_batch, dict):
        if not metadata_batch:
            raise ValueError('metadata_batch dict must contain at least one key.')
        columns = []
        batch_size = None
        for key in key_tuple:
            if key not in metadata_batch:
                raise KeyError(f"Missing geology metadata key '{key}' in batch metadata.")
            value = metadata_batch[key]
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy().reshape(-1)
            else:
                arr = np.asarray(value).reshape(-1)
            if batch_size is None:
                batch_size = int(arr.shape[0])
            elif int(arr.shape[0]) != batch_size:
                raise ValueError('Collated metadata arrays must share the same batch axis length.')
            columns.append(arr.astype(np.float64, copy=False))
        matrix = np.stack(columns, axis=1)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if len(metadata_batch) == 0:
        raise ValueError('metadata_batch must contain at least one sample.')
    rows = []
    for sample in metadata_batch:
        if not isinstance(sample, dict):
            raise ValueError('Each metadata item must be a dict keyed by geology metadata name.')
        vals = []
        for key in key_tuple:
            if key not in sample:
                raise KeyError(f"Missing geology metadata key '{key}' in batch metadata.")
            vals.append(float(sample[key]))
        rows.append(vals)
    matrix = np.asarray(rows, dtype=np.float64)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def resolve_background_key_indices(metadata_keys, background_keys):
    key_tuple = tuple(str(v) for v in metadata_keys)
    wanted = tuple(str(v) for v in (background_keys or ()))
    index_map = {key: idx for idx, key in enumerate(key_tuple)}
    indices = [index_map[key] for key in wanted if key in index_map]
    if indices:
        return tuple(indices)
    return tuple(range(len(key_tuple)))


def weighted_percentile(values, q, weights=None):
    """Percentile q in [0, 100]; with weights, interpolates the weighted CDF (equals np.percentile for unit weights)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        return float(np.percentile(values, q))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind='stable')
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    denom = cum[-1] - w[-1]
    if denom <= 0.0:
        return float(v[-1])
    ranks = (cum - w) / denom
    return float(np.interp(float(q) / 100.0, ranks, v))


def fit_geology_metadata_calibration(
    dataset,
    metadata_keys,
    strategy='robust_log1p',
    eps=1e-6,
    clip=6.0,
    background_keys=(),
    sample_weights=None,
):
    key_tuple = tuple(str(v) for v in metadata_keys)
    if not key_tuple:
        return None
    if strategy not in {'none', 'robust_log1p'}:
        raise ValueError("geology calibration strategy must be one of: 'none', 'robust_log1p'.")
    if not hasattr(dataset, '_metadata_arrays'):
        raise ValueError('dataset does not expose metadata arrays required for calibration fitting.')
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0) or sample_weights.sum() <= 0.0:
            raise ValueError('calibration sample_weights must be finite, non-negative, and not all zero.')
        if np.allclose(sample_weights, sample_weights[0]):
            sample_weights = None

    columns = []
    for key in key_tuple:
        if key not in dataset._metadata_arrays:
            raise KeyError(f"Missing geology metadata key '{key}' in dataset metadata arrays.")
        arr = np.asarray(dataset._metadata_arrays[key][:], dtype=np.float64).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(arr)
    raw_matrix = np.stack(columns, axis=1)

    if strategy == 'none':
        nonzero_scale = np.ones((raw_matrix.shape[1],), dtype=np.float64)
        center = np.zeros((raw_matrix.shape[1],), dtype=np.float64)
        scale = np.ones((raw_matrix.shape[1],), dtype=np.float64)
    else:
        nonzero_scale = np.ones((raw_matrix.shape[1],), dtype=np.float64)
        for idx in range(raw_matrix.shape[1]):
            col = raw_matrix[:, idx]
            positive_mask = col > 0.0
            if not np.any(positive_mask):
                nonzero_scale[idx] = 1.0
            elif sample_weights is None:
                nonzero_scale[idx] = max(float(np.median(col[positive_mask])), float(eps))
            else:
                nonzero_scale[idx] = max(weighted_percentile(col[positive_mask], 50.0, sample_weights[positive_mask]), float(eps))

        transformed = np.log1p(np.clip(raw_matrix, a_min=0.0, a_max=None) / nonzero_scale.reshape(1, -1))
        if sample_weights is None:
            center = np.median(transformed, axis=0)
            q25 = np.percentile(transformed, 25.0, axis=0)
            q75 = np.percentile(transformed, 75.0, axis=0)
        else:
            columns_t = [transformed[:, i] for i in range(transformed.shape[1])]
            center = np.array([weighted_percentile(c, 50.0, sample_weights) for c in columns_t])
            q25 = np.array([weighted_percentile(c, 25.0, sample_weights) for c in columns_t])
            q75 = np.array([weighted_percentile(c, 75.0, sample_weights) for c in columns_t])
        scale = np.maximum(q75 - q25, float(eps))

    return {
        'schema_version': 1,
        'keys': list(key_tuple),
        'strategy': str(strategy),
        'eps': float(eps),
        'clip': float(clip),
        'nonzero_scale': nonzero_scale.astype(np.float64).tolist(),
        'center': center.astype(np.float64).tolist(),
        'scale': scale.astype(np.float64).tolist(),
        'background_keys': list(tuple(str(v) for v in (background_keys or ()))),
    }


def prepare_geology_metadata_vectors(
    metadata_batch,
    metadata_keys,
    device,
    dtype,
    geology_metadata_calibration=None,
    background_threshold=1e-6,
    background_key_indices=(),
):
    raw_matrix = _metadata_batch_to_matrix(metadata_batch, metadata_keys)
    key_count = int(raw_matrix.shape[1])

    mask_indices = tuple(int(v) for v in (background_key_indices or ()))
    if mask_indices:
        selected_matrix = raw_matrix[:, list(mask_indices)]
    else:
        selected_matrix = raw_matrix
    has_selected = np.any(selected_matrix > float(background_threshold), axis=1)

    transformed = raw_matrix.copy()
    if geology_metadata_calibration is not None:
        calibration_keys = tuple(str(v) for v in geology_metadata_calibration.get('keys', ()))
        expected_keys = tuple(str(v) for v in metadata_keys)
        if calibration_keys != expected_keys:
            raise ValueError(
                'geology calibration keys do not match active metadata keys. '
                f'calibration_keys={calibration_keys}, active_keys={expected_keys}'
            )

        strategy = str(geology_metadata_calibration.get('strategy', 'none'))
        eps = float(geology_metadata_calibration.get('eps', 1e-6))
        clip = float(geology_metadata_calibration.get('clip', 0.0))
        nonzero_scale = np.asarray(geology_metadata_calibration.get('nonzero_scale', np.ones((key_count,))), dtype=np.float64)
        center = np.asarray(geology_metadata_calibration.get('center', np.zeros((key_count,))), dtype=np.float64)
        scale = np.asarray(geology_metadata_calibration.get('scale', np.ones((key_count,))), dtype=np.float64)

        if strategy == 'robust_log1p':
            transformed = np.log1p(np.clip(transformed, a_min=0.0, a_max=None) / np.maximum(nonzero_scale.reshape(1, -1), eps))
        elif strategy != 'none':
            raise ValueError(f'Unsupported geology calibration strategy: {strategy}')

        transformed = (transformed - center.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), eps)
        if clip > 0.0:
            transformed = np.clip(transformed, a_min=-clip, a_max=clip)

    normalized = np.stack([_normalize_metadata_vector(row) for row in transformed], axis=0)
    metadata_vectors = torch.as_tensor(normalized, dtype=dtype, device=device)
    has_selected_tensor = torch.as_tensor(has_selected, dtype=torch.bool, device=device)
    return metadata_vectors, has_selected_tensor


def compute_geology_similarity_loss(
    latent_vectors,
    metadata_batch,
    metadata_keys,
    geology_metadata_calibration=None,
    background_threshold=1e-6,
    background_key_indices=(),
    loss_type='mse',
    huber_delta=0.1,
    offdiag_only=True,
):
    if latent_vectors.ndim != 2:
        raise ValueError('latent_vectors must be a 2D tensor of shape [batch, latent_dim].')
    if not metadata_keys:
        return latent_vectors.new_zeros(())

    if isinstance(metadata_batch, dict):
        if not metadata_batch:
            raise ValueError('metadata_batch dict must contain at least one key.')
        first_value = next(iter(metadata_batch.values()))
        sample_count = int(np.asarray(first_value).reshape(-1).shape[0])
    else:
        sample_count = len(metadata_batch)
    if sample_count != int(latent_vectors.shape[0]):
        raise ValueError('metadata_batch length must match latent_vectors batch size.')

    target, has_selected = prepare_geology_metadata_vectors(
        metadata_batch,
        metadata_keys,
        device=latent_vectors.device,
        dtype=latent_vectors.dtype,
        geology_metadata_calibration=geology_metadata_calibration,
        background_threshold=background_threshold,
        background_key_indices=background_key_indices,
    )
    # eps=1e-6 prevents NaN from F.normalize on MPS/CUDA when metadata vectors
    # are zero (happens when most patches have no structural signal).
    latent_norm = F.normalize(latent_vectors, dim=1, eps=1e-6)
    target_norm = F.normalize(target, dim=1, eps=1e-6)

    # Dimension-agnostic alignment: preserve relative neighborhood geometry
    # by matching pairwise cosine-similarity structure in latent vs metadata space.
    latent_sim = torch.clamp(latent_norm @ latent_norm.transpose(0, 1), -1.0, 1.0)
    target_sim = torch.clamp(target_norm @ target_norm.transpose(0, 1), -1.0, 1.0)

    pair_mask = torch.ones(
        (int(latent_vectors.shape[0]), int(latent_vectors.shape[0])),
        dtype=torch.bool,
        device=latent_vectors.device,
    )
    if offdiag_only:
        pair_mask = torch.triu(pair_mask, diagonal=1)
    if has_selected is not None:
        pair_mask = pair_mask & has_selected.unsqueeze(1) & has_selected.unsqueeze(0)
    if int(pair_mask.sum().item()) < 1:
        return latent_vectors.new_zeros(())

    diff = latent_sim[pair_mask] - target_sim[pair_mask]
    if loss_type == 'mse':
        loss = torch.mean(diff.square())
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=float(huber_delta), reduction='mean')
    else:
        raise ValueError("loss_type must be one of: 'mse', 'huber'.")
    if not torch.isfinite(loss):
        return latent_vectors.new_zeros(())
    return loss


def compute_batch_strata_labels(
    metadata_batch,
    metadata_keys,
    geology_metadata_calibration=None,
    background_threshold=1e-6,
    background_key_indices=(),
    strata_presence_threshold=0.5,
    strata_max_active_keys=2,
):
    """Derive per-sample multilabel strata labels for a collated batch.

    Mirrors the dataset-level strata construction used by the geology-aware sampler so
    the supervised-contrastive objective groups positives/negatives the same way. Label
    0 denotes background (no active geology feature).
    """
    if not metadata_keys:
        return None
    metadata_vectors, _ = prepare_geology_metadata_vectors(
        metadata_batch,
        metadata_keys,
        device='cpu',
        dtype=torch.float32,
        geology_metadata_calibration=geology_metadata_calibration,
        background_threshold=float(background_threshold),
        background_key_indices=tuple(background_key_indices),
    )
    strata = build_multilabel_strata(
        metadata_vectors.detach().cpu().numpy(),
        tuple(str(k) for k in metadata_keys),
        threshold=float(strata_presence_threshold),
        active_key_indices=tuple(background_key_indices),
        max_active_keys_per_stratum=int(strata_max_active_keys),
    )
    return np.asarray(strata.labels, dtype=np.int64)


def compute_supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.1,
    ignore_label=0,
):
    """Supervised contrastive (SupCon) loss on unit-norm geology embeddings.

    For each anchor, positives are other in-batch samples sharing its strata label.
    Background samples (``ignore_label``) are excluded as anchors and positives so they
    do not form an artificial cluster. Returns a zero scalar when no valid positive pair
    exists in the batch.
    """
    if embeddings.ndim != 2:
        raise ValueError('embeddings must be a 2D tensor of shape [batch, dim].')
    device = embeddings.device
    labels_t = torch.as_tensor(np.asarray(labels).reshape(-1), device=device, dtype=torch.long)
    if int(labels_t.shape[0]) != int(embeddings.shape[0]):
        raise ValueError('labels length must match embeddings batch size.')

    temperature = max(float(temperature), 1e-6)
    valid = labels_t != int(ignore_label)
    if int(valid.sum().item()) < 2:
        return embeddings.new_zeros(())

    logits = (embeddings @ embeddings.transpose(0, 1)) / temperature
    # Numerical stability: subtract row-wise max before exp.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch = int(embeddings.shape[0])
    self_mask = torch.eye(batch, dtype=torch.bool, device=device)
    label_eq = labels_t.unsqueeze(0) == labels_t.unsqueeze(1)
    valid_pair = valid.unsqueeze(0) & valid.unsqueeze(1)

    positive_mask = label_eq & valid_pair & (~self_mask)
    candidate_mask = valid_pair & (~self_mask)

    anchors_with_pos = positive_mask.any(dim=1) & valid
    if int(anchors_with_pos.sum().item()) < 1:
        return embeddings.new_zeros(())

    exp_logits = torch.exp(logits) * candidate_mask.to(logits.dtype)
    denom = exp_logits.sum(dim=1)
    log_prob = logits - torch.log(denom.clamp_min(1e-12)).unsqueeze(1)

    pos_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1) / pos_counts

    loss = -mean_log_prob_pos[anchors_with_pos].mean()
    if not torch.isfinite(loss):
        return embeddings.new_zeros(())
    return loss


def compute_uniformity_loss(
    embeddings,
    labels=None,
    t=2.0,
    ignore_label=0,
):
    """Hypersphere uniformity regularizer (Wang & Isola, 2020).

    Encourages unit-norm embeddings to spread over the hypersphere, counteracting the
    representational collapse where every embedding points in nearly the same direction.
    Computes ``log E[exp(-t * ||z_i - z_j||^2)]`` over distinct sample pairs. When ``labels``
    are provided, background samples (``ignore_label``) are excluded so they do not anchor the
    spread. Returns a zero scalar when fewer than two valid samples exist.
    """
    if embeddings.ndim != 2:
        raise ValueError('embeddings must be a 2D tensor of shape [batch, dim].')
    if labels is not None:
        device = embeddings.device
        labels_t = torch.as_tensor(np.asarray(labels).reshape(-1), device=device, dtype=torch.long)
        if int(labels_t.shape[0]) != int(embeddings.shape[0]):
            raise ValueError('labels length must match embeddings batch size.')
        embeddings = embeddings[labels_t != int(ignore_label)]
    if int(embeddings.shape[0]) < 2:
        return embeddings.new_zeros(())
    # Pairwise squared Euclidean distances via the Gram matrix. Avoids torch.pdist,
    # which is not implemented on the MPS backend.
    n = int(embeddings.shape[0])
    sq_norms = embeddings.pow(2).sum(dim=1)
    gram = embeddings @ embeddings.transpose(0, 1)
    sq_dist_mat = (sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * gram).clamp_min(0.0)
    iu = torch.triu_indices(n, n, offset=1, device=embeddings.device)
    sq_dists = sq_dist_mat[iu[0], iu[1]]
    loss = torch.logsumexp(-float(t) * sq_dists, dim=0) - math.log(float(sq_dists.shape[0]))
    if not torch.isfinite(loss):
        return embeddings.new_zeros(())
    return loss


def compute_latent_geology_diagnostics(
    latent_vectors,
    metadata_vectors,
    neighbor_k=5,
    tail_fraction=0.10,
    neighbor_ks: Optional[Sequence[int]] = None,
    has_selected_geology: Optional[torch.Tensor] = None,
):
    if latent_vectors.ndim != 2 or metadata_vectors.ndim != 2:
        raise ValueError('latent_vectors and metadata_vectors must both be 2D.')
    if latent_vectors.shape[0] != metadata_vectors.shape[0]:
        raise ValueError('latent_vectors and metadata_vectors must have the same number of samples.')

    if has_selected_geology is not None:
        if has_selected_geology.ndim != 1 or int(has_selected_geology.shape[0]) != int(latent_vectors.shape[0]):
            raise ValueError('has_selected_geology must be a 1D mask aligned with sample count.')
        selected_mask = has_selected_geology.to(dtype=torch.bool, device=latent_vectors.device)
        latent_vectors = latent_vectors[selected_mask]
        metadata_vectors = metadata_vectors[selected_mask]

    sample_count = int(latent_vectors.shape[0])
    ks_requested = list(neighbor_ks) if neighbor_ks is not None else [int(neighbor_k)]
    if not ks_requested:
        ks_requested = [int(neighbor_k)]
    ks_requested = [int(max(1, v)) for v in ks_requested]
    if int(neighbor_k) not in ks_requested:
        ks_requested.insert(0, int(neighbor_k))

    empty_neighbor_metrics = {
        'neighbor_overlap': 0.0,
    }
    for k_value in ks_requested:
        empty_neighbor_metrics[f'neighbor_overlap_at_{int(k_value)}'] = 0.0

    if sample_count < 3:
        metrics = {
            'pair_cosine_correlation': 0.0,
            'similar_latent_cosine': 0.0,
            'dissimilar_latent_cosine': 0.0,
            'cosine_separation': 0.0,
        }
        metrics.update(empty_neighbor_metrics)
        return metrics
    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError('tail_fraction must be in (0, 0.5).')

    latent_norm = F.normalize(latent_vectors.float(), dim=1, eps=1e-6)
    metadata_norm = F.normalize(metadata_vectors.float(), dim=1, eps=1e-6)
    latent_similarity = torch.clamp(latent_norm @ latent_norm.transpose(0, 1), -1.0, 1.0)
    metadata_similarity = torch.clamp(metadata_norm @ metadata_norm.transpose(0, 1), -1.0, 1.0)

    pair_mask = torch.triu(
        torch.ones((sample_count, sample_count), dtype=torch.bool, device=latent_vectors.device),
        diagonal=1,
    )
    latent_pairs = latent_similarity[pair_mask]
    metadata_pairs = metadata_similarity[pair_mask]
    latent_centered = latent_pairs - latent_pairs.mean()
    metadata_centered = metadata_pairs - metadata_pairs.mean()
    correlation_denom = torch.sqrt(
        torch.sum(latent_centered.square()) * torch.sum(metadata_centered.square())
    )
    if float(correlation_denom.item()) > 1e-12:
        correlation = torch.sum(latent_centered * metadata_centered) / correlation_denom
    else:
        correlation = latent_pairs.new_zeros(())

    lower_threshold = torch.quantile(metadata_pairs, float(tail_fraction))
    upper_threshold = torch.quantile(metadata_pairs, 1.0 - float(tail_fraction))
    similar_latent_cosine = latent_pairs[metadata_pairs >= upper_threshold].mean()
    dissimilar_latent_cosine = latent_pairs[metadata_pairs <= lower_threshold].mean()

    diagonal_mask = torch.eye(sample_count, dtype=torch.bool, device=latent_vectors.device)
    neighbor_metrics = {}
    for raw_k in ks_requested:
        effective_k = min(max(1, int(raw_k)), sample_count - 1)
        latent_neighbors = latent_similarity.masked_fill(diagonal_mask, float('-inf')).topk(effective_k, dim=1).indices
        metadata_neighbors = metadata_similarity.masked_fill(diagonal_mask, float('-inf')).topk(effective_k, dim=1).indices
        overlap = (
            (latent_neighbors.unsqueeze(2) == metadata_neighbors.unsqueeze(1)).any(dim=2).float().sum(dim=1)
            / float(effective_k)
        ).mean()
        neighbor_metrics[f'neighbor_overlap_at_{int(raw_k)}'] = float(overlap.item())

    metrics = {
        'pair_cosine_correlation': float(correlation.item()),
        'similar_latent_cosine': float(similar_latent_cosine.item()),
        'dissimilar_latent_cosine': float(dissimilar_latent_cosine.item()),
        'cosine_separation': float((similar_latent_cosine - dissimilar_latent_cosine).item()),
        'neighbor_overlap': float(neighbor_metrics.get(f'neighbor_overlap_at_{int(neighbor_k)}', 0.0)),
    }
    metrics.update(neighbor_metrics)
    return metrics


def compute_vae_losses(
    recon,
    targets,
    mu,
    logvar,
    kl_weight,
    deep_supervision_loss=None,
    rec_loss_fn=None,
    lpips_loss_fn=None,
    lpips_weight=0.0,
    geology_metadata_batch=None,
    geology_metadata_keys=None,
    geology_loss_weight=0.0,
    latent_vectors=None,
    geology_metadata_calibration=None,
    geology_background_threshold=1e-6,
    geology_background_key_indices=(),
    geology_loss_type='mse',
    geology_huber_delta=0.1,
    geology_offdiag_only=True,
):
    if deep_supervision_loss is not None:
        rec_loss = deep_supervision_loss(recon, targets)
    else:
        _rec_fn = rec_loss_fn if rec_loss_fn is not None else torch.nn.functional.mse_loss
        if isinstance(recon, (list, tuple)):
            rec_loss = _rec_fn(recon[0], targets)
        else:
            rec_loss = _rec_fn(recon, targets)
    lpips_loss = rec_loss.new_zeros(())
    if lpips_loss_fn is not None and float(lpips_weight) > 0.0:
        lpips_loss = lpips_loss_fn(_get_primary_prediction(recon), targets)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.numel()
    geology_loss = rec_loss.new_zeros(())
    if geology_loss_weight > 0.0 and geology_metadata_batch is not None and geology_metadata_keys and latent_vectors is not None:
        geology_loss = compute_geology_similarity_loss(
            latent_vectors,
            geology_metadata_batch,
            geology_metadata_keys,
            geology_metadata_calibration=geology_metadata_calibration,
            background_threshold=geology_background_threshold,
            background_key_indices=geology_background_key_indices,
            loss_type=geology_loss_type,
            huber_delta=geology_huber_delta,
            offdiag_only=geology_offdiag_only,
        )
    loss = rec_loss + (float(lpips_weight) * lpips_loss) + kl_weight * kld + float(geology_loss_weight) * geology_loss
    return loss, rec_loss, kld, lpips_loss, geology_loss


def compute_discriminator_gan_loss(discriminator, real_cubes, fake_cubes_detached):
    real_logits = discriminator(real_cubes)
    fake_logits = discriminator(fake_cubes_detached)

    # Mix and shuffle so each cube is randomly real/fake for discriminator classification.
    all_logits = torch.cat([real_logits, fake_logits], dim=0)
    all_labels = torch.cat([
        torch.ones_like(real_logits),
        torch.zeros_like(fake_logits),
    ], dim=0)
    perm = torch.randperm(all_logits.size(0), device=all_logits.device)
    all_logits = all_logits[perm]
    all_labels = all_labels[perm]

    d_gan_loss = torch.nn.functional.binary_cross_entropy_with_logits(all_logits, all_labels)
    predictions = (all_logits >= 0.0).to(all_labels.dtype)
    d_gan_accuracy = (predictions == all_labels).to(all_labels.dtype).mean()
    return d_gan_loss, d_gan_accuracy


def compute_generator_gan_loss(discriminator, fake_cubes):
    fake_logits = discriminator(fake_cubes)
    real_labels = torch.ones_like(fake_logits)
    g_gan_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, real_labels)
    return g_gan_loss


def append_real_batch(inputs, targets, real_batch):
    """Append an unlabeled real batch while preserving the synthetic prefix."""
    synthetic_count = int(inputs.shape[0])
    if real_batch is None:
        return inputs, targets, synthetic_count
    real_inputs, real_targets = real_batch
    if tuple(real_inputs.shape[1:]) != tuple(inputs.shape[1:]):
        raise ValueError(
            f"real batch shape {tuple(real_inputs.shape[1:])} does not match synthetic shape {tuple(inputs.shape[1:])}"
        )
    return (
        torch.cat((inputs, real_inputs.to(inputs.device)), dim=0),
        torch.cat((targets, real_targets.to(targets.device)), dim=0),
        synthetic_count,
    )


def weighted_real_reconstruction_adjustment(
    recon,
    targets,
    synthetic_count,
    real_recon_weight,
    rec_loss_fn=None,
    deep_supervision_loss=None,
    lpips_loss_fn=None,
    lpips_weight=0.0,
):
    real_count = int(targets.shape[0]) - int(synthetic_count)
    if real_count <= 0 or float(real_recon_weight) == 1.0:
        return targets.new_zeros(())
    real_targets = targets[int(synthetic_count):]
    real_recon = tuple(value[int(synthetic_count):] for value in recon) if isinstance(recon, (list, tuple)) else recon[int(synthetic_count):]
    if deep_supervision_loss is not None:
        real_loss = deep_supervision_loss(real_recon, real_targets)
    else:
        loss_fn = rec_loss_fn if rec_loss_fn is not None else torch.nn.functional.mse_loss
        real_loss = loss_fn(_get_primary_prediction(real_recon), real_targets)
    if lpips_loss_fn is not None and float(lpips_weight) > 0.0:
        real_loss = real_loss + float(lpips_weight) * lpips_loss_fn(_get_primary_prediction(real_recon), real_targets)
    real_fraction = float(real_count) / float(targets.shape[0])
    return (float(real_recon_weight) - 1.0) * real_fraction * real_loss


def compute_average_loss(
    model,
    dataloader,
    device,
    steps,
    kl_weight,
    deep_supervision=False,
    deep_supervision_loss=None,
    rec_loss_fn=None,
    lpips_loss_fn=None,
    lpips_weight=0.0,
):
    if steps is None:
        raise ValueError('steps must be provided for compute_average_loss.')
    steps = int(steps)
    if steps <= 0:
        raise ValueError('steps must be a positive integer.')
    model.eval()
    total_loss = 0.0
    batch_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if deep_supervision:
                recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
                loss, _, _, _, _ = compute_vae_losses(
                    ds_outputs,
                    targets,
                    mu,
                    logvar,
                    kl_weight,
                    deep_supervision_loss,
                    rec_loss_fn=rec_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=lpips_weight,
                )
            else:
                recon, mu, logvar = model(inputs)
                loss, _, _, _, _ = compute_vae_losses(
                    recon,
                    targets,
                    mu,
                    logvar,
                    kl_weight,
                    rec_loss_fn=rec_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=lpips_weight,
                )
            total_loss += loss.item()
    return total_loss / steps


def get_kl_weight(epoch_idx, args):
    if args.kl_schedule == 'fixed':
        return float(args.kl_fixed)

    # Linear warmup from kl_start to kl_end.
    warmup_epochs = max(1, int(args.kl_warmup_epochs))
    progress = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return float(args.kl_start + progress * (args.kl_end - args.kl_start))


def get_named_group_lr(optimizer, group_name, fallback=float('nan')):
    for param_group in optimizer.param_groups:
        if param_group.get('name') == group_name:
            return float(param_group['lr'])
    return float(fallback)


def apply_parameter_freezing(model, args):
    """Freeze encoder and/or decoder parameters for two-phase geology training.

    Returns a short human-readable summary of which submodules were frozen.
    """
    frozen = []
    if bool(getattr(args, 'freeze_encoder', False)) and int(getattr(args, 'freeze_encoder_epochs', 0)) > 0:
        raise ValueError('--freeze_encoder cannot be combined with --freeze_encoder_epochs.')
    if bool(getattr(args, 'freeze_encoder', False)):
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen.append('encoder')
    freeze_epochs = int(getattr(args, 'freeze_encoder_epochs', 0))
    if freeze_epochs > 0 and not bool(getattr(args, 'freeze_encoder', False)):
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen.append(f'encoder for {freeze_epochs} epochs')
    if bool(getattr(args, 'freeze_decoder', False)):
        for param in model.decoder.parameters():
            param.requires_grad = False
        frozen.append('decoder')
    return frozen


def build_optimizer(model, args):
    if args.encoder_lr_mult <= 0.0:
        raise ValueError('--encoder_lr_mult must be positive.')
    if args.decoder_lr_mult <= 0.0:
        raise ValueError('--decoder_lr_mult must be positive.')

    base_lr = float(args.learning_rate)
    scheduled_encoder = int(getattr(args, 'freeze_encoder_epochs', 0)) > 0
    if args.encoder_lr_mult == 1.0 and args.decoder_lr_mult == 1.0 and not scheduled_encoder:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError('No trainable parameters remain after freezing; check --freeze_encoder/--freeze_decoder.')
        return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=args.weight_decay)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad or scheduled_encoder]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    tracked_ids = {id(p) for p in list(model.encoder.parameters()) + list(model.decoder.parameters())}
    other_params = [p for p in model.parameters() if id(p) not in tracked_ids and p.requires_grad]

    param_groups = []
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': base_lr * float(args.encoder_lr_mult),
            'name': 'encoder',
        })
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr * float(args.decoder_lr_mult),
            'name': 'decoder',
        })
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other'})
    if not param_groups:
        raise ValueError('No trainable parameters remain after freezing; check --freeze_encoder/--freeze_decoder.')
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=args.weight_decay)


def unfreeze_encoder_after_warmup(model, optimizer, epoch_idx, args):
    freeze_epochs = int(getattr(args, 'freeze_encoder_epochs', 0))
    if freeze_epochs <= 0 or bool(getattr(args, 'freeze_encoder', False)) or int(epoch_idx) < freeze_epochs:
        return False
    if any(param.requires_grad for param in model.encoder.parameters()):
        return False
    for param in model.encoder.parameters():
        param.requires_grad = True
    return True


def build_discriminator(args):
    if not args.use_discriminator:
        return None
    return CubeDiscriminator(in_ch=1, base_ch=args.discriminator_base_ch)


def build_discriminator_optimizer(discriminator, args):
    if discriminator is None:
        return None
    disc_lr = args.discriminator_learning_rate
    if disc_lr is None:
        disc_lr = args.learning_rate
    disc_weight_decay = args.discriminator_weight_decay
    if disc_weight_decay is None:
        disc_weight_decay = args.weight_decay
    return torch.optim.AdamW(discriminator.parameters(), lr=disc_lr, weight_decay=disc_weight_decay)


def build_scheduler(optimizer, args):
    if args.lr_scheduler == 'none':
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )


def format_elapsed_time(seconds):
    total_seconds = int(max(0.0, seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


METRICS_CSV_COLUMNS = [
    'epoch',
    'examples_this_epoch',
    'cumulative_examples',
    'train_loss',
    'train_lpips_loss',
    'val_loss',
    'val_lpips_loss',
    'real_validation_mae',
    'real_test_mae',
    'kl_weight',
    'learning_rate',
    'discriminator_learning_rate',
    'gan_weight',
    'g_gan_loss',
    'd_gan_loss',
    'd_gan_acc_pct',
    'val_geo_latent_cosine_corr',
    'val_geo_latent_cosine_gap',
    'val_geo_neighbor_overlap',
    'val_geo_neighbor_overlap_at_5',
    'val_geo_neighbor_overlap_at_10',
    'val_geo_neighbor_overlap_at_20',
    'best_model',
]


def migrate_metrics_csv_if_needed(csv_path: Path, expected_header):
    if not csv_path.exists():
        return False

    with csv_path.open('r', newline='') as csv_file:
        rows = list(csv.reader(csv_file))

    if not rows:
        return False

    current_header = rows[0]
    if current_header == list(expected_header):
        return False

    migrated_rows = [list(expected_header)]
    for row in rows[1:]:
        row_map = {
            column_name: row[idx] if idx < len(row) else ''
            for idx, column_name in enumerate(current_header)
        }
        migrated_rows.append([row_map.get(column_name, '') for column_name in expected_header])

    with csv_path.open('w', newline='') as csv_file:
        csv.writer(csv_file).writerows(migrated_rows)
    return True


def clamp_float(value, lower, upper):
    return max(lower, min(upper, value))


def build_checkpoint_payload(model, epoch=None, geology_metadata_calibration=None):
    payload = {
        'model_state_dict': model.state_dict(),
        'patch_shape': [int(v) for v in model.patch_shape],
        'latent_dim': int(model.latent_dim),
        'base_ch': int(model.base_ch),
        'deep_supervision': bool(getattr(model, 'deep_supervision', False)),
        'geology_projection': bool(getattr(model, 'geology_projection', False)),
        'geology_proj_hidden': int(getattr(model, 'geology_proj_hidden', 128)),
        'geology_proj_dim': int(getattr(model, 'geology_proj_dim', 64)),
        'geology_classifier': bool(getattr(model, 'geology_classifier_enabled', False)),
        'geology_classifier_mode': str(getattr(model, 'geology_classifier_mode', 'patch')),
        'geology_classifier_hidden': int(getattr(model, 'geology_classifier_hidden', 256)),
        'model_config': dict(getattr(model, 'model_config', {})),
        'encoder_init': dict(getattr(model, 'encoder_init_metadata', {})),
    }
    if epoch is not None:
        payload['epoch'] = int(epoch)
    if geology_metadata_calibration is not None:
        payload['geology_metadata_calibration'] = geology_metadata_calibration
    return payload


def load_pretrained_encoder(model, checkpoint_path):
    if getattr(model, 'encoder_arch', None) != 'resnetv2':
        raise ValueError('--init_encoder_from requires --encoder_arch resnetv2.')
    if model.encoder.norm_type != 'instance' or model.encoder_stem != 'pretrain_v2' or model.encoder_input_axes != 'zxy':
        raise ValueError(
            '--init_encoder_from requires --encoder_norm instance, --encoder_stem pretrain_v2, '
            'and --encoder_input_axes zxy.'
        )
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Pretrained checkpoint {checkpoint_path} must contain a dictionary.')
    ema_state = checkpoint.get('ema_state')
    if isinstance(ema_state, dict) and isinstance(ema_state.get('shadow'), dict):
        source_state = ema_state['shadow']
        source_kind = 'ema_state.shadow'
    elif isinstance(checkpoint.get('model'), dict):
        source_state = checkpoint['model']
        source_kind = 'model'
    else:
        raise ValueError(f'Pretrained checkpoint {checkpoint_path} contains neither ema_state.shadow nor model weights.')
    source = {
        key.removeprefix('encoder.'): value
        for key, value in source_state.items()
        if str(key).startswith('encoder.')
    }
    target = model.encoder.trunk.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    mismatched = sorted(
        key for key in set(target).intersection(source)
        if tuple(target[key].shape) != tuple(source[key].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f'missing={missing[:8]}')
        if unexpected:
            details.append(f'unexpected={unexpected[:8]}')
        if mismatched:
            details.append(f'shape_mismatch={[(key, tuple(source[key].shape), tuple(target[key].shape)) for key in mismatched[:8]]}')
        raise ValueError(f'Pretrained encoder state is incompatible: {"; ".join(details)}')
    model.encoder.trunk.load_state_dict(source, strict=True)
    model.encoder_init_metadata = {
        'path': str(Path(checkpoint_path).resolve()),
        'epoch': int(checkpoint.get('epoch', -1)),
        'train_paths_count': len(checkpoint.get('train_paths', ())),
        'source_state': source_kind,
        'tensors_loaded': len(source),
    }
    return dict(model.encoder_init_metadata)


@dataclass
class BatchSnapshot:
    inputs: torch.Tensor
    targets: torch.Tensor
    recon: torch.Tensor
    per_example_mse: torch.Tensor


@dataclass
class RepresentativeExample:
    split: str
    percentile: int
    source_epoch: int
    batch_index: int
    selection_mse: float
    input_cube: torch.Tensor
    target_cube: torch.Tensor


def compute_per_example_mse(recon, targets):
    return ((recon - targets) ** 2).mean(dim=(1, 2, 3, 4))


def compute_per_example_pmse(recon, targets, eps=1e-8):
    mse_num = ((recon - targets) ** 2).mean(dim=(1, 2, 3, 4))
    mse_den = (targets ** 2).mean(dim=(1, 2, 3, 4))
    return mse_num / torch.clamp(mse_den, min=eps)


def compute_per_example_combined_recon_loss(recon, targets, mse_weight, eps=1e-8):
    mse_values = compute_per_example_mse(recon, targets)
    pmse_values = compute_per_example_pmse(recon, targets, eps=eps)
    return float(mse_weight) * mse_values + (1.0 - float(mse_weight)) * pmse_values


def compute_per_example_recon_loss(recon, targets, loss_type, mse_weight, eps=1e-8):
    if loss_type == 'mse_pmse':
        return compute_per_example_combined_recon_loss(recon, targets, mse_weight, eps=eps)
    if loss_type in {'mae', 'multi_component'}:
        return (recon - targets).abs().mean(dim=(1, 2, 3, 4))
    raise ValueError(f"Unsupported reconstruction loss: {loss_type!r}")


def compute_per_example_deep_supervision_recon_loss(outputs, target, weights, loss_type, mse_weight, eps=1e-8):
    if isinstance(outputs, torch.Tensor):
        return compute_per_example_recon_loss(outputs, target, loss_type, mse_weight, eps=eps)
    if outputs is None:
        raise ValueError('outputs must not be None')

    predictions = list(outputs)
    if len(predictions) == 0:
        raise ValueError('outputs must contain at least one tensor')
    if len(predictions) != len(weights):
        raise ValueError(
            f'outputs length ({len(predictions)}) must match weights length ({len(weights)})'
        )

    total = torch.zeros((target.shape[0],), dtype=target.dtype, device=target.device)
    for weight, pred in zip(weights, predictions):
        target_for_scale = target
        if pred.shape[2:] != target.shape[2:]:
            target_for_scale = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='trilinear', align_corners=False)
        total = total + (float(weight) * compute_per_example_recon_loss(pred, target_for_scale, loss_type, mse_weight, eps=eps))
    return total


def _build_representative_examples(snapshot, split, epoch_number, percentiles):
    if snapshot is None:
        return []
    batch_size = int(snapshot.per_example_mse.shape[0])
    if batch_size == 0:
        return []

    mse_values = snapshot.per_example_mse.detach().cpu().numpy()
    selected = []
    used_indices = set()
    candidate_indices = np.arange(batch_size)
    for percentile in percentiles:
        percentile_mse = float(np.percentile(mse_values, percentile))
        rank_order = np.argsort(np.abs(mse_values - percentile_mse))
        chosen_idx = None
        for rank_idx in rank_order.tolist():
            idx = int(candidate_indices[rank_idx])
            if idx not in used_indices:
                chosen_idx = idx
                break
        if chosen_idx is None:
            chosen_idx = int(candidate_indices[int(rank_order[0])])
        used_indices.add(chosen_idx)

        selected.append(
            RepresentativeExample(
                split=split,
                percentile=int(percentile),
                source_epoch=int(epoch_number),
                batch_index=int(chosen_idx),
                selection_mse=float(mse_values[chosen_idx]),
                input_cube=snapshot.inputs[chosen_idx:chosen_idx+1].detach().cpu().clone(),
                target_cube=snapshot.targets[chosen_idx:chosen_idx+1].detach().cpu().clone(),
            )
        )
    return selected


def _build_composite_slices(input_cube, pred_cube, target_cube):
    mid_x = int(input_cube.shape[0] // 2)
    mid_y = int(input_cube.shape[1] // 2)

    # Use [depth, lateral] orientation so plots are 64 (vertical) x 32 (horizontal).
    inline_input = input_cube[:, mid_y, :].T
    inline_pred = pred_cube[:, mid_y, :].T
    inline_target = target_cube[:, mid_y, :].T
    inline_composite = np.concatenate([inline_input, inline_pred, inline_target], axis=1)

    crossline_input = input_cube[mid_x, :, :].T
    crossline_pred = pred_cube[mid_x, :, :].T
    crossline_target = target_cube[mid_x, :, :].T
    crossline_composite = np.concatenate([crossline_input, crossline_pred, crossline_target], axis=1)

    return inline_composite, crossline_composite


def _format_latent_percentiles(name, latent_values):
    percentile_levels = [5, 20, 50, 80, 95]
    stats = [f"{level}%={np.percentile(latent_values, level):.4f}" for level in percentile_levels]
    return f"{name:>10}[" + ", ".join(stats) + "]"


def _plot_representative_example(
    model,
    device,
    example,
    epoch_number,
    rec_loss_fn,
    lpips_loss_fn=None,
    lpips_weight=0.0,
    vmin=-3.1,
    vmax=3.1,
):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        x = example.input_cube.to(device)
        y = example.target_cube.to(device)
        recon, _, _ = model(x)
        mse_value = float(compute_per_example_mse(recon, y)[0].item())
        pmse_value = float(compute_per_example_pmse(recon, y)[0].item())
        rec_loss_value = float(rec_loss_fn(recon, y).item())
        lpips_value = 0.0
        if lpips_loss_fn is not None and float(lpips_weight) > 0.0:
            lpips_value = float(lpips_loss_fn(recon, y).item())
        combined_loss_value = rec_loss_value + (float(lpips_weight) * lpips_value)
        latent_input_mu, _ = model.encoder(x)
        latent_pred_mu, _ = model.encoder(recon)
        latent_label_mu, _ = model.encoder(y)

    input_cube = example.input_cube[0, 0].cpu().numpy()
    pred_cube = recon[0, 0].detach().cpu().numpy()
    target_cube = example.target_cube[0, 0].cpu().numpy()
    latent_input = latent_input_mu[0].detach().cpu().numpy()
    latent_pred = latent_pred_mu[0].detach().cpu().numpy()
    latent_label = latent_label_mu[0].detach().cpu().numpy()
    # header = (
    #     f"Representative example | split={example.split} | epoch={epoch_number} | "
    #     f"percentile={example.percentile}% | batch_idx={example.batch_index} |"
    # )
    # input_stats = _format_latent_percentiles('input', latent_input)
    # pred_stats = _format_latent_percentiles('prediction', latent_pred)
    # label_stats = _format_latent_percentiles('label', latent_label)
    # print(f"{header}\n{input_stats} |\n{pred_stats} |\n{label_stats}")

    inline_composite, crossline_composite = _build_composite_slices(input_cube, pred_cube, target_cube)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [3.0, 0.56, 3.0]},
    )
    im0 = axes[0].imshow(inline_composite, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
    axes[0].set_title('Middle inline: input | prediction | label')
    axes[0].set_xlabel('Trace / Section')
    axes[0].set_ylabel('Depth sample')

    latent_axis = axes[1]
    latent_positions = np.arange(latent_input.shape[0], dtype=np.float32)
    shade_color = (50.0 / 255.0, 50.0 / 255.0, 50.0 / 255.0)
    input_baseline = -20.0
    pred_baseline = 0.0
    label_baseline = 20.0
    latent_input_shifted = latent_input - 20.0
    latent_pred_shifted = latent_pred
    latent_label_shifted = latent_label + 20.0

    # Use explicit half-sample bin edges so fill and line share identical vertical registration.
    latent_edges = np.arange(latent_input.shape[0] + 1, dtype=np.float32) - 0.5
    latent_axis.stairs(latent_input_shifted, latent_edges, orientation='horizontal', baseline=input_baseline, fill=True, color=shade_color, alpha=0.9)
    latent_axis.stairs(latent_pred_shifted, latent_edges, orientation='horizontal', baseline=pred_baseline, fill=True, color=shade_color, alpha=0.9)
    latent_axis.stairs(latent_label_shifted, latent_edges, orientation='horizontal', baseline=label_baseline, fill=True, color=shade_color, alpha=0.9)
    latent_axis.stairs(latent_input_shifted, latent_edges, orientation='horizontal', baseline=input_baseline, fill=False, color='black', linewidth=1.0)
    latent_axis.stairs(latent_pred_shifted, latent_edges, orientation='horizontal', baseline=pred_baseline, fill=False, color='black', linewidth=1.0)
    latent_axis.stairs(latent_label_shifted, latent_edges, orientation='horizontal', baseline=label_baseline, fill=False, color='black', linewidth=1.0)

    latent_axis.set_title('Latents I/P/L')
    latent_axis.set_xticks([])
    latent_axis.set_xlabel('')
    latent_axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    latent_axis.spines['bottom'].set_visible(False)
    latent_axis.spines['top'].set_visible(False)
    latent_axis.spines['right'].set_visible(False)
    latent_axis.spines['left'].set_position(('data', -50.0))
    latent_axis.set_ylabel('Latent index')
    latent_axis.set_yticks(np.arange(0, latent_input.shape[0], 16, dtype=int))
    latent_axis.set_ylim(latent_input.shape[0] - 0.5, -0.5)
    latent_axis.set_xlim(-60.0, 60.0)

    im1 = axes[2].imshow(crossline_composite, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
    axes[2].set_title('Middle crossline: input | prediction | label')
    axes[2].set_xlabel('Trace / Section')
    axes[2].set_ylabel('Depth sample')

    fig.colorbar(im1, ax=[axes[0], axes[2]], shrink=0.9, label='Amplitude')
    fig.suptitle(
        (
            f"{example.split} representative | epoch={epoch_number} | "
            f"percentile(epoch4)={example.percentile}% | mse={mse_value:.6f} | pmse={pmse_value:.6f} | "
            f"lpips={lpips_value:.6f} | loss={combined_loss_value:.6f} | "
            f"epoch4_recon={example.selection_mse:.6f} | batch_idx={example.batch_index}"
        ),
        fontsize=10,
    )
    return fig, mse_value, pmse_value, lpips_value, combined_loss_value


def _log_representative_examples(
    writer,
    model,
    device,
    examples,
    epoch_number,
    output_dir,
    rec_loss_fn,
    lpips_loss_fn=None,
    lpips_weight=0.0,
):
    if not examples:
        return
    plot_root = Path(output_dir) / 'representative_plots' / f'epoch_{epoch_number:04d}'
    plot_root.mkdir(parents=True, exist_ok=True)

    for example in examples:
        fig, mse_value, pmse_value, lpips_value, combined_loss_value = _plot_representative_example(
            model,
            device,
            example,
            epoch_number,
            rec_loss_fn=rec_loss_fn,
            lpips_loss_fn=lpips_loss_fn,
            lpips_weight=lpips_weight,
        )
        filename = f"{example.split.lower()}_p{example.percentile:02d}.png"
        fig.savefig(plot_root / filename, dpi=140)
        writer.add_figure(
            f"representative/{example.split.lower()}/p{example.percentile:02d}",
            fig,
            global_step=epoch_number,
        )
        writer.add_scalar(
            f"representative/{example.split.lower()}/p{example.percentile:02d}_mse",
            mse_value,
            epoch_number,
        )
        writer.add_scalar(
            f"representative/{example.split.lower()}/p{example.percentile:02d}_pmse",
            pmse_value,
            epoch_number,
        )
        writer.add_scalar(
            f"representative/{example.split.lower()}/p{example.percentile:02d}_lpips",
            lpips_value,
            epoch_number,
        )
        writer.add_scalar(
            f"representative/{example.split.lower()}/p{example.percentile:02d}_loss",
            combined_loss_value,
            epoch_number,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)


def _save_representative_example_metadata(examples_by_split, output_path):
    serializable = {}
    for split_name, split_examples in examples_by_split.items():
        serializable[split_name] = [
            {
                'split': ex.split,
                'percentile': ex.percentile,
                'source_epoch': ex.source_epoch,
                'batch_index': ex.batch_index,
                'selection_mse': ex.selection_mse,
                'input_cube': ex.input_cube,
                'target_cube': ex.target_cube,
            }
            for ex in split_examples
        ]
    torch.save(serializable, output_path)


def _load_representative_example_metadata(input_path):
    payload = torch.load(input_path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('Representative metadata payload must be a dict.')

    loaded = {'training': [], 'validation': []}
    for split_name in ('training', 'validation'):
        split_items = payload.get(split_name, [])
        if not isinstance(split_items, list):
            continue
        for item in split_items:
            loaded[split_name].append(
                RepresentativeExample(
                    split=str(item['split']),
                    percentile=int(item['percentile']),
                    source_epoch=int(item['source_epoch']),
                    batch_index=int(item['batch_index']),
                    selection_mse=float(item['selection_mse']),
                    input_cube=item['input_cube'].detach().cpu().clone(),
                    target_cube=item['target_cube'].detach().cpu().clone(),
                )
            )
    return loaded


def update_gan_balance_controller(
    args,
    d_gan_acc_epoch,
    d_gan_acc_history,
    current_gan_weight,
    disc_optimizer,
    disc_lr_min,
    disc_lr_max,
):
    if not args.gan_balance_controller or disc_optimizer is None:
        return current_gan_weight, None, 'off', float(d_gan_acc_epoch)

    current_disc_lr = float(disc_optimizer.param_groups[0]['lr'])
    next_gan_weight = current_gan_weight
    next_disc_lr = current_disc_lr
    status = 'hold'
    control_acc = float(d_gan_acc_epoch)
    used_prediction = False
    target_low = float(args.gan_balance_target_low)
    target_high = float(args.gan_balance_target_high)

    if args.gan_balance_lookahead and len(d_gan_acc_history) >= args.gan_balance_lookahead_window:
        y = np.asarray(list(d_gan_acc_history)[-args.gan_balance_lookahead_window:], dtype=np.float64)
        x = np.arange(y.shape[0], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        lookahead_x = float(y.shape[0] - 1 + args.gan_balance_lookahead_horizon)
        predicted_acc = float(intercept + slope * lookahead_x)
        control_acc = clamp_float(predicted_acc, 0.0, 1.0)
        used_prediction = True
        deadband = max(0.0, float(args.gan_balance_lookahead_deadband))
        target_low = min(target_high, target_low + deadband)
        target_high = max(target_low, target_high - deadband)

    if control_acc > target_high:
        # D is too strong: increase G adversarial pressure and slow D slightly.
        next_gan_weight = clamp_float(
            current_gan_weight * args.gan_balance_gan_weight_up_mult,
            args.gan_balance_gan_weight_min,
            args.gan_balance_gan_weight_max,
        )
        next_disc_lr = clamp_float(
            current_disc_lr * args.gan_balance_disc_lr_down_mult,
            disc_lr_min,
            disc_lr_max,
        )
        status = 'd_strong_pred' if used_prediction else 'd_strong'
    elif control_acc < target_low:
        # D is too weak: reduce G adversarial pressure and speed D slightly.
        next_gan_weight = clamp_float(
            current_gan_weight * args.gan_balance_gan_weight_down_mult,
            args.gan_balance_gan_weight_min,
            args.gan_balance_gan_weight_max,
        )
        next_disc_lr = clamp_float(
            current_disc_lr * args.gan_balance_disc_lr_up_mult,
            disc_lr_min,
            disc_lr_max,
        )
        status = 'd_weak_pred' if used_prediction else 'd_weak'

    for param_group in disc_optimizer.param_groups:
        param_group['lr'] = next_disc_lr

    return next_gan_weight, next_disc_lr, status, control_acc


def train_one_epoch(
    model,
    discriminator,
    dataloader,
    device,
    optimizer,
    disc_optimizer,
    steps_per_epoch,
    grad_clip,
    kl_weight,
    gan_weight,
    deep_supervision=False,
    deep_supervision_loss=None,
    rec_loss_fn=None,
    lpips_loss_fn=None,
    lpips_weight=0.0,
    reconstruction_loss='mse_pmse',
    mse_weight=0.6,
    deep_supervision_weights=None,
    geology_metadata_keys=(),
    geology_loss_weight=0.0,
    geology_metadata_calibration=None,
    geology_background_threshold=1e-6,
    geology_background_key_indices=(),
    geology_loss_type='mse',
    geology_huber_delta=0.1,
    geology_offdiag_only=True,
    geology_contrastive_weight=0.0,
    geology_contrastive_temperature=0.1,
    geology_uniformity_weight=0.0,
    geology_uniformity_t=2.0,
    geology_strata_presence_threshold=1e-4,
    geology_strata_max_active_keys=2,
    geology_classifier_weight=0.0,
    geology_classifier_targets=ALL_CLASSIFIER_TARGETS,
    geology_classifier_loss='bce',
    geology_classifier_focal_gamma=2.0,
    geology_classifier_label_smoothing=0.05,
    geology_presence_strata=None,
    epoch_stats=None,
    real_dataloader=None,
    real_recon_weight=1.0,
):
    if steps_per_epoch is None:
        raise ValueError('steps_per_epoch must be provided for train_one_epoch.')
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError('steps_per_epoch must be a positive integer.')
    model.train()
    # Keep fully-frozen submodules in eval mode so their BatchNorm running stats
    # do not drift while their weights are frozen (two-phase geology training).
    for submodule in (getattr(model, 'encoder', None), getattr(model, 'decoder', None)):
        if submodule is not None:
            params = list(submodule.parameters())
            if params and all(not p.requires_grad for p in params):
                submodule.eval()
    if discriminator is not None:
        discriminator.train()
    total_loss = 0.0
    total_lpips_loss = 0.0
    total_g_gan_loss = 0.0
    total_d_gan_loss = 0.0
    total_d_gan_acc = 0.0
    total_geology_contrastive_loss = 0.0
    total_geology_uniformity_loss = 0.0
    total_geology_classifier_loss = 0.0
    batch_iter = itertools.cycle(dataloader)
    real_batch_iter = itertools.cycle(real_dataloader) if real_dataloader is not None else None

    last_snapshot = None
    for _ in range(steps_per_epoch):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)
        geology_metadata_batch = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, geology_metadata_batch = batch
        else:
            inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        real_batch = next(real_batch_iter) if real_batch_iter is not None else None
        inputs, targets, synthetic_count = append_real_batch(inputs, targets, real_batch)
        ds_outputs = None

        d_gan_loss_value = 0.0
        d_gan_acc_value = 0.0
        if discriminator is not None:
            # Discriminator step.
            with torch.no_grad():
                if deep_supervision:
                    recon_for_d, _, _, _ = model(inputs, return_deep_supervision=True)
                else:
                    recon_for_d, _, _ = model(inputs)
            disc_optimizer.zero_grad()
            d_gan_loss, d_gan_accuracy = compute_discriminator_gan_loss(
                discriminator, targets[:synthetic_count], recon_for_d[:synthetic_count].detach()
            )
            d_gan_loss.backward()
            disc_optimizer.step()
            d_gan_loss_value = float(d_gan_loss.item())
            d_gan_acc_value = float(d_gan_accuracy.item())

        # Generator (VAE) step.
        if deep_supervision:
            recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
            vae_loss, _, _, lpips_loss, _ = compute_vae_losses(
                ds_outputs,
                targets,
                mu,
                logvar,
                kl_weight,
                deep_supervision_loss,
                rec_loss_fn=rec_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                lpips_weight=lpips_weight,
                geology_metadata_batch=geology_metadata_batch,
                geology_metadata_keys=geology_metadata_keys,
                geology_loss_weight=geology_loss_weight,
                latent_vectors=mu[:synthetic_count],
                geology_metadata_calibration=geology_metadata_calibration,
                geology_background_threshold=geology_background_threshold,
                geology_background_key_indices=geology_background_key_indices,
                geology_loss_type=geology_loss_type,
                geology_huber_delta=geology_huber_delta,
                geology_offdiag_only=geology_offdiag_only,
            )
        else:
            recon, mu, logvar = model(inputs)
            vae_loss, _, _, lpips_loss, _ = compute_vae_losses(
                recon,
                targets,
                mu,
                logvar,
                kl_weight,
                rec_loss_fn=rec_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                lpips_weight=lpips_weight,
                geology_metadata_batch=geology_metadata_batch,
                geology_metadata_keys=geology_metadata_keys,
                geology_loss_weight=geology_loss_weight,
                latent_vectors=mu[:synthetic_count],
                geology_metadata_calibration=geology_metadata_calibration,
                geology_background_threshold=geology_background_threshold,
                geology_background_key_indices=geology_background_key_indices,
                geology_loss_type=geology_loss_type,
                geology_huber_delta=geology_huber_delta,
                geology_offdiag_only=geology_offdiag_only,
            )

        g_gan_loss_value = 0.0
        total_g_loss = vae_loss + weighted_real_reconstruction_adjustment(
            ds_outputs if deep_supervision else recon,
            targets,
            synthetic_count,
            real_recon_weight,
            rec_loss_fn=rec_loss_fn,
            deep_supervision_loss=deep_supervision_loss,
            lpips_loss_fn=lpips_loss_fn,
            lpips_weight=lpips_weight,
        )
        if discriminator is not None:
            g_gan_loss = compute_generator_gan_loss(discriminator, recon[:synthetic_count])
            g_gan_loss_value = float(g_gan_loss.item())
            total_g_loss = total_g_loss + gan_weight * g_gan_loss

        geology_contrastive_value = 0.0
        geology_uniformity_value = 0.0
        if (
            (float(geology_contrastive_weight) > 0.0 or float(geology_uniformity_weight) > 0.0)
            and getattr(model, 'geology_head', None) is not None
            and geology_metadata_batch is not None
            and (geology_metadata_keys or geology_presence_strata is not None)
        ):
            if geology_presence_strata is not None:
                strata_classes, rarity = geology_presence_strata
                strata_labels = build_presence_strata(
                    presence_matrix_from_labels(geology_metadata_batch, strata_classes),
                    strata_classes,
                    rarity,
                    max_active_keys_per_stratum=int(geology_strata_max_active_keys),
                ).labels
            else:
                strata_labels = compute_batch_strata_labels(
                    geology_metadata_batch,
                    geology_metadata_keys,
                    geology_metadata_calibration=geology_metadata_calibration,
                    background_threshold=geology_background_threshold,
                    background_key_indices=geology_background_key_indices,
                    strata_presence_threshold=geology_strata_presence_threshold,
                    strata_max_active_keys=geology_strata_max_active_keys,
                )
            if strata_labels is not None:
                z_geo = model.encode_geo(mu[:synthetic_count])
                if float(geology_contrastive_weight) > 0.0:
                    contrastive_loss = compute_supervised_contrastive_loss(
                        z_geo,
                        strata_labels,
                        temperature=geology_contrastive_temperature,
                    )
                    geology_contrastive_value = float(contrastive_loss.item())
                    total_g_loss = total_g_loss + float(geology_contrastive_weight) * contrastive_loss
                if float(geology_uniformity_weight) > 0.0:
                    uniformity_loss = compute_uniformity_loss(
                        z_geo,
                        strata_labels,
                        t=geology_uniformity_t,
                    )
                    geology_uniformity_value = float(uniformity_loss.item())
                    total_g_loss = total_g_loss + float(geology_uniformity_weight) * uniformity_loss

        classifier = getattr(model, 'geology_classifier', None)
        if float(geology_classifier_weight) > 0.0 and classifier is not None and geology_metadata_batch is not None:
            classifier_loss, _ = compute_geology_classifier_loss(
                model.classify(mu[:synthetic_count]),
                geology_metadata_batch,
                targets=geology_classifier_targets,
                loss_type=geology_classifier_loss,
                focal_gamma=geology_classifier_focal_gamma,
                label_smoothing=geology_classifier_label_smoothing,
                pos_weight=classifier.pos_weight,
            )
            total_geology_classifier_loss += float(classifier_loss.item())
            total_g_loss = total_g_loss + float(geology_classifier_weight) * classifier_loss

        optimizer.zero_grad()
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += total_g_loss.item()
        total_lpips_loss += float(lpips_loss.item())
        total_g_gan_loss += g_gan_loss_value
        total_d_gan_loss += d_gan_loss_value
        total_d_gan_acc += d_gan_acc_value
        total_geology_contrastive_loss += geology_contrastive_value
        total_geology_uniformity_loss += geology_uniformity_value
        if deep_supervision:
            if deep_supervision_weights is None:
                raise ValueError('deep_supervision_weights must be provided when deep supervision is enabled.')
            if ds_outputs is None:
                raise ValueError('deep supervision outputs are required when deep supervision is enabled.')
            ds_outputs_detached = tuple(out.detach() for out in ds_outputs)
            per_example_mse = compute_per_example_deep_supervision_recon_loss(
                ds_outputs_detached,
                targets.detach(),
                weights=tuple(float(v) for v in deep_supervision_weights),
                loss_type=reconstruction_loss,
                mse_weight=mse_weight,
            ).detach().cpu()
        else:
            per_example_mse = compute_per_example_recon_loss(
                recon.detach(),
                targets.detach(),
                loss_type=reconstruction_loss,
                mse_weight=mse_weight,
            ).detach().cpu()
        last_snapshot = BatchSnapshot(
            inputs=inputs.detach().cpu().clone(),
            targets=targets.detach().cpu().clone(),
            recon=recon.detach().cpu().clone(),
            per_example_mse=per_example_mse,
        )

    if epoch_stats is not None:
        epoch_stats['geology_classifier_loss'] = total_geology_classifier_loss / steps_per_epoch
    return (
        total_loss / steps_per_epoch,
        total_lpips_loss / steps_per_epoch,
        total_g_gan_loss / steps_per_epoch,
        total_d_gan_loss / steps_per_epoch,
        total_d_gan_acc / steps_per_epoch,
        total_geology_contrastive_loss / steps_per_epoch,
        total_geology_uniformity_loss / steps_per_epoch,
        last_snapshot,
    )


@dataclass
class EarlyStoppingState:
    best_val_loss: float
    epochs_without_improvement: int


def update_early_stopping(state, val_loss, min_delta):
    improved = val_loss < (state.best_val_loss - min_delta)
    if improved:
        state.best_val_loss = val_loss
        state.epochs_without_improvement = 0
    else:
        state.epochs_without_improvement += 1
    return improved


def discover_model_data_volumes(root_path):
    root = Path(root_path)
    if not root.exists():
        return []
    discovered = []
    for candidate in sorted(root.rglob('model_data.zarr')):
        if not candidate.is_dir():
            continue
        vol_dir = candidate.parent
        if vol_dir.name.startswith('seismic__'):
            temp_name = vol_dir.name.replace('seismic__', 'temp_folder__', 1)
            if (vol_dir.parent / temp_name).exists():
                continue
        discovered.append(candidate)
    return discovered


def parse_class_quota(items):
    quota = {}
    for item in items or ():
        name, sep, value = str(item).partition('=')
        if not sep or name not in GEOLOGY_PRESENCE_CLASSES:
            raise ValueError(
                f"--geology_batch_class_quota entries must be CLASS=COUNT with CLASS in {GEOLOGY_PRESENCE_CLASSES}; got {item!r}"
            )
        quota[name] = int(value)
    return quota


def resolve_label_target_keys(args):
    """Per-patch label arrays the training dataset must load (classifier targets, presence strata, class quotas)."""
    keys = []
    if float(getattr(args, 'geology_classifier_weight', 0.0)) > 0.0:
        keys.extend(classifier_target_keys(args.geology_classifier_classes))
    if bool(getattr(args, 'geology_batch_sampler', False)):
        classes = []
        if getattr(args, 'geology_strata_source', 'metadata') == 'presence_labels':
            classes.extend(args.geology_strata_classes)
        classes.extend(parse_class_quota(getattr(args, 'geology_batch_class_quota', None)))
        keys.extend(PRESENCE_TARGET_KEYS[c] for c in classes)
    return tuple(dict.fromkeys(keys))


def _read_zarr_attrs(path):
    return dict(cast(Any, zarr.open(str(path), mode='r')).attrs)


def assert_train_validation_consistency(data_path, validation_data_path, skip=False):
    """Fail fast if --data and --validation_data were sampled from overlapping synthoseis volumes,
    or use different amplitude scaling / label conventions, so a leak or a normalization mismatch is
    caught before training rather than showing up as an unexplained metric. --skip_data_location_checks
    bypasses this (e.g. for datasets made before these attrs existed).
    """
    if skip:
        return
    train_attrs = _read_zarr_attrs(data_path)
    val_attrs = _read_zarr_attrs(validation_data_path)

    train_vols = set(str(v) for v in train_attrs.get('source_volumes', ()))
    val_vols = set(str(v) for v in val_attrs.get('source_volumes', ()))
    if train_vols and val_vols:
        overlap = sorted(train_vols & val_vols)
        if overlap:
            raise ValueError(
                f"--data ({data_path}) and --validation_data ({validation_data_path}) share "
                f"{len(overlap)} source synthoseis volume(s), e.g. {overlap[0]}. Training and validation "
                "patches must come from disjoint volumes (sample --validation_data from a separate "
                "--source, e.g. the synthoseis 'validation' folder, or with --disjoint_from). "
                "Pass --skip_data_location_checks to override."
            )
    else:
        print(
            "WARNING: --data or --validation_data lacks a 'source_volumes' attr "
            "(dataset predates scripts/sample_patches.py provenance tracking); cannot verify they are disjoint."
        )

    train_scale = (train_attrs.get('scaling_mode'), train_attrs.get('scaling_mean'), train_attrs.get('scaling_std'))
    val_scale = (val_attrs.get('scaling_mode'), val_attrs.get('scaling_mean'), val_attrs.get('scaling_std'))
    if train_scale[0] is not None and val_scale[0] is not None:
        if train_scale[0] != val_scale[0]:
            raise ValueError(
                f"--data scaling_mode={train_scale[0]!r} does not match --validation_data "
                f"scaling_mode={val_scale[0]!r}. Pass --skip_data_location_checks to override."
            )
        # scaling_mean only affects the baked-in patch values under 'zscore'; 'divide_by_std'
        # (and 'none') ignore it, so comparing it there would flag harmless near-zero noise.
        checks = [('scaling_std', train_scale[2], val_scale[2])]
        if train_scale[0] == 'zscore':
            checks.append(('scaling_mean', train_scale[1], val_scale[1]))
        for name, t, v in checks:
            if t is None or v is None:
                continue
            if abs(float(t) - float(v)) > 1e-3 * max(abs(float(t)), 1e-8):
                raise ValueError(
                    f"--data {name}={t} does not match --validation_data {name}={v} (>0.1% relative "
                    "difference). Regenerate --validation_data with --no_derive_dataset_stats and the "
                    "--dataset_mean/--dataset_std recorded on --data, or pass --skip_data_location_checks."
                )

    for key in ('label_z_offset', 'dip_mean_class_edges_deg', 'dip_range_class_edges_deg', 'label_class_order'):
        t, v = train_attrs.get(key), val_attrs.get(key)
        if t is None or v is None:
            continue
        t_cmp = list(t) if isinstance(t, (list, tuple)) else t
        v_cmp = list(v) if isinstance(v, (list, tuple)) else v
        if t_cmp != v_cmp:
            raise ValueError(
                f"--data and --validation_data disagree on {key}: {t!r} vs {v!r}. "
                "Pass --skip_data_location_checks to override."
            )


def build_dataset(args, data_path, augment=False):
    geology_keys = tuple(str(v) for v in (args.geology_metadata_keys or ()))
    include_metadata = bool(
        len(geology_keys) > 0
        and (
            args.geology_loss_weight > 0.0
            or args.geology_contrastive_weight > 0.0
            or args.geology_uniformity_weight > 0.0
            or bool(getattr(args, 'geology_batch_sampler', False))
        )
    )
    return ZarrPatchDataset(
        data_path,
        scaling=args.input_scaling,
        scaling_mean=args.input_mean,
        scaling_std=args.input_std,
        augment=augment,
        swap_xy_prob=args.swap_xy_prob,
        flip_x_prob=args.flip_x_prob,
        flip_y_prob=args.flip_y_prob,
        vertical_warp_prob=args.vertical_warp_prob,
        phase_rotation_prob=getattr(args, 'phase_rotation_prob', 0.0),
        phase_range=getattr(args, 'phase_range', (-60.0, 0.0, 40.0)),
        stretch_prob=getattr(args, 'stretch_prob', 0.0),
        stretch_xy=getattr(args, 'stretch_xy', (1.0, 1.25)),
        stretch_z=getattr(args, 'stretch_z', (1.0, 1.5)),
        zero_cluster_min=args.zero_cluster_min,
        zero_cluster_max=args.zero_cluster_max,
        extrema_only=None,
        input_extrema_prob=args.input_extrema_prob,
        input_sparse_keep_prob=args.input_sparse_keep_prob,
        input_decimate_trilinear_prob=args.input_decimate_trilinear_prob,
        sparse_keep_fraction_min=args.sparse_keep_fraction_min,
        sparse_keep_fraction_max=args.sparse_keep_fraction_max,
        sparse_poisson_radius_scale=args.sparse_poisson_radius_scale,
        mixup_augment_prob=args.mixup_augment_prob,
        include_metadata=include_metadata,
        geology_metadata_keys=geology_keys,
        label_target_keys=resolve_label_target_keys(args),
        dip_label_policy=getattr(args, 'dip_label_policy', 'adjust'),
    )


def build_real_dataset(args, data_path, augment=False):
    return ZarrPatchDataset(
        data_path,
        scaling='none',
        augment=augment,
        swap_xy_prob=args.swap_xy_prob,
        flip_x_prob=args.flip_x_prob,
        flip_y_prob=args.flip_y_prob,
        vertical_warp_prob=args.vertical_warp_prob,
        phase_rotation_prob=args.phase_rotation_prob,
        phase_range=args.phase_range,
        stretch_prob=args.stretch_prob,
        stretch_xy=args.stretch_xy,
        stretch_z=args.stretch_z,
        zero_cluster_min=args.zero_cluster_min,
        zero_cluster_max=args.zero_cluster_max,
        extrema_only=None if augment else False,
        input_extrema_prob=args.input_extrema_prob,
        input_sparse_keep_prob=args.input_sparse_keep_prob,
        input_decimate_trilinear_prob=args.input_decimate_trilinear_prob,
        sparse_keep_fraction_min=args.sparse_keep_fraction_min,
        sparse_keep_fraction_max=args.sparse_keep_fraction_max,
        sparse_poisson_radius_scale=args.sparse_poisson_radius_scale,
        mixup_augment_prob=args.mixup_augment_prob if augment else 0.0,
        include_metadata=False,
        geology_metadata_keys=(),
        label_target_keys=(),
    )


def compute_real_mae(model, args, data_path, device, max_steps):
    if not data_path:
        return float('nan')
    dataset = build_real_dataset(args, data_path, augment=False)
    if dataset.patch_shape != args.patch_size_xyz:
        raise ValueError(f"real patch shape {dataset.patch_shape} does not match --patch_size {args.patch_size_xyz}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    steps = min(len(dataloader), max(1, int(max_steps)))
    total_absolute_error = 0.0
    total_voxels = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in itertools.islice(dataloader, steps):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, return_deep_supervision=True)[0] if args.deep_supervision else model(inputs)[0]
            total_absolute_error += float(torch.sum(torch.abs(outputs - targets)).item())
            total_voxels += int(targets.numel())
    return total_absolute_error / float(max(1, total_voxels))


def presence_matrix_from_labels(label_arrays, class_names):
    """(N, C) 0/1 matrix from label_presence_<class> arrays (dict of arrays or collated batch tensors)."""
    columns = []
    for name in class_names:
        value = label_arrays[PRESENCE_TARGET_KEYS[name]]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        columns.append((np.asarray(value).reshape(-1) > 0).astype(np.float64))
    return np.stack(columns, axis=1)


def build_metadata_strata(dataset, args, geology_metadata_calibration, geology_background_key_indices):
    metadata_keys = tuple(str(v) for v in getattr(dataset, 'geology_metadata_keys', ()))
    metadata_dict = {
        key: np.asarray(dataset._metadata_arrays[key][:], dtype=np.float64)
        for key in metadata_keys
    }
    metadata_vectors, _ = prepare_geology_metadata_vectors(
        metadata_dict,
        metadata_keys,
        device='cpu',
        dtype=torch.float32,
        geology_metadata_calibration=geology_metadata_calibration,
        background_threshold=float(args.geology_background_threshold),
        background_key_indices=geology_background_key_indices,
    )
    return build_multilabel_strata(
        metadata_vectors.detach().cpu().numpy(),
        metadata_keys,
        threshold=float(args.geology_strata_presence_threshold),
        active_key_indices=geology_background_key_indices,
        max_active_keys_per_stratum=int(args.geology_strata_max_active_keys),
    )


def build_train_dataloader(
    dataset,
    args,
    sample_weights=None,
    geology_metadata_calibration=None,
    geology_background_key_indices=(),
    presence_strata=None,
):
    if bool(getattr(args, 'geology_batch_sampler', False)):
        if not getattr(dataset, 'include_metadata', False):
            raise ValueError('geology_batch_sampler requires metadata-enabled training dataset.')

        if presence_strata is not None:
            strata_classes, rarity = presence_strata
            strata = build_presence_strata(
                presence_matrix_from_labels(dataset._label_arrays, strata_classes),
                strata_classes,
                rarity,
                max_active_keys_per_stratum=int(args.geology_strata_max_active_keys),
            )
        else:
            strata = build_metadata_strata(dataset, args, geology_metadata_calibration, geology_background_key_indices)
        class_quota = parse_class_quota(getattr(args, 'geology_batch_class_quota', None))
        membership_classes: list[str] = list(dict.fromkeys(
            list(presence_strata[0] if presence_strata is not None else ()) + list(class_quota)
        ))
        class_membership: dict[str, np.ndarray] = {
            name: np.asarray(dataset._label_arrays[PRESENCE_TARGET_KEYS[name]]) > 0
            for name in membership_classes
        }

        if args.number_batches is not None:
            num_batches = int(args.number_batches)
        else:
            num_batches = int(math.ceil(len(dataset) / float(max(1, int(args.batch_size)))))

        sampler = GeologyAwareBatchSampler(
            strata_labels=strata.labels,
            batch_size=int(args.batch_size),
            num_batches=num_batches,
            seed=int(args.seed),
            sample_weights=np.asarray(sample_weights, dtype=np.float64) if sample_weights is not None else None,
            background_fraction=float(args.geology_batch_background_fraction),
            hard_fraction=float(args.geology_batch_hard_fraction),
            hard_top_quantile=float(args.geology_batch_hard_top_quantile),
            min_negative_strata=int(args.geology_batch_min_negative_strata),
            require_positive_pair=bool(args.geology_batch_require_positive_pair),
            allow_duplicates=bool(args.geology_batch_allow_duplicates),
            class_membership=class_membership,
            class_quota=class_quota,
        )
        return DataLoader(dataset, batch_sampler=sampler, num_workers=2)

    if sample_weights is None:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.ndim != 1 or int(weights.shape[0]) != len(dataset):
        raise ValueError('sample_weights must be a 1D array with length equal to training dataset size.')
    if np.any(weights < 0.0):
        raise ValueError('sample_weights must be non-negative.')
    if not np.isfinite(weights).all():
        raise ValueError('sample_weights must be finite.')

    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
        weight_sum = float(weights.sum())
    normalized = weights / weight_sum

    sampler = WeightedRandomSampler(
        weights=normalized.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=2)


def build_sampling_eval_dataset(args):
    return ZarrPatchDataset(
        args.data,
        scaling=args.input_scaling,
        scaling_mean=args.input_mean,
        scaling_std=args.input_std,
        augment=False,
        swap_xy_prob=0.0,
        flip_x_prob=0.0,
        flip_y_prob=0.0,
        vertical_warp_prob=0.0,
        zero_cluster_min=0,
        zero_cluster_max=0,
        extrema_only=False,
        input_extrema_prob=args.input_extrema_prob,
        input_sparse_keep_prob=args.input_sparse_keep_prob,
        input_decimate_trilinear_prob=args.input_decimate_trilinear_prob,
        sparse_keep_fraction_min=args.sparse_keep_fraction_min,
        sparse_keep_fraction_max=args.sparse_keep_fraction_max,
        sparse_poisson_radius_scale=args.sparse_poisson_radius_scale,
        mixup_augment_prob=0.0,
    )


def compute_full_dataset_recon_snapshot(model, dataset, batch_size, device, mse_weight, reconstruction_loss='mse_pmse', deep_supervision=False, deep_supervision_weights=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    recon_snapshot = np.zeros((len(dataset),), dtype=np.float32)

    model.eval()
    write_offset = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if deep_supervision:
                if deep_supervision_weights is None:
                    raise ValueError('deep_supervision_weights must be provided when deep_supervision is enabled.')
                recon, _, _, ds_outputs = model(inputs, return_deep_supervision=True)
                batch_recon = compute_per_example_deep_supervision_recon_loss(
                    ds_outputs,
                    targets,
                    weights=tuple(float(v) for v in deep_supervision_weights),
                    loss_type=reconstruction_loss,
                    mse_weight=mse_weight,
                )
            else:
                recon, _, _ = model(inputs)

                batch_recon = compute_per_example_recon_loss(recon, targets, reconstruction_loss, mse_weight)

            batch_recon_np = batch_recon.detach().cpu().numpy().astype(np.float32)
            batch_count = int(batch_recon_np.shape[0])
            recon_snapshot[write_offset:write_offset + batch_count] = batch_recon_np
            write_offset += batch_count

    return recon_snapshot


def compute_adaptive_sampling_scores(recon_history, improvement_weight=1.0):
    if not recon_history:
        raise ValueError('recon_history must contain at least one snapshot.')

    current_recon = np.asarray(recon_history[-1], dtype=np.float32)
    if len(recon_history) < 2:
        average_improvement = np.zeros_like(current_recon, dtype=np.float32)
    else:
        improvements = []
        for previous_snapshot, next_snapshot in zip(recon_history[:-1], recon_history[1:]):
            prev_arr = np.asarray(previous_snapshot, dtype=np.float32)
            next_arr = np.asarray(next_snapshot, dtype=np.float32)
            improvements.append(prev_arr - next_arr)
        average_improvement = np.mean(np.stack(improvements, axis=0), axis=0).astype(np.float32)

    score = current_recon + float(improvement_weight) * average_improvement
    score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)
    score = np.clip(score, a_min=0.0, a_max=None)

    if float(score.sum()) <= 0.0:
        score = np.where(np.isfinite(current_recon), current_recon, 0.0).astype(np.float32)
        score = np.clip(score, a_min=0.0, a_max=None)
    if float(score.sum()) <= 0.0:
        score = np.ones_like(current_recon, dtype=np.float32)

    probabilities = score / float(score.sum())
    return probabilities.astype(np.float64), average_improvement, score


def save_adaptive_sampling_snapshots(output_path, snapshot_records):
    serializable = {
        'snapshots': [
            {
                'epoch': int(record['epoch']),
                'recon_loss': np.asarray(record.get('recon_loss', record.get('mse')), dtype=np.float32),
                'average_improvement': np.asarray(record['average_improvement'], dtype=np.float32),
                'score': np.asarray(record['score'], dtype=np.float32),
                'probability': np.asarray(record['probability'], dtype=np.float32),
            }
            for record in snapshot_records
        ]
    }
    torch.save(serializable, output_path)


def infer_completed_epochs_from_resume_path(path):
    match = re.search(r'vae_epoch(\d+)$', path.stem)
    if match is None:
        return 0
    return int(match.group(1))


def validate(
    model,
    args,
    device,
    train_steps_per_epoch,
    deep_supervision_loss=None,
    rec_loss_fn=None,
    lpips_loss_fn=None,
    geology_metadata_calibration=None,
    geology_background_key_indices=(),
):
    validation_extrema_mode = None if args.validation_extrema_only else False
    validation_ds = ZarrPatchDataset(
        args.validation_data,
        scaling=args.input_scaling,
        scaling_mean=args.input_mean,
        scaling_std=args.input_std,
        augment=False,
        swap_xy_prob=0.0,
        flip_x_prob=0.0,
        flip_y_prob=0.0,
        vertical_warp_prob=0.0,
        zero_cluster_min=0,
        zero_cluster_max=0,
        extrema_only=validation_extrema_mode,
        input_extrema_prob=args.input_extrema_prob,
        input_sparse_keep_prob=args.input_sparse_keep_prob,
        input_decimate_trilinear_prob=args.input_decimate_trilinear_prob,
        sparse_keep_fraction_min=args.sparse_keep_fraction_min,
        sparse_keep_fraction_max=args.sparse_keep_fraction_max,
        sparse_poisson_radius_scale=args.sparse_poisson_radius_scale,
        mixup_augment_prob=0.0,
        include_metadata=bool(args.geology_loss_weight > 0.0 and len(args.geology_metadata_keys) > 0),
        geology_metadata_keys=tuple(str(v) for v in args.geology_metadata_keys),
    )
    validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    if validation_ds.patch_shape != args.patch_size_xyz:
        raise ValueError(
            f"validation patch shape {validation_ds.patch_shape} does not match --patch_size {args.patch_size_xyz}"
        )
    validation_steps = max(1, int(math.ceil(0.2 * train_steps_per_epoch)))
    model.eval()
    total_loss = 0.0
    total_lpips_loss = 0.0
    batch_iter = itertools.cycle(validation_dl)
    last_snapshot = None
    diagnostic_latents = []
    diagnostic_metadata = []
    diagnostic_has_selected = []
    diagnostic_max_samples = max(3, int(args.geology_diagnostic_max_samples))
    with torch.no_grad():
        for _ in range(validation_steps):
            batch = next(batch_iter)
            geology_metadata_batch = None
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inputs, targets, geology_metadata_batch = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            ds_outputs = None
            if args.deep_supervision:
                recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
                loss, _, _, lpips_loss, _ = compute_vae_losses(
                    ds_outputs,
                    targets,
                    mu,
                    logvar,
                    args.current_kl_weight,
                    deep_supervision_loss,
                    rec_loss_fn=rec_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=args.lpips_weight,
                    geology_metadata_batch=geology_metadata_batch,
                    geology_metadata_keys=args.geology_metadata_keys,
                    geology_loss_weight=float(args.geology_loss_weight),
                    latent_vectors=mu,
                    geology_metadata_calibration=geology_metadata_calibration,
                    geology_background_threshold=float(args.geology_background_threshold),
                    geology_background_key_indices=geology_background_key_indices,
                    geology_loss_type=str(args.geology_loss_type),
                    geology_huber_delta=float(args.geology_huber_delta),
                    geology_offdiag_only=bool(args.geology_offdiag_only),
                )
            else:
                recon, mu, logvar = model(inputs)
                loss, _, _, lpips_loss, _ = compute_vae_losses(
                    recon,
                    targets,
                    mu,
                    logvar,
                    args.current_kl_weight,
                    rec_loss_fn=rec_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=args.lpips_weight,
                    geology_metadata_batch=geology_metadata_batch,
                    geology_metadata_keys=args.geology_metadata_keys,
                    geology_loss_weight=float(args.geology_loss_weight),
                    latent_vectors=mu,
                    geology_metadata_calibration=geology_metadata_calibration,
                    geology_background_threshold=float(args.geology_background_threshold),
                    geology_background_key_indices=geology_background_key_indices,
                    geology_loss_type=str(args.geology_loss_type),
                    geology_huber_delta=float(args.geology_huber_delta),
                    geology_offdiag_only=bool(args.geology_offdiag_only),
                )
            total_loss += float(loss.item())
            total_lpips_loss += float(lpips_loss.item())
            diagnostic_count = sum(item.shape[0] for item in diagnostic_latents)
            if geology_metadata_batch is not None and diagnostic_count < diagnostic_max_samples:
                diagnostic_batch_size = min(int(targets.shape[0]), diagnostic_max_samples - diagnostic_count)
                diagnostic_inputs = np.stack([
                    preprocess_for_token(targets[idx, 0].detach().cpu().numpy())
                    for idx in range(diagnostic_batch_size)
                ])
                diagnostic_mu, _ = model.encoder(
                    torch.from_numpy(diagnostic_inputs[:, np.newaxis]).to(device)
                )
                diagnostic_latents.append(diagnostic_mu.detach().cpu())
                diagnostic_batch = {
                    key: (value[:diagnostic_batch_size] if isinstance(value, torch.Tensor) else np.asarray(value)[:diagnostic_batch_size])
                    for key, value in geology_metadata_batch.items()
                }
                metadata_vectors, has_selected = prepare_geology_metadata_vectors(
                    diagnostic_batch,
                    args.geology_metadata_keys,
                    device='cpu',
                    dtype=torch.float32,
                    geology_metadata_calibration=geology_metadata_calibration,
                    background_threshold=float(args.geology_background_threshold),
                    background_key_indices=geology_background_key_indices,
                )
                diagnostic_metadata.append(metadata_vectors)
                diagnostic_has_selected.append(has_selected)
            if args.deep_supervision:
                if ds_outputs is None:
                    raise ValueError('deep supervision outputs are required when deep supervision is enabled.')
                per_example_mse = compute_per_example_deep_supervision_recon_loss(
                    ds_outputs,
                    targets,
                    weights=tuple(float(v) for v in args.deep_supervision_weights),
                    loss_type=args.reconstruction_loss,
                    mse_weight=float(args.loss_mse_weight),
                ).detach().cpu()
            else:
                per_example_mse = compute_per_example_recon_loss(
                    recon,
                    targets,
                    loss_type=args.reconstruction_loss,
                    mse_weight=float(args.loss_mse_weight),
                ).detach().cpu()
            last_snapshot = BatchSnapshot(
                inputs=inputs.detach().cpu().clone(),
                targets=targets.detach().cpu().clone(),
                recon=recon.detach().cpu().clone(),
                per_example_mse=per_example_mse,
            )
    geology_diagnostics = None
    if diagnostic_latents:
        latent_values = torch.cat(diagnostic_latents, dim=0)[:diagnostic_max_samples]
        metadata_values = torch.cat(diagnostic_metadata, dim=0)[:diagnostic_max_samples]
        selected_values = torch.cat(diagnostic_has_selected, dim=0)[:diagnostic_max_samples] if diagnostic_has_selected else None
        geology_diagnostics = compute_latent_geology_diagnostics(
            latent_values,
            metadata_values,
            neighbor_k=args.geology_diagnostic_neighbor_k,
            neighbor_ks=args.geology_diagnostic_topk,
            has_selected_geology=selected_values,
        )
    return total_loss / validation_steps, total_lpips_loss / validation_steps, last_snapshot, geology_diagnostics


RESUME_TOLERATED_KEYS = {
    'decoder.aux_head_coarse.weight',
    'decoder.aux_head_coarse.bias',
    'decoder.aux_head_mid.weight',
    'decoder.aux_head_mid.bias',
}
# Optional heads are randomly initialized when warm-starting from a checkpoint without them.
RESUME_TOLERATED_PREFIXES = ('geology_head.', 'geology_classifier.')


def resume_state_dict_incompatibilities(missing_keys, unexpected_keys):
    """Return (invalid_missing, invalid_unexpected) after removing keys of optional heads."""
    def invalid(keys):
        return [
            k for k in keys
            if k not in RESUME_TOLERATED_KEYS and not k.startswith(RESUME_TOLERATED_PREFIXES)
        ]
    return invalid(missing_keys), invalid(unexpected_keys)


def validate_resume_model_config(checkpoint_model_config, model):
    if checkpoint_model_config:
        mismatched = {
            key: (checkpoint_model_config.get(key), model.model_config.get(key))
            for key in model.model_config
            if checkpoint_model_config.get(key) != model.model_config.get(key)
        }
        if mismatched:
            raise ValueError(f'Resume checkpoint architecture config does not match active model: {mismatched}')
    elif model.encoder_arch != 'conv' or model.residual_encoder:
        raise ValueError('Resume checkpoint has no architecture config and can only load the legacy conv encoder.')


def train(args):
    args.encoder_arch = str(getattr(args, 'encoder_arch', 'conv'))
    args.encoder_hidden_dims = getattr(args, 'encoder_hidden_dims', None)
    args.encoder_depth_profile = str(getattr(args, 'encoder_depth_profile', 'baseline'))
    args.encoder_stage_blocks = getattr(args, 'encoder_stage_blocks', None)
    args.encoder_norm = getattr(args, 'encoder_norm', None)
    args.encoder_stem = str(getattr(args, 'encoder_stem', 'pretrain_v2'))
    args.encoder_input_axes = str(getattr(args, 'encoder_input_axes', 'xyz'))
    args.decoder_hidden_dims = getattr(args, 'decoder_hidden_dims', None)
    args.decoder_block = str(getattr(args, 'decoder_block', 'conv'))
    args.init_encoder_from = getattr(args, 'init_encoder_from', None)
    args.freeze_encoder_epochs = int(getattr(args, 'freeze_encoder_epochs', 0))
    if args.freeze_encoder_epochs < 0:
        raise ValueError('--freeze_encoder_epochs must be non-negative.')
    args.real_batch_count = int(getattr(args, 'real_batch_count', 0))
    args.real_recon_weight = float(getattr(args, 'real_recon_weight', 1.0))
    args.real_data = getattr(args, 'real_data', None)
    args.real_validation_data = getattr(args, 'real_validation_data', None)
    args.real_test_data = getattr(args, 'real_test_data', None)
    if args.real_batch_count < 0:
        raise ValueError('--real_batch_count must be non-negative.')
    if args.real_recon_weight < 0.0:
        raise ValueError('--real_recon_weight must be non-negative.')
    if args.real_batch_count > 0 and not args.real_data:
        raise ValueError('--real_data is required when --real_batch_count > 0.')
    if args.geology_classifier_weight < 0.0:
        raise ValueError('--geology_classifier_weight must be non-negative.')
    if args.geology_classifier_weight > 0.0 and not args.geology_classifier:
        raise ValueError('--geology_classifier_weight > 0 requires --geology_classifier to build the classifier head.')
    if args.geology_loss_weight < 0.0:
        raise ValueError('--geology_loss_weight must be non-negative.')
    if args.geology_loss_weight > 0.0 and not args.geology_metadata_keys:
        raise ValueError('--geology_metadata_keys must be provided when --geology_loss_weight > 0.')
    if args.geology_diagnostic_max_samples < 3:
        raise ValueError('--geology_diagnostic_max_samples must be at least 3.')
    if args.geology_diagnostic_neighbor_k < 1:
        raise ValueError('--geology_diagnostic_neighbor_k must be positive.')
    if args.geology_huber_delta <= 0.0:
        raise ValueError('--geology_huber_delta must be positive.')
    if args.geology_background_threshold < 0.0:
        raise ValueError('--geology_background_threshold must be non-negative.')
    if not args.geology_diagnostic_topk:
        raise ValueError('--geology_diagnostic_topk must contain at least one value.')
    if any(int(v) <= 0 for v in args.geology_diagnostic_topk):
        raise ValueError('--geology_diagnostic_topk values must be positive integers.')
    if not 0.0 <= float(args.geology_batch_background_fraction) <= 1.0:
        raise ValueError('--geology_batch_background_fraction must be in [0, 1].')
    if not 0.0 <= float(args.geology_batch_hard_fraction) <= 1.0:
        raise ValueError('--geology_batch_hard_fraction must be in [0, 1].')
    if not 0.0 < float(args.geology_batch_hard_top_quantile) <= 1.0:
        raise ValueError('--geology_batch_hard_top_quantile must be in (0, 1].')
    if int(args.geology_batch_min_negative_strata) < 1:
        raise ValueError('--geology_batch_min_negative_strata must be >= 1.')
    if int(args.geology_strata_max_active_keys) < 1:
        raise ValueError('--geology_strata_max_active_keys must be >= 1.')
    uses_presence_strata = args.geology_strata_source == 'presence_labels'
    if (uses_presence_strata or args.geology_batch_class_quota) and not args.geology_batch_sampler:
        raise ValueError('--geology_strata_source presence_labels and --geology_batch_class_quota require --geology_batch_sampler.')
    if sum(parse_class_quota(args.geology_batch_class_quota).values()) > int(args.batch_size):
        raise ValueError('--geology_batch_class_quota counts must not exceed --batch_size.')

    ds = build_dataset(args, args.data, augment=args.augment)
    args.patch_size_xyz = resolve_patch_size_xyz(args.patch_size, ds.patch_shape)
    args.geology_diagnostic_topk = [int(v) for v in args.geology_diagnostic_topk]
    if int(args.geology_diagnostic_neighbor_k) not in args.geology_diagnostic_topk:
        args.geology_diagnostic_topk.insert(0, int(args.geology_diagnostic_neighbor_k))

    if Path(args.data).exists() and Path(args.validation_data).exists():
        assert_train_validation_consistency(
            args.data, args.validation_data, skip=bool(getattr(args, 'skip_data_location_checks', False))
        )

    geology_metadata_calibration = None
    geology_background_key_indices = ()
    if bool((args.geology_loss_weight > 0.0 or bool(getattr(args, 'geology_batch_sampler', False))) and len(args.geology_metadata_keys) > 0):
        calibration_weights = None
        if args.geology_calibration_inclusion_weight:
            train_store = cast(Any, zarr.open(str(args.data), mode='r'))
            if 'inclusion_weight' in train_store:
                calibration_weights = np.asarray(train_store['inclusion_weight'][:], dtype=np.float64)
                print('Geology calibration: weighting examples by inclusion_weight (natural prevalence).')
        geology_metadata_calibration = fit_geology_metadata_calibration(
            ds,
            args.geology_metadata_keys,
            strategy=str(args.geology_calibration_strategy),
            eps=float(args.geology_calibration_eps),
            clip=float(args.geology_calibration_clip),
            background_keys=args.geology_background_keys,
            sample_weights=calibration_weights,
        )
        geology_background_key_indices = resolve_background_key_indices(
            args.geology_metadata_keys,
            args.geology_background_keys,
        )
        calibration_path = Path(args.out_dir) / args.geology_calibration_filename
        torch.save(geology_metadata_calibration, calibration_path)
        print(f'Geology metadata calibration saved: {calibration_path}')
    if args.adaptive_sampling_by_mse and args.sampling_snapshot_interval <= 0:
        raise ValueError('--sampling_snapshot_interval must be a positive integer when adaptive sampling is enabled.')
    if args.sampling_improvement_window < 2:
        raise ValueError('--sampling_improvement_window must be at least 2.')

    adaptive_sample_weights = None
    adaptive_recon_history = deque(maxlen=int(args.sampling_improvement_window))
    adaptive_snapshot_records = []
    adaptive_eval_ds = None
    adaptive_snapshot_path = Path(args.out_dir) / args.sampling_snapshot_filename
    if args.adaptive_sampling_by_mse:
        adaptive_eval_ds = build_sampling_eval_dataset(args)
        if adaptive_eval_ds.patch_shape != args.patch_size_xyz:
            raise ValueError(
                f"adaptive eval patch shape {adaptive_eval_ds.patch_shape} does not match --patch_size {args.patch_size_xyz}"
            )
        adaptive_sample_weights = np.ones((len(ds),), dtype=np.float64) / float(max(1, len(ds)))
        print(
            'Adaptive sampling:',
            f"enabled={args.adaptive_sampling_by_mse}",
            f"snapshot_interval={args.sampling_snapshot_interval}",
            f"improvement_window={args.sampling_improvement_window}",
            f"improvement_weight={args.sampling_improvement_weight}",
            f"snapshot_file={adaptive_snapshot_path}",
        )

    presence_strata = None
    if uses_presence_strata:
        strata_classes = tuple(args.geology_strata_classes)
        rarity = presence_rarity_weights(presence_matrix_from_labels(ds._label_arrays, strata_classes))
        presence_strata = (strata_classes, rarity)
        print(
            'Geology strata: source=presence_labels',
            f"classes={list(strata_classes)}",
            f"rarity={[round(float(v), 2) for v in rarity]}",
            f"class_quota={parse_class_quota(args.geology_batch_class_quota)}",
        )

    dl = build_train_dataloader(
        ds,
        args,
        sample_weights=adaptive_sample_weights,
        geology_metadata_calibration=geology_metadata_calibration,
        geology_background_key_indices=geology_background_key_indices,
        presence_strata=presence_strata,
    )
    real_dl = None
    if args.real_batch_count > 0:
        real_ds = build_real_dataset(args, args.real_data, augment=args.augment)
        if real_ds.patch_shape != args.patch_size_xyz:
            raise ValueError(
                f"real patch shape {real_ds.patch_shape} does not match --patch_size {args.patch_size_xyz}"
            )
        real_dl = DataLoader(
            real_ds,
            batch_size=args.real_batch_count,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

    if args.number_batches is not None and args.number_batches <= 0:
        raise ValueError('--number_batches must be a positive integer when provided.')
    steps_per_epoch = args.number_batches if args.number_batches is not None else len(dl)
    samples_per_epoch = steps_per_epoch * (args.batch_size + args.real_batch_count)

    model = VAE3D(
        in_ch=1,
        out_ch=1,
        base_ch=16,
        latent_dim=128,
        patch_shape=args.patch_size_xyz,
        deep_supervision=args.deep_supervision,
        residual_encoder=args.residual_encoder,
        geology_projection=bool(args.geology_projection),
        geology_proj_hidden=int(args.geology_proj_hidden),
        geology_proj_dim=int(args.geology_proj_dim),
        geology_classifier=bool(args.geology_classifier),
        geology_classifier_mode=str(args.geology_classifier_mode),
        geology_classifier_hidden=int(args.geology_classifier_hidden),
        encoder_arch=args.encoder_arch,
        encoder_hidden_dims=tuple(args.encoder_hidden_dims) if args.encoder_hidden_dims else None,
        encoder_depth_profile=args.encoder_depth_profile,
        encoder_stage_blocks=tuple(args.encoder_stage_blocks) if args.encoder_stage_blocks else None,
        encoder_norm=args.encoder_norm,
        encoder_stem=args.encoder_stem,
        encoder_input_axes=args.encoder_input_axes,
        decoder_hidden_dims=tuple(args.decoder_hidden_dims) if args.decoder_hidden_dims else None,
        decoder_block=args.decoder_block,
    )
    if args.init_encoder_from is not None:
        if args.resume is not None:
            raise ValueError('--init_encoder_from cannot be combined with --resume.')
        transfer = load_pretrained_encoder(model, args.init_encoder_from)
        print(
            'Initialized pretrained encoder:',
            f"source={transfer['path']}",
            f"epoch={transfer['epoch']}",
            f"train_paths={transfer['train_paths_count']}",
            f"tensors={transfer['tensors_loaded']}",
        )
    if float(args.geology_contrastive_weight) > 0.0 and not bool(args.geology_projection):
        raise ValueError('--geology_contrastive_weight > 0 requires --geology_projection to build the z_geo head.')
    if float(args.geology_uniformity_weight) > 0.0 and not bool(args.geology_projection):
        raise ValueError('--geology_uniformity_weight > 0 requires --geology_projection to build the z_geo head.')
    discriminator = build_discriminator(args)
    device = resolve_device(args.device)
    resume_completed_epochs = 0

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f'Checkpoint {ckpt_path} is invalid. Expected a dict with keys '
                "['model_state_dict', 'patch_shape', 'latent_dim', 'base_ch']."
            )
        required_keys = {'model_state_dict', 'patch_shape', 'latent_dim', 'base_ch'}
        missing_keys = required_keys.difference(checkpoint.keys())
        if missing_keys:
            missing_keys_display = ', '.join(sorted(missing_keys))
            available_keys_display = ', '.join(sorted(str(k) for k in checkpoint.keys()))
            if {'training', 'validation'}.issubset(checkpoint.keys()):
                raise ValueError(
                    f'Checkpoint {ckpt_path} looks like representative-example metadata, not model weights. '
                    'Pass a VAE checkpoint such as vae_best.pt or vae_epoch<N>.pt to --resume. '
                    f'Available keys in the provided file: [{available_keys_display}].'
                )
            raise ValueError(
                f'Checkpoint {ckpt_path} is missing required keys: {missing_keys_display}. '
                "Expected keys: ['model_state_dict', 'patch_shape', 'latent_dim', 'base_ch']. "
                f'Available keys in the provided file: [{available_keys_display}]. '
                'If you intended to resume training, pass a VAE checkpoint such as vae_best.pt or vae_epoch<N>.pt.'
            )

        checkpoint_patch_shape = tuple(int(v) for v in checkpoint['patch_shape'])
        checkpoint_latent_dim = int(checkpoint['latent_dim'])
        checkpoint_base_ch = int(checkpoint['base_ch'])
        expected_patch_shape = tuple(int(v) for v in args.patch_size_xyz)
        expected_latent_dim = int(model.latent_dim)
        expected_base_ch = int(model.base_ch)

        if checkpoint_patch_shape != expected_patch_shape:
            raise ValueError(
                f'Resume checkpoint patch_shape {checkpoint_patch_shape} does not match '
                f'active training patch shape {expected_patch_shape}.'
            )
        if checkpoint_latent_dim != expected_latent_dim:
            raise ValueError(
                f'Resume checkpoint latent_dim {checkpoint_latent_dim} does not match '
                f'active model latent_dim {expected_latent_dim}.'
            )
        if checkpoint_base_ch != expected_base_ch:
            raise ValueError(
                f'Resume checkpoint base_ch {checkpoint_base_ch} does not match '
                f'active model base_ch {expected_base_ch}.'
            )

        validate_resume_model_config(checkpoint.get('model_config'), model)

        state_dict = checkpoint['model_state_dict']
        load_result = model.load_state_dict(state_dict, strict=False)
        invalid_missing, invalid_unexpected = resume_state_dict_incompatibilities(
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
        if invalid_missing or invalid_unexpected:
            raise ValueError(
                'Resume checkpoint model_state_dict is incompatible with current architecture. '
                f'invalid missing keys={invalid_missing}, invalid unexpected keys={invalid_unexpected}'
            )
        checkpoint_epoch = checkpoint.get('epoch', None)
        setattr(model, 'encoder_init_metadata', dict(checkpoint.get('encoder_init', {})))
        if isinstance(checkpoint.get('geology_metadata_calibration', None), dict):
            geology_metadata_calibration = checkpoint['geology_metadata_calibration']
            print('Loaded geology metadata calibration from resume checkpoint.')
        if args.resume_epoch is not None:
            resume_completed_epochs = int(args.resume_epoch)
            print(f"Overriding resume epoch numbering with --resume_epoch={resume_completed_epochs}")
        elif checkpoint_epoch is not None:
            resume_completed_epochs = max(0, int(checkpoint_epoch))
        else:
            resume_completed_epochs = infer_completed_epochs_from_resume_path(ckpt_path)
        print(f"Resumed model weights from {ckpt_path}")
        print(f"Resuming epoch numbering from {resume_completed_epochs + 1}")

    print(f"Using device: {device}")
    print(
        'Model architecture:',
        f"config={model.model_config}",
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}",
        f"encoder_parameters={sum(parameter.numel() for parameter in model.encoder.parameters())}",
        f"decoder_parameters={sum(parameter.numel() for parameter in model.decoder.parameters())}",
    )
    if float(args.geology_classifier_weight) > 0.0:
        classifier = cast(Any, model.geology_classifier)
        pos_weight = classifier.pos_weight.clone()
        for i, name in enumerate(GEOLOGY_PRESENCE_CLASSES):
            key = PRESENCE_TARGET_KEYS[name]
            if key in ds._label_arrays:
                pos_weight[i] = compute_pos_weight(ds._label_arrays[key][:, None])[0]
        classifier.pos_weight.copy_(pos_weight)
        print(
            'Geology classifier:',
            f"mode={args.geology_classifier_mode}",
            f"weight={args.geology_classifier_weight}",
            f"loss={args.geology_classifier_loss}",
            f"targets={list(args.geology_classifier_classes)}",
            f"pos_weight={[round(float(v), 2) for v in classifier.pos_weight]}",
        )
    print(f"Training seed: {args.seed}")
    print(f"Batch size (B): {args.batch_size}, batches/epoch: {steps_per_epoch}, examples/epoch: {samples_per_epoch}")
    print(
        "Real seismic mixing:",
        f"K={args.real_batch_count}",
        f"recon_weight={args.real_recon_weight}",
        f"train={args.real_data}",
        f"validation={args.real_validation_data}",
        f"test={args.real_test_data}",
    )
    print(
        "Augmentations:",
        f"enabled={args.augment}",
        f"swap_xy_prob={args.swap_xy_prob}",
        f"flip_x_prob={args.flip_x_prob}",
        f"flip_y_prob={args.flip_y_prob}",
        f"vertical_warp_prob={args.vertical_warp_prob}",
        f"phase_rotation_prob={args.phase_rotation_prob}",
        f"phase_range={list(args.phase_range)}",
        f"stretch_prob={args.stretch_prob}",
        f"stretch_xy={list(args.stretch_xy)}",
        f"stretch_z={list(args.stretch_z)}",
        f"dip_label_policy={args.dip_label_policy}",
        f"mixup_augment_prob={args.mixup_augment_prob}",
        f"zero_cluster_range=[{args.zero_cluster_min},{args.zero_cluster_max}]",
        f"input_extrema_prob={args.input_extrema_prob}",
        f"input_sparse_keep_prob={args.input_sparse_keep_prob}",
        f"input_decimate_trilinear_prob={args.input_decimate_trilinear_prob}",
        f"sparse_keep_fraction_range=[{args.sparse_keep_fraction_min},{args.sparse_keep_fraction_max}]",
        f"sparse_poisson_radius_scale={args.sparse_poisson_radius_scale}",
    )
    print("Train input transform mode=one-of-three (extrema/sparse/decimate) with normalized positive weights")
    if args.validation_extrema_only:
        print("Validation input transform mode=shared train weights (extrema/sparse/decimate)")
    else:
        print("Validation input transform mode=disabled")
    print(f"Discriminator enabled={args.use_discriminator}")
    print(
        'Deep supervision:',
        f"enabled={args.deep_supervision}",
        f"weights={args.deep_supervision_weights}",
    )
    checkpoint_keys = ['model_state_dict', 'patch_shape', 'latent_dim', 'base_ch', 'deep_supervision', 'geology_projection', 'geology_proj_hidden', 'geology_proj_dim', 'geology_classifier', 'geology_classifier_mode', 'geology_classifier_hidden']
    print(f"Checkpoint schema keys={checkpoint_keys}")
    print("base_ch = base channel count for the VAE's convolution layers")
    print(
        'Optimizer LR multipliers:',
        f"encoder_lr_mult={args.encoder_lr_mult}",
        f"decoder_lr_mult={args.decoder_lr_mult}",
    )
    print(
        "Checkpoint metadata:",
        f"patch_shape={list(model.patch_shape)}",
        f"latent_dim={model.latent_dim}",
        f"base_ch={model.base_ch}",
    )
    print(
        'Reconstruction loss:',
        f"type={args.reconstruction_loss}",
        f"mse_weight={args.loss_mse_weight:.4f}",
        f"pmse_weight={1.0 - args.loss_mse_weight:.4f}",
        f"lpips_weight={args.lpips_weight:.4f}",
    )
    representative_percentiles = (
        5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
    )
    if args.representative_selection_epoch <= 0:
        raise ValueError('--representative_selection_epoch must be a positive integer.')
    if args.representative_plot_interval <= 0:
        raise ValueError('--representative_plot_interval must be a positive integer.')
    print(
        "Representative plots:",
        f"selection_epoch={args.representative_selection_epoch}",
        f"plot_every={args.representative_plot_interval}",
        f"percentiles={list(representative_percentiles)}",
        "source=last_batch_reconstruction_distribution",
    )
    model.to(device)
    if discriminator is not None:
        discriminator.to(device)

    if args.learning_rate <= 0:
        raise ValueError('--learning_rate must be positive.')
    if args.grad_clip <= 0:
        raise ValueError('--grad_clip must be positive.')
    if args.weight_decay < 0:
        raise ValueError('--weight_decay must be non-negative.')
    if args.resume_epoch is not None and args.resume_epoch < 0:
        raise ValueError('--resume_epoch must be non-negative when provided.')
    if args.lpips_weight < 0.0:
        raise ValueError('--lpips_weight must be non-negative.')
    if args.lpips_min_size <= 0:
        raise ValueError('--lpips_min_size must be positive.')
    if not 0.0 <= args.vertical_warp_prob <= 1.0:
        raise ValueError('--vertical_warp_prob must be in [0, 1].')
    if not 0.0 <= args.phase_rotation_prob <= 1.0:
        raise ValueError('--phase_rotation_prob must be in [0, 1].')
    phase_min, phase_mode, phase_max = (float(v) for v in args.phase_range)
    if not (phase_min <= phase_mode <= phase_max):
        raise ValueError('--phase_range must be ordered as (min, mode, max) with min <= mode <= max.')
    if not 0.0 <= args.stretch_prob <= 1.0:
        raise ValueError('--stretch_prob must be in [0, 1].')
    if not (1.0 <= args.stretch_xy[0] <= args.stretch_xy[1]):
        raise ValueError('--stretch_xy must be ordered MIN MAX with both values >= 1.')
    if not (1.0 <= args.stretch_z[0] <= args.stretch_z[1]):
        raise ValueError('--stretch_z must be ordered MIN MAX with both values >= 1.')
    if not 0.0 <= args.mixup_augment_prob <= 1.0:
        raise ValueError('--mixup_augment_prob must be in [0, 1].')
    if not 0.0 <= args.input_extrema_prob <= 1.0:
        raise ValueError('--input_extrema_prob must be in [0, 1].')
    if not 0.0 <= args.input_sparse_keep_prob <= 1.0:
        raise ValueError('--input_sparse_keep_prob must be in [0, 1].')
    if not 0.0 <= args.input_decimate_trilinear_prob <= 1.0:
        raise ValueError('--input_decimate_trilinear_prob must be in [0, 1].')
    if (args.input_extrema_prob + args.input_sparse_keep_prob + args.input_decimate_trilinear_prob) <= 0.0:
        raise ValueError('At least one of input transform probabilities must be > 0.')
    if args.sparse_keep_fraction_min < 0.01 or args.sparse_keep_fraction_max > 1.0:
        raise ValueError('--sparse_keep_fraction_min/max must be in [0.01, 1.0].')
    if args.sparse_keep_fraction_min > args.sparse_keep_fraction_max:
        raise ValueError('--sparse_keep_fraction_min must be <= --sparse_keep_fraction_max.')
    if args.sparse_poisson_radius_scale < 0.1 or args.sparse_poisson_radius_scale > 2.0:
        raise ValueError('--sparse_poisson_radius_scale must be in [0.1, 2.0].')
    if args.early_stopping_patience <= 0:
        raise ValueError('--early_stopping_patience must be positive.')
    if not 0.0 <= args.loss_mse_weight <= 1.0:
        raise ValueError('--loss_mse_weight must be in [0, 1].')
    if args.gan_weight < 0:
        raise ValueError('--gan_weight must be non-negative.')
    if len(args.deep_supervision_weights) != 3:
        raise ValueError('--deep_supervision_weights must contain exactly 3 values.')
    if any(weight < 0.0 for weight in args.deep_supervision_weights):
        raise ValueError('--deep_supervision_weights must be non-negative.')
    if args.deep_supervision and sum(args.deep_supervision_weights) <= 0.0:
        raise ValueError('--deep_supervision_weights sum must be > 0 when --deep_supervision is enabled.')
    if not 0.0 < args.gan_balance_target_low < 1.0:
        raise ValueError('--gan_balance_target_low must be in (0, 1).')
    if not 0.0 < args.gan_balance_target_high < 1.0:
        raise ValueError('--gan_balance_target_high must be in (0, 1).')
    if args.gan_balance_target_low >= args.gan_balance_target_high:
        raise ValueError('--gan_balance_target_low must be less than --gan_balance_target_high.')
    if args.gan_balance_lookahead_window < 2:
        raise ValueError('--gan_balance_lookahead_window must be >= 2.')
    if args.gan_balance_lookahead_horizon < 1:
        raise ValueError('--gan_balance_lookahead_horizon must be >= 1.')
    if args.gan_balance_lookahead_deadband < 0.0:
        raise ValueError('--gan_balance_lookahead_deadband must be non-negative.')
    if args.gan_balance_lookahead_deadband >= 0.5 * (args.gan_balance_target_high - args.gan_balance_target_low):
        raise ValueError('--gan_balance_lookahead_deadband is too large for the target band width.')
    if args.gan_balance_gan_weight_min < 0.0:
        raise ValueError('--gan_balance_gan_weight_min must be non-negative.')
    if args.gan_balance_gan_weight_min > args.gan_balance_gan_weight_max:
        raise ValueError('--gan_balance_gan_weight_min must be <= --gan_balance_gan_weight_max.')
    if args.gan_balance_gan_weight_down_mult <= 0.0 or args.gan_balance_gan_weight_up_mult <= 0.0:
        raise ValueError('--gan_balance_gan_weight_down_mult and --gan_balance_gan_weight_up_mult must be positive.')
    if args.gan_balance_disc_lr_down_mult <= 0.0 or args.gan_balance_disc_lr_up_mult <= 0.0:
        raise ValueError('--gan_balance_disc_lr_down_mult and --gan_balance_disc_lr_up_mult must be positive.')
    if args.gan_balance_disc_lr_min is not None and args.gan_balance_disc_lr_min <= 0.0:
        raise ValueError('--gan_balance_disc_lr_min must be positive when provided.')
    if args.gan_balance_disc_lr_max is not None and args.gan_balance_disc_lr_max <= 0.0:
        raise ValueError('--gan_balance_disc_lr_max must be positive when provided.')
    if (
        args.gan_balance_disc_lr_min is not None
        and args.gan_balance_disc_lr_max is not None
        and args.gan_balance_disc_lr_min > args.gan_balance_disc_lr_max
    ):
        raise ValueError('--gan_balance_disc_lr_min must be <= --gan_balance_disc_lr_max.')

    frozen_submodules = apply_parameter_freezing(model, args)
    if frozen_submodules:
        print(f"Frozen submodules (no gradient updates): {', '.join(frozen_submodules)}")
    opt = build_optimizer(model, args)
    disc_opt = build_discriminator_optimizer(discriminator, args)
    scheduler = build_scheduler(opt, args)
    rec_loss_fn = build_reconstruction_loss(
        args.reconstruction_loss,
        mse_weight=float(args.loss_mse_weight),
        recon_mae_weight=float(args.recon_mae_weight),
        recon_tv_weight=float(args.recon_tv_weight),
        recon_gdl_weight=float(args.recon_gdl_weight),
    )
    lpips_loss_fn = None
    if args.lpips_weight > 0.0:
        lpips_loss_fn = SliceLPIPSLoss(min_spatial_size=int(args.lpips_min_size)).to(device)
        lpips_loss_fn.eval()
    deep_supervision_loss = None
    if args.deep_supervision:
        deep_supervision_loss = DeepSupervisionLoss(
            base_loss=rec_loss_fn,
            weights=tuple(float(v) for v in args.deep_supervision_weights),
        )

    current_gan_weight = float(args.gan_weight)
    disc_lr_min = None
    disc_lr_max = None
    if disc_opt is not None:
        initial_disc_lr = float(disc_opt.param_groups[0]['lr'])
        disc_lr_min = args.gan_balance_disc_lr_min
        if disc_lr_min is None:
            disc_lr_min = initial_disc_lr * 0.5
        disc_lr_max = args.gan_balance_disc_lr_max
        if disc_lr_max is None:
            disc_lr_max = initial_disc_lr * 1.25
        if disc_lr_min > disc_lr_max:
            raise ValueError('--gan_balance_disc_lr_min must be <= --gan_balance_disc_lr_max (effective values).')

    if args.gan_balance_controller and not args.use_discriminator:
        print('GAN balance controller requested, but discriminator is disabled; controller will be ignored.')
    if args.gan_balance_controller and disc_opt is not None:
        print(
            'GAN balance controller:',
            f"target_acc_range=[{100.0*args.gan_balance_target_low:.1f}%,{100.0*args.gan_balance_target_high:.1f}%]",
            f"gan_weight_bounds=[{args.gan_balance_gan_weight_min:.6f},{args.gan_balance_gan_weight_max:.6f}]",
            f"disc_lr_bounds=[{disc_lr_min:.2e},{disc_lr_max:.2e}]",
            f"lookahead={args.gan_balance_lookahead}",
            f"lookahead_window={args.gan_balance_lookahead_window}",
            f"lookahead_horizon={args.gan_balance_lookahead_horizon}",
            f"lookahead_deadband={args.gan_balance_lookahead_deadband}",
        )

    d_gan_acc_history = deque(maxlen=max(2, int(args.gan_balance_lookahead_window)))

    early_stopping = EarlyStoppingState(best_val_loss=float('inf'), epochs_without_improvement=0)
    best_ckpt_path = Path(args.out_dir) / args.best_checkpoint_name

    metrics_csv_path = Path(args.out_dir) / 'training_metrics.csv'
    tensorboard_dir = Path(args.out_dir) / 'tensorboard'
    representative_metadata_path = Path(args.out_dir) / f'representative_examples_epoch{args.representative_selection_epoch}.pt'
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard log dir: {tensorboard_dir}")
    print(f"Run TensorBoard: uv run tensorboard --logdir {tensorboard_dir}")
    representative_examples = {'training': [], 'validation': []}
    representative_examples_ready = False
    if args.resume is not None and representative_metadata_path.exists() and not args.refresh_representative_examples:
        try:
            representative_examples = _load_representative_example_metadata(representative_metadata_path)
            representative_examples_ready = bool(representative_examples['training']) and bool(representative_examples['validation'])
            if representative_examples_ready:
                print(f"Loaded representative metadata from {representative_metadata_path}")
        except Exception as exc:
            print(f"Warning: failed to load representative metadata from {representative_metadata_path}: {exc}")
    if migrate_metrics_csv_if_needed(metrics_csv_path, METRICS_CSV_COLUMNS):
        print(f"Migrated metrics CSV schema at {metrics_csv_path}")
    csv_exists = metrics_csv_path.exists()
    training_start_time = time.time()
    with metrics_csv_path.open('a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(METRICS_CSV_COLUMNS)

        def print_epoch_header():
            print(
                f"{'Epoch':>9} {'loss':>9} {'lpips':>9} {'val_loss':>10} {'val_lpips':>11} {'kl_weight':>11} {'lr':>9} "
                f"{'gan_weight':>11} {'d_gan_lr':>9} {'g_gan_loss':>11} {'d_gan_loss':>11} {'d_gan_acc':>10} "
                f"{'best_val':>10} {'gan_status':>10} {'elapsed / eta / est finish':>33}"
            )

        print_epoch_header()

        total_display_epochs = resume_completed_epochs + args.epochs

        for epoch_offset in range(args.epochs):
            if epoch_offset > 0 and epoch_offset % 25 == 0:
                print()
                print_epoch_header()

            epoch_idx = resume_completed_epochs + epoch_offset
            epoch_number = epoch_idx + 1
            if unfreeze_encoder_after_warmup(model, opt, epoch_idx, args):
                print(f"Unfroze encoder after {args.freeze_encoder_epochs} completed epochs.")

            batch_sampler = getattr(dl, 'batch_sampler', None)
            set_epoch_fn = getattr(batch_sampler, 'set_epoch', None)
            if callable(set_epoch_fn):
                set_epoch_fn(epoch_idx)

            kl_weight = get_kl_weight(epoch_idx, args)
            args.current_kl_weight = kl_weight
            gan_weight_for_epoch = current_gan_weight
            train_epoch_stats = {}

            (
                train_loss,
                train_lpips_loss,
                g_gan_loss_epoch,
                d_gan_loss_epoch,
                d_gan_acc_epoch,
                train_contrastive_loss,
                train_uniformity_loss,
                train_last_snapshot,
            ) = train_one_epoch(
                model,
                discriminator,
                dl,
                device,
                opt,
                disc_opt,
                steps_per_epoch,
                args.grad_clip,
                kl_weight,
                gan_weight_for_epoch,
                deep_supervision=args.deep_supervision,
                deep_supervision_loss=deep_supervision_loss,
                rec_loss_fn=rec_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                lpips_weight=float(args.lpips_weight),
                reconstruction_loss=args.reconstruction_loss,
                mse_weight=float(args.loss_mse_weight),
                deep_supervision_weights=args.deep_supervision_weights,
                geology_metadata_keys=tuple(str(v) for v in args.geology_metadata_keys),
                geology_loss_weight=float(args.geology_loss_weight),
                geology_metadata_calibration=geology_metadata_calibration,
                geology_background_threshold=float(args.geology_background_threshold),
                geology_background_key_indices=geology_background_key_indices,
                geology_loss_type=str(args.geology_loss_type),
                geology_huber_delta=float(args.geology_huber_delta),
                geology_offdiag_only=bool(args.geology_offdiag_only),
                geology_contrastive_weight=float(args.geology_contrastive_weight),
                geology_contrastive_temperature=float(args.geology_contrastive_temperature),
                geology_uniformity_weight=float(args.geology_uniformity_weight),
                geology_uniformity_t=float(args.geology_uniformity_t),
                geology_strata_presence_threshold=float(args.geology_strata_presence_threshold),
                geology_strata_max_active_keys=int(args.geology_strata_max_active_keys),
                geology_classifier_weight=float(args.geology_classifier_weight),
                geology_classifier_targets=tuple(args.geology_classifier_classes),
                geology_classifier_loss=str(args.geology_classifier_loss),
                geology_classifier_focal_gamma=float(args.geology_classifier_focal_gamma),
                geology_classifier_label_smoothing=float(args.geology_classifier_label_smoothing),
                geology_presence_strata=presence_strata,
                epoch_stats=train_epoch_stats,
                real_dataloader=real_dl,
                real_recon_weight=float(args.real_recon_weight),
            )
            val_loss, val_lpips_loss, val_last_snapshot, geology_diagnostics = validate(
                model,
                args,
                device,
                steps_per_epoch,
                deep_supervision_loss=deep_supervision_loss,
                rec_loss_fn=rec_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                geology_metadata_calibration=geology_metadata_calibration,
                geology_background_key_indices=geology_background_key_indices,
            )
            real_metric_steps = max(1, int(math.ceil(0.2 * steps_per_epoch)))
            real_validation_mae = compute_real_mae(
                model, args, args.real_validation_data, device, real_metric_steps
            )
            real_test_mae = compute_real_mae(
                model, args, args.real_test_data, device, real_metric_steps
            )
            examples_this_epoch = samples_per_epoch
            cumulative_examples = epoch_number * samples_per_epoch

            next_disc_lr = float(disc_opt.param_groups[0]['lr']) if disc_opt is not None else None
            controller_status = 'off'
            controller_acc = float(d_gan_acc_epoch)
            d_gan_acc_history.append(float(d_gan_acc_epoch))
            current_gan_weight, next_disc_lr, controller_status, controller_acc = update_gan_balance_controller(
                args,
                d_gan_acc_epoch,
                d_gan_acc_history,
                current_gan_weight,
                disc_opt,
                disc_lr_min,
                disc_lr_max,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            improved = update_early_stopping(early_stopping, val_loss, args.early_stopping_min_delta)
            if improved:
                torch.save(
                    build_checkpoint_payload(
                        model,
                        epoch=epoch_number,
                        geology_metadata_calibration=geology_metadata_calibration,
                    ),
                    best_ckpt_path,
                )

            if args.save_epoch_checkpoints:
                torch.save(
                    build_checkpoint_payload(
                        model,
                        epoch=epoch_number,
                        geology_metadata_calibration=geology_metadata_calibration,
                    ),
                    Path(args.out_dir)/f"vae_epoch{epoch_number}.pt",
                )

            current_lr = opt.param_groups[0]['lr']
            current_disc_lr = disc_opt.param_groups[0]['lr'] if disc_opt is not None else float('nan')

            writer.add_scalar('train/loss', float(train_loss), epoch_number)
            writer.add_scalar('train/lpips_loss', float(train_lpips_loss), epoch_number)
            writer.add_scalar('train/geology_contrastive_loss', float(train_contrastive_loss), epoch_number)
            writer.add_scalar('train/geology_uniformity_loss', float(train_uniformity_loss), epoch_number)
            if float(args.geology_classifier_weight) > 0.0:
                writer.add_scalar('train/geology_classifier_loss', float(train_epoch_stats.get('geology_classifier_loss', 0.0)), epoch_number)
                print(f"  geology_classifier_loss={train_epoch_stats.get('geology_classifier_loss', 0.0):.4f}")
            writer.add_scalar('validation/loss', float(val_loss), epoch_number)
            writer.add_scalar('validation/lpips_loss', float(val_lpips_loss), epoch_number)
            if math.isfinite(real_validation_mae):
                writer.add_scalar('real/validation_mae', real_validation_mae, epoch_number)
            if math.isfinite(real_test_mae):
                writer.add_scalar('real/test_mae', real_test_mae, epoch_number)
            if geology_diagnostics is not None:
                writer.add_scalar('validation/geology_latent_pair_cosine_correlation', geology_diagnostics['pair_cosine_correlation'], epoch_number)
                writer.add_scalar('validation/geology_latent_similar_cosine', geology_diagnostics['similar_latent_cosine'], epoch_number)
                writer.add_scalar('validation/geology_latent_dissimilar_cosine', geology_diagnostics['dissimilar_latent_cosine'], epoch_number)
                writer.add_scalar('validation/geology_latent_cosine_separation', geology_diagnostics['cosine_separation'], epoch_number)
                writer.add_scalar('validation/geology_latent_neighbor_overlap', geology_diagnostics['neighbor_overlap'], epoch_number)
                for k_value in args.geology_diagnostic_topk:
                    metric_key = f'neighbor_overlap_at_{int(k_value)}'
                    writer.add_scalar(
                        f'validation/geology_latent_{metric_key}',
                        float(geology_diagnostics.get(metric_key, 0.0)),
                        epoch_number,
                    )
            writer.add_scalar('train/lr', float(current_lr), epoch_number)
            writer.add_scalar('train/gan_weight', float(gan_weight_for_epoch), epoch_number)
            writer.add_scalar('train/kl_weight', float(kl_weight), epoch_number)
            writer.add_scalar('train/d_gan_accuracy', float(d_gan_acc_epoch), epoch_number)
            writer.add_scalar('train/d_gan_controller_accuracy', float(controller_acc), epoch_number)
            writer.add_scalar('train/d_gan_lr', float(current_disc_lr), epoch_number)
            writer.add_scalar('train/encoder_lr', float(get_named_group_lr(opt, 'encoder', current_lr)), epoch_number)
            writer.add_scalar('train/decoder_lr', float(get_named_group_lr(opt, 'decoder', current_lr)), epoch_number)
            batch_sampler = getattr(dl, 'batch_sampler', None)
            get_sampler_stats_fn = getattr(batch_sampler, 'get_last_epoch_stats', None)
            if callable(get_sampler_stats_fn):
                sampler_stats_raw = get_sampler_stats_fn()
                sampler_stats = sampler_stats_raw if isinstance(sampler_stats_raw, dict) else {}
                writer.add_scalar('sampling/positive_pair_batch_rate', float(sampler_stats.get('positive_pair_batch_rate', 0.0)), epoch_number)
                writer.add_scalar('sampling/negative_strata_batch_rate', float(sampler_stats.get('negative_strata_batch_rate', 0.0)), epoch_number)
                writer.add_scalar('sampling/avg_unique_strata_per_batch', float(sampler_stats.get('avg_unique_strata_per_batch', 0.0)), epoch_number)
                writer.add_scalar('sampling/fallback_positive_pair_count', float(sampler_stats.get('fallback_positive_pair_count', 0.0)), epoch_number)
                writer.add_scalar('sampling/fallback_duplicate_fill_count', float(sampler_stats.get('fallback_duplicate_fill_count', 0.0)), epoch_number)
                writer.add_scalar('sampling/background_fraction_achieved', float(sampler_stats.get('background_fraction_achieved', 0.0)), epoch_number)
                writer.add_scalar('sampling/hard_fraction_achieved', float(sampler_stats.get('hard_fraction_achieved', 0.0)), epoch_number)
                for stat_key, stat_value in sampler_stats.items():
                    if stat_key.startswith('class_'):
                        writer.add_scalar(f'sampling/{stat_key}', float(stat_value), epoch_number)

            if args.adaptive_sampling_by_mse and epoch_number % args.sampling_snapshot_interval == 0:
                snapshot_recon = compute_full_dataset_recon_snapshot(
                    model,
                    adaptive_eval_ds,
                    args.batch_size,
                    device,
                    reconstruction_loss=args.reconstruction_loss,
                    mse_weight=args.loss_mse_weight,
                    deep_supervision=args.deep_supervision,
                    deep_supervision_weights=args.deep_supervision_weights,
                )
                adaptive_recon_history.append(snapshot_recon)
                adaptive_sample_weights, avg_improvement, score = compute_adaptive_sampling_scores(
                    list(adaptive_recon_history),
                    improvement_weight=args.sampling_improvement_weight,
                )

                adaptive_snapshot_records.append(
                    {
                        'epoch': epoch_number,
                        'recon_loss': snapshot_recon,
                        'average_improvement': avg_improvement,
                        'score': score,
                        'probability': adaptive_sample_weights,
                    }
                )
                save_adaptive_sampling_snapshots(adaptive_snapshot_path, adaptive_snapshot_records)
                dl = build_train_dataloader(
                    ds,
                    args,
                    sample_weights=adaptive_sample_weights,
                    geology_metadata_calibration=geology_metadata_calibration,
                    geology_background_key_indices=geology_background_key_indices,
                    presence_strata=presence_strata,
                )

                writer.add_scalar('adaptive_sampling/recon_mean', float(np.mean(snapshot_recon)), epoch_number)
                writer.add_scalar('adaptive_sampling/improvement_mean', float(np.mean(avg_improvement)), epoch_number)
                writer.add_scalar('adaptive_sampling/score_mean', float(np.mean(score)), epoch_number)
                writer.add_scalar('adaptive_sampling/probability_max', float(np.max(adaptive_sample_weights)), epoch_number)
                writer.add_scalar('adaptive_sampling/probability_min', float(np.min(adaptive_sample_weights)), epoch_number)
                writer.add_scalar('adaptive_sampling/probability_entropy', float(-np.sum(adaptive_sample_weights * np.log(np.clip(adaptive_sample_weights, 1e-12, 1.0)))), epoch_number)

                print(
                    f"Adaptive sampling snapshot @ epoch {epoch_number}: "
                    f"recon_mean={float(np.mean(snapshot_recon)):.6f}, "
                    f"improvement_mean={float(np.mean(avg_improvement)):.6f}, "
                    f"score_mean={float(np.mean(score)):.6f}"
                )

            if (not representative_examples_ready) and epoch_number >= args.representative_selection_epoch:
                representative_examples['training'] = _build_representative_examples(
                    train_last_snapshot,
                    split='training',
                    epoch_number=epoch_number,
                    percentiles=representative_percentiles,
                )
                representative_examples['validation'] = _build_representative_examples(
                    val_last_snapshot,
                    split='validation',
                    epoch_number=epoch_number,
                    percentiles=representative_percentiles,
                )
                representative_examples_ready = bool(representative_examples['training']) and bool(representative_examples['validation'])
                if representative_examples_ready:
                    _save_representative_example_metadata(representative_examples, representative_metadata_path)
                    print(
                        f"Saved representative metadata from epoch {epoch_number} "
                        f"to {representative_metadata_path}"
                    )

            if (
                epoch_number % args.representative_plot_interval == 0
                and representative_examples['training']
                and representative_examples['validation']
            ):
                _log_representative_examples(
                    writer,
                    model,
                    device,
                    representative_examples['training'] + representative_examples['validation'],
                    epoch_number,
                    args.out_dir,
                    rec_loss_fn=rec_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=float(args.lpips_weight),
                )

            csv_writer.writerow([
                epoch_number,
                examples_this_epoch,
                cumulative_examples,
                f"{train_loss:.6f}",
                f"{train_lpips_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_lpips_loss:.6f}",
                f"{real_validation_mae:.6f}" if math.isfinite(real_validation_mae) else '',
                f"{real_test_mae:.6f}" if math.isfinite(real_test_mae) else '',
                f"{kl_weight:.6f}",
                f"{current_lr:.8f}",
                f"{current_disc_lr:.8f}",
                f"{gan_weight_for_epoch:.6f}",
                f"{g_gan_loss_epoch:.6f}",
                f"{d_gan_loss_epoch:.6f}",
                f"{100.0 * d_gan_acc_epoch:.2f}",
                f"{geology_diagnostics['pair_cosine_correlation']:.6f}" if geology_diagnostics is not None else '',
                f"{geology_diagnostics['cosine_separation']:.6f}" if geology_diagnostics is not None else '',
                f"{geology_diagnostics['neighbor_overlap']:.6f}" if geology_diagnostics is not None else '',
                f"{geology_diagnostics.get('neighbor_overlap_at_5', 0.0):.6f}" if geology_diagnostics is not None else '',
                f"{geology_diagnostics.get('neighbor_overlap_at_10', 0.0):.6f}" if geology_diagnostics is not None else '',
                f"{geology_diagnostics.get('neighbor_overlap_at_20', 0.0):.6f}" if geology_diagnostics is not None else '',
                'best' if improved else '',
            ])
            csv_file.flush()
            writer.flush()

            elapsed_summary = ''
            if (epoch_offset + 1) % 5 == 0:
                elapsed_seconds = time.time() - training_start_time
                average_epoch_seconds = elapsed_seconds / float(epoch_offset + 1)
                remaining_epochs = max(0, args.epochs - (epoch_offset + 1))
                remaining_seconds = average_epoch_seconds * float(remaining_epochs)
                estimated_finish = datetime.now() + timedelta(seconds=remaining_seconds)
                elapsed_summary = (
                    f"{format_elapsed_time(elapsed_seconds)} / "
                    f"{format_elapsed_time(remaining_seconds)} / "
                    f"{estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            gan_status_display = controller_status
            if not args.gan_balance_controller or disc_opt is None:
                gan_status_display = 'off'
            d_gan_lr_display = f"{current_disc_lr:9.2e}" if disc_opt is not None else f"{'n/a':>9}"

            print(
                f"{(f'{epoch_number:>{len(str(total_display_epochs))}d}/{total_display_epochs}'):>9} "
                f"{train_loss:9.6f} "
                f"{train_lpips_loss:9.6f} "
                f"{val_loss:10.6f} "
                f"{val_lpips_loss:11.6f} "
                f"{kl_weight:11.6f} "
                f"{current_lr:9.2e} "
                f"{gan_weight_for_epoch:11.6f} "
                f"{d_gan_lr_display} "
                f"{g_gan_loss_epoch:11.6f} "
                f"{d_gan_loss_epoch:11.6f} "
                f"{(100.0 * d_gan_acc_epoch):9.2f}% "
                f"{early_stopping.best_val_loss:10.6f} "
                f"{gan_status_display:>10} "
                f"{elapsed_summary:>33}"
            )
            if geology_diagnostics is not None:
                print(
                    "  Latent geology validation: "
                    f"pair_cos_corr={geology_diagnostics['pair_cosine_correlation']:.4f} "
                    f"similar_cos={geology_diagnostics['similar_latent_cosine']:.4f} "
                    f"dissimilar_cos={geology_diagnostics['dissimilar_latent_cosine']:.4f} "
                    f"gap={geology_diagnostics['cosine_separation']:.4f} "
                    f"neighbor_overlap@{args.geology_diagnostic_neighbor_k}={geology_diagnostics['neighbor_overlap']:.4f} "
                    f"n@5={geology_diagnostics.get('neighbor_overlap_at_5', 0.0):.4f} "
                    f"n@10={geology_diagnostics.get('neighbor_overlap_at_10', 0.0):.4f} "
                    f"n@20={geology_diagnostics.get('neighbor_overlap_at_20', 0.0):.4f}"
                )
            if math.isfinite(real_validation_mae) or math.isfinite(real_test_mae):
                print(f"  Real seismic: validation_mae={real_validation_mae:.6f} test_mae={real_test_mae:.6f}")
            batch_sampler = getattr(dl, 'batch_sampler', None)
            get_sampler_stats_fn = getattr(batch_sampler, 'get_last_epoch_stats', None)
            if callable(get_sampler_stats_fn):
                sampler_stats_raw = get_sampler_stats_fn()
                sampler_stats = sampler_stats_raw if isinstance(sampler_stats_raw, dict) else {}
                print(
                    "  Geology sampler: "
                    f"pos_batch_rate={sampler_stats.get('positive_pair_batch_rate', 0.0):.3f} "
                    f"neg_batch_rate={sampler_stats.get('negative_strata_batch_rate', 0.0):.3f} "
                    f"unique_strata={sampler_stats.get('avg_unique_strata_per_batch', 0.0):.2f} "
                    f"bg_frac={sampler_stats.get('background_fraction_achieved', 0.0):.3f} "
                    f"hard_frac={sampler_stats.get('hard_fraction_achieved', 0.0):.3f} "
                    f"fallback_pos={sampler_stats.get('fallback_positive_pair_count', 0.0):.0f} "
                    f"fallback_dup={sampler_stats.get('fallback_duplicate_fill_count', 0.0):.0f}"
                )
                class_stats = {k: v for k, v in sampler_stats.items() if k.startswith('class_')}
                if class_stats:
                    print('  sampler class stats: ' + ' '.join(f"{k[len('class_'):]}={v:.3f}" for k, v in class_stats.items()))

            if early_stopping.epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping triggered after epoch {epoch_number} "
                    f"(no val improvement for {early_stopping.epochs_without_improvement} epochs)."
                )
                break

    writer.close()

    print(f"Best checkpoint: {best_ckpt_path} (best_val_loss={early_stopping.best_val_loss:.6f})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--patch_size', type=int, nargs='+', default=None, help='Patch size: one value for cubic or three values X Y Z. If omitted, infer from training dataset.')
    p.add_argument('--data', required=True)
    p.add_argument('--real_data', type=str, default=None, help='Real-seismic training patch store appended to each synthetic batch.')
    p.add_argument('--real_validation_data', type=str, default=None, help='Held-out real-seismic validation patch store.')
    p.add_argument('--real_test_data', type=str, default=None, help='Real-seismic test patch store; guardrail metric only.')
    p.add_argument('--real_batch_count', type=int, default=0, help='Real examples K appended to each synthetic batch without geology labels.')
    p.add_argument('--real_recon_weight', type=float, default=1.0, help='Relative reconstruction weight for appended real examples.')
    p.add_argument('--batch_size', '--examples_per_batch', dest='batch_size', type=int, default=100)
    p.add_argument('--number_batches', type=int, default=None, help='Number of batches per epoch. If omitted, uses full dataloader length.')
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--encoder_lr_mult', type=float, default=1.0, help='Multiplier applied to encoder LR relative to --learning_rate.')
    p.add_argument('--decoder_lr_mult', type=float, default=1.0, help='Multiplier applied to decoder LR relative to --learning_rate.')
    p.add_argument('--grad_clip', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=None, help='Training and augmentation seed. If omitted, generate a new seed from system entropy.')
    p.add_argument('--input_scaling', type=str, default='none', choices=['none', 'divide_by_std', 'zscore'])
    p.add_argument('--input_mean', type=float, default=0.0)
    p.add_argument('--input_std', type=float, default=1.0)
    p.add_argument('--augment', action='store_true', help='Enable on-the-fly paired augmentations and input-only trace dropout.')
    p.add_argument('--swap_xy_prob', type=float, default=0.5)
    p.add_argument('--flip_x_prob', type=float, default=0.5)
    p.add_argument('--flip_y_prob', type=float, default=0.5)
    p.add_argument('--vertical_warp_prob', type=float, default=0.5, help='Probability of applying non-linear depth stretch/squeeze to label and paired input.')
    p.add_argument('--phase_rotation_prob', type=float, default=0.0, help='Probability of applying a constant phase rotation (along z) to label and paired input.')
    p.add_argument('--phase_range', type=float, nargs=3, default=[-60.0, 0.0, 40.0], metavar=('MIN_DEG', 'MODE_DEG', 'MAX_DEG'), help='Triangular-distribution (min, mode, max) phase rotation range in degrees.')
    p.add_argument('--stretch_prob', type=float, default=0.0, help='Probability of applying paired zoom-in stretch followed by a center crop.')
    p.add_argument('--stretch_xy', type=float, nargs=2, default=[1.0, 1.25], metavar=('MIN', 'MAX'), help='Uniform XY zoom range; values must be >= 1.')
    p.add_argument('--stretch_z', type=float, nargs=2, default=[1.0, 1.5], metavar=('MIN', 'MAX'), help='Z zoom range; values must be >= 1.')
    p.add_argument(
        '--dip_label_policy',
        choices=list(DIP_LABEL_POLICIES),
        default='adjust',
        help='How dip-dependent labels follow vertical warp: adjust (re-derive from dip_samples_*; see '
             'scripts/add_dip_samples.py), mask (ignore dip-class targets for warped samples), ignore (unchanged). '
             'Azimuth is always adjusted for x/y flips and swaps.',
    )
    p.add_argument('--mixup_augment_prob', type=float, default=0.10, help='Probability of adding extrema-only signal from a random second zarr example into the input volume.')
    p.add_argument('--input_extrema_prob', type=float, default=1.0, help='Weight for selecting extrema-only input transform in one-of-three input transform mode.')
    p.add_argument('--input_sparse_keep_prob', type=float, default=0.0, help='Weight for selecting sparse-keep input transform in one-of-three input transform mode.')
    p.add_argument('--input_decimate_trilinear_prob', type=float, default=0.0, help='Weight for selecting parity decimate+trilinear input transform in one-of-three input transform mode.')
    p.add_argument('--sparse_keep_fraction_min', type=float, default=0.10, help='Minimum per-sample kept-voxel fraction for sparse-keep transform.')
    p.add_argument('--sparse_keep_fraction_max', type=float, default=0.30, help='Maximum per-sample kept-voxel fraction for sparse-keep transform.')
    p.add_argument('--sparse_poisson_radius_scale', type=float, default=0.85, help='Radius scale for Poisson-like branch of sparse selector; the other branch uses uniform thresholding.')
    p.add_argument('--zero_cluster_min', type=int, default=8)
    p.add_argument('--zero_cluster_max', type=int, default=12)
    p.add_argument('--resume', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    p.add_argument('--resume_epoch', type=int, default=None, help='Override the completed epoch count used for resumed runs.')
    p.add_argument('--validation_data', type=str, default='data/validation.zarr', help='Path to validation zarr patches.')
    p.add_argument(
        '--skip_data_location_checks',
        action='store_true',
        help='Skip the startup check that --data and --validation_data come from disjoint synthoseis '
             'volumes and use matching amplitude scaling / label conventions.',
    )
    p.add_argument('--validation_extrema_only', dest='validation_extrema_only', action='store_true', help='Use the same input transform family and probabilities as training for validation data (default).')
    p.add_argument('--no_validation_extrema_only', dest='validation_extrema_only', action='store_false', help='Disable input transforms for validation data.')
    p.set_defaults(validation_extrema_only=True)
    p.add_argument('--kl_schedule', type=str, default='warmup', choices=['warmup', 'fixed'])
    p.add_argument('--kl_start', type=float, default=0.0)
    p.add_argument('--kl_end', type=float, default=1e-3)
    p.add_argument('--kl_warmup_epochs', type=int, default=15)
    p.add_argument('--kl_fixed', type=float, default=1e-3)
    p.add_argument('--deep_supervision', action='store_true', help='Enable MONAI-style decoder deep supervision with auxiliary heads during training.')
    p.add_argument('--residual_encoder', action='store_true', help='Use the residual encoder variant while keeping the external latent contract unchanged.')
    p.add_argument('--encoder_arch', choices=['conv', 'residual', 'resnetv2'], default='conv')
    p.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=None)
    p.add_argument('--encoder_depth_profile', choices=['baseline', 'deeper'], default='baseline')
    p.add_argument('--encoder_stage_blocks', type=int, nargs='+', default=None)
    p.add_argument('--encoder_norm', choices=['batch', 'instance', 'group'], default=None)
    p.add_argument('--encoder_stem', choices=['pretrain_v2', 'light'], default='pretrain_v2')
    p.add_argument('--encoder_input_axes', choices=['xyz', 'zxy'], default='xyz')
    p.add_argument('--decoder_hidden_dims', type=int, nargs='+', default=None)
    p.add_argument('--decoder_block', choices=['conv', 'res'], default='conv')
    p.add_argument('--init_encoder_from', type=str, default=None, help='Pretrain-v2 checkpoint; loads EMA encoder tensors into an E2 ResNetV2 trunk.')
    p.add_argument('--freeze_encoder_epochs', type=int, default=0, help='Freeze encoder for N completed epochs, then unfreeze it.')
    p.add_argument('--geology_projection', action='store_true', help='Add an MLP projection head g(mu)->z_geo (unit-norm) for the contrastive geology embedding used in retrieval.')
    p.add_argument('--geology_proj_hidden', type=int, default=128, help='Hidden width of the geology projection head.')
    p.add_argument('--geology_proj_dim', type=int, default=64, help='Output dimension of the geology projection embedding z_geo.')
    p.add_argument('--geology_contrastive_weight', type=float, default=0.0, help='Weight for the supervised-contrastive (SupCon) loss on z_geo. Requires --geology_projection.')
    p.add_argument('--geology_contrastive_temperature', type=float, default=0.1, help='Temperature for the supervised-contrastive geology loss.')
    p.add_argument('--geology_uniformity_weight', type=float, default=0.0, help='Weight for the hypersphere uniformity regularizer on z_geo (anti-collapse). Requires --geology_projection.')
    p.add_argument('--geology_uniformity_t', type=float, default=2.0, help='Temperature t for the geology uniformity regularizer.')
    p.add_argument('--geology_classifier', action='store_true', help='Add a multi-task geology classifier decoder on mu (presence of 7 classes + dip-mean/dip-range classes).')
    p.add_argument('--geology_classifier_mode', type=str, default='patch', choices=['patch'], help='Classifier head type; voxel mode is planned (WP6).')
    p.add_argument('--geology_classifier_hidden', type=int, default=256, help='Hidden width of the geology classifier trunk.')
    p.add_argument('--geology_classifier_weight', type=float, default=0.0, help='Weight for the geology classifier loss. Requires --geology_classifier and label_presence_* arrays in the training data.')
    p.add_argument('--geology_classifier_loss', type=str, default='bce', choices=['bce', 'focal'], help='Presence loss: BCE with pos_weight, or focal BCE with pos_weight.')
    p.add_argument('--geology_classifier_focal_gamma', type=float, default=2.0, help='Focal loss gamma for --geology_classifier_loss focal.')
    p.add_argument('--geology_classifier_label_smoothing', type=float, default=0.05, help='Label smoothing for the dip-class cross-entropy terms.')
    p.add_argument('--geology_classifier_classes', nargs='+', default=list(ALL_CLASSIFIER_TARGETS), choices=list(ALL_CLASSIFIER_TARGETS), help='Classifier targets included in the loss.')
    p.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights (Phase 1 head-only geology training).')
    p.add_argument('--freeze_decoder', action='store_true', help='Freeze decoder weights (e.g. when only shaping the geology embedding).')
    p.add_argument('--geology_loss_weight', type=float, default=0.0, help='Weight for metadata-to-latent geology similarity loss. Set >0 to enable geology-aware latent shaping.')
    p.add_argument('--geology_loss_type', type=str, default='mse', choices=['mse', 'huber'], help='Pairwise geology-similarity regression loss type for latent-vs-metadata geometry.')
    p.add_argument('--geology_huber_delta', type=float, default=0.1, help='Huber delta for geology loss when --geology_loss_type huber.')
    p.add_argument('--geology_offdiag_only', dest='geology_offdiag_only', action='store_true', help='Exclude similarity-matrix diagonal in geology loss (recommended).')
    p.add_argument('--no_geology_offdiag_only', dest='geology_offdiag_only', action='store_false', help='Include similarity-matrix diagonal in geology loss.')
    p.set_defaults(geology_offdiag_only=True)
    p.add_argument('--geology_calibration_strategy', type=str, default='robust_log1p', choices=['none', 'robust_log1p'], help='Training-split metadata calibration applied before geology cosine targets.')
    p.add_argument('--geology_calibration_eps', type=float, default=1e-6, help='Numerical epsilon for geology metadata calibration stability.')
    p.add_argument('--geology_calibration_clip', type=float, default=6.0, help='Absolute clip applied after calibration standardization; <=0 disables clipping.')
    p.add_argument('--geology_calibration_filename', type=str, default='geology_metadata_calibration.pt', help='Output calibration artifact filename under --out_dir.')
    p.add_argument('--geology_background_threshold', type=float, default=1e-6, help='Raw metadata threshold used to classify selected geology vs neutral background.')
    p.add_argument('--geology_background_keys', nargs='+', default=list(DEFAULT_BACKGROUND_METADATA_KEYS), help='Metadata keys used to decide has-selected-geology mask for pair filtering.')
    p.add_argument('--geology_diagnostic_max_samples', type=int, default=512, help='Maximum validation patches used for latent/geology cosine diagnostics each epoch.')
    p.add_argument('--geology_diagnostic_neighbor_k', type=int, default=5, help='Neighbor count for validation latent/geology top-k overlap.')
    p.add_argument('--geology_diagnostic_topk', type=int, nargs='+', default=[5, 10, 20], help='Neighbor-overlap k values reported during latent geology diagnostics.')
    p.add_argument('--geology_batch_sampler', action='store_true', help='Enable geology-aware constrained batch sampling (Phase 2).')
    p.add_argument('--geology_strata_source', type=str, default='metadata', choices=['metadata', 'presence_labels'], help='Build sampler and SupCon strata from thresholded calibrated metadata (default) or from label_presence_* arrays.')
    p.add_argument('--geology_strata_classes', nargs='+', default=['fault', 'fault_x', 'channel', 'closure', 'onlap', 'flat_spot'], choices=list(GEOLOGY_PRESENCE_CLASSES), help='Presence classes used for strata with --geology_strata_source presence_labels.')
    p.add_argument('--geology_batch_class_quota', nargs='+', default=None, metavar='CLASS=COUNT', help='Minimum samples per batch containing each listed presence class (e.g. fault_x=1 flat_spot=1), filled before the other batch constraints.')
    p.add_argument('--geology_calibration_inclusion_weight', dest='geology_calibration_inclusion_weight', action='store_true', help='Weight metadata calibration by the dataset inclusion_weight array when present (default).')
    p.add_argument('--no_geology_calibration_inclusion_weight', dest='geology_calibration_inclusion_weight', action='store_false', help='Fit metadata calibration without inclusion weights.')
    p.set_defaults(geology_calibration_inclusion_weight=True)
    p.add_argument('--geology_strata_presence_threshold', type=float, default=1e-4, help='Presence threshold on calibrated metadata features for stratum assignment.')
    p.add_argument('--geology_strata_max_active_keys', type=int, default=2, help='Maximum active geology feature keys retained in multi-label stratum signatures.')
    p.add_argument('--geology_batch_background_fraction', type=float, default=0.20, help='Target fraction of neutral/background examples in each geology-aware batch.')
    p.add_argument('--geology_batch_hard_fraction', type=float, default=0.20, help='Target fraction of reconstruction-hard examples in each geology-aware batch.')
    p.add_argument('--geology_batch_hard_top_quantile', type=float, default=0.20, help='Top quantile of adaptive score weights considered hard examples.')
    p.add_argument('--geology_batch_min_negative_strata', type=int, default=2, help='Minimum distinct non-anchor strata sampled as negatives per batch when available.')
    p.add_argument('--geology_batch_require_positive_pair', dest='geology_batch_require_positive_pair', action='store_true', help='Require sampling a same-stratum positive pair in each geology-aware batch when feasible.')
    p.add_argument('--no_geology_batch_require_positive_pair', dest='geology_batch_require_positive_pair', action='store_false', help='Disable required positive-pair constraint in geology-aware batches.')
    p.set_defaults(geology_batch_require_positive_pair=True)
    p.add_argument('--geology_batch_allow_duplicates', action='store_true', help='Allow duplicate dataset indices within a geology-aware batch if constraints are otherwise infeasible.')
    p.add_argument(
        '--geology_metadata_keys',
        nargs='+',
        default=list(DEFAULT_DERIVED_METADATA_KEYS),
        help='Per-patch metadata array keys stored in sampled zarr datasets for geology-aware loss.',
    )
    p.add_argument('--deep_supervision_weights', type=float, nargs=3, default=[1.0, 0.5, 0.25], help='Three deep supervision reconstruction loss weights (fine, mid, coarse).')
    p.add_argument('--reconstruction_loss', type=str, default='mse_pmse', choices=['mse_pmse', 'mae', 'multi_component'], help='Voxelwise reconstruction objective. multi_component is MAE plus optional TV/GDL terms.')
    p.add_argument('--loss_mse_weight', type=float, default=0.6, help='Weight for MSE component of reconstruction loss in [0, 1]; PMSE weight = 1 - this value.')
    p.add_argument('--recon_mae_weight', type=float, default=1.0, help='MAE weight for --reconstruction_loss multi_component.')
    p.add_argument('--recon_tv_weight', type=float, default=0.0, help='Total-variation weight for --reconstruction_loss multi_component.')
    p.add_argument('--recon_gdl_weight', type=float, default=0.0, help='Gradient-difference weight for --reconstruction_loss multi_component.')
    p.add_argument('--lpips_weight', type=float, default=0.0, help='Weight for optional slice-wise LPIPS perceptual loss. Default 0 keeps baseline behavior.')
    p.add_argument('--lpips_min_size', type=int, default=64, help='Minimum LPIPS slice height/width. Smaller slices are bilinearly upsampled before LPIPS.')
    p.add_argument('--lr_scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr_scheduler_patience', type=int, default=3)
    p.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    p.add_argument('--lr_scheduler_min_lr', type=float, default=1e-6)
    p.add_argument('--early_stopping_patience', type=int, default=8)
    p.add_argument('--early_stopping_min_delta', type=float, default=0.0)
    p.add_argument('--use_discriminator', action='store_true', help='Enable GAN-style discriminator training on real vs reconstructed cubes.')
    p.add_argument('--discriminator_base_ch', type=int, default=16)
    p.add_argument('--gan_weight', type=float, default=1e-3)
    p.add_argument('--discriminator_learning_rate', type=float, default=None)
    p.add_argument('--discriminator_weight_decay', type=float, default=None)
    p.add_argument('--gan_balance_controller', action='store_true', help='Enable automatic epoch-level balancing of gan_weight and discriminator LR using d_gan_acc.')
    p.add_argument('--gan_balance_target_low', type=float, default=0.60, help='Lower bound of target discriminator accuracy band (fraction).')
    p.add_argument('--gan_balance_target_high', type=float, default=0.80, help='Upper bound of target discriminator accuracy band (fraction).')
    p.add_argument('--gan_balance_gan_weight_min', type=float, default=0.01)
    p.add_argument('--gan_balance_gan_weight_max', type=float, default=0.20)
    p.add_argument('--gan_balance_gan_weight_down_mult', type=float, default=0.98)
    p.add_argument('--gan_balance_gan_weight_up_mult', type=float, default=1.02)
    p.add_argument('--gan_balance_disc_lr_min', type=float, default=None)
    p.add_argument('--gan_balance_disc_lr_max', type=float, default=None)
    p.add_argument('--gan_balance_disc_lr_down_mult', type=float, default=0.98)
    p.add_argument('--gan_balance_disc_lr_up_mult', type=float, default=1.02)
    p.add_argument('--gan_balance_lookahead', action='store_true', help='Use trend look-ahead for GAN balance controller decisions based on recent discriminator accuracy.')
    p.add_argument('--gan_balance_lookahead_window', type=int, default=5, help='Number of recent epochs used for linear fit of d_gan_acc when look-ahead is enabled.')
    p.add_argument('--gan_balance_lookahead_horizon', type=int, default=1, help='Prediction horizon in epochs for d_gan_acc look-ahead control.')
    p.add_argument('--gan_balance_lookahead_deadband', type=float, default=0.01, help='Predictive-mode-only deadband (fraction) applied inward from target edges to reduce control oscillation.')
    p.add_argument('--best_checkpoint_name', type=str, default='vae_best.pt')
    p.add_argument('--save_epoch_checkpoints', dest='save_epoch_checkpoints', action='store_true', help='Save per-epoch checkpoints in addition to best checkpoint.')
    p.add_argument('--no_save_epoch_checkpoints', dest='save_epoch_checkpoints', action='store_false', help='Disable per-epoch checkpoint saving and keep only best checkpoint.')
    p.set_defaults(save_epoch_checkpoints=True)
    p.add_argument('--representative_selection_epoch', type=int, default=4, help='Epoch number used to select representative examples from last-batch MSE percentiles.')
    p.add_argument('--representative_plot_interval', type=int, default=5, help='Generate representative plots every N epochs, reusing the selected examples.')
    p.add_argument('--refresh_representative_examples', action='store_true', help='Ignore saved representative examples when resuming and select new examples from the active datasets.')
    p.add_argument('--adaptive_sampling_by_mse', action='store_true', help='Enable adaptive training sampling probabilities from full-dataset blended reconstruction-loss snapshots.')
    p.add_argument('--sampling_snapshot_interval', type=int, default=5, help='Recompute full-dataset blended reconstruction loss every N epochs when adaptive sampling is enabled.')
    p.add_argument('--sampling_improvement_window', type=int, default=3, help='Number of recent reconstruction-loss snapshots kept to compute average improvement (>=2).')
    p.add_argument('--sampling_improvement_weight', type=float, default=1.0, help='Weight on average improvement term in score: score = current_recon + weight * avg_improvement.')
    p.add_argument('--sampling_snapshot_filename', type=str, default='adaptive_sampling_snapshots.pt', help='Output filename under --out_dir for stored per-example adaptive sampling snapshots.')
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    args.seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.patch_size_xyz = normalize_patch_size(args.patch_size) if args.patch_size is not None else None
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

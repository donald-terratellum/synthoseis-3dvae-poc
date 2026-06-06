import argparse
from pathlib import Path
import sys
from typing import Any, cast, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import itertools
import math
import csv
import time
import re
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
import torch
import torch.nn as nn
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
from src.model import VAE3D


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

    def __getitem__(self, idx):
        arr = self._load_scaled_example(int(idx))

        # For denoising-style augmentation, label stays clean while input is perturbed.
        x = arr.copy()
        y = arr.copy()
        if self.augment:
            x, y = apply_pair_augmentations(
                x,
                y,
                self.swap_xy_prob,
                self.flip_x_prob,
                self.flip_y_prob,
                self.vertical_warp_prob,
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
        return torch.from_numpy(x), torch.from_numpy(y)


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

    def __init__(self, mse_weight: float = 0.6, eps: float = 1e-8):
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
        pmse = mse / torch.clamp(label_energy, min=self.eps)
        return self.mse_weight * mse + self.pmse_weight * pmse


def compute_vae_losses(recon, targets, mu, logvar, kl_weight, deep_supervision_loss=None, rec_loss_fn=None):
    if deep_supervision_loss is not None:
        rec_loss = deep_supervision_loss(recon, targets)
    else:
        _rec_fn = rec_loss_fn if rec_loss_fn is not None else torch.nn.functional.mse_loss
        if isinstance(recon, (list, tuple)):
            rec_loss = _rec_fn(recon[0], targets)
        else:
            rec_loss = _rec_fn(recon, targets)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.numel()
    loss = rec_loss + kl_weight * kld
    return loss, rec_loss, kld


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


def compute_average_loss(model, dataloader, device, steps, kl_weight, deep_supervision=False, deep_supervision_loss=None, rec_loss_fn=None):
    model.eval()
    total_loss = 0.0
    batch_iter = iter(dataloader) if steps is None else itertools.cycle(dataloader)
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if deep_supervision:
                recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
                loss, _, _ = compute_vae_losses(ds_outputs, targets, mu, logvar, kl_weight, deep_supervision_loss, rec_loss_fn=rec_loss_fn)
            else:
                recon, mu, logvar = model(inputs)
                loss, _, _ = compute_vae_losses(recon, targets, mu, logvar, kl_weight, rec_loss_fn=rec_loss_fn)
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


def build_optimizer(model, args):
    if args.encoder_lr_mult <= 0.0:
        raise ValueError('--encoder_lr_mult must be positive.')
    if args.decoder_lr_mult <= 0.0:
        raise ValueError('--decoder_lr_mult must be positive.')

    base_lr = float(args.learning_rate)
    if args.encoder_lr_mult == 1.0 and args.decoder_lr_mult == 1.0:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    tracked_ids = {id(p) for p in encoder_params + decoder_params}
    other_params = [p for p in model.parameters() if id(p) not in tracked_ids]

    param_groups = [
        {
            'params': encoder_params,
            'lr': base_lr * float(args.encoder_lr_mult),
            'name': 'encoder',
        },
        {
            'params': decoder_params,
            'lr': base_lr * float(args.decoder_lr_mult),
            'name': 'decoder',
        },
    ]
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other'})
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=args.weight_decay)


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


def clamp_float(value, lower, upper):
    return max(lower, min(upper, value))


def build_checkpoint_payload(model, epoch=None):
    payload = {
        'model_state_dict': model.state_dict(),
        'patch_shape': [int(v) for v in model.patch_shape],
        'latent_dim': int(model.latent_dim),
        'base_ch': int(model.base_ch),
        'deep_supervision': bool(getattr(model, 'deep_supervision', False)),
    }
    if epoch is not None:
        payload['epoch'] = int(epoch)
    return payload


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


def compute_per_example_deep_supervision_combined_recon_loss(outputs, target, weights, mse_weight, eps=1e-8):
    if isinstance(outputs, torch.Tensor):
        return compute_per_example_combined_recon_loss(outputs, target, mse_weight, eps=eps)
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
        total = total + (float(weight) * compute_per_example_combined_recon_loss(pred, target_for_scale, mse_weight, eps=eps))
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


def _plot_representative_example(model, device, example, epoch_number, vmin=-3.1, vmax=3.1):
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
            f"epoch4_recon={example.selection_mse:.6f} | batch_idx={example.batch_index}"
        ),
        fontsize=10,
    )
    return fig, mse_value, pmse_value


def _log_representative_examples(writer, model, device, examples, epoch_number, output_dir):
    if not examples:
        return
    plot_root = Path(output_dir) / 'representative_plots' / f'epoch_{epoch_number:04d}'
    plot_root.mkdir(parents=True, exist_ok=True)

    for example in examples:
        fig, mse_value, pmse_value = _plot_representative_example(model, device, example, epoch_number)
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
    mse_weight=0.6,
    deep_supervision_weights=None,
):
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_loss = 0.0
    total_g_gan_loss = 0.0
    total_d_gan_loss = 0.0
    total_d_gan_acc = 0.0
    batch_iter = iter(dataloader) if steps_per_epoch is None else itertools.cycle(dataloader)

    last_snapshot = None
    for _ in range(steps_per_epoch):
        inputs, targets = next(batch_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)
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
            d_gan_loss, d_gan_accuracy = compute_discriminator_gan_loss(discriminator, targets, recon_for_d.detach())
            d_gan_loss.backward()
            disc_optimizer.step()
            d_gan_loss_value = float(d_gan_loss.item())
            d_gan_acc_value = float(d_gan_accuracy.item())

        # Generator (VAE) step.
        if deep_supervision:
            recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
            vae_loss, _, _ = compute_vae_losses(ds_outputs, targets, mu, logvar, kl_weight, deep_supervision_loss, rec_loss_fn=rec_loss_fn)
        else:
            recon, mu, logvar = model(inputs)
            vae_loss, _, _ = compute_vae_losses(recon, targets, mu, logvar, kl_weight, rec_loss_fn=rec_loss_fn)

        g_gan_loss_value = 0.0
        total_g_loss = vae_loss
        if discriminator is not None:
            g_gan_loss = compute_generator_gan_loss(discriminator, recon)
            g_gan_loss_value = float(g_gan_loss.item())
            total_g_loss = total_g_loss + gan_weight * g_gan_loss

        optimizer.zero_grad()
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += total_g_loss.item()
        total_g_gan_loss += g_gan_loss_value
        total_d_gan_loss += d_gan_loss_value
        total_d_gan_acc += d_gan_acc_value
        if deep_supervision:
            if deep_supervision_weights is None:
                raise ValueError('deep_supervision_weights must be provided when deep supervision is enabled.')
            if ds_outputs is None:
                raise ValueError('deep supervision outputs are required when deep supervision is enabled.')
            ds_outputs_detached = tuple(out.detach() for out in ds_outputs)
            per_example_mse = compute_per_example_deep_supervision_combined_recon_loss(
                ds_outputs_detached,
                targets.detach(),
                weights=tuple(float(v) for v in deep_supervision_weights),
                mse_weight=mse_weight,
            ).detach().cpu()
        else:
            per_example_mse = compute_per_example_combined_recon_loss(
                recon.detach(),
                targets.detach(),
                mse_weight,
            ).detach().cpu()
        last_snapshot = BatchSnapshot(
            inputs=inputs.detach().cpu().clone(),
            targets=targets.detach().cpu().clone(),
            recon=recon.detach().cpu().clone(),
            per_example_mse=per_example_mse,
        )

    return (
        total_loss / steps_per_epoch,
        total_g_gan_loss / steps_per_epoch,
        total_d_gan_loss / steps_per_epoch,
        total_d_gan_acc / steps_per_epoch,
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


def build_dataset(args, data_path, augment=False):
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
    )


def build_train_dataloader(dataset, args, sample_weights=None):
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


def compute_full_dataset_recon_snapshot(model, dataset, batch_size, device, mse_weight, deep_supervision=False, deep_supervision_weights=None):
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
                batch_recon = compute_per_example_deep_supervision_combined_recon_loss(
                    ds_outputs,
                    targets,
                    weights=tuple(float(v) for v in deep_supervision_weights),
                    mse_weight=mse_weight,
                )
            else:
                recon, _, _ = model(inputs)

                batch_recon = compute_per_example_combined_recon_loss(recon, targets, mse_weight)

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


def validate(model, args, device, train_steps_per_epoch, deep_supervision_loss=None, rec_loss_fn=None):
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
    )
    validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    if validation_ds.patch_shape != args.patch_size_xyz:
        raise ValueError(
            f"validation patch shape {validation_ds.patch_shape} does not match --patch_size {args.patch_size_xyz}"
        )
    validation_steps = max(1, int(math.ceil(0.2 * train_steps_per_epoch)))
    model.eval()
    total_loss = 0.0
    batch_iter = itertools.cycle(validation_dl)
    last_snapshot = None
    with torch.no_grad():
        for _ in range(validation_steps):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            ds_outputs = None
            if args.deep_supervision:
                recon, mu, logvar, ds_outputs = model(inputs, return_deep_supervision=True)
                loss, _, _ = compute_vae_losses(
                    ds_outputs,
                    targets,
                    mu,
                    logvar,
                    args.current_kl_weight,
                    deep_supervision_loss,
                    rec_loss_fn=rec_loss_fn,
                )
            else:
                recon, mu, logvar = model(inputs)
                loss, _, _ = compute_vae_losses(recon, targets, mu, logvar, args.current_kl_weight, rec_loss_fn=rec_loss_fn)
            total_loss += float(loss.item())
            if args.deep_supervision:
                if ds_outputs is None:
                    raise ValueError('deep supervision outputs are required when deep supervision is enabled.')
                per_example_mse = compute_per_example_deep_supervision_combined_recon_loss(
                    ds_outputs,
                    targets,
                    weights=tuple(float(v) for v in args.deep_supervision_weights),
                    mse_weight=float(args.loss_mse_weight),
                ).detach().cpu()
            else:
                per_example_mse = compute_per_example_combined_recon_loss(
                    recon,
                    targets,
                    mse_weight=float(args.loss_mse_weight),
                ).detach().cpu()
            last_snapshot = BatchSnapshot(
                inputs=inputs.detach().cpu().clone(),
                targets=targets.detach().cpu().clone(),
                recon=recon.detach().cpu().clone(),
                per_example_mse=per_example_mse,
            )
    return total_loss / validation_steps, last_snapshot


def train(args):
    ds = build_dataset(args, args.data, augment=args.augment)
    args.patch_size_xyz = resolve_patch_size_xyz(args.patch_size, ds.patch_shape)
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

    dl = build_train_dataloader(ds, args, sample_weights=adaptive_sample_weights)

    if args.number_batches is not None and args.number_batches <= 0:
        raise ValueError('--number_batches must be a positive integer when provided.')
    steps_per_epoch = args.number_batches if args.number_batches is not None else len(dl)
    samples_per_epoch = steps_per_epoch * args.batch_size

    model = VAE3D(
        in_ch=1,
        out_ch=1,
        base_ch=16,
        latent_dim=128,
        patch_shape=args.patch_size_xyz,
        deep_supervision=args.deep_supervision,
    )
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

        state_dict = checkpoint['model_state_dict']
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        allowed_ds_missing = {
            'decoder.aux_head_coarse.weight',
            'decoder.aux_head_coarse.bias',
            'decoder.aux_head_mid.weight',
            'decoder.aux_head_mid.bias',
        }
        allowed_ds_unexpected = allowed_ds_missing
        invalid_missing = [k for k in missing_keys if k not in allowed_ds_missing]
        invalid_unexpected = [k for k in unexpected_keys if k not in allowed_ds_unexpected]
        if invalid_missing or invalid_unexpected:
            raise ValueError(
                'Resume checkpoint model_state_dict is incompatible with current architecture. '
                f'invalid missing keys={invalid_missing}, invalid unexpected keys={invalid_unexpected}'
            )
        checkpoint_epoch = checkpoint.get('epoch', None)
        if checkpoint_epoch is not None:
            resume_completed_epochs = max(0, int(checkpoint_epoch))
        else:
            resume_completed_epochs = infer_completed_epochs_from_resume_path(ckpt_path)
        print(f"Resumed model weights from {ckpt_path}")
        print(f"Resuming epoch numbering from {resume_completed_epochs + 1}")

    print(f"Using device: {device}")
    print(f"Batch size (B): {args.batch_size}, batches/epoch: {steps_per_epoch}, examples/epoch: {samples_per_epoch}")
    print(
        "Augmentations:",
        f"enabled={args.augment}",
        f"swap_xy_prob={args.swap_xy_prob}",
        f"flip_x_prob={args.flip_x_prob}",
        f"flip_y_prob={args.flip_y_prob}",
        f"vertical_warp_prob={args.vertical_warp_prob}",
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
    checkpoint_keys = ['model_state_dict', 'patch_shape', 'latent_dim', 'base_ch', 'deep_supervision']
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
        f"mse_weight={args.loss_mse_weight:.4f}",
        f"pmse_weight={1.0 - args.loss_mse_weight:.4f}",
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
    if not 0.0 <= args.vertical_warp_prob <= 1.0:
        raise ValueError('--vertical_warp_prob must be in [0, 1].')
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

    opt = build_optimizer(model, args)
    disc_opt = build_discriminator_optimizer(discriminator, args)
    scheduler = build_scheduler(opt, args)
    rec_loss_fn = CombinedReconLoss(mse_weight=float(args.loss_mse_weight))
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
    representative_metadata_path = Path(args.out_dir) / 'representative_examples_epoch4.pt'
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard log dir: {tensorboard_dir}")
    print(f"Run TensorBoard: uv run tensorboard --logdir {tensorboard_dir}")
    representative_examples = {'training': [], 'validation': []}
    representative_examples_ready = False
    if args.resume is not None and representative_metadata_path.exists():
        try:
            representative_examples = _load_representative_example_metadata(representative_metadata_path)
            representative_examples_ready = bool(representative_examples['training']) and bool(representative_examples['validation'])
            if representative_examples_ready:
                print(f"Loaded representative metadata from {representative_metadata_path}")
        except Exception as exc:
            print(f"Warning: failed to load representative metadata from {representative_metadata_path}: {exc}")
    csv_exists = metrics_csv_path.exists()
    training_start_time = time.time()
    with metrics_csv_path.open('a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow([
                'epoch',
                'examples_this_epoch',
                'cumulative_examples',
                'train_loss',
                'val_loss',
                'kl_weight',
                'learning_rate',
                'discriminator_learning_rate',
                'gan_weight',
                'g_gan_loss',
                'd_gan_loss',
                'd_gan_acc_pct',
                'best_model',
            ])

        def print_epoch_header():
            print(
                f"{'Epoch':>9} {'loss':>9} {'val_loss':>10} {'kl_weight':>11} {'lr':>9} "
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

            kl_weight = get_kl_weight(epoch_idx, args)
            args.current_kl_weight = kl_weight
            gan_weight_for_epoch = current_gan_weight

            (
                train_loss,
                g_gan_loss_epoch,
                d_gan_loss_epoch,
                d_gan_acc_epoch,
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
                mse_weight=float(args.loss_mse_weight),
                deep_supervision_weights=args.deep_supervision_weights,
            )
            val_loss, val_last_snapshot = validate(
                model,
                args,
                device,
                steps_per_epoch,
                deep_supervision_loss=deep_supervision_loss,
                rec_loss_fn=rec_loss_fn,
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
                torch.save(build_checkpoint_payload(model, epoch=epoch_number), best_ckpt_path)

            if args.save_epoch_checkpoints:
                torch.save(build_checkpoint_payload(model, epoch=epoch_number), Path(args.out_dir)/f"vae_epoch{epoch_number}.pt")

            current_lr = opt.param_groups[0]['lr']
            current_disc_lr = disc_opt.param_groups[0]['lr'] if disc_opt is not None else float('nan')

            writer.add_scalar('train/loss', float(train_loss), epoch_number)
            writer.add_scalar('validation/loss', float(val_loss), epoch_number)
            writer.add_scalar('train/lr', float(current_lr), epoch_number)
            writer.add_scalar('train/gan_weight', float(gan_weight_for_epoch), epoch_number)
            writer.add_scalar('train/kl_weight', float(kl_weight), epoch_number)
            writer.add_scalar('train/d_gan_accuracy', float(d_gan_acc_epoch), epoch_number)
            writer.add_scalar('train/d_gan_controller_accuracy', float(controller_acc), epoch_number)
            writer.add_scalar('train/d_gan_lr', float(current_disc_lr), epoch_number)
            writer.add_scalar('train/encoder_lr', float(get_named_group_lr(opt, 'encoder', current_lr)), epoch_number)
            writer.add_scalar('train/decoder_lr', float(get_named_group_lr(opt, 'decoder', current_lr)), epoch_number)

            if args.adaptive_sampling_by_mse and epoch_number % args.sampling_snapshot_interval == 0:
                snapshot_recon = compute_full_dataset_recon_snapshot(
                    model,
                    adaptive_eval_ds,
                    args.batch_size,
                    device,
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
                dl = build_train_dataloader(ds, args, sample_weights=adaptive_sample_weights)

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
                )

            csv_writer.writerow([
                epoch_number,
                examples_this_epoch,
                cumulative_examples,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{kl_weight:.6f}",
                f"{current_lr:.8f}",
                f"{current_disc_lr:.8f}",
                f"{gan_weight_for_epoch:.6f}",
                f"{g_gan_loss_epoch:.6f}",
                f"{d_gan_loss_epoch:.6f}",
                f"{100.0 * d_gan_acc_epoch:.2f}",
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
                f"{val_loss:10.6f} "
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
    p.add_argument('--batch_size', '--examples_per_batch', dest='batch_size', type=int, default=100)
    p.add_argument('--number_batches', type=int, default=None, help='Number of batches per epoch. If omitted, uses full dataloader length.')
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--encoder_lr_mult', type=float, default=1.0, help='Multiplier applied to encoder LR relative to --learning_rate.')
    p.add_argument('--decoder_lr_mult', type=float, default=1.0, help='Multiplier applied to decoder LR relative to --learning_rate.')
    p.add_argument('--grad_clip', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--input_scaling', type=str, default='none', choices=['none', 'divide_by_std', 'zscore'])
    p.add_argument('--input_mean', type=float, default=0.0)
    p.add_argument('--input_std', type=float, default=1.0)
    p.add_argument('--augment', action='store_true', help='Enable on-the-fly paired augmentations and input-only trace dropout.')
    p.add_argument('--swap_xy_prob', type=float, default=0.5)
    p.add_argument('--flip_x_prob', type=float, default=0.5)
    p.add_argument('--flip_y_prob', type=float, default=0.5)
    p.add_argument('--vertical_warp_prob', type=float, default=0.5, help='Probability of applying non-linear depth stretch/squeeze to label and paired input.')
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
    p.add_argument('--validation_data', type=str, default='data/validation.zarr', help='Path to validation zarr patches.')
    p.add_argument('--validation_extrema_only', dest='validation_extrema_only', action='store_true', help='Use the same input transform family and probabilities as training for validation data (default).')
    p.add_argument('--no_validation_extrema_only', dest='validation_extrema_only', action='store_false', help='Disable input transforms for validation data.')
    p.set_defaults(validation_extrema_only=True)
    p.add_argument('--kl_schedule', type=str, default='warmup', choices=['warmup', 'fixed'])
    p.add_argument('--kl_start', type=float, default=0.0)
    p.add_argument('--kl_end', type=float, default=1e-3)
    p.add_argument('--kl_warmup_epochs', type=int, default=15)
    p.add_argument('--kl_fixed', type=float, default=1e-3)
    p.add_argument('--deep_supervision', action='store_true', help='Enable MONAI-style decoder deep supervision with auxiliary heads during training.')
    p.add_argument('--deep_supervision_weights', type=float, nargs=3, default=[1.0, 0.5, 0.25], help='Three deep supervision reconstruction loss weights (fine, mid, coarse).')
    p.add_argument('--loss_mse_weight', type=float, default=0.6, help='Weight for MSE component of reconstruction loss in [0, 1]; PMSE weight = 1 - this value.')
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
    p.add_argument('--adaptive_sampling_by_mse', action='store_true', help='Enable adaptive training sampling probabilities from full-dataset blended reconstruction-loss snapshots.')
    p.add_argument('--sampling_snapshot_interval', type=int, default=5, help='Recompute full-dataset blended reconstruction loss every N epochs when adaptive sampling is enabled.')
    p.add_argument('--sampling_improvement_window', type=int, default=3, help='Number of recent reconstruction-loss snapshots kept to compute average improvement (>=2).')
    p.add_argument('--sampling_improvement_weight', type=float, default=1.0, help='Weight on average improvement term in score: score = current_recon + weight * avg_improvement.')
    p.add_argument('--sampling_snapshot_filename', type=str, default='adaptive_sampling_snapshots.pt', help='Output filename under --out_dir for stored per-example adaptive sampling snapshots.')
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    args.patch_size_xyz = normalize_patch_size(args.patch_size) if args.patch_size is not None else None
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

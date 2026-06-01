import argparse
from pathlib import Path
import itertools
import math
import csv
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import zarr
import numpy as np
from typing import Any, cast
from src.augmentations import apply_input_trace_dropout
from src.augmentations import apply_input_extrema_mixup
from src.augmentations import apply_pair_augmentations
from src.augmentations import keep_trace_extrema_only
from src.augmentations import sample_mixup_corpus_index
from src.model import VAE3D


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
        extrema_only=False,
        mixup_augment_prob=0.10,
    ):
        z = cast(Any, zarr.open(str(zarr_path), mode='r'))
        self.data = cast(Any, z['patches'])
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
        self.extrema_only = bool(extrema_only)
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
        if not 0.0 <= self.mixup_augment_prob <= 1.0:
            raise ValueError('--mixup_augment_prob must be in [0, 1].')

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
        if self.extrema_only:
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


def compute_vae_losses(recon, targets, mu, logvar, kl_weight):
    rec_loss = torch.nn.functional.mse_loss(recon, targets)
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


def compute_average_loss(model, dataloader, device, steps, kl_weight):
    model.eval()
    total_loss = 0.0
    batch_iter = iter(dataloader) if steps is None else itertools.cycle(dataloader)
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            recon, mu, logvar = model(inputs)
            loss, _, _ = compute_vae_losses(recon, targets, mu, logvar, kl_weight)
            total_loss += loss.item()
    return total_loss / steps


def get_kl_weight(epoch_idx, args):
    if args.kl_schedule == 'fixed':
        return float(args.kl_fixed)

    # Linear warmup from kl_start to kl_end.
    warmup_epochs = max(1, int(args.kl_warmup_epochs))
    progress = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return float(args.kl_start + progress * (args.kl_end - args.kl_start))


def build_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


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


def update_gan_balance_controller(
    args,
    d_gan_acc_epoch,
    current_gan_weight,
    disc_optimizer,
    disc_lr_min,
    disc_lr_max,
):
    if not args.gan_balance_controller or disc_optimizer is None:
        return current_gan_weight, None, 'off'

    current_disc_lr = float(disc_optimizer.param_groups[0]['lr'])
    next_gan_weight = current_gan_weight
    next_disc_lr = current_disc_lr
    status = 'hold'

    if d_gan_acc_epoch > args.gan_balance_target_high:
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
        status = 'd_strong'
    elif d_gan_acc_epoch < args.gan_balance_target_low:
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
        status = 'd_weak'

    for param_group in disc_optimizer.param_groups:
        param_group['lr'] = next_disc_lr

    return next_gan_weight, next_disc_lr, status


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
):
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_loss = 0.0
    total_g_gan_loss = 0.0
    total_d_gan_loss = 0.0
    total_d_gan_acc = 0.0
    batch_iter = iter(dataloader) if steps_per_epoch is None else itertools.cycle(dataloader)

    for _ in range(steps_per_epoch):
        inputs, targets = next(batch_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)

        d_gan_loss_value = 0.0
        d_gan_acc_value = 0.0
        if discriminator is not None:
            # Discriminator step.
            with torch.no_grad():
                recon_for_d, _, _ = model(inputs)
            disc_optimizer.zero_grad()
            d_gan_loss, d_gan_accuracy = compute_discriminator_gan_loss(discriminator, targets, recon_for_d.detach())
            d_gan_loss.backward()
            disc_optimizer.step()
            d_gan_loss_value = float(d_gan_loss.item())
            d_gan_acc_value = float(d_gan_accuracy.item())

        # Generator (VAE) step.
        recon, mu, logvar = model(inputs)
        vae_loss, _, _ = compute_vae_losses(recon, targets, mu, logvar, kl_weight)
        g_gan_loss_value = 0.0
        if discriminator is not None:
            g_gan_loss = compute_generator_gan_loss(discriminator, recon)
            g_gan_loss_value = float(g_gan_loss.item())
            total_g_loss = vae_loss + gan_weight * g_gan_loss
        else:
            total_g_loss = vae_loss

        optimizer.zero_grad()
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += total_g_loss.item()
        total_g_gan_loss += g_gan_loss_value
        total_d_gan_loss += d_gan_loss_value
        total_d_gan_acc += d_gan_acc_value

    return (
        total_loss / steps_per_epoch,
        total_g_gan_loss / steps_per_epoch,
        total_d_gan_loss / steps_per_epoch,
        total_d_gan_acc / steps_per_epoch,
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
        extrema_only=True,
        mixup_augment_prob=args.mixup_augment_prob,
    )


def validate(model, args, device, train_steps_per_epoch):
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
        extrema_only=args.validation_extrema_only,
        mixup_augment_prob=0.0,
    )
    validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    validation_steps = max(1, int(math.ceil(0.2 * train_steps_per_epoch)))
    return compute_average_loss(model, validation_dl, device, validation_steps, args.current_kl_weight)


def train(args):
    ds = build_dataset(args, args.data, augment=args.augment)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.number_batches is not None and args.number_batches <= 0:
        raise ValueError('--number_batches must be a positive integer when provided.')
    steps_per_epoch = args.number_batches if args.number_batches is not None else len(dl)
    samples_per_epoch = steps_per_epoch * args.batch_size

    model = VAE3D(in_ch=1, out_ch=1, base_ch=16, latent_dim=128)
    discriminator = build_discriminator(args)
    device = resolve_device(args.device)

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {ckpt_path}")

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
    )
    print("Train extrema-only input=True")
    print(f"Validation extrema-only input={args.validation_extrema_only}")
    print(f"Discriminator enabled={args.use_discriminator}")
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
    if args.early_stopping_patience <= 0:
        raise ValueError('--early_stopping_patience must be positive.')
    if args.gan_weight < 0:
        raise ValueError('--gan_weight must be non-negative.')
    if not 0.0 < args.gan_balance_target_low < 1.0:
        raise ValueError('--gan_balance_target_low must be in (0, 1).')
    if not 0.0 < args.gan_balance_target_high < 1.0:
        raise ValueError('--gan_balance_target_high must be in (0, 1).')
    if args.gan_balance_target_low >= args.gan_balance_target_high:
        raise ValueError('--gan_balance_target_low must be less than --gan_balance_target_high.')
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
        )

    early_stopping = EarlyStoppingState(best_val_loss=float('inf'), epochs_without_improvement=0)
    best_ckpt_path = Path(args.out_dir) / args.best_checkpoint_name

    metrics_csv_path = Path(args.out_dir) / 'training_metrics.csv'
    csv_exists = metrics_csv_path.exists()
    training_start_time = time.time()
    with metrics_csv_path.open('a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow([
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

        for epoch in range(args.epochs):
            if epoch > 0 and epoch % 25 == 0:
                print()
                print_epoch_header()

            kl_weight = get_kl_weight(epoch, args)
            args.current_kl_weight = kl_weight
            gan_weight_for_epoch = current_gan_weight

            train_loss, g_gan_loss_epoch, d_gan_loss_epoch, d_gan_acc_epoch = train_one_epoch(
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
            )
            val_loss = validate(model, args, device, steps_per_epoch)
            examples_this_epoch = samples_per_epoch
            cumulative_examples = (epoch + 1) * samples_per_epoch

            next_disc_lr = float(disc_opt.param_groups[0]['lr']) if disc_opt is not None else None
            controller_status = 'off'
            current_gan_weight, next_disc_lr, controller_status = update_gan_balance_controller(
                args,
                d_gan_acc_epoch,
                current_gan_weight,
                disc_opt,
                disc_lr_min,
                disc_lr_max,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            improved = update_early_stopping(early_stopping, val_loss, args.early_stopping_min_delta)
            if improved:
                torch.save(model.state_dict(), best_ckpt_path)

            if args.save_epoch_checkpoints:
                torch.save(model.state_dict(), Path(args.out_dir)/f"vae_epoch{epoch+1}.pt")

            current_lr = opt.param_groups[0]['lr']
            current_disc_lr = disc_opt.param_groups[0]['lr'] if disc_opt is not None else float('nan')

            writer.writerow([
                epoch + 1,
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

            elapsed_summary = ''
            if (epoch + 1) % 5 == 0:
                elapsed_seconds = time.time() - training_start_time
                average_epoch_seconds = elapsed_seconds / float(epoch + 1)
                remaining_epochs = max(0, args.epochs - (epoch + 1))
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
                f"{(f'{epoch+1:>{len(str(args.epochs))}d}/{args.epochs}'):>9} "
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
                    f"Early stopping triggered after epoch {epoch+1} "
                    f"(no val improvement for {early_stopping.epochs_without_improvement} epochs)."
                )
                break

    print(f"Best checkpoint: {best_ckpt_path} (best_val_loss={early_stopping.best_val_loss:.6f})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--batch_size', '--examples_per_batch', dest='batch_size', type=int, default=100)
    p.add_argument('--number_batches', type=int, default=None, help='Number of batches per epoch. If omitted, uses full dataloader length.')
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
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
    p.add_argument('--zero_cluster_min', type=int, default=8)
    p.add_argument('--zero_cluster_max', type=int, default=12)
    p.add_argument('--resume', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    p.add_argument('--validation_data', type=str, default='data/validation.zarr', help='Path to validation zarr patches.')
    p.add_argument('--validation_extrema_only', dest='validation_extrema_only', action='store_true', help='Use extrema-only input transform for validation data (default).')
    p.add_argument('--no_validation_extrema_only', dest='validation_extrema_only', action='store_false', help='Disable extrema-only input transform for validation data.')
    p.set_defaults(validation_extrema_only=True)
    p.add_argument('--kl_schedule', type=str, default='warmup', choices=['warmup', 'fixed'])
    p.add_argument('--kl_start', type=float, default=0.0)
    p.add_argument('--kl_end', type=float, default=1e-3)
    p.add_argument('--kl_warmup_epochs', type=int, default=15)
    p.add_argument('--kl_fixed', type=float, default=1e-3)
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
    p.add_argument('--best_checkpoint_name', type=str, default='vae_best.pt')
    p.add_argument('--save_epoch_checkpoints', dest='save_epoch_checkpoints', action='store_true', help='Save per-epoch checkpoints in addition to best checkpoint.')
    p.add_argument('--no_save_epoch_checkpoints', dest='save_epoch_checkpoints', action='store_false', help='Disable per-epoch checkpoint saving and keep only best checkpoint.')
    p.set_defaults(save_epoch_checkpoints=True)
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

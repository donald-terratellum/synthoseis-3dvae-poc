import argparse
from pathlib import Path
import itertools
import math
import torch
from torch.utils.data import DataLoader, Dataset
import zarr
import numpy as np
from typing import Any, cast
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
        zero_cluster_min=8,
        zero_cluster_max=12,
        extrema_only=False,
    ):
        z = cast(Any, zarr.open(str(zarr_path), mode='r'))
        self.data = cast(Any, z['patches'])
        self.scaling = scaling
        self.scaling_mean = float(scaling_mean)
        self.scaling_std = float(scaling_std)
        self.augment = bool(augment)
        self.swap_xy_prob = float(swap_xy_prob)
        self.flip_x_prob = float(flip_x_prob)
        self.flip_y_prob = float(flip_y_prob)
        self.zero_cluster_min = int(zero_cluster_min)
        self.zero_cluster_max = int(zero_cluster_max)
        self.extrema_only = bool(extrema_only)

        if self.scaling not in {'none', 'divide_by_std', 'zscore'}:
            raise ValueError("--input_scaling must be one of: none, divide_by_std, zscore")
        if self.scaling != 'none' and abs(self.scaling_std) <= 0.0:
            raise ValueError('--input_std must be non-zero when input scaling is enabled.')
        if self.zero_cluster_min < 0 or self.zero_cluster_max < 0:
            raise ValueError('--zero_cluster_min and --zero_cluster_max must be non-negative.')
        if self.zero_cluster_min > self.zero_cluster_max:
            raise ValueError('--zero_cluster_min must be <= --zero_cluster_max.')

    def _keep_trace_extrema_only(self, x):
        # Keep only local peak/trough samples along each trace (z axis).
        if x.shape[-1] < 3:
            return np.zeros_like(x)

        left = x[:, :, :-2]
        mid = x[:, :, 1:-1]
        right = x[:, :, 2:]
        extrema_mask = ((mid > left) & (mid > right)) | ((mid < left) & (mid < right))

        out = np.zeros_like(x)
        out[:, :, 1:-1][extrema_mask] = x[:, :, 1:-1][extrema_mask]
        return out

    def _apply_pair_augmentations(self, x, y):
        # Geometric transforms are applied to both input and target.
        if np.random.random() < self.swap_xy_prob:
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
        if np.random.random() < self.flip_x_prob:
            x = x[::-1, :, :]
            y = y[::-1, :, :]
        if np.random.random() < self.flip_y_prob:
            x = x[:, ::-1, :]
            y = y[:, ::-1, :]
        return x, y

    def _apply_input_trace_dropout(self, x):
        # Zero 3x3 XY trace clusters through all Z samples on input only.
        nx, ny, _ = x.shape
        if nx < 3 or ny < 3 or self.zero_cluster_max == 0:
            return x

        n_clusters = np.random.randint(self.zero_cluster_min, self.zero_cluster_max + 1)
        center_x = np.arange(1, nx - 1)
        center_y = np.arange(1, ny - 1)
        max_centers = int(center_x.size * center_y.size)
        if max_centers == 0:
            return x
        n_clusters = min(n_clusters, max_centers)

        flat_choices = np.random.choice(max_centers, size=n_clusters, replace=False)
        for idx in flat_choices:
            cx = int(idx // center_y.size) + 1
            cy = int(idx % center_y.size) + 1
            x[cx - 1:cx + 2, cy - 1:cy + 2, :] = 0.0
        return x

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        arr = np.asarray(arr, dtype='f4')
        if self.scaling == 'divide_by_std':
            arr = arr / self.scaling_std
        elif self.scaling == 'zscore':
            arr = (arr - self.scaling_mean) / self.scaling_std

        # For denoising-style augmentation, label stays clean while input is perturbed.
        x = arr.copy()
        y = arr.copy()
        if self.augment:
            x, y = self._apply_pair_augmentations(x, y)
            x = self._apply_input_trace_dropout(x)
        elif self.extrema_only:
            x = self._keep_trace_extrema_only(x)

        x = np.ascontiguousarray(x[np.newaxis, ...])
        y = np.ascontiguousarray(y[np.newaxis, ...])
        return torch.from_numpy(x), torch.from_numpy(y)


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


def compute_average_loss(model, dataloader, device, steps):
    model.eval()
    total_loss = 0.0
    batch_iter = iter(dataloader) if steps is None else itertools.cycle(dataloader)
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            recon, mu, logvar = model(inputs)
            rec_loss = torch.nn.functional.mse_loss(recon, targets)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.numel()
            loss = rec_loss + 1e-4 * kld
            total_loss += loss.item()
    return total_loss / steps


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
        zero_cluster_min=args.zero_cluster_min,
        zero_cluster_max=args.zero_cluster_max,
        extrema_only=False,
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
        zero_cluster_min=0,
        zero_cluster_max=0,
        extrema_only=True,
    )
    validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    validation_steps = max(1, int(math.ceil(0.2 * train_steps_per_epoch)))
    return compute_average_loss(model, validation_dl, device, validation_steps)


def train(args):
    ds = build_dataset(args, args.data, augment=args.augment)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.number_batches is not None and args.number_batches <= 0:
        raise ValueError('--number_batches must be a positive integer when provided.')
    steps_per_epoch = args.number_batches if args.number_batches is not None else len(dl)
    samples_per_epoch = steps_per_epoch * args.batch_size

    model = VAE3D(in_ch=1, out_ch=1, base_ch=16, latent_dim=128)
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
        f"zero_cluster_range=[{args.zero_cluster_min},{args.zero_cluster_max}]",
    )
    model.to(device)

    if args.learning_rate <= 0:
        raise ValueError('--learning_rate must be positive.')
    if args.grad_clip <= 0:
        raise ValueError('--grad_clip must be positive.')

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_iter = iter(dl) if args.number_batches is None else itertools.cycle(dl)
        for _ in range(steps_per_epoch):
            inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            recon, mu, logvar = model(inputs)
            rec_loss = torch.nn.functional.mse_loss(recon, targets)
            # KLD normalized per-element
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.numel()
            loss = rec_loss + 1e-4 * kld
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            total_loss += loss.item()

        train_loss = total_loss / steps_per_epoch
        val_loss = validate(model, args, device, steps_per_epoch)
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.6f} val_loss={val_loss:.6f}")
        torch.save(model.state_dict(), Path(args.out_dir)/f"vae_epoch{epoch+1}.pt")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--batch_size', '--examples_per_batch', dest='batch_size', type=int, default=100)
    p.add_argument('--number_batches', type=int, default=None, help='Number of batches per epoch. If omitted, uses full dataloader length.')
    p.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=1e-4)
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
    p.add_argument('--zero_cluster_min', type=int, default=8)
    p.add_argument('--zero_cluster_max', type=int, default=12)
    p.add_argument('--resume', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    p.add_argument('--validation_data', type=str, default='data/validation.zarr', help='Path to validation zarr patches.')
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

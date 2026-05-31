import argparse
from pathlib import Path
import itertools
import torch
from torch.utils.data import DataLoader, Dataset
import zarr
import numpy as np
from typing import Any, cast
from src.model import VAE3D


class ZarrPatchDataset(Dataset):
    def __init__(self, zarr_path, scaling='none', scaling_mean=0.0, scaling_std=1.0):
        z = cast(Any, zarr.open(str(zarr_path), mode='r'))
        self.data = cast(Any, z['patches'])
        self.scaling = scaling
        self.scaling_mean = float(scaling_mean)
        self.scaling_std = float(scaling_std)

        if self.scaling not in {'none', 'divide_by_std', 'zscore'}:
            raise ValueError("--input_scaling must be one of: none, divide_by_std, zscore")
        if self.scaling != 'none' and abs(self.scaling_std) <= 0.0:
            raise ValueError('--input_std must be non-zero when input scaling is enabled.')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        arr = np.asarray(arr, dtype='f4')
        if self.scaling == 'divide_by_std':
            arr = arr / self.scaling_std
        elif self.scaling == 'zscore':
            arr = (arr - self.scaling_mean) / self.scaling_std
        arr = arr[np.newaxis, ...]  # add channel dim
        return torch.from_numpy(arr)


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


def train(args):
    ds = ZarrPatchDataset(
        args.data,
        scaling=args.input_scaling,
        scaling_mean=args.input_mean,
        scaling_std=args.input_std,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.number_batches is not None and args.number_batches <= 0:
        raise ValueError('--number_batches must be a positive integer when provided.')
    steps_per_epoch = args.number_batches if args.number_batches is not None else len(dl)
    samples_per_epoch = steps_per_epoch * args.batch_size

    model = VAE3D(in_ch=1, out_ch=1, base_ch=16, latent_dim=128)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Batch size (B): {args.batch_size}, batches/epoch: {steps_per_epoch}, examples/epoch: {samples_per_epoch}")
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
        for i in range(steps_per_epoch):
            batch = next(batch_iter)
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            rec_loss = torch.nn.functional.mse_loss(recon, batch)
            # KLD normalized per-element
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.numel()
            loss = rec_loss + 1e-4 * kld
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/steps_per_epoch:.6f}")
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
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

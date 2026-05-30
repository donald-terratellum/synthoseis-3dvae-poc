import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import zarr
import numpy as np
from src.model import VAE3D
import torch.nn.functional as F


class ZarrPatchDataset(Dataset):
    def __init__(self, zarr_path):
        z = zarr.open(str(zarr_path), mode='r')
        self.data = z['patches']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        arr = np.asarray(arr, dtype='f4')
        arr = arr[np.newaxis, ...]  # add channel dim
        return torch.from_numpy(arr)


def train(args):
    ds = ZarrPatchDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = VAE3D(in_ch=1, out_ch=1, base_ch=16, latent_dim=128)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(dl):
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            # ensure reconstruction matches input spatial size (decoder may use different upsampling)
            if recon.shape[2:] != batch.shape[2:]:
                recon = F.interpolate(recon, size=batch.shape[2:], mode='trilinear', align_corners=False)
            rec_loss = torch.nn.functional.mse_loss(recon, batch)
            # KLD normalized per-element
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.numel()
            loss = rec_loss + 1e-4 * kld
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(dl):.6f}")
        torch.save(model.state_dict(), Path(args.out_dir)/f"vae_epoch{epoch+1}.pt")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out_dir', type=str, default='checkpoints')
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)

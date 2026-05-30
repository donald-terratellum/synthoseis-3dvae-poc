#!/usr/bin/env python3
"""Generate reconstructions from a saved VAE checkpoint and create an HTML preview.

Saves reconstructions and inputs as a zarr at --out and writes an HTML file with PNG previews.
"""
from pathlib import Path
import argparse
import zarr
import numpy as np
import torch
from src.model import VAE3D
import matplotlib.pyplot as plt


def save_png_comparison(inp, recon, outpath):
    # inp, recon: numpy arrays shape (D,H,W) or (32,32,32)
    k = inp.shape[2] // 2
    fig, axes = plt.subplots(1,2, figsize=(6,3))
    axes[0].imshow(inp[:,:,k].T, cmap='gray', origin='lower')
    axes[0].set_title('input (Z mid)')
    axes[0].axis('off')
    axes[1].imshow(recon[:,:,k].T, cmap='gray', origin='lower')
    axes[1].set_title('recon (Z mid)')
    axes[1].axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='input patches zarr')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='output zarr path')
    p.add_argument('--html', required=True, help='output html preview path')
    p.add_argument('--n_samples', type=int, default=8)
    args = p.parse_args()

    src = Path(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgdir = Path(args.html).parent / 'images'
    imgdir.mkdir(parents=True, exist_ok=True)

    g = zarr.open(str(src), mode='r')
    patches = np.asarray(g['patches'])
    n = min(args.n_samples, patches.shape[0])
    idxs = np.linspace(0, patches.shape[0]-1, n, dtype=int)

    device = torch.device('cpu')
    model = VAE3D(in_ch=1, out_ch=1, base_ch=16, latent_dim=128)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    dst = zarr.open(str(out), mode='w')
    dst.create_array('inputs', shape=(n,)+patches.shape[1:], dtype='f4', chunks=(1,)+patches.shape[1:])
    dst.create_array('recons', shape=(n,)+patches.shape[1:], dtype='f4', chunks=(1,)+patches.shape[1:])

    html_lines = ["<html><head><meta charset='utf-8'><title>VAE Recon Preview</title></head><body>",
                  "<h1>VAE Reconstructions</h1>", "<div style='display:flex;flex-wrap:wrap'>"]

    for i, idx in enumerate(idxs):
        arr = patches[idx]
        inp = arr.astype('f4')
        tensor = torch.from_numpy(inp[np.newaxis, np.newaxis, ...])
        with torch.no_grad():
            recon, mu, logvar = model(tensor)
            # recon shape [1,1,D,H,W]
            recon_np = recon.squeeze().cpu().numpy()
        dst['inputs'][i] = inp
        dst['recons'][i] = recon_np
        imgpath = imgdir / f'recon_{i:03d}.png'
        save_png_comparison(inp, recon_np, str(imgpath))
        html_lines.append(f"<div style='margin:8px'><img src='images/{imgpath.name}' width=320><p style='text-align:center'>sample {i}</p></div>")

    html_lines.append("</div></body></html>")
    Path(args.html).write_text('\n'.join(html_lines))
    print('Wrote reconstructions to', out)
    print('Wrote HTML preview to', args.html)


if __name__ == '__main__':
    main()

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import SliceLPIPSLoss, compute_vae_losses
from src.model import Decoder
from src.tokenizer.core.model_adapter import VaeLatentAdapter


CHECKPOINTS = {
    'p0b': 'checkpoints/geoaware_p0b_r2_seed20260925_r2/vae_epoch30.pt',
    'p5a': 'checkpoints/p5_p6_encoder/p5a_seed20260925/stage2_geology/vae_epoch40.pt',
    'p5b': 'checkpoints/p5_p6_encoder/p5b_seed20260925/stage2_geology/vae_epoch40.pt',
    'p6': 'checkpoints/p5_p6_encoder/p6_seed20260925/stage2_geology/vae_epoch30.pt',
}


def select_indices(size, count, seed):
    if not 2 <= count <= size:
        raise ValueError('sample count must be between 2 and dataset size')
    return np.random.default_rng(seed).choice(size, count, replace=False).tolist()


def load_patches(path, count, seed):
    root = zarr.open(str(path), mode='r')
    mode = root.attrs.get('scaling_mode')
    if mode != 'divide_by_std':
        raise ValueError(f'{path}: expected baked-in divide_by_std scaling, got {mode!r}')
    indices = select_indices(root['patches'].shape[0], count, seed)
    patches = torch.from_numpy(np.stack([root['patches'][index] for index in indices]).astype('float32'))[:, None]
    if not torch.isfinite(patches).all():
        raise ValueError(f'{path}: non-finite patches')
    return patches, {'path': str(path), 'indices': indices, 'scaling': dict(root.attrs),
                     'target_std': float(patches.std()), 'zero_output_mae': float(patches.abs().mean())}


def axis_audit(model, device):
    if model.encoder.arch != 'resnetv2':
        return {'applicable': False}
    shape = model.patch_shape
    coordinates = torch.meshgrid(*(torch.arange(size) for size in shape), indexing='ij')
    fixture = (coordinates[0] * 10000 + coordinates[1] * 100 + coordinates[2]).float()[None, None].to(device)
    fixture = fixture / fixture.max()
    with torch.no_grad():
        adapted = fixture.permute(0, 1, 4, 2, 3).contiguous() if model.encoder.input_axes == 'zxy' else fixture
        features = model.encoder.trunk(adapted).flatten(1)
        actual_mu, actual_logvar = model.encoder(fixture)
        torch.testing.assert_close(actual_mu, model.encoder.fc_mu(features))
        torch.testing.assert_close(actual_logvar, model.encoder.fc_logvar(features))
        decoder = model.decoder
        native = decoder.unflatten(decoder.fc(actual_mu))
        for upsample, block in zip(decoder.ups, decoder.blocks):
            native = block(upsample(native))
        native = decoder.out_conv(native)
        expected = native.permute(0, 1, 3, 4, 2).contiguous() if decoder.output_axes == 'zxy' else native
        torch.testing.assert_close(decoder(actual_mu)[0], expected)
        if tuple(expected.shape) != tuple(fixture.shape):
            raise ValueError('decoder axis inversion changed patch shape')
        if model.encoder.input_axes == 'zxy':
            torch.testing.assert_close(adapted.permute(0, 1, 3, 4, 2), fixture)
    return {'applicable': True, 'passed': True, 'encoded_shape': list(model.encoder._encoded_shape)}


def component_metrics(prediction, targets, mu, logvar, lpips_fn):
    if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
        raise ValueError('non-finite or shape-mismatched reconstruction')
    total, mae, kl, perceptual, _ = compute_vae_losses(
        prediction, targets, mu, logvar, 0.001,
        rec_loss_fn=torch.nn.functional.l1_loss, lpips_loss_fn=lpips_fn, lpips_weight=0.1,
    )
    latent_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    result = {'mae': float(mae), 'mse': float((prediction - targets).square().mean()),
              'lpips': float(perceptual), 'kl_per_voxel': float(kl),
              'kl_per_example': float(latent_kl.mean()), 'recipe_total': float(total),
              'prediction_std': float(prediction.flatten(1).std(dim=1).mean()),
              'target_std': float(targets.flatten(1).std(dim=1).mean()),
              'zero_output_mae': float(targets.abs().mean())}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('non-finite component metrics')
    return result


def evaluate(model, patches, device, batch_size, seed, lpips_fn):
    model.eval()
    noise = torch.randn((len(patches), model.latent_dim), generator=torch.Generator().manual_seed(seed))
    records = {'mean': [], 'sampled': []}
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            targets = patches[start:start + batch_size].to(device)
            mu, logvar = model.encoder(targets)
            for name, latent in [('mean', mu), ('sampled', mu + (0.5 * logvar).exp() * noise[start:start + len(targets)].to(device))]:
                prediction = model.decoder(latent)[0]
                records[name].append((len(targets), component_metrics(prediction, targets, mu, logvar, lpips_fn)))
            print(f'    evaluated {min(start + batch_size, len(patches))}/{len(patches)}', flush=True)
    return {name: {key: sum(size * values[key] for size, values in batches) / len(patches)
                   for key in batches[0][1]} for name, batches in records.items()}


def fit_decoder(model, targets, device, steps, seed, legacy=False):
    if steps < 1:
        raise ValueError('fit steps must be positive')
    model.eval()
    before = {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}
    targets = targets.to(device)
    with torch.no_grad():
        mu = model.encoder(targets)[0].detach()
    torch.manual_seed(seed)
    decoder = Decoder(base_ch=model.base_ch, latent_dim=model.latent_dim, patch_shape=model.patch_shape).to(device) if legacy else copy.deepcopy(model.decoder)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(decoder(mu)[0], targets)
        if not torch.isfinite(loss):
            raise ValueError('non-finite decoder fit loss')
        gradient_norm = None
        if step < steps:
            loss.backward()
            squared_norms = [parameter.grad.square().sum() for parameter in decoder.parameters() if parameter.grad is not None]
            gradient_norm = float(torch.stack(squared_norms).sum().sqrt())
            if not np.isfinite(gradient_norm):
                raise ValueError('non-finite decoder gradients')
        if step == 0 or step % 10 == 0 or step == steps:
            record = {'step': step, 'mae': float(loss.detach()), 'gradient_norm': gradient_norm}
            history.append(record)
            print(f'    fit {"fresh_legacy" if legacy else "native"}: {record}', flush=True)
        if step < steps:
            optimizer.step()
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    return {'history': history, 'encoder_unchanged': True,
            'relative_mae_reduction': 1 - history[-1]['mae'] / max(history[0]['mae'], 1e-12),
            'normalization_mode': 'eval (fixed BatchNorm running statistics)'}


def csv_audit(root):
    result = {}
    for path in sorted(root.glob('*/stage*/training_metrics.csv')):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'empty training metrics: {path}')
        losses = [float(row['val_loss']) for row in rows]
        window = min(10, len(rows) // 2)
        if window < 1:
            raise ValueError(f'too few training rows: {path}')
        previous = float(np.mean(losses[-2 * window:-window]))
        late = float(np.mean(losses[-window:]))
        best_index = int(np.argmin(losses))
        result[str(path)] = {'epochs': len(rows), 'first_val_loss': losses[0],
                             'best_val_loss': losses[best_index], 'best_epoch': rows[best_index]['epoch'],
                             'last_val_loss': losses[-1], 'last_learning_rate': rows[-1]['learning_rate'],
                             'previous_window_mean': previous, 'late_window_mean': late,
                             'late_change_fraction': late / previous - 1,
                             'components_available': ['val_lpips_loss'],
                             'mae_and_kl_not_separable_from_csv': True}
        print(f'CSV {path}: {result[str(path)]}', flush=True)
    if len(result) != 6:
        raise ValueError(f'expected six P5/P6 stage CSVs, found {len(result)}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_data', default='data/synth_val_v2_uniform_32-32-64.zarr')
    parser.add_argument('--real_data', default='data/real_test_32-32-64.zarr')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--fit_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--out_json', default='logs/p5_p6_reconstruction_diagnostics.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.fit_steps < 1:
        parser.error('batch_size and fit_steps must be positive')
    for path in [*CHECKPOINTS.values(), args.synthetic_data, args.real_data]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    torch.set_num_threads(2)
    report = {'settings': vars(args), 'csv_audit': csv_audit(Path('checkpoints/p5_p6_encoder')),
              'datasets': {}, 'arms': {}, 'scope': 'small-subset diagnostic, not an adoption or full-set guardrail evaluation'}
    datasets = {}
    for name, path in [('synthetic', args.synthetic_data), ('real', args.real_data)]:
        datasets[name], report['datasets'][name] = load_patches(path, args.samples, args.seed)
        print(f'DATA {name}: std={report["datasets"][name]["target_std"]:.6f} baked scaling; no extra normalization', flush=True)
    device = torch.device(args.device)
    lpips_fn = SliceLPIPSLoss().to(device).eval()
    for arm, checkpoint in CHECKPOINTS.items():
        print(f'ARM {arm}: {checkpoint}', flush=True)
        torch.manual_seed(args.seed)
        adapter = VaeLatentAdapter(checkpoint, device=args.device, load_classifier=True)
        model = adapter.model
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        arm_report = {'checkpoint': checkpoint, 'model_config': model.model_config,
                      'encoder_parameters': sum(parameter.numel() for parameter in model.encoder.parameters()),
                      'decoder_parameters': sum(parameter.numel() for parameter in model.decoder.parameters()),
                      'axis_audit': axis_audit(model, device), 'evaluation': {}, 'decoder_fit': {}}
        for name, patches in datasets.items():
            print(f'  SPLIT {name}', flush=True)
            arm_report['evaluation'][name] = evaluate(model, patches, device, args.batch_size, args.seed, lpips_fn)
            print(f'  METRICS {name}: {arm_report["evaluation"][name]}', flush=True)
        if arm in {'p0b', 'p5a', 'p6'}:
            arm_report['decoder_fit']['native'] = fit_decoder(model, datasets['synthetic'][:2], device, args.fit_steps, args.seed)
            if arm != 'p0b':
                arm_report['decoder_fit']['fresh_legacy'] = fit_decoder(model, datasets['synthetic'][:2], device, args.fit_steps, args.seed, legacy=True)
        report['arms'][arm] = arm_report
        del model, adapter
    print('SUMMARY (same fixed subset; mean / sampled MAE and sampled recipe total)', flush=True)
    for arm, values in report['arms'].items():
        values['deltas_vs_p0b'] = {}
        for split, metrics in values['evaluation'].items():
            control = report['arms']['p0b']['evaluation'][split]
            values['deltas_vs_p0b'][split] = {mode: metrics[mode]['mae'] / control[mode]['mae'] - 1 for mode in ('mean', 'sampled')}
            print(f'{arm} {split}: {metrics["mean"]["mae"]:.6f} / {metrics["sampled"]["mae"]:.6f}; total={metrics["sampled"]["recipe_total"]:.6f}; MAE delta={values["deltas_vs_p0b"][split]}', flush=True)
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'COMPLETE: {destination}. No training checkpoint or dataset modified.', flush=True)


if __name__ == '__main__':
    main()
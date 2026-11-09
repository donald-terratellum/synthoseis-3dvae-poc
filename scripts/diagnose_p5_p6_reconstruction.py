import argparse
import copy
import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import SliceLPIPSLoss, compute_vae_losses
from src.augmentations import keep_trace_extrema_only
from src.model import Decoder, VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from src.tokenizer.core.preprocess import DEFAULT_STD_EPS, preprocess_for_token
from src.tokenizer.core.search_engine import run_similarity_search_on_padded_volume


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


def load_patches(path, count, seed, input_mode='clean'):
    root = zarr.open(str(path), mode='r')
    mode = root.attrs.get('scaling_mode')
    if mode != 'divide_by_std':
        raise ValueError(f'{path}: expected baked-in divide_by_std scaling, got {mode!r}')
    indices = select_indices(root['patches'].shape[0], count, seed)
    patches = torch.from_numpy(np.stack([root['patches'][index] for index in indices]).astype('float32'))[:, None]
    if not torch.isfinite(patches).all():
        raise ValueError(f'{path}: non-finite patches')
    if input_mode not in {'clean', 'extrema', 'tokenizer'}:
        raise ValueError('input_mode must be clean, extrema or tokenizer')
    metadata = {'path': str(path), 'dataset_size': int(root['patches'].shape[0]), 'indices': indices,
                'input_mode': input_mode, 'scaling': dict(root.attrs),
                'target_std': float(patches.std()), 'zero_output_mae': float(patches.abs().mean()),
                'target_convention': 'clean baked-scaled patches; no output rescaling',
                'input_function': {'clean': 'identity', 'extrema': 'keep_trace_extrema_only',
                                   'tokenizer': 'src.tokenizer.core.preprocess.preprocess_for_token'}[input_mode]}
    if input_mode == 'tokenizer':
        metadata['input_normalization_divisors'] = [max(float(cube[0].numpy().std()), DEFAULT_STD_EPS) for cube in patches]
    if 'source_volume_index' in root:
        source_indices = np.asarray(root['source_volume_index'][:], dtype=np.int64)
        metadata['source_volume_indices'] = source_indices[indices].tolist()
        metadata['source_volume_names'] = [str(name) for name in root.attrs.get('source_volumes', [])]
    return patches, metadata


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
              'prediction_abs_max': float(prediction.abs().max()),
              'target_abs_max': float(targets.abs().max()),
              'zero_output_mae': float(targets.abs().mean())}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('non-finite component metrics')
    return result


def evaluation_inputs(targets, input_mode):
    if input_mode == 'clean':
        return targets
    if input_mode not in {'extrema', 'tokenizer'}:
        raise ValueError('input_mode must be clean, extrema or tokenizer')
    arrays = targets.detach().cpu().numpy()
    transform = preprocess_for_token if input_mode == 'tokenizer' else keep_trace_extrema_only
    masked = np.stack([transform(cube[0]) for cube in arrays])[:, None]
    return torch.from_numpy(np.ascontiguousarray(masked)).to(targets.device)


def evaluate(model, patches, device, batch_size, seed, lpips_fn, input_mode='clean', source_indices=None, source_names=()):
    model.eval()
    noise = torch.randn((len(patches), model.latent_dim), generator=torch.Generator().manual_seed(seed))
    records = {'mean': [], 'sampled': []}
    volume_records = {'mean': {}, 'sampled': {}}
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            targets = patches[start:start + batch_size].to(device)
            mu, logvar = model.encoder(evaluation_inputs(targets, input_mode))
            for name, latent in [('mean', mu), ('sampled', mu + (0.5 * logvar).exp() * noise[start:start + len(targets)].to(device))]:
                prediction = model.decoder(latent)[0]
                records[name].append((len(targets), component_metrics(prediction, targets, mu, logvar, lpips_fn)))
                if source_indices is not None:
                    errors = (prediction - targets).abs().flatten(1).mean(dim=1).cpu().tolist()
                    for index, error in zip(source_indices[start:start + len(targets)], errors):
                        values = volume_records[name].setdefault(int(index), [0.0, 0])
                        values[0] += float(error)
                        values[1] += 1
            print(f'    evaluated {min(start + batch_size, len(patches))}/{len(patches)}', flush=True)
    result = {name: {key: (max(values[key] for size, values in batches) if key.endswith('_abs_max') else
                          sum(size * values[key] for size, values in batches) / len(patches))
                     for key in batches[0][1]} for name, batches in records.items()}
    if source_indices is not None:
        result['per_volume'] = {}
        for name, volumes in volume_records.items():
            grouped = {str(index): {'count': values[1], 'mae': values[0] / values[1],
                                   'source': source_names[index] if 0 <= index < len(source_names) else f'index_{index}'}
                       for index, values in volumes.items()}
            result['per_volume'][name] = {'volumes': grouped,
                                         'volume_balanced_mae': sum(values['mae'] for values in grouped.values()) / len(grouped)}
    return result


def verify_tokenizer_adapter_parity(adapter, patches):
    prepared = evaluation_inputs(patches[:2], 'tokenizer').cpu().numpy()[:, 0]
    prepared = np.ascontiguousarray(prepared, dtype=np.float32)
    with torch.no_grad():
        mu, _ = adapter.model.encoder(torch.from_numpy(prepared[:, None]).to(adapter.device))
        expected_mu = mu.cpu().numpy()
        expected_geo = adapter.model.encode_geo(mu).cpu().numpy()
    actual_mu = adapter.encode_batch(prepared)
    actual_geo = adapter.encode_geo_batch(prepared)
    np.testing.assert_allclose(actual_mu, expected_mu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual_geo, expected_geo, rtol=1e-4, atol=1e-5)
    return {'passed': True, 'mu_max_abs_delta': float(np.max(np.abs(actual_mu - expected_mu))),
            'z_geo_max_abs_delta': float(np.max(np.abs(actual_geo - expected_geo))),
            'rtol': 1e-4, 'atol': 1e-5, 'preprocessing_count': 1,
            'retrieval_quality_measured': False}


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


def run_training_diagnostic(model, args, device, epoch, lpips_fn):
    interval = int(getattr(args, 'reconstruction_diagnostic_interval', 0))
    if interval <= 0 or epoch % interval:
        return None
    report = {'epoch': epoch, 'input_mode': 'clean', 'scope': 'fixed-subset reconstruction diagnostic', 'evaluation': {}}
    modes = tuple(getattr(args, 'reconstruction_diagnostic_modes', ('clean',)))
    report['input_mode'] = modes[0] if len(modes) == 1 else 'multiple'
    report['input_modes'] = list(modes)
    splits = [('synthetic', args.validation_data), ('real', args.real_test_data)]
    if bool(getattr(args, 'reconstruction_diagnostic_train', False)):
        splits.insert(0, ('training', args.data))
    was_training = model.training
    try:
        for name, path in splits:
            if not path:
                continue
            patches, metadata = load_patches(path, args.reconstruction_diagnostic_samples, 20261003)
            metrics = {mode: evaluate(model, patches, device, 2, 20261003, lpips_fn, mode) for mode in modes}
            report['evaluation'][name] = {'dataset': metadata, 'metrics': metrics[modes[0]], 'metrics_by_input_mode': metrics}
    finally:
        model.train(was_training)
    destination = Path(args.out_dir) / f'reconstruction_diagnostic_epoch{epoch}.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'RECONSTRUCTION MILESTONE {epoch}: {destination}', flush=True)
    for name, values in report['evaluation'].items():
        for mode, metrics in values['metrics_by_input_mode'].items():
            print(f'  {name}/{mode}: {metrics}', flush=True)
    return report


def exposure_summary(dataset_size, batch_size=12, batches=450, epochs=150):
    per_epoch = batch_size * batches
    if dataset_size < 1 or min(batch_size, batches, epochs) < 1 or per_epoch > dataset_size:
        raise ValueError('exposure audit requires positive counts and a partial or full shuffled pass')
    return {'dataset_size': dataset_size, 'examples_per_epoch': per_epoch,
            'epochs': epochs, 'total_exposures': per_epoch * epochs,
            'corpus_equivalent_passes': per_epoch * epochs / dataset_size,
            'expected_unique_fraction_if_independent_shuffles': 1 - (1 - per_epoch / dataset_size) ** epochs,
            'actual_historical_index_coverage': 'unknown: indices and RNG state were not logged',
            'assumptions': 'uniform independent shuffle, reconstruction-stage sampler; not geology-stage coverage'}


def pre_p7_audit(training_data, reference_checkpoint):
    size = int(zarr.open(training_data, mode='r')['patches'].shape[0])
    payload = torch.load(reference_checkpoint, map_location='cpu', weights_only=False)
    config = payload.get('model_config') or {}
    result = {'exposure': exposure_summary(size), 'latent_parameter_counts': {}}
    for architecture, architecture_config in [('E0', {}), ('E2', config)]:
        result['latent_parameter_counts'][architecture] = {}
        for latent_dim in (128, 256, 512):
            model = VAE3D(patch_shape=(32, 32, 64), latent_dim=latent_dim, geology_projection=True,
                          geology_classifier=True, **architecture_config)
            result['latent_parameter_counts'][architecture][str(latent_dim)] = {
                'total': sum(parameter.numel() for parameter in model.parameters()),
                'trunk': sum(parameter.numel() for name, parameter in model.encoder.named_parameters() if not name.startswith('fc_')),
                'mu_projection': model.encoder.fc_mu.weight.numel() + model.encoder.fc_mu.bias.numel(),
                'logvar_projection': model.encoder.fc_logvar.weight.numel() + model.encoder.fc_logvar.bias.numel(),
                'decoder_projection': model.decoder.fc.weight.numel() + model.decoder.fc.bias.numel(),
            }
            del model
    print(f'PRE-P7 AUDIT: {result}', flush=True)
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
    parser.add_argument('--checkpoints', nargs='+', help='Override checkpoints as ARM=PATH; first arm is the control')
    parser.add_argument('--evaluation_only', action='store_true', help='Skip historical CSV audit and decoder fitting')
    parser.add_argument('--input_mode', choices=['clean', 'extrema', 'tokenizer'], default='clean')
    parser.add_argument('--contract_audit', action='store_true', help='Verify adapter parity, fixed contract and per-source reconstruction metrics without training')
    parser.add_argument('--audit_training_data', help='Record theoretical reconstruction sampler exposure and 128/256/512 parameter counts')
    parser.add_argument('--out_json', default='logs/p5_p6_reconstruction_diagnostics.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.fit_steps < 1:
        parser.error('batch_size and fit_steps must be positive')
    checkpoints = dict(CHECKPOINTS)
    if args.checkpoints:
        checkpoints = {}
        for entry in args.checkpoints:
            arm, separator, path = entry.partition('=')
            if not separator or not arm or not path or arm in checkpoints:
                parser.error('checkpoints must be unique ARM=PATH entries')
            checkpoints[arm] = path
    control_arm = next(iter(checkpoints))
    for path in [*checkpoints.values(), args.synthetic_data, args.real_data]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    torch.set_num_threads(2)
    report = {'settings': vars(args), 'csv_audit': {} if args.evaluation_only else csv_audit(Path('checkpoints/p5_p6_encoder')),
              'datasets': {}, 'arms': {}, 'scope': 'reconstruction comparison; dataset coverage recorded per split'}
    if args.contract_audit:
        if not args.evaluation_only:
            parser.error('contract_audit requires evaluation_only')
        report['tokenizer_contract'] = {'preprocessor': 'src.tokenizer.core.preprocess.preprocess_for_token',
                                      'route': 'jobs.py passes preprocess_for_token to search_engine.run_similarity_search_on_padded_volume',
                                      'std_eps': DEFAULT_STD_EPS, 'original_p7_gate_modified': False}
        actual_default = inspect.signature(run_similarity_search_on_padded_volume).parameters['preprocess_fn'].default
        if actual_default is not preprocess_for_token:
            raise ValueError('search-engine default preprocessing does not match the audited tokenizer contract')
        report['tokenizer_contract']['search_default_verified'] = True
    if args.audit_training_data:
        report['pre_p7_audit'] = pre_p7_audit(args.audit_training_data, list(checkpoints.values())[-1])
    datasets = {}
    for name, path in [('synthetic', args.synthetic_data), ('real', args.real_data)]:
        datasets[name], report['datasets'][name] = load_patches(path, args.samples, args.seed, args.input_mode)
        print(f'DATA {name}: std={report["datasets"][name]["target_std"]:.6f} baked scaling; no extra normalization', flush=True)
    device = torch.device(args.device)
    lpips_fn = SliceLPIPSLoss().to(device).eval()
    for arm, checkpoint in checkpoints.items():
        print(f'ARM {arm}: {checkpoint}', flush=True)
        torch.manual_seed(args.seed)
        adapter = VaeLatentAdapter(checkpoint, device=args.device, load_classifier=True)
        model = adapter.model
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        if args.contract_audit and (model.latent_dim != 128 or tuple(model.patch_shape) != (32, 32, 64)):
            raise ValueError('P7 contract audit requires latent_dim=128 and patch_shape=32x32x64')
        arm_report = {'checkpoint': checkpoint, 'model_config': model.model_config,
                      'encoder_parameters': sum(parameter.numel() for parameter in model.encoder.parameters()),
                      'decoder_parameters': sum(parameter.numel() for parameter in model.decoder.parameters()),
                      'axis_audit': axis_audit(model, device), 'evaluation': {}, 'decoder_fit': {}}
        if args.contract_audit:
            arm_report['adapter_parity'] = verify_tokenizer_adapter_parity(adapter, datasets['synthetic'])
            print(f'  ADAPTER PARITY: {arm_report["adapter_parity"]}', flush=True)
        for name, patches in datasets.items():
            print(f'  SPLIT {name}', flush=True)
            metadata = report['datasets'][name]
            if args.contract_audit and not metadata.get('source_volume_indices'):
                raise ValueError(f'{name}: source_volume_index is required for this audit')
            arm_report['evaluation'][name] = evaluate(
                model, patches, device, args.batch_size, args.seed, lpips_fn, args.input_mode,
                metadata.get('source_volume_indices') if args.contract_audit else None,
                metadata.get('source_volume_names', ()),
            )
            print(f'  METRICS {name}: {arm_report["evaluation"][name]}', flush=True)
        if not args.evaluation_only and args.input_mode != 'clean':
            raise ValueError('decoder-fitting diagnostics require clean mode; use --evaluation_only for extrema')
        if not args.evaluation_only and arm in {'p0b', 'p5a', 'p6'}:
            arm_report['decoder_fit']['native'] = fit_decoder(model, datasets['synthetic'][:2], device, args.fit_steps, args.seed)
            if arm != 'p0b':
                arm_report['decoder_fit']['fresh_legacy'] = fit_decoder(model, datasets['synthetic'][:2], device, args.fit_steps, args.seed, legacy=True)
        report['arms'][arm] = arm_report
        del model, adapter
    print('SUMMARY (same fixed subset; mean / sampled MAE and sampled recipe total)', flush=True)
    for arm, values in report['arms'].items():
        values['deltas_vs_p0b'] = {}
        for split, metrics in values['evaluation'].items():
            control = report['arms'][control_arm]['evaluation'][split]
            values['deltas_vs_p0b'][split] = {mode: metrics[mode]['mae'] / control[mode]['mae'] - 1 for mode in ('mean', 'sampled')}
            print(f'{arm} {split}: {metrics["mean"]["mae"]:.6f} / {metrics["sampled"]["mae"]:.6f}; total={metrics["sampled"]["recipe_total"]:.6f}; MAE delta={values["deltas_vs_p0b"][split]}', flush=True)
    destination = Path(args.out_json)
    report['reconstruction_gate'] = {
        'control': control_arm,
        'synthetic_recipe_and_real_mae_within_2pct': {
            arm: values['evaluation']['synthetic']['sampled']['recipe_total'] <= 1.02 * report['arms'][control_arm]['evaluation']['synthetic']['sampled']['recipe_total']
            and values['evaluation']['real']['sampled']['mae'] <= 1.02 * report['arms'][control_arm]['evaluation']['real']['sampled']['mae']
            for arm, values in report['arms'].items() if arm != control_arm
        },
        'full_dataset_coverage': all(len(values['indices']) == zarr.open(values['path'], mode='r')['patches'].shape[0] for values in report['datasets'].values()),
        'geology_stage_launched': False,
    }
    print(f'RECONSTRUCTION GATE: {report["reconstruction_gate"]}', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'COMPLETE: {destination}. No training checkpoint or dataset modified.', flush=True)


if __name__ == '__main__':
    main()
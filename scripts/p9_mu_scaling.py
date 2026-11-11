#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import threading
import time
from pathlib import Path

import torch

from scripts.train import (
    SliceLPIPSLoss,
    build_checkpoint_payload,
    compute_geology_classifier_loss,
    compute_supervised_contrastive_loss,
)
from src.model import GEOLOGY_PRESENCE_CLASSES, VAE3D

PATCH_SHAPE = (32, 32, 64)


def build_e1_model(latent_dim: int, config: dict, checkpoint: dict) -> VAE3D:
    return VAE3D(
        base_ch=int(checkpoint['base_ch']),
        latent_dim=latent_dim,
        patch_shape=tuple(checkpoint['patch_shape']),
        deep_supervision=bool(checkpoint.get('deep_supervision', False)),
        geology_projection=bool(checkpoint.get('geology_projection', False)),
        geology_proj_hidden=int(checkpoint.get('geology_proj_hidden', 128)),
        geology_proj_dim=int(checkpoint.get('geology_proj_dim', 64)),
        geology_classifier=bool(checkpoint.get('geology_classifier', False)),
        geology_classifier_mode=str(checkpoint.get('geology_classifier_mode', 'patch')),
        geology_classifier_hidden=int(checkpoint.get('geology_classifier_hidden', 256)),
        encoder_arch=str(config.get('encoder_arch', 'conv')),
        encoder_hidden_dims=config.get('encoder_hidden_dims'),
        encoder_depth_profile=str(config.get('encoder_depth_profile', 'baseline')),
        encoder_stage_blocks=config.get('encoder_stage_blocks'),
        encoder_norm=config.get('encoder_norm'),
        encoder_stem=str(config.get('encoder_stem', 'pretrain_v2')),
        encoder_input_axes=str(config.get('encoder_input_axes', 'xyz')),
        decoder_hidden_dims=config.get('decoder_hidden_dims'),
        decoder_block=str(config.get('decoder_block', 'conv')),
    )


def initialize(args: argparse.Namespace) -> None:
    source_path = Path(args.source_checkpoint)
    destination = Path(args.output)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if destination.exists():
        raise FileExistsError(destination)
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    if int(source.get('latent_dim', -1)) != 128:
        raise ValueError('P9 initialization source must be the seed-matched 128-D E1 initialization.')
    if tuple(source.get('patch_shape', ())) != PATCH_SHAPE:
        raise ValueError('P9 initialization source must use patch_shape=32x32x64.')
    config = source.get('model_config') or {}
    if config.get('encoder_arch') != 'residual' or config.get('decoder_hidden_dims') is not None:
        raise ValueError('P9 requires the E1 residual encoder and original decoder.')
    if not source.get('geology_projection') or not source.get('geology_classifier'):
        raise ValueError('P9 E1 initialization requires geology projection and classifier heads.')

    torch.manual_seed(args.seed)
    model = build_e1_model(args.latent_dim, config, source)
    target_state = model.state_dict()
    source_state = source['model_state_dict']
    trunk_keys = [
        key for key in target_state
        if key.startswith('encoder.') and not key.startswith(('encoder.fc_mu.', 'encoder.fc_logvar.'))
    ]
    if not trunk_keys:
        raise ValueError('E1 encoder trunk state was unexpectedly empty.')
    for key in trunk_keys:
        if key not in source_state or source_state[key].shape != target_state[key].shape:
            raise ValueError(f'Seed-matched E1 trunk state is incompatible at {key}.')
        target_state[key] = source_state[key].clone()
    model.load_state_dict(target_state, strict=True)
    payload = build_checkpoint_payload(model, epoch=0)
    payload['p9_initialization'] = {
        'seed': int(args.seed),
        'source_checkpoint': str(source_path),
        'copied_encoder_tensors': len(trunk_keys),
        'copied_latent_or_head_weights': False,
        'latent_dim': int(args.latent_dim),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(payload, stream)
    print(json.dumps(payload['p9_initialization'], indent=2))


class DeviceMemorySampler:
    def __init__(self, device: torch.device):
        self.device = device
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_driver_bytes = 0
        self.peak_allocated_bytes = 0

    def _sample(self):
        while not self.stop_event.is_set():
            if self.device.type == 'mps':
                self.peak_driver_bytes = max(self.peak_driver_bytes, torch.mps.driver_allocated_memory())
                self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.mps.current_allocated_memory())
            self.stop_event.wait(0.02)

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


def process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == 'Darwin' else peak * 1024


def profile_latent_size(latent_dim: int, device: torch.device, batch_size: int) -> dict:
    torch.mps.empty_cache() if device.type == 'mps' else None
    torch.manual_seed(20261109)
    model = VAE3D(
        latent_dim=latent_dim,
        patch_shape=PATCH_SHAPE,
        residual_encoder=True,
        geology_projection=True,
        geology_classifier=True,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    batch = torch.randn(batch_size, 1, *PATCH_SHAPE, device=device)
    targets = torch.randn_like(batch)
    labels = {
        f'label_presence_{name}': torch.randint(0, 2, (batch_size,), device=device)
        for name in GEOLOGY_PRESENCE_CLASSES
    }
    labels.update({
        'meta_dip_mean_class': torch.randint(0, 6, (batch_size,), device=device),
        'meta_dip_range_class': torch.randint(0, 6, (batch_size,), device=device),
    })
    strata = [index // 2 + 1 for index in range(batch_size)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    lpips = SliceLPIPSLoss().to(device).eval()
    sampler = DeviceMemorySampler(device)
    before_driver = torch.mps.driver_allocated_memory() if device.type == 'mps' else 0
    start = time.perf_counter()
    outcome = {'latent_dim': latent_dim, 'device': str(device), 'batch_size': batch_size, 'fits_required_batch': False}
    try:
        sampler.start()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(batch)
        z_geo = model.encode_geo(mu)
        classifier, _ = compute_geology_classifier_loss(model.classify(mu), labels, loss_type='focal')
        contrastive = compute_supervised_contrastive_loss(z_geo, strata, temperature=0.2)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / targets.numel()
        loss = torch.nn.functional.l1_loss(reconstruction, targets) + 0.1 * lpips(reconstruction, targets)
        loss = loss + 0.001 * kl + 0.1 * classifier + 0.5 * contrastive
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite profile loss at latent_dim={latent_dim}')
        loss.backward()
        optimizer.step()
        if device.type == 'mps':
            torch.mps.synchronize()
        outcome.update({
            'fits_required_batch': True,
            'parameter_count': parameters,
            'model_parameter_bytes_fp32': parameters * 4,
            'input_and_target_bytes_fp32': batch.numel() * batch.element_size() * 2,
            'elapsed_seconds': time.perf_counter() - start,
            'process_peak_rss_bytes': process_peak_rss_bytes(),
            'mps_driver_bytes_before': before_driver,
            'mps_driver_bytes_after': torch.mps.driver_allocated_memory() if device.type == 'mps' else None,
            'mps_peak_driver_bytes_sampled': sampler.peak_driver_bytes if device.type == 'mps' else None,
            'mps_peak_allocated_bytes_sampled': sampler.peak_allocated_bytes if device.type == 'mps' else None,
            'mps_recommended_max_bytes': torch.mps.recommended_max_memory() if device.type == 'mps' else None,
            'loss': float(loss.detach().cpu()),
        })
        if device.type == 'mps':
            recommended = torch.mps.recommended_max_memory()
            peak = max(sampler.peak_driver_bytes, outcome['mps_driver_bytes_after'])
            outcome['fits_required_batch'] = peak <= int(recommended * 0.8)
            outcome['memory_headroom_bytes'] = max(0, recommended - peak)
            outcome['required_headroom_fraction'] = 0.2
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower():
            raise
        outcome.update({'fits_required_batch': False, 'failure': str(error), 'elapsed_seconds': time.perf_counter() - start})
    finally:
        sampler.stop()
        del optimizer, lpips, model, batch, targets, labels, strata
        if device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
    return outcome


def memory_probe(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device(args.device)
    reports = [profile_latent_size(size, device, args.batch_size) for size in args.latent_dims]
    result = {
        'device': str(device),
        'patch_shape': list(PATCH_SHAPE),
        'training_batch_size': args.batch_size,
        'data_residency': 'Zarr patches are streamed by the training DataLoader; only the current batch is resident.',
        'method': 'one full-precision E1 forward/backward/AdamW step with reconstruction, LPIPS, classifier, and SupCon losses',
        'mps_fit_rule': 'sampled peak driver allocation must stay below 80% of recommended MPS working-set limit',
        'results': reports,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


def reconstruction_gate(args: argparse.Namespace) -> None:
    report_path = Path(args.report)
    report = json.loads(report_path.read_text())
    if report.get('settings', {}).get('input_mode') != 'tokenizer':
        raise ValueError('reconstruction gate requires exact tokenizer-mode metrics.')
    if not report.get('reconstruction_gate', {}).get('full_dataset_coverage'):
        raise ValueError('reconstruction gate requires full dataset coverage.')
    arms = report.get('arms', {})
    if args.candidate not in arms or any(control not in arms for control in args.controls):
        raise ValueError('reconstruction report is missing the candidate or a required control.')
    candidate = arms[args.candidate]['evaluation']
    comparisons = {}
    for control_name in args.controls:
        control = arms[control_name]['evaluation']
        comparisons[control_name] = {
            'synthetic_recipe_total_delta': (
                float(candidate['synthetic']['sampled']['recipe_total'])
                / float(control['synthetic']['sampled']['recipe_total']) - 1
            ),
            'real_mae_delta': (
                float(candidate['real']['sampled']['mae'])
                / float(control['real']['sampled']['mae']) - 1
            ),
        }
    result = {
        'candidate': args.candidate,
        'controls': args.controls,
        'maximum_regression': args.maximum_regression,
        'full_dataset_coverage': True,
        'comparisons': comparisons,
        'passed': all(
            values['synthetic_recipe_total_delta'] <= args.maximum_regression
            and values['real_mae_delta'] <= args.maximum_regression
            for values in comparisons.values()
        ),
    }
    destination = report_path.with_name('latent_reconstruction_gate.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


def finalize(args: argparse.Namespace) -> None:
    root = Path(args.root)
    reconstruction_path = root / 'stage1_tokenizer_reconstruction.json'
    stage1_gate_path = root / 'latent_reconstruction_gate.json'
    if not reconstruction_path.is_file() or not stage1_gate_path.is_file():
        raise FileNotFoundError('stage-1 tokenizer evaluation and gate are both required.')
    reconstruction = json.loads(reconstruction_path.read_text())
    gate = json.loads(stage1_gate_path.read_text())
    if reconstruction['settings'].get('input_mode') != 'tokenizer':
        raise ValueError('finalization requires tokenizer-mode reconstruction reports.')
    if not reconstruction.get('reconstruction_gate', {}).get('full_dataset_coverage'):
        raise ValueError('finalization requires full-coverage reconstruction evaluation.')
    candidate_name = 'mu512'
    baseline_name = 'p0b'
    stage1_control_name = 'mu128'
    baseline_report = json.loads(Path(args.p0b_benchmark).read_text())
    baseline_n5 = float(baseline_report['diagnostics']['neighbor_overlap_at_5'])
    records = []

    def reconstruction_values(report_path: Path, arm_name: str, geology_control_name: str | None = None):
        report = json.loads(report_path.read_text())
        candidate = report['arms'][arm_name]['evaluation']
        baseline = report['arms'][baseline_name]['evaluation']
        values = {
            'synthetic': float(candidate['synthetic']['sampled']['recipe_total']),
            'synthetic_delta': float(candidate['synthetic']['sampled']['recipe_total'])
            / float(baseline['synthetic']['sampled']['recipe_total']) - 1,
            'real': float(candidate['real']['sampled']['mae']),
            'real_delta': float(candidate['real']['sampled']['mae'])
            / float(baseline['real']['sampled']['mae']) - 1,
            'kl_unweighted': float(candidate['synthetic']['mean']['kl_per_example']),
            'kl_weighted': 0.001 * float(candidate['synthetic']['mean']['kl_per_voxel']),
            'full_coverage': bool(report['reconstruction_gate']['full_dataset_coverage']),
        }
        control_name = geology_control_name or stage1_control_name
        if control_name in report['arms']:
            control = report['arms'][control_name]['evaluation']
            values['synthetic_delta_vs_mu128'] = (
                float(candidate['synthetic']['sampled']['recipe_total'])
                / float(control['synthetic']['sampled']['recipe_total']) - 1
            )
            values['real_delta_vs_mu128'] = (
                float(candidate['real']['sampled']['mae'])
                / float(control['real']['sampled']['mae']) - 1
            )
        else:
            values['synthetic_delta_vs_mu128'] = ''
            values['real_delta_vs_mu128'] = ''
        return values

    stage1 = reconstruction_values(reconstruction_path, candidate_name, 'mu128')
    records.append({
        'seed': args.seed, 'latent_dim': 512, 'phase': 'stage1_reconstruction',
        'epoch': '', 'synthetic_tokenizer_recipe_total': stage1['synthetic'],
        'synthetic_delta_vs_p0b': stage1['synthetic_delta'],
        'synthetic_delta_vs_mu128': stage1['synthetic_delta_vs_mu128'],
        'real_tokenizer_mae': stage1['real'], 'real_delta_vs_p0b': stage1['real_delta'],
        'real_delta_vs_mu128': stage1['real_delta_vs_mu128'],
        'n5': '', 'n10': '', 'eligible_query_count': '', 'macro_auroc': '',
        'kl_unweighted': stage1['kl_unweighted'], 'kl_weighted': stage1['kl_weighted'],
        'reconstruction_gate': bool(gate['passed']), 'retrieval_gate': '',
        'checkpoint': str(root / 'e1_mu512/stage1_reconstruction/vae_best.pt'),
    })

    baseline_n10 = float(baseline_report['diagnostics']['neighbor_overlap_at_10'])
    stage2_candidates = []
    for epoch in args.epochs:
        benchmark_path = root / f'mu512_ep{epoch}_zgeo.json'
        reconstruction_epoch_path = root / f'mu512_ep{epoch}_tokenizer.json'
        if not benchmark_path.is_file() or not reconstruction_epoch_path.is_file():
            raise FileNotFoundError(f'missing epoch {epoch} retrieval or tokenizer evaluation report.')
        benchmark = json.loads(benchmark_path.read_text())
        metrics = reconstruction_values(reconstruction_epoch_path, f'mu512_ep{epoch}', 'mu128_ep30')
        n5 = float(benchmark['diagnostics']['neighbor_overlap_at_5'])
        n10 = float(benchmark['diagnostics']['neighbor_overlap_at_10'])
        query_count = int(benchmark['global_metrics']['query_count'])
        classifier = benchmark.get('classifier_metrics', {})
        macro_auroc = classifier.get('macro_auroc', '')
        reconstruction_pass = (
            metrics['full_coverage'] and metrics['synthetic_delta'] <= 0.02
            and metrics['real_delta'] <= 0.02
            and metrics['synthetic_delta_vs_mu128'] <= 0.02
            and metrics['real_delta_vs_mu128'] <= 0.02
        )
        retrieval_pass = reconstruction_pass and n5 > baseline_n5
        stage2_candidates.append({
            'epoch': epoch, 'n5': n5, 'n10': n10, 'query_count': query_count,
            'macro_auroc': macro_auroc, 'reconstruction_pass': reconstruction_pass,
            'retrieval_pass': retrieval_pass,
        })
        records.append({
            'seed': args.seed, 'latent_dim': 512, 'phase': 'stage2_geology',
            'epoch': epoch, 'synthetic_tokenizer_recipe_total': metrics['synthetic'],
            'synthetic_delta_vs_p0b': metrics['synthetic_delta'],
            'synthetic_delta_vs_mu128': metrics['synthetic_delta_vs_mu128'],
            'real_tokenizer_mae': metrics['real'], 'real_delta_vs_p0b': metrics['real_delta'],
            'real_delta_vs_mu128': metrics['real_delta_vs_mu128'],
            'n5': n5, 'n10': n10, 'eligible_query_count': query_count,
            'macro_auroc': macro_auroc, 'kl_unweighted': metrics['kl_unweighted'],
            'kl_weighted': metrics['kl_weighted'],
            'reconstruction_gate': reconstruction_pass, 'retrieval_gate': retrieval_pass,
            'checkpoint': str(root / f'e1_mu512/stage2_geology/vae_epoch{epoch}.pt'),
        })
    eligible = [item for item in stage2_candidates if item['retrieval_pass']]
    best = max(eligible, key=lambda item: (item['n5'], item['n10'])) if eligible else None
    decision = {
        'seed': args.seed,
        'latent_dim': 512,
        'same_seed_p0b_n5': baseline_n5,
        'same_seed_p0b_n10': baseline_n10,
        'stage1_reconstruction_gate_passed': bool(gate['passed']),
        'stage2_milestones': stage2_candidates,
        'eligible_milestone': best,
        'screening_success': best is not None,
        'adoption': False,
        'adoption_note': 'One seed is a screen only; an eligible candidate requires independent-seed replication.',
    }
    csv_path = root / 'latent_size_comparison.csv'
    fields = [
        'seed', 'latent_dim', 'phase', 'epoch', 'parameter_count',
        'synthetic_tokenizer_recipe_total', 'synthetic_delta_vs_p0b', 'synthetic_delta_vs_mu128',
        'real_tokenizer_mae', 'real_delta_vs_p0b', 'real_delta_vs_mu128', 'n5', 'n10',
        'eligible_query_count', 'macro_auroc', 'kl_unweighted', 'kl_weighted',
        'reconstruction_gate', 'retrieval_gate', 'checkpoint',
    ]
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        checkpoint_payload = torch.load(
            root / 'e1_mu512/stage1_reconstruction/vae_best.pt',
            map_location='cpu',
            weights_only=True,
        )
        parameter_count = sum(
            int(value.numel()) for name, value in checkpoint_payload['model_state_dict'].items()
            if not name.endswith(('running_mean', 'running_var', 'num_batches_tracked', 'pos_weight'))
        )
        for record in records:
            record['parameter_count'] = parameter_count
            writer.writerow(record)
    decision_path = root / 'p9_mu512_decision.json'
    with decision_path.open('x') as stream:
        json.dump(decision, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description='P9 latent-width initialization and memory preflight.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    init_parser = subparsers.add_parser('initialize', help='Create a fresh wide E1 checkpoint from seed-matched encoder trunk weights.')
    init_parser.add_argument('--source-checkpoint', required=True)
    init_parser.add_argument('--output', required=True)
    init_parser.add_argument('--latent-dim', type=int, required=True, choices=(512, 2048))
    init_parser.add_argument('--seed', type=int, default=20260926)
    init_parser.set_defaults(function=initialize)
    probe_parser = subparsers.add_parser('memory-probe', help='Profile a realistic full-batch training step for 512/2048.')
    probe_parser.add_argument('--device', choices=('mps', 'cuda', 'cpu'), default='mps')
    probe_parser.add_argument('--batch-size', type=int, default=12)
    probe_parser.add_argument('--latent-dims', type=int, nargs='+', default=(512, 2048), choices=(512, 2048))
    probe_parser.add_argument('--out-json', required=True)
    probe_parser.set_defaults(function=memory_probe)
    gate_parser = subparsers.add_parser('reconstruction-gate', help='Compare full-tokenizer metrics against every matched control.')
    gate_parser.add_argument('--report', required=True)
    gate_parser.add_argument('--candidate', required=True)
    gate_parser.add_argument('--controls', nargs='+', required=True)
    gate_parser.add_argument('--maximum-regression', type=float, default=0.02)
    gate_parser.set_defaults(function=reconstruction_gate)
    finalize_parser = subparsers.add_parser('finalize', help='Write the comparison CSV and screening decision after all evaluations.')
    finalize_parser.add_argument('--root', required=True)
    finalize_parser.add_argument('--seed', type=int, default=20260926)
    finalize_parser.add_argument('--p0b-benchmark', required=True)
    finalize_parser.add_argument('--epochs', type=int, nargs='+', default=(10, 20, 30, 40))
    finalize_parser.set_defaults(function=finalize)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()

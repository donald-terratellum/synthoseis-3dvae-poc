import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from src.model import Decoder, VAE3D


class PairedBatchSampler(Sampler):
    def __init__(self, size, batch_size, batches, seed):
        if min(size, batch_size, batches) < 1 or batch_size * batches > size:
            raise ValueError('paired sampling requires a positive budget no larger than the dataset')
        self.size, self.batch_size, self.batches, self.seed = size, batch_size, batches, seed
        self.epoch = 0
        self.schedule = []
        self.consumed = []
        self.seen = set()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)[:self.batch_size * self.batches].tolist()
        self.schedule = [indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
        self.consumed = []
        yield from self.schedule

    def record_consumed(self):
        position = len(self.consumed) // self.batch_size
        if position >= len(self.schedule):
            raise RuntimeError('paired sample budget exceeded')
        self.consumed.extend(self.schedule[position])
        self.seen.update(self.schedule[position])


def write_sampling_audit(sampler, dataset_path, destination):
    import zarr

    root = zarr.open(str(dataset_path), mode='r')
    indices = np.asarray(sampler.consumed, dtype=np.int64)
    sources = np.asarray(root['source_volume_index'][:])
    volumes, counts = np.unique(sources[indices], return_counts=True)
    report = {'epoch': sampler.epoch + 1, 'indices': sampler.consumed,
              'consumed': len(indices), 'unique_this_epoch': len(set(sampler.consumed)),
              'unique_cumulative': len(sampler.seen), 'dataset_size': sampler.size,
              'source_volume_counts': {str(volume): int(count) for volume, count in zip(volumes, counts)}}
    Path(destination).write_text(json.dumps(report) + '\n')
    print(f'P7 COVERAGE: epoch={report["epoch"]} consumed={report["consumed"]} cumulative_unique={report["unique_cumulative"]}/{sampler.size}', flush=True)


def capture_training_state(optimizer, scheduler, loader_generator, early_stopping):
    numpy_state = np.random.get_state()
    state = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'torch_rng': torch.get_rng_state(),
        'python_rng': random.getstate(),
        'numpy_rng': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'loader_rng': loader_generator.get_state() if loader_generator is not None else None,
        'early_stopping': dict(vars(early_stopping)),
    }
    if torch.backends.mps.is_available():
        state['mps_rng'] = torch.mps.get_rng_state()
    return state


def restore_training_state(state, optimizer, scheduler, loader_generator, early_stopping):
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        if state['scheduler'] is None:
            raise ValueError('resume checkpoint has no scheduler state')
        scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'].cpu())
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if loader_generator is not None:
        loader_generator.set_state(state['loader_rng'].cpu())
    if 'mps_rng' in state:
        if not torch.backends.mps.is_available():
            raise ValueError('MPS training-state resume requires MPS')
        torch.mps.set_rng_state(state['mps_rng'].cpu())
    for key, value in state['early_stopping'].items():
        setattr(early_stopping, key, value)


def prepare_initializations(root, seed):
    from scripts.train import build_checkpoint_payload

    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    torch.manual_seed(seed)
    control = VAE3D(patch_shape=(32, 32, 64), geology_projection=True, geology_classifier=True)
    candidate = VAE3D(patch_shape=(32, 32, 64), encoder_arch='residual', geology_projection=True, geology_classifier=True)
    prefixes = ('encoder.fc_mu.', 'encoder.fc_logvar.', 'decoder.', 'geology_head.', 'geology_classifier.')
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in control_state.items():
        if key.startswith(prefixes):
            candidate_state[key].copy_(value)
            torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)
    for name, model in [('e0', control), ('e1', candidate)]:
        if not isinstance(model.decoder, Decoder):
            raise ValueError('P7 must retain the original decoder')
        destination = root / f'{name}_initial.pt'
        torch.save(build_checkpoint_payload(model, epoch=0), destination)
        print(f'INITIALIZATION {name}: {destination}; parameters={sum(parameter.numel() for parameter in model.parameters())}', flush=True)


def reconstruction_gate(report_dir):
    results = {}
    for mode in ('clean', 'extrema'):
        report = json.loads((Path(report_dir) / f'reconstruction_{mode}.json').read_text())
        if not report['reconstruction_gate']['full_dataset_coverage']:
            raise ValueError('full dataset evaluation is required before P7 geology')
        arms = report['arms']
        results[mode] = {}
        for candidate in ('e0', 'e1'):
            references = ('p0b',) if candidate == 'e0' else ('p0b', 'e0')
            results[mode][candidate] = all(
                arms[candidate]['evaluation']['synthetic']['sampled']['recipe_total'] <= 1.02 * arms[reference]['evaluation']['synthetic']['sampled']['recipe_total']
                and arms[candidate]['evaluation']['real']['sampled']['mae'] <= 1.02 * arms[reference]['evaluation']['real']['sampled']['mae']
                for reference in references
            )
    decision = {'modes': results, 'geology_eligible': all(values[arm] for values in results.values() for arm in ('e0', 'e1'))}
    (Path(report_dir) / 'reconstruction_gate.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(f'P7 RECONSTRUCTION GATE: {decision}', flush=True)
    return decision


def tokenizer_reconstruction_gate(report_path, candidate='e1', control='p0b', maximum_regression=0.02):
    report = json.loads(Path(report_path).read_text())
    if report.get('settings', {}).get('input_mode') != 'tokenizer':
        raise ValueError('replication gate requires exact tokenizer input mode')
    if report.get('settings', {}).get('contract_audit') is not True:
        raise ValueError('replication gate requires tokenizer contract parity audit')
    if not {candidate, control} <= set(report.get('arms', {})):
        raise ValueError('replication gate requires candidate and same-seed control checkpoints')
    if not report.get('reconstruction_gate', {}).get('full_dataset_coverage'):
        raise ValueError('replication gate requires full dataset coverage')
    candidate_values = report['arms'][candidate]['evaluation']
    control_values = report['arms'][control]['evaluation']
    comparisons = {
        'synthetic_recipe_total_regression': candidate_values['synthetic']['sampled']['recipe_total'] / control_values['synthetic']['sampled']['recipe_total'] - 1,
        'real_mae_regression': candidate_values['real']['sampled']['mae'] / control_values['real']['sampled']['mae'] - 1,
    }
    passed = all(value <= maximum_regression for value in comparisons.values())
    result = {'candidate': candidate, 'control': control, 'input_mode': 'tokenizer',
              'maximum_regression': maximum_regression, 'comparisons': comparisons,
              'passed': passed, 'full_dataset_coverage': True}
    destination = Path(report_path).with_name('tokenizer_reconstruction_gate.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(f'SAME-SEED TOKENIZER RECONSTRUCTION GATE: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'gate', 'tokenizer-gate'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--candidate', default='e1')
    parser.add_argument('--control', default='p0b')
    parser.add_argument('--maximum-regression', type=float, default=0.02)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_initializations(args.root, args.seed)
    elif args.command == 'gate':
        reconstruction_gate(args.root)
    else:
        tokenizer_reconstruction_gate(args.root, args.candidate, args.control, args.maximum_regression)


if __name__ == '__main__':
    main()
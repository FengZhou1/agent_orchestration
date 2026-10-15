"""Isolated toy three-phase PPO equivalence against the pinned old source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def compare(a, b, path='checkpoint'):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6, msg=path)
    elif isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6, err_msg=path)
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{index}]')
    elif isinstance(a, float):
        assert np.isclose(a, b, atol=1e-7, rtol=1e-6, equal_nan=True), path
    else:
        assert a == b, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    for label, source in (('reference', args.reference_source.resolve()), ('optimized', repository)):
        env = dict(os.environ, PYTHONPATH=str(source / 'src'), PYTHONIOENCODING='utf-8')
        command = [sys.executable, str(source / 'scripts/run_slot_joint.py'),
            '--scenario', str(source / 'configs/toy.yaml'), '--output', str(output / label),
            '--smoke', '--updates', '3', '--route-updates', '1', '--deploy-updates', '1',
            '--slots', '2', '--mapping-samples', '7', '--seed', '17',
            '--hidden-size', '16', '--hidden-layers', '1', '--minibatch-size', '8',
            '--update-epochs', '1', '--workflow-engine', 'array', '--workflow-workers', '1']
        subprocess.run(command, cwd=source, env=env, check=True)
    a = torch.load(output / 'reference/checkpoint.pt', map_location='cpu', weights_only=False)
    b = torch.load(output / 'optimized/checkpoint.pt', map_location='cpu', weights_only=False)
    keys = ('next_update', 'policy_state_dict', 'optimizer_state_dict', 'lagrange_multipliers',
            'environment_reward_state', 'numpy_random_state', 'python_random_state', 'torch_random_state')
    for key in keys:
        compare(a[key], b[key], key)
    for left, right in zip(a['history'], b['history']):
        for key in left:
            if key.startswith('timing_') or key in ('collection_time_s', 'optimization_time_s'):
                continue
            compare(left[key], right[key], f'history.{key}')
    report = {'status': 'PASS', 'scope': 'toy 3-phase PPO, not convergence evidence',
              'phases': [row['training_phase'] for row in a['history']],
              'compared': keys, 'float32_atol': 1e-7, 'float32_rtol': 1e-6}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

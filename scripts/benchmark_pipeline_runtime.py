"""Bind one saved run, then compare the pinned pre-task source in isolation.

Reference code is mechanically exported from Git, never mixed with optimized
modules. Original training assets are read only; snapshots/reports are new.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def prepare_reference(root, commit, repository):
    marker = root / 'reference_commit.txt'
    if root.exists():
        if not marker.exists() or marker.read_text(encoding='utf-8') != commit:
            raise ValueError('Reference source directory does not match pinned commit')
        return
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit], cwd=repository)
    root.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if not (root / member.name).resolve().is_relative_to(root.resolve()):
                raise ValueError('Archive path escapes the reference source directory')
        stream.extractall(root, filter='data')
    marker.write_text(commit, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--reference-commit', default='de4a017')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference_root = args.reference_source.resolve()
    prepare_reference(reference_root, args.reference_commit, repository)
    run = args.run.resolve()
    snapshot = output / 'input_snapshot'
    snapshot.mkdir()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    source_hashes = {}
    for name in ('train_trajectory.json', 'warmup_deployments.json', 'checkpoint.pt'):
        path = run / name
        if not path.exists() and name == 'warmup_deployments.json':
            continue
        data = path.read_bytes()
        source_hashes[name] = hashlib.sha256(data).hexdigest()
        (snapshot / name).write_bytes(data)
    scenario = Path(manifest['scenario']).read_bytes()
    if hashlib.sha256(scenario).hexdigest() != manifest['scenario_hash']:
        raise ValueError('Scenario differs from run manifest')
    (snapshot / 'scenario.yaml').write_bytes(scenario)
    manifest['scenario'] = str(snapshot / 'scenario.yaml')
    (snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'binding.json').write_text(json.dumps({'original_run': str(run),
        'reference_commit': args.reference_commit, 'scenario_hash': manifest['scenario_hash'],
        'asset_sha256': source_hashes}, indent=2), encoding='utf-8')
    for label, source in (('reference', reference_root), ('optimized', repository)):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(source / 'src')
        env['PYTHONIOENCODING'] = 'utf-8'
        command = [sys.executable, str(source / 'scripts/benchmark_slot_runtime.py'),
                   '--run', str(snapshot), '--workflow-engine', 'array', '--workflow-workers', '1',
                   '--output', str(output / label)]
        print(f'[pipeline] {label}: {args.run}', flush=True)
        subprocess.run(command, cwd=source, env=env, check=True)
    a = json.loads((output / 'reference/report.json').read_text(encoding='utf-8'))
    b = json.loads((output / 'optimized/report.json').read_text(encoding='utf-8'))
    for key in ('metrics_digest', 'record_digest', 'trajectory_digest', 'mapping_samples',
                'warmup_deployment_digest', 'checkpoint_next_update', 'transitions'):
        if a[key] != b[key]:
            raise AssertionError(f'Pipeline mismatch: {key}')
    report = {'status': 'PASS', 'scope': '32-slot rollout, no PPO update',
              'reference_commit': args.reference_commit, 'source_assets': source_hashes,
              'byte_exact_metrics_and_records': True, 'reference_s': a['collection_s'],
              'optimized_s': b['collection_s'], 'speedup': a['collection_s'] / b['collection_s'],
              'reference_timings_s': a['timings_s'], 'optimized_timings_s': b['timings_s'],
              'reference_memory': a['memory'], 'optimized_memory': b['memory']}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Mechanical gates for the paired probes and full no-update rollouts."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scenes = {}
    for count in (20, 50):
        pair_dir = args.input / f'final_pairs_n20_a{count}'
        pairs = load(pair_dir / 'report.json')
        labels = ('reference_1', 'array_1', 'array_2', 'array_4')
        grouped = {label: [r for r in pairs['rows'] if r['engine'] == label] for label in labels}
        assert all(len(rows) == 25 for rows in grouped.values()), 'Incomplete five-pair checks'
        for label, rows in grouped.items():
            assert {(r['repeat'], r['slot']) for r in rows} == {
                (repeat, slot) for repeat in range(5) for slot in (0, 1, 2, 17, 31)}
        # Byte-exact complete metric hashes are stronger than the float64 gate.
        exact = all({r['metrics_sha256'] for rows in grouped.values() for r in rows if r['slot'] == slot}
                    == {grouped['reference_1'][index]['metrics_sha256']}
                    for index, slot in enumerate((0, 1, 2, 17, 31)))
        timing = {label: {'workflow_median_s': median(r['timings_s']['workflow'] for r in rows),
                         'warm_workflow_median_s': median(r['timings_s']['workflow'] for r in rows if r['repeat'] > 0),
                         'cold_first_slot_s': rows[0]['timings_s']['workflow']}
                  for label, rows in grouped.items()}
        rollouts = {label: load(args.input / f'rollout_n20_a{count}_{label.replace("_", "")}' / 'report.json') for label in labels}
        reference = rollouts['reference_1']
        for record in rollouts.values():
            assert record['trajectory_digest'] == pairs['trajectory_digest']
            assert record['warmup_deployment_digest'] == pairs['warmup_digest']
            assert record['checkpoint_next_update'] == pairs['checkpoint_next_update']
            for path, digest in pairs['source_sha256'].items():
                assert record['source_sha256_at_start'][path] == digest, 'Source changed during verification'
            checkpoints = [digest for path, digest in record['source_sha256_at_start'].items()
                           if Path(path).name == 'checkpoint.pt']
            assert checkpoints == [pairs['checkpoint_sha256']], 'Different policy checkpoint'
        rollout_exact = all(record['metrics_digest'] == reference['metrics_digest']
                            and record['record_digest'] == reference['record_digest']
                            and record['transitions'] == reference['transitions']
                            for record in rollouts.values())
        for label, record in rollouts.items():
            timing[label]['rollout_s'] = record['collection_s']
            timing[label]['rollout_workflow_s'] = record['timings_s']['workflow']
        speed = timing['reference_1']['workflow_median_s'] / timing['array_1']['workflow_median_s']
        rollout_speed = reference['collection_s'] / rollouts['array_1']['collection_s']
        recommendations = {}
        for workers in (2, 4):
            label = f'array_{workers}'
            memory = load(pair_dir / f'{label}_memory.json')
            serial_memory = load(pair_dir / 'array_1_memory.json')
            full_memory = rollouts[label]['memory']
            full_serial = rollouts['array_1']['memory']
            extra = max(memory['peak_children_private_bytes'], full_memory['peak_children_private_bytes'],
                memory['peak_private_bytes'] - serial_memory['peak_private_bytes'],
                full_memory['peak_private_bytes'] - full_serial['peak_private_bytes'])
            available = min(memory['minimum_system_available_bytes'], full_memory['minimum_system_available_bytes'])
            gain = rollouts['array_1']['collection_s'] / rollouts[label]['collection_s']
            workflow_gain = timing['array_1']['warm_workflow_median_s'] / timing[label]['warm_workflow_median_s']
            reduction = 1.0 - 1.0 / gain
            workflow_reduction = 1.0 - 1.0 / workflow_gain
            memory_ok = extra <= (1 if workers == 2 else 2) * 1024 ** 3 and available >= 1024 ** 3
            recommendations[label] = {'rollout_speedup_vs_array1': gain, 'warm_workflow_speedup_vs_array1': workflow_gain,
                'rollout_time_reduction_vs_array1': reduction, 'warm_workflow_time_reduction_vs_array1': workflow_reduction,
                'extra_private_bytes': extra, 'minimum_system_available_bytes': available,
                'memory_gate_pass': memory_ok,
                'recommended': reduction >= 0.15 and workflow_reduction >= 0.15 and memory_ok}
        scenes[f'N20_A{count}'] = {'timings': timing,
            'paired_metrics_byte_exact': exact, 'rollout_records_and_metrics_byte_exact': rollout_exact,
            'workflow_speedup_array1': speed, 'rollout_speedup_array1': rollout_speed,
            'array1_promotion_gate': exact and rollout_exact and speed >= 2 and rollout_speed >= 1.3,
            'parallel': recommendations, 'source_sha256': pairs['source_sha256']}
    report = {'scope': 'runtime verification, no scientific protocol change or convergence claim',
        'environment': {'python': sys.version, 'executable': sys.executable,
                        **{name: version(name) for name in ('numpy', 'torch', 'networkx', 'psutil')}},
        'scenes': scenes, 'promote_new_runs_to_array1': all(s['array1_promotion_gate'] for s in scenes.values())}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

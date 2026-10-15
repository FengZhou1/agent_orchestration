"""Paired, real-manifest workflow checks; no training update or result overwrite."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent_orch.agents import StructuredActorCritic
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.parameters import ppo_config_from_manifest
from agent_orch.performance.workflow_runtime import WorkflowOptions, WorkflowRuntime
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from benchmark_slot_runtime import canonical
from run_slot_joint import load_training_trajectory, load_warmup_deployments, environment_options
from workflow_probe_memory import MemoryMonitor


def compare(expected, actual, path='root'):
    if isinstance(expected, dict):
        assert list(expected) == list(actual), path
        for key in expected:
            compare(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual), path
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, (float, np.floating)):
        assert np.isclose(expected, actual, atol=1e-12, rtol=1e-12, equal_nan=True), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


def fixed_decisions(env, policy, slot):
    """Generate policy actions without evaluating the last action's workflow.

    This preparation is NOT timed. Use the same normalization as env.step;
    all engines then receive the identical, serialized decisions and loads.
    The separate full-rollout probe exercises the unmodified final transition.
    """
    env._period_index = slot
    env._begin_deployment_cycle()
    obs = env._observation()
    for group in range(len(env.layout.model_groups)):
        action, _, _ = policy.act(obs, deterministic=True, device='cpu')
        if group + 1 < len(env.layout.model_groups):
            obs = env.step(action)[0]
        else:
            raw = np.asarray(action['model'], dtype=float).reshape(env._working_share.shape)
            row = np.maximum(0.0, raw[group])
            active = env.action_masks()['model_mask'].reshape(env._working_share.shape)[group]
            row = np.where(active, row, 0.0)
            total = float(row.sum())
            env._working_share[group] = row / total if total > 0 else active.astype(float) / max(1, int(active.sum()))
    routing = env.decode_routing({'model': env._working_share.reshape(-1).copy()})
    return env.scenario, env.current_deployment.copy(), routing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--slots', default='0,1,2,17,31')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source_sha256 = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path('src/agent_orch/performance').glob('workflow*.py')}
    manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
    scene = ScenarioLoader.load(manifest['scenario'])
    assert hashlib.sha256(Path(manifest['scenario']).read_bytes()).hexdigest() == manifest['scenario_hash']
    trace = load_training_trajectory(args.run, manifest)
    bank = load_warmup_deployments(args.run, manifest, scene, trace)
    checkpoint_bytes = (args.run / 'checkpoint.pt').read_bytes()
    import io
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    assert checkpoint.get('warmup_deployment_digest') == bank.digest()
    fields_ = {f.name for f in fields(ObjectiveSpec)}
    objective = ObjectiveSpec(**{k: v for k, v in manifest['objective'].items() if k in fields_})
    env = SlotSequentialJointEnv(scene, trace, mapping_samples=manifest['training_settings']['mapping_samples'],
        objective=objective, warmup_deployments=bank, **environment_options(manifest))
    env.training_phase = 'composition'
    policy = StructuredActorCritic(env, ppo_config_from_manifest(manifest))
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    decisions = {slot: fixed_decisions(env, policy, slot) for slot in map(int, args.slots.split(','))}
    expected = {}
    rows = []
    for option in [WorkflowOptions('reference', 1), *[WorkflowOptions('array', n) for n in (1, 2, 4)]]:
        label = f'{option.engine}_{option.workers}'
        with WorkflowRuntime(option) as runtime, MemoryMonitor() as memory:
            for repeat in range(args.repeats):
                for slot, (scene_at, deployment, routing) in decisions.items():
                    simulator = Simulator(scene_at, env.simulator.workflow.max_mapping_samples, workflow_runtime=runtime)
                    simulator.slot = slot
                    if slot:
                        simulator.previous_deployment = bank.deployment_at(slot - 1)
                    started = time.perf_counter()
                    metrics = simulator.step(deployment, routing).metrics
                    wall = time.perf_counter() - started
                    value = env.objective.evaluate(metrics, simulator.current_arrival_rates())
                    current = {'metrics': asdict(metrics), 'utility': value.utility, 'constraints': list(value.constraints)}
                    previous = checkpoint['environment_reward_state']['previous_round'][slot]
                    current['delta_utility'] = float(value.utility) - previous['utility']
                    current['penalized_reward'] = current['delta_utility'] - float(np.asarray(value.constraints) @ checkpoint['lagrange_multipliers'])
                    if slot not in expected:
                        expected[slot] = current
                    compare(expected[slot], current)
                    # Discrete flow SLO/completion values must agree exactly.
                    for key, flow in metrics.diagnostics.get('flow_metrics', {}).items():
                        for field in ('slo', 'completion_probability'):
                            assert flow[field] == expected[slot]['metrics']['diagnostics']['flow_metrics'][key][field]
                    row = {'engine': label, 'repeat': repeat, 'slot': slot,
                        'wall_s': wall, 'timings_s': simulator.last_timings,
                        'cache': simulator.last_workflow_cache,
                        'metrics_sha256': hashlib.sha256(json.dumps(canonical(current), sort_keys=True).encode()).hexdigest()}
                    rows.append(row)
                    print(f'{label} repeat={repeat+1} slot={slot} workflow={simulator.last_timings["workflow"]:.3f}s exactness=PASS', flush=True)
            (args.output / f'{label}_memory.json').write_text(json.dumps(memory.report(), indent=2), encoding='utf-8')
        report = {'run': str(args.run), 'scenario_hash': manifest['scenario_hash'],
            'source_sha256': source_sha256,
            'trajectory_digest': trace.digest(), 'warmup_digest': bank.digest(),
            'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
            'checkpoint_next_update': checkpoint['next_update'], 'rows': rows,
            'exactness': 'PASS for completed rows', 'float64_atol': 1e-12, 'float64_rtol': 1e-12}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'expected_metrics.json').write_text(json.dumps(canonical(expected), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

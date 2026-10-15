"""Bounded exactness/runtime probe: one rollout, no PPO update or teacher.

Run before/after an implementation optimization using the same saved trajectory.
Digests cover observations, sampled actions, values, rewards and final metrics.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import cProfile
from dataclasses import asdict, fields, replace
import hashlib
import io
import json
from pathlib import Path
import pstats
import random
import time

import numpy as np
import torch

from agent_orch.agents import StructuredActorCritic
from agent_orch.agents.rollout import collect_rollout
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.parameters import load_settings, make_ppo_config, ppo_config_from_manifest
from agent_orch.performance.workflow_runtime import WorkflowOptions, WorkflowRuntime
from run_slot_joint import load_training_trajectory, load_warmup_deployments, environment_options, checkpoint_trajectory_digest
from workflow_probe_memory import MemoryMonitor
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def _main(stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--run", type=Path, help="Bind to actual manifest, warm-up bank and policy checkpoint")
    parser.add_argument("--workflow-engine", choices=("reference", "array"))
    parser.add_argument("--workflow-workers", type=int, choices=(1, 2, 4))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--scalar-workflow", action="store_true", help="Use the retained scalar oracle for an exactness comparison")
    args = parser.parse_args()
    if args.run and (args.scenario or args.trajectory):
        parser.error('--run cannot be combined with --scenario/--trajectory')
    if not args.run and not (args.scenario and args.trajectory):
        parser.error('Specify --run OR both --scenario and --trajectory')
    manifest = {}
    checkpoint = None
    reference_reused = False
    warmup = None
    options = {}
    seed = 0
    if args.run:
        manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
        args.scenario = Path(manifest['scenario'])
        args.trajectory = args.run / 'train_trajectory.json'
        if hashlib.sha256(args.scenario.read_bytes()).hexdigest() != manifest['scenario_hash']:
            raise ValueError('Run scenario hash mismatch')
        options = environment_options(manifest)
        seed = manifest['ppo_seed']
        checkpoint_bytes = (args.run / 'checkpoint.pt').read_bytes()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    runtime_options = WorkflowOptions.resolve(manifest, args.workflow_engine, args.workflow_workers)
    if args.scalar_workflow and runtime_options.engine != 'reference':
        parser.error('--scalar-workflow requires --workflow-engine reference')
    runtime = stack.enter_context(WorkflowRuntime(runtime_options))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.scenario, args.trajectory, Path('configs/experiment_settings.yaml'),
        *[Path('src/agent_orch') / p for p in ('envs/base.py', 'envs/slot_joint_env.py',
            'performance/analytical.py', 'performance/workflow.py', 'performance/workflow_batch.py',
            'performance/workflow_mapping.py', 'performance/workflow_runtime.py', 'performance/llm_queueing.py',
            'performance/fixed_reference.py', 'routing/capacity.py', 'agents/networks.py', 'agents/rollout.py')])}
    settings = load_settings()
    scene = ScenarioLoader.load(args.scenario)
    if args.run:
        trajectory = load_training_trajectory(args.run, manifest)
        warmup = load_warmup_deployments(args.run, manifest, scene, trajectory)
        if (checkpoint_trajectory_digest(checkpoint) != trajectory.digest()
                or checkpoint.get('protocol') != manifest['protocol']
                or checkpoint.get('warmup_deployment_digest') != manifest.get('warmup_deployment_digest')
                or checkpoint.get('physical_routing') != manifest['physical_routing']
                or checkpoint.get('scenario_hash') != manifest['scenario_hash']):
            raise ValueError('Checkpoint provenance differs from run')
        config = ppo_config_from_manifest(manifest)
        spec_fields = {f.name for f in fields(ObjectiveSpec)}
        objective = ObjectiveSpec(**{k: v for k, v in manifest['objective'].items() if k in spec_fields})
        phase = checkpoint.get('optimizer_training_phase',
            manifest['training_settings']['phase_schedule'][-1][0])
        config = replace(config, training_phase=phase)
        mapping_samples = manifest['training_settings']['mapping_samples']
        provenance[str(args.run / 'checkpoint.pt')] = hashlib.sha256(checkpoint_bytes).hexdigest()
    else:
        trajectory = SlotTrajectory.from_dict(json.loads(args.trajectory.read_text(encoding="utf-8")))
        config = make_ppo_config(settings, training_phase="composition").for_constraint_count(4)
        objective = ObjectiveSpec(**settings['objective'])
        mapping_samples = 128
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    WorkflowEvaluator.DEFAULT_BATCH_EVALUATION = not args.scalar_workflow
    started = time.perf_counter()
    env = SlotSequentialJointEnv(scene, trajectory, mapping_samples=mapping_samples,
        seed=seed, objective=objective, warmup_deployments=warmup,
        workflow_runtime=runtime, **options)
    env.gamma = config.gamma
    policy = StructuredActorCritic(env, config)
    if checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        if checkpoint.get('environment_reward_state'):
            env.restore_reward_state(checkpoint['environment_reward_state'])
            # Every resumed slot already uses previous_round, not the first-
            # episode greedy reference. Avoid re-running that unused reference
            # in this diagnostic only; production initialization is unchanged.
            if set(env._previous_round) == set(range(len(trajectory))):
                env._baseline = {k: dict(v) for k, v in env._previous_round.items()}
                reference_reused = True
    env.training_phase = config.training_phase
    policy.set_training_phase(config.training_phase)
    captured = []
    original = env._step_composition

    def evaluate(action):
        result = original(action)
        captured.append(asdict(result[-1]["metrics"]))
        return result

    env._step_composition = evaluate
    obs, _ = env.reset(seed=seed)
    env.on_evaluation_phase = lambda event: print(f'[slot] {env._period_index + 1}/{len(trajectory)} {event}', flush=True)
    initialization_s = time.perf_counter() - started
    print(f"Initialization {initialization_s:.3f}s; collecting one rollout", flush=True)
    profiler = cProfile.Profile() if args.profile else None
    memory = stack.enter_context(MemoryMonitor())
    if profiler:
        profiler.enable()
    batch = collect_rollout(policy, env, update=0, rollout_steps=1,
        rollout_periods=len(trajectory), config=config, device="cpu", exploration_weight=0.,
        lagrange_multipliers=np.asarray(checkpoint['lagrange_multipliers']) if checkpoint else np.zeros(4),
        observation=obs, seed=seed, episode_counter=0)
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.output / "profile.pstats"))
    h = hashlib.sha256()
    for record in batch.records:
        for state in (record["observation"], record["next_observation"], record["action"]):
            for key, value in state.items():
                h.update(key.encode())
                h.update(np.asarray(value).tobytes())
        h.update(np.asarray([record[k] for k in ("log_prob", "value", "reward", "discount")]).tobytes())
    scientific = canonical(captured)
    metrics_digest = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    top = []
    if profiler:
        stats = pstats.Stats(profiler)
        for (file, line, name), row in sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[:25]:
            top.append({"function": f"{file}:{line}:{name}", "calls": row[1], "self_s": row[2], "cumulative_s": row[3]})
    report = {"scope": "one-rollout functional/runtime probe, not convergence evidence",
        "source_sha256_at_start": provenance,
        "trajectory_digest": trajectory.digest(), "mapping_samples": mapping_samples,
        "warmup_deployment_digest": warmup.digest() if warmup else None,
        "workflow_runtime": runtime_options.to_dict(),
        "checkpoint_next_update": checkpoint.get('next_update') if checkpoint else None,
        "fixed_reference_initialization": ('checkpoint previous-round reference for all slots'
            if reference_reused else 'computed greedy reference'),
        "initialization_s": initialization_s, "collection_s": batch.collection_time_s,
        "transitions": len(batch.records), "record_digest": h.hexdigest(),
        "metrics_digest": metrics_digest, "timings_s": getattr(batch, "timings_s", {}),
        "profile_enabled": args.profile, "profile_top": top}
    report['memory'] = memory.report()
    report['workflow_cache'] = runtime.last_cache
    # collect_rollout resets the env after its terminal slot; the new Simulator
    # has no last-evaluation diagnostic. These counters are process-cumulative.
    from agent_orch.performance.analytical import _reference_llm_capacity_cached, _llm_instance_cached
    report['analytical_cache'] = {'reference_capacity': _reference_llm_capacity_cached.cache_info()._asdict(),
                                 'instance': _llm_instance_cached.cache_info()._asdict()}
    report["workflow_evaluation"] = "scalar" if args.scalar_workflow else "indexed_batch"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(scientific, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    with ExitStack() as stack:
        return _main(stack)


if __name__ == "__main__":
    main()

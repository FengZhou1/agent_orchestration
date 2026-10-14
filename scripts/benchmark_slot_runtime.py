"""Bounded exactness/runtime probe: one rollout, no PPO update or teacher.

Run before/after an implementation optimization using the same saved trajectory.
Digests cover observations, sampled actions, values, rewards and final metrics.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import hashlib
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
from agent_orch.parameters import load_settings, make_ppo_config
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.scenario, args.trajectory, Path('configs/experiment_settings.yaml'),
        *[Path('src/agent_orch') / p for p in ('envs/base.py', 'envs/slot_joint_env.py',
            'performance/analytical.py', 'performance/workflow.py', 'agents/rollout.py')])}
    settings = load_settings()
    scene = ScenarioLoader.load(args.scenario)
    trajectory = SlotTrajectory.from_dict(json.loads(args.trajectory.read_text(encoding="utf-8")))
    config = make_ppo_config(settings, training_phase="composition").for_constraint_count(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.perf_counter()
    env = SlotSequentialJointEnv(scene, trajectory, mapping_samples=128,
        seed=0, objective=ObjectiveSpec(**settings["objective"]))
    policy = StructuredActorCritic(env, config)
    env.training_phase = "composition"
    captured = []
    original = env._step_composition

    def evaluate(action):
        result = original(action)
        captured.append(asdict(result[-1]["metrics"]))
        return result

    env._step_composition = evaluate
    obs, _ = env.reset(seed=0)
    initialization_s = time.perf_counter() - started
    print(f"Initialization {initialization_s:.3f}s; collecting one rollout", flush=True)
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    batch = collect_rollout(policy, env, update=0, rollout_steps=1,
        rollout_periods=len(trajectory), config=config, device="cpu", exploration_weight=0.,
        lagrange_multipliers=np.zeros(4), observation=obs, seed=0, episode_counter=0)
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
        "trajectory_digest": trajectory.digest(), "mapping_samples": 128,
        "initialization_s": initialization_s, "collection_s": batch.collection_time_s,
        "transitions": len(batch.records), "record_digest": h.hexdigest(),
        "metrics_digest": metrics_digest, "timings_s": getattr(batch, "timings_s", {}),
        "profile_enabled": args.profile, "profile_top": top}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(scientific, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

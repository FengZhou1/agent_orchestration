"""CPU routing timings and solver counts at a fixed deployment/model choice."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from agent_orch.capacity import CapacityPlanner
from agent_orch.performance import analytical
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.simulator import Simulator
from agent_orch.schema.loader import ScenarioLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True,
                        help="Explicit generated scenario; no inherited Abilene default")
    parser.add_argument("--mapping-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-slots", type=int, default=0,
                        help="Also time one CPU model-selection episode")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scenario = ScenarioLoader.load(args.scenario)
    placement = CapacityPlanner(scenario).initial_deployment()
    shares = {(app.id, g, model): 1 / len(scenario.models)
              for app in scenario.applications.values() for g in app.ingress_rates
              for model in scenario.models}
    rates = {(app.id, g): rate for app in scenario.applications.values()
             for g, rate in app.ingress_rates.items()}
    report = {"scenario_hash": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
              "active_instances": sum(placement.llm_active.values()),
              "mapping_samples": args.mapping_samples, "timings": {}}
    for name, router_type in (("capacity_proportional", CapacityProportionalRouter),):
        analytical._reference_llm_capacity_cached.cache_clear()
        with patch.object(analytical, "evaluate_llm_instance", wraps=analytical.evaluate_llm_instance) as full, \
             patch.object(analytical, "two_mode_capacity", wraps=analytical.two_mode_capacity) as scans:
            router = router_type(scenario)
            start = perf_counter()
            decision = router.route(placement, shares, None, rates)
            cold = perf_counter() - start
            calls, capacity_scans = full.call_count, scans.call_count
            start = perf_counter()
            repeat = router_type(scenario).route(placement, shares, None, rates)
            repeated = perf_counter() - start
            assert decision == repeat
            report["timings"][name] = {
                "cold_route_s": cold, "rebuilt_router_same_inputs_s": repeated,
                "cold_full_llm_evaluations": calls, "cold_capacity_scans": capacity_scans,
                "repeat_capacity_scans": scans.call_count - capacity_scans,
            }
        simulator = Simulator(scenario, max_mapping_samples=args.mapping_samples)
        start = perf_counter()
        result = simulator.backend.evaluate(placement, decision, rates)
        report["timings"][name]["actual_load_backend_s"] = perf_counter() - start
        start = perf_counter()
        simulator.workflow.evaluate(placement, decision, result, rates)
        report["timings"][name]["workflow_s"] = perf_counter() - start
    if args.rollout_slots:
        report["rollout"] = profile_rollout(scenario, args.rollout_slots, args.mapping_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def profile_rollout(scenario, slots, mapping_samples):
    import random
    import numpy as np
    import torch
    from agent_orch.agents import PPOConfig, StructuredActorCritic
    from agent_orch.agents.rollout import collect_rollout
    from agent_orch.envs import SlotSequentialJointEnv
    from agent_orch.objective import ObjectiveSpec
    from dataclasses import replace
    from agent_orch.workload import SlotTrajectory

    trace = SlotTrajectory.sample(scenario, slots, 101)
    objective = replace(ObjectiveSpec.slo_constrained(.9), quality_weight=.25,
                        goodput_weight=.25, network_utilization_target=.9)
    config = replace(PPOConfig().for_constraint_count(4), training_phase="composition",
                     gae_lambda=1., exploration_mode="none", shared_composition_head=True)
    results = {"trajectory_digest": trace.digest(), "ppo_seed": 0}
    for method, replay in (("capacity_proportional", False),):
        analytical._reference_llm_capacity_cached.cache_clear()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        env = SlotSequentialJointEnv(scenario, trace, mapping_samples=mapping_samples,
                                    routing_method=method, replay_frozen_deployment_steps=replay,
                                    objective=objective)
        env.training_phase = "composition"
        observation, _ = env.reset(seed=0)
        policy = StructuredActorCritic(env, config)
        policy.set_training_phase("composition")
        batch = collect_rollout(
            policy, env, update=0, rollout_steps=1, rollout_periods=slots,
            config=config, device="cpu", exploration_weight=0.,
            lagrange_multipliers=np.zeros(4), observation=observation,
            seed=0, episode_counter=0,
        )
        results[method] = {"collection_s": batch.collection_time_s,
                           "transitions": len(batch.records),
                           "model_steps": len(batch.composition_indices),
                           "completed_slots": batch.completed_periods}
    return results


if __name__ == "__main__":
    main()

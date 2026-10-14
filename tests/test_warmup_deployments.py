from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from agent_orch.capacity import CapacityPlanner
from agent_orch.deployment.warmup import (FixedWarmupDeployments, WarmupSpec,
    _capacities, capacity_probes, coverage_report, generate_warmup_deployments,
    physical_signature, scenario_digest, validate_placement, _coverage_profile)
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.performance.analytical import AnalyticalBackend
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.workload import SlotTrajectory


@pytest.fixture(scope="module")
def bundle(scenario):
    trajectory = SlotTrajectory.sample(scenario, 4, 101)
    return trajectory, generate_warmup_deployments(scenario, trajectory, 20261014)


def test_generation_deterministic_and_measures_actual_fleet(scenario, bundle):
    trajectory, bank = bundle
    again = generate_warmup_deployments(scenario, trajectory, 20261014)
    assert bank.digest() == again.digest()
    assert coverage_report(bank)["unique_physical_deployments"] == 4
    for slot, e in enumerate(bank.entries):
        current = trajectory.scenario_at(slot, scenario)
        assert e["available_gpu"] == sum(s.gpu_count for s in current.servers.values())
        assert e["gpu_used"] <= e["available_gpu"]
        assert e["active_models"] == ["large", "small"]
        assert CapacityPlanner(current).deployment_feasible(bank.deployment_at(slot))
        capacities, _ = _capacities(current)
        measured = {m: sum(capacities[current.candidates[c].config]
            for c, active in bank.deployment_at(slot).llm_active.items()
            if active and current.candidates[c].model == m) for m in e["active_models"]}
        assert e["model_capacities_rps"] == pytest.approx(measured, rel=1e-12)
    assert len({e["llm_instances"] for e in bank.entries}) > 1


def test_profiles_are_not_locked_to_model_cardinality():
    spec = WarmupSpec()
    profiles = [_coverage_profile(t, 3, tuple(range(2 + t % 3)), spec) for t in range(32)]
    for n in (2, 3, 4):
        group = [p for p in profiles if len(p["models"]) == n]
        assert {p["model_bias"] for p in group} == set(spec.model_biases)
        assert {p["gpu_fraction"] for p in group} == set(spec.gpu_fractions)
        assert {p["placement"] for p in group} == set(spec.placements)
        assert {p["tool_utilization"] for p in group} == set(spec.tool_utilization_targets)
        assert {p["extra_tool_probability"] for p in group} == set(spec.extra_tool_probabilities)


def test_replica_index_permutations_do_not_inflate_diversity(scenario):
    original = scenario.candidates["small-n0"]
    replica = replace(original, id="small-n0-r1", replica_index=1)
    current = replace(scenario, candidates={**scenario.candidates, replica.id: replica})
    d = CapacityPlanner(current).empty_deployment()
    d.llm_active[original.id] = 1
    other = d.copy()
    other.llm_active[original.id] = 0
    other.llm_active[replica.id] = 1
    assert physical_signature(current, d) == physical_signature(current, other)


def test_roundtrip_and_reject_tamper_or_changed_workload(scenario, bundle, tmp_path):
    trajectory, bank = bundle
    path = tmp_path / "bank.json"
    bank.save(path)
    loaded = FixedWarmupDeployments.load(path, scenario, trajectory)
    assert loaded.digest() == bank.digest()
    with pytest.raises(FileExistsError): bank.save(path)
    app = next(iter(scenario.applications.values()))
    changed = replace(scenario, applications={**scenario.applications,
        app.id: replace(app, quality={m: 0.0 for m in app.quality})})
    assert scenario_digest(changed) != scenario_digest(scenario)
    with pytest.raises(ValueError, match="different scenario"):
        loaded.validate(changed, trajectory)
    with pytest.raises(ValueError, match="different scenario or trajectory"):
        loaded.validate(scenario, SlotTrajectory.sample(scenario, 4, 102))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["deployment"]["llm_active"]["small-n0"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        FixedWarmupDeployments.load(path, scenario, trajectory)


def test_resource_masks_and_positive_tool_coverage(scenario, bundle):
    trajectory, bank = bundle
    current = trajectory.scenario_at(0, scenario)
    d = bank.deployment_at(0)
    h = next(iter(current.tools))
    for key in d.tool_replicas:
        if key[0] == h: d.tool_replicas[key] = 0
    with pytest.raises(ValueError, match="Missing positive-workload tool"):
        validate_placement(current, d)
    d = bank.deployment_at(0)
    d.tool_replicas[(h, "n0")] = current.tool_replica_limit(h, "n0") + 1
    with pytest.raises(ValueError, match="physical resources"):
        validate_placement(current, d)


def test_capacity_probes_match_main_router_and_backend(scenario, bundle):
    trajectory, bank = bundle
    current = trajectory.scenario_at(1, scenario)
    deployment = bank.deployment_at(1)
    capacities, calls = _capacities(current)
    probes = capacity_probes(current, deployment, capacities, calls)
    for probe in probes:
        shares = {(a.id, g, m): x for a in current.applications.values()
                  for g in a.ingress_rates for m, x in probe["model_shares"].items()}
        router = CapacityProportionalRouter(current)
        routing = router.route(deployment, shares)
        result = AnalyticalBackend(current).evaluate(deployment, routing)
        assert probe["max_network_utilization"] == pytest.approx(
            max(AnalyticalBackend(current).network.utilization(result.link_load_mbps).values()), rel=1e-10)
        assert probe["max_tool_utilization"] == pytest.approx(max(result.tool_utilization.values()), rel=1e-10)
        for c, perf in result.llm_instance_performance.items():
            m = current.candidates[c].model
            assert perf.utilization == pytest.approx(probe["llm_capacity_ratios"][m], rel=1e-10)


def test_slo_scores_do_not_filter_contexts(scenario, bundle):
    trajectory, bank = bundle
    changed = replace(scenario, applications={a.id: replace(a, slo=replace(a.slo,
        ttft_s=1e-9 if a.slo.ttft_s is not None else None,
        tbt_s=1e-9 if a.slo.tbt_s is not None else None,
        deadline_s=1e-9 if a.slo.deadline_s is not None else None))
        for a in scenario.applications.values()})
    tighter = generate_warmup_deployments(changed, trajectory, 20261014)
    assert [e["deployment"] for e in tighter.entries] == [e["deployment"] for e in bank.entries]


def test_fixed_contexts_replay_across_episodes_and_do_not_restrict_deployment(scenario, bundle):
    trajectory, bank = bundle
    env = SlotSequentialJointEnv(scenario, trajectory, mapping_samples=2, warmup_deployments=bank)
    env.training_phase = "composition"
    first = []
    for episode_seed in (1, 999):
        observation, info = env.reset(seed=episode_seed)
        assert info["warmup_deployment_digest"] == bank.digest()
        for slot in range(len(trajectory)):
            assert env.current_deployment == bank.deployment_at(slot)
            assert observation["action_type"] == env.COMPOSITION
            while True:
                observation, reward, done, _, info = env.step({"deploy": 0,
                    "model": np.ones(env.layout.model_action_size)})
                if info.get("period_complete"): break
            assert info["deployment_steps"] == 0
            if episode_seed == 1: first.append(info["utility"])
            else:
                assert info["utility"] == first[slot]
                assert reward == pytest.approx(0, abs=1e-12)
                assert info["reward_baseline"] == "previous_round"
        assert done
    for phase in ("deployment", "joint"):
        env.training_phase = phase
        observation, _ = env.reset(seed=0)
        assert env._stage_deployment is None
        assert sum(env.current_deployment.llm_active.values()) == 0
        assert observation["action_type"] == env.LLM_DEPLOYMENT


def test_reward_resume_rejects_different_contexts(scenario, bundle):
    trajectory, bank = bundle
    a = SlotSequentialJointEnv(scenario, trajectory, warmup_deployments=bank)
    b = SlotSequentialJointEnv(scenario, trajectory)
    with pytest.raises(ValueError, match="different warm-up"):
        b.restore_reward_state(a.reward_state())
    b = SlotSequentialJointEnv(scenario, trajectory, warmup_deployments=bank)
    b.restore_reward_state(copy.deepcopy(a.reward_state()))


def test_bad_generator_settings_fail_closed():
    with pytest.raises(ValueError): WarmupSpec(gpu_fractions=(1.1,)).validate()
    with pytest.raises(ValueError): WarmupSpec(placements=("optimal",)).validate()


def test_composition_training_does_not_update_deployment_branch(scenario, bundle):
    import torch
    from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
    trajectory, bank = bundle
    env = SlotSequentialJointEnv(scenario, trajectory, mapping_samples=2, warmup_deployments=bank)
    config = replace(PPOConfig(), hidden_size=16, hidden_layers=1, update_epochs=1,
        minibatch_size=32, training_phase="composition").for_constraint_count(len(env.constraint_names))
    torch.manual_seed(91)
    initial = StructuredActorCritic(env, config).state_dict()
    trained, _ = train_ppo(env, updates=1, rollout_steps=1, rollout_periods=len(trajectory),
        seed=91, config=config, phase_schedule=(("composition", 1),))
    final = trained.state_dict()
    for name in initial:
        if name.startswith(("deployment_encoder.", "deploy_head.", "deployment_value_head.")):
            assert torch.equal(initial[name], final[name]), name
    assert any(not torch.equal(initial[k], final[k]) for k in initial
               if k.startswith(("composition_encoder.", "model_head.", "routing_value_head.")))

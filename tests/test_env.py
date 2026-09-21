import math

import numpy as np
import pytest

from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv


def _blank_routing_action(env):
    return {
        "deploy": 0,
        "model": np.ones(env.layout.model_action_size, dtype=np.float32),
    }


def _valid_action(env, observation):
    action = _blank_routing_action(env)
    if observation["action_type"] == env.DEPLOYMENT:
        valid = np.flatnonzero(observation["deploy_mask"])
        assert len(valid) > 0
        action["deploy"] = int(valid[0])
    return action


def _complete_deployment(env, observation):
    infos = []
    while observation["action_type"] < env.COMPOSITION:
        previous_slot = env.simulator.slot
        observation, reward, terminated, truncated, info = env.step(
            _valid_action(env, observation)
        )
        assert reward == 0.0
        assert env.simulator.slot == previous_slot
        assert not terminated
        assert not truncated
        infos.append(info)
    return observation, infos


def test_environment_completes_deployment_and_routing_phases(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, potential_shaping=True, seed=3)
    observation, _ = env.reset(seed=3)
    assert observation["action_type"] == env.DEPLOYMENT
    observation, deployment_infos = _complete_deployment(env, observation)
    assert deployment_infos
    assert all(info["discount"] == env.deployment_gamma for info in deployment_infos)
    assert observation["action_type"] == env.COMPOSITION
    observation, reward, _, _, info = env.step(_blank_routing_action(env))
    assert math.isfinite(reward)
    assert info["phase"] == "composition"
    assert info["metrics"].total_arrival_rps > 0.0


def test_structured_policy_produces_finite_actions(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=5)
    observation, _ = env.reset(seed=5)
    policy = StructuredActorCritic(env, PPOConfig(hidden_size=32))
    action, log_prob, value = policy.act(observation, deterministic=False)
    assert env.action_space.contains(action)
    assert math.isfinite(log_prob)
    assert math.isfinite(value)


def test_ppo_smoke_update(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=9)
    config = PPOConfig(update_epochs=1, minibatch_size=32, hidden_size=32)
    _, history = train_ppo(
        env,
        updates=1,
        rollout_steps=32,
        seed=9,
        config=config,
    )
    assert len(history) == 1
    assert math.isfinite(history[0]["mean_loss"])
    assert history[0]["routing_steps"] >= 1
    assert history[0]["mean_rnd_loss"] >= 0.0


def test_ppo_can_budget_rollouts_by_complete_orchestration_period(scenario):
    env = AgentOrchestrationEnv(
        scenario, max_slots=4, seed=29, mapping_samples=8
    )
    config = PPOConfig(update_epochs=1, minibatch_size=256, hidden_size=32)
    _, history = train_ppo(
        env,
        updates=1,
        rollout_steps=1,
        rollout_periods=2,
        seed=29,
        config=config,
    )
    assert history[0]["routing_steps"] == 2
    assert history[0]["transition_steps"] > history[0]["routing_steps"]
    assert history[0]["collection_time_s"] >= 0.0
    assert history[0]["optimization_time_s"] >= 0.0


def test_ppo_reports_rollout_optimization_and_update_progress(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=19)
    config = PPOConfig(update_epochs=1, minibatch_size=32, hidden_size=32)
    phases = []
    rollout_steps = []
    optimization_steps = []
    updates = []
    step_count = 32
    train_ppo(
        env,
        updates=1,
        rollout_steps=step_count,
        seed=19,
        config=config,
        on_phase=lambda update, phase: phases.append((update, phase)),
        on_rollout_step=lambda update, step: rollout_steps.append((update, step)),
        on_optimization_step=lambda update, step, total: optimization_steps.append(
            (update, step, total)
        ),
        on_update=updates.append,
    )
    assert phases == [(0, "collecting"), (0, "optimizing")]
    assert rollout_steps[-1][1] >= step_count
    assert len(updates) == 1
    expected_steps = math.ceil(updates[0]["transition_steps"] / config.minibatch_size)
    assert optimization_steps == [
        (0, step, expected_steps) for step in range(1, expected_steps + 1)
    ]


def test_ppo_resumes_from_update_checkpoint(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=23)
    config = PPOConfig(update_epochs=1, minibatch_size=32, hidden_size=32)
    checkpoint = {}

    class Interrupted(RuntimeError):
        pass

    def stop_after_first_update(state):
        checkpoint.update(state)
        raise Interrupted

    with pytest.raises(Interrupted):
        train_ppo(
            env,
            updates=2,
            rollout_steps=32,
            seed=23,
            config=config,
            on_checkpoint=stop_after_first_update,
        )
    assert checkpoint["next_update"] == 1
    assert len(checkpoint["history"]) == 1

    resumed_env = AgentOrchestrationEnv(scenario, max_slots=2, seed=23)
    saved = []
    _, history = train_ppo(
        resumed_env,
        updates=2,
        rollout_steps=32,
        seed=23,
        config=config,
        resume_state=checkpoint,
        on_checkpoint=saved.append,
    )
    assert len(history) == 2
    assert saved[-1]["next_update"] == 2


def test_ppo_icm_smoke_update(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=10)
    config = PPOConfig(
        update_epochs=1,
        minibatch_size=32,
        hidden_size=32,
        exploration_mode="icm",
    )
    _, history = train_ppo(
        env,
        updates=1,
        rollout_steps=32,
        seed=10,
        config=config,
    )
    assert math.isfinite(history[0]["mean_icm_loss"])
    assert history[0]["mean_intrinsic_reward"] >= 0.0


def test_routing_only_environment_never_enters_deployment(scenario):
    env = RoutingOnlyEnv(scenario, max_slots=2, seed=12)
    observation, _ = env.reset(seed=12)
    deployment = env.current_deployment.copy()
    assert observation["action_type"] == env.COMPOSITION
    observation, _, _, _, _ = env.step(_blank_routing_action(env))
    assert observation["action_type"] == env.COMPOSITION
    assert env.current_deployment == deployment


def test_routing_only_environment_rotates_feasible_fixed_deployments(scenario):
    env = RoutingOnlyEnv(scenario, max_slots=1, seed=0)
    signatures = set()
    for seed in range(8):
        _, info = env.reset(seed=seed)
        assert env.planner.deployment_feasible(env.current_deployment)
        signatures.add(env._deployment_signature(env.current_deployment))
        assert info["fixed_deployment_count"] >= 2
    assert len(signatures) >= 2


def test_deployment_only_environment_evaluates_internal_interval(scenario):
    env = DeploymentOnlyEnv(scenario, max_slots=2, seed=14)
    observation, _ = env.reset(seed=14)
    terminated = False
    last_info = {}
    reward = 0.0
    while not terminated:
        observation, reward, terminated, _, last_info = env.step(
            _valid_action(env, observation)
        )
    assert terminated
    assert last_info["period_complete"]
    assert env.simulator.slot == 2
    assert math.isfinite(reward)


def test_normalized_reward_has_separate_constraint_cost(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=15)
    observation, _ = env.reset(seed=15)
    observation, deployment_infos = _complete_deployment(env, observation)
    assert all(info["constraint_cost"] == 0.0 for info in deployment_infos)
    _, reward, _, _, routing_info = env.step(_blank_routing_action(env))
    components = routing_info["reward_components"]
    expected = (
        scenario.reward.goodput_weight * components["goodput_normalized"]
        + scenario.reward.quality_weight * components["quality_normalized"]
        - scenario.reward.cost_weight * components["cost_normalized"]
        - scenario.reward.latency_weight * components["latency_normalized"]
    )
    assert reward == pytest.approx(expected)
    assert -1.0 <= reward <= 1.0
    assert set(components) == {
        "utility",
        "cost_normalized",
        "latency_normalized",
        "goodput_normalized",
        "quality_normalized",
    }
    assert routing_info["constraint_cost"] >= 0.0


def test_normalized_reward_uses_fixed_metric_scales(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=22)
    arrival_rates = {
        (app.id, ingress): rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    app_latency = {
        app.id: 0.5 * env._app_latency_reference(app.id)
        for app in scenario.applications.values()
    }
    utility, components = env._normalized_utility(
        cost=0.5 * (env.cost_min + env.cost_max),
        mean_latency=0.5 * env.latency_reference,
        attainment=0.8,
        quality=0.6,
        app_latency=app_latency,
        arrival_rates=arrival_rates,
    )
    assert components["cost_normalized"] == pytest.approx(0.5)
    assert components["latency_normalized"] == pytest.approx(0.5)
    assert components["goodput_normalized"] == pytest.approx(0.8)
    assert components["quality_normalized"] == pytest.approx(0.6)
    expected = 0.25 * (0.8 + 0.6 - 0.5 - 0.5)
    assert utility == pytest.approx(expected)


def test_feature_vector_excludes_horizon_progress_and_duplicate_phase(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=31)
    observation, _ = env.reset(seed=31)
    features = observation["features"].copy()
    env._period_index = 1_000
    env.phase = env.COMPOSITION
    changed = env._observation()
    np.testing.assert_allclose(changed["features"], features)
    assert changed["action_type"] == env.COMPOSITION


def test_sequential_deployment_builds_a_feasible_capacity_plan(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=16)
    observation, _ = env.reset(seed=16)
    observation, infos = _complete_deployment(env, observation)
    assert infos[-1]["deployment_complete"]
    assert env.planner.deployment_feasible(env.current_deployment)
    assert sum(env.current_deployment.llm_active.values()) > 0
    assert sum(env.current_deployment.tool_replicas.values()) > 0
    assert observation["action_type"] == env.ROUTING


def test_sequential_deployment_rejects_a_masked_target(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=17)
    observation, _ = env.reset(seed=17)
    invalid = np.flatnonzero(observation["deploy_mask"] == 0)
    assert len(invalid) > 0
    action = _blank_routing_action(env)
    action["deploy"] = int(invalid[0])
    with pytest.raises(ValueError, match="masked or invalid"):
        env.step(action)


def test_deployment_and_first_routing_share_the_same_physical_slot(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=18)
    observation, _ = env.reset(seed=18)
    assert env.simulator.slot == 0
    observation, deployment_infos = _complete_deployment(env, observation)
    assert all(info["discount"] == env.deployment_gamma for info in deployment_infos)
    assert env.simulator.slot == 0
    _, _, _, _, routing_info = env.step(_blank_routing_action(env))
    assert routing_info["discount"] == env.gamma
    assert env.simulator.slot == 1


def test_each_deployment_item_is_processed_exactly_once(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=1, seed=41)
    observation, _ = env.reset(seed=41)
    visited = []
    while observation["action_type"] < env.COMPOSITION:
        visited.append(env._current_demand)
        action = _blank_routing_action(env)
        action["deploy"] = 0
        observation, _, _, _, _ = env.step(action)
    expected = len(env.layout.candidates) + len(env.layout.tools) * len(
        env.layout.servers
    )
    assert len(visited) == expected
    assert len(set(visited)) == expected


def test_actor_branches_can_be_frozen_independently(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=1, seed=42)
    policy = StructuredActorCritic(env, PPOConfig(hidden_size=32))
    policy.set_training_phase("deployment")
    assert all(
        parameter.requires_grad
        for parameter in policy.deployment_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in policy.composition_encoder.parameters()
    )
    policy.set_training_phase("composition")
    assert all(
        not parameter.requires_grad
        for parameter in policy.deployment_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in policy.composition_encoder.parameters()
    )

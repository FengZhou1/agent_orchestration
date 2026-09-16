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
    assert optimization_steps == [(0, 1, 2), (0, 2, 2)]
    assert len(updates) == 1


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
    assert observation["action_type"] == env.COMPOSITION
    observation, _, _, _, _ = env.step(_blank_routing_action(env))
    assert observation["action_type"] == env.COMPOSITION


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
    assert -1.0 <= reward <= 1.0
    assert set(components) == {
        "utility",
        "cost_normalized",
        "latency_normalized",
        "goodput_normalized",
        "quality_normalized",
    }
    assert routing_info["constraint_cost"] >= 0.0


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

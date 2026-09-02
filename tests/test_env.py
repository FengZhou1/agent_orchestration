import math
from dataclasses import replace

import numpy as np

from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv


def _blank_routing_action(env):
    return {
        "deploy": env.encode_deployment(env.current_deployment),
        "model": np.ones(env.layout.model_action_size, dtype=np.float32),
    }


def test_environment_completes_deployment_and_routing_phases(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, potential_shaping=True, seed=3)
    observation, _ = env.reset(seed=3)
    assert observation["action_type"] == env.DEPLOYMENT
    observation, reward, terminated, truncated, info = env.step(
        _blank_routing_action(env)
    )
    assert math.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["discount"] == 1.0
    assert observation["action_type"] == env.ROUTING
    observation, reward, _, _, info = env.step(_blank_routing_action(env))
    assert math.isfinite(reward)
    assert info["phase"] == "routing"
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
        rollout_steps=scenario.simulation.deployment_period_slots + 2,
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
    step_count = scenario.simulation.deployment_period_slots + 2
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
    assert rollout_steps[-1] == (0, step_count)
    assert optimization_steps == [(0, 1, 1)]
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
        rollout_steps=scenario.simulation.deployment_period_slots + 2,
        seed=10,
        config=config,
    )
    assert math.isfinite(history[0]["mean_icm_loss"])
    assert history[0]["mean_intrinsic_reward"] >= 0.0


def test_routing_only_environment_never_enters_deployment(scenario):
    env = RoutingOnlyEnv(scenario, max_slots=2, seed=12)
    observation, _ = env.reset(seed=12)
    assert observation["action_type"] == env.ROUTING
    observation, _, _, _, _ = env.step(_blank_routing_action(env))
    assert observation["action_type"] == env.ROUTING


def test_deployment_only_environment_evaluates_internal_interval(scenario):
    env = DeploymentOnlyEnv(scenario, max_slots=2, seed=14)
    observation, _ = env.reset(seed=14)
    observation, reward, terminated, _, last_info = env.step(
        _blank_routing_action(env)
    )
    assert terminated
    assert last_info["evaluated_slots"] == 2
    assert len(last_info["interval_metrics"]) == 2
    assert [metrics.slot for metrics in last_info["interval_metrics"]] == [0, 1]
    assert math.isfinite(reward)


def test_normalized_reward_has_separate_constraint_cost(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=15)
    observation, _ = env.reset(seed=15)
    observation, _, _, _, deployment_info = env.step(_blank_routing_action(env))
    assert deployment_info["constraint_cost"] == 0.0
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


def test_missing_services_are_applied_and_exposed_as_constraint_cost(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=16)
    env.reset(seed=16)
    missing = {
        "deploy": np.zeros(len(env.layout.deployment_groups), dtype=np.int64),
        "model": np.ones(env.layout.model_action_size, dtype=np.float32),
    }
    observation, reward, _, _, info = env.step(missing)
    assert reward == 0.0
    assert not info["invalid_action"]
    assert info["constraint_cost"] > 0.0
    assert sum(env.current_deployment.llm_active.values()) == 0
    assert observation["features"][0] == 0.0
    _, _, _, _, routing_info = env.step(missing)
    assert routing_info["constraint_cost"] > 0.0
    assert any(
        "unserved" in label
        for label in routing_info["metrics"].diagnostics["violation_labels"]
    )


def test_resource_infeasible_deployment_is_rejected_without_changing_state(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=17)
    env.reset(seed=17)
    before = env.current_deployment.copy()
    env.scenario.servers["n0"] = replace(
        env.scenario.servers["n0"], cpu_cores=1, memory_gb=1.0
    )
    selected = np.asarray(
        [width - 1 for width in env.layout.deployment_widths], dtype=np.int64
    )
    invalid = {
        "deploy": selected,
        "model": np.ones(env.layout.model_action_size, dtype=np.float32),
    }
    _, reward, _, _, info = env.step(invalid)
    assert reward == 0.0
    assert info["invalid_action"]
    assert info["constraint_cost"] > 0.0
    assert env.current_deployment == before


def test_deployment_and_first_routing_share_the_same_physical_slot(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=18)
    observation, _ = env.reset(seed=18)
    assert env.simulator.slot == 0
    observation, _, _, _, deployment_info = env.step(_blank_routing_action(env))
    assert deployment_info["discount"] == 1.0
    assert env.simulator.slot == 0
    _, _, _, _, routing_info = env.step(_blank_routing_action(env))
    assert routing_info["discount"] == env.gamma
    assert env.simulator.slot == 1

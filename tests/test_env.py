import math

import numpy as np

from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv


def _blank_routing_action(env):
    return {
        "deploy": 0,
        "model": np.ones(env.layout.model_action_size, dtype=np.float32),
        "llm": np.ones(env.layout.llm_action_size, dtype=np.float32),
        "tool": np.ones(env.layout.tool_action_size, dtype=np.float32),
    }


def test_environment_completes_deployment_and_routing_phases(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, potential_shaping=True, seed=3)
    observation, _ = env.reset(seed=3)
    while observation["action_type"] == env.DEPLOYMENT:
        selected = int(observation["deploy_mask"][1] == 1)
        observation, reward, terminated, truncated, info = env.step(
            {**_blank_routing_action(env), "deploy": selected}
        )
        assert math.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["discount"] == 1.0
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
        rollout_steps=len(env.layout.deployment_items) + 2,
        seed=9,
        config=config,
    )
    assert len(history) == 1
    assert math.isfinite(history[0]["mean_loss"])
    assert history[0]["routing_steps"] >= 1


def test_ppo_icm_smoke_update(scenario):
    env = AgentOrchestrationEnv(scenario, max_slots=2, seed=10)
    config = PPOConfig(update_epochs=1, minibatch_size=32, hidden_size=32)
    _, history = train_ppo(
        env,
        updates=1,
        rollout_steps=len(env.layout.deployment_items) + 2,
        seed=10,
        config=config,
        use_icm=True,
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
    last_info = {}
    while not last_info.get("evaluated_slots"):
        selected = int(observation["deploy_mask"][1] == 1)
        observation, reward, terminated, _, last_info = env.step(
            {**_blank_routing_action(env), "deploy": selected}
        )
    assert terminated
    assert last_info["evaluated_slots"] == 2
    assert len(last_info["interval_metrics"]) == 2
    assert [metrics.slot for metrics in last_info["interval_metrics"]] == [0, 1]
    assert math.isfinite(reward)

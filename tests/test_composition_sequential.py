"""The sequential composition environment: one group per step, one slot per pass."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.agents.distributions import (
    _evaluate_grouped_dirichlet,
    _sample_grouped_dirichlet,
)
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs import CompositionSequentialEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader


@pytest.fixture(scope="module")
def scenario():
    return ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")


@pytest.fixture(scope="module")
def library(scenario):
    return DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))


def _env(scenario, library, **kwargs):
    return CompositionSequentialEnv(
        scenario,
        max_slots=4,
        seed=0,
        mapping_samples=16,
        objective=ObjectiveSpec.slo_constrained(0.9),
        deployment_library=library.subset(list(range(8))),
        **kwargs,
    )


def test_one_group_per_step_and_one_slot_per_pass(scenario, library):
    """A pass over the groups yields exactly one slot, with no external reward
    until the units is complete."""

    env = _env(scenario, library)
    observation, info = env.reset(seed=0)
    groups = env.n_groups
    assert groups > 1
    assert observation["model_group"] == 0

    rewards: list[float] = []
    slots = 0
    for step in range(groups):
        action = {
            "deploy": 0,
            "model": np.zeros(env.layout.model_action_size, dtype=np.float32),
        }
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info.get("period_complete"):
            slots += 1
            assert step == groups - 1, "only the last sub-step may complete a slot"
    assert slots == 1, "one pass must run exactly one slot"
    assert rewards[:-1] == [0.0] * (groups - 1), "intermediate steps carry no reward"
    assert env._period_index == 1


def test_intermediate_steps_report_no_constraint_and_no_utility(scenario, library):
    env = _env(scenario, library)
    env.reset(seed=0)
    action = {"deploy": 0, "model": np.zeros(env.layout.model_action_size, dtype=np.float32)}
    _, reward, _, _, info = env.step(action)
    assert reward == 0.0
    assert info["period_complete"] is False
    assert info["utility"] == 0.0 if "utility" in info else True
    assert info["constraint_steps"] == 0
    assert info["discount"] == 1.0
    assert set(info["reward_components"]) >= {"utility", "delta_round"}


def test_hard_choice_mode_puts_the_group_on_one_model(scenario, library):
    """``action_mode='model'`` is the ablation: the best active model takes all."""

    env = _env(scenario, library, action_mode="model")
    observation, _ = env.reset(seed=0)
    groups = env.n_groups
    models = len(env.layout.models)
    for _ in range(groups):
        row = np.zeros(env.n_groups * models, dtype=np.float32)
        row[0] = 1.0  # only the first model is scored; the env takes the best active
        env.step({"deploy": 0, "model": row})
    for group_row in env._working_share:
        active = [value for value in group_row if value > 0]
        assert active and max(active) == pytest.approx(1.0)


def test_reward_is_the_same_slot_in_the_previous_round(scenario, library):
    """The baseline is the previous *round*, not the previous slot.

    The reward is how much better this slot did than the same slot in the previous
    training round (the reference chapter's baseline).  A cell the previous round
    never visited contributes zero rather than a spurious improvement.
    """

    env = _env(scenario, library)
    env.reset(seed=0)
    groups = env.n_groups
    rewards: list[float] = []
    components: list[dict] = []
    for slot in range(3):
        for _ in range(groups):
            action = {
                "deploy": 0,
                "model": np.zeros(env.layout.model_action_size, dtype=np.float32),
            }
            _, reward, _, _, info = env.step(action)
        rewards.append(reward)
        components.append(dict(info["reward_components"]))
        observation, _ = env.reset(seed=0)
    # Every slot of the first round has no baseline yet, so every difference is zero.
    assert rewards == pytest.approx([0.0, 0.0, 0.0])
    assert all(set(entry) >= {"delta_round", "utility"} for entry in components)

    # Second round: the same (context, slot) now has a baseline, so a slot whose
    # utility differs from the previous round's shows a non-zero difference, and the
    # difference is exactly utility_now - utility_then.
    first_round = [entry["utility"] for entry in components]
    deltas = []
    for slot in range(3):
        for _ in range(groups):
            action = {
                "deploy": 0,
                "model": np.zeros(env.layout.model_action_size, dtype=np.float32),
            }
            _, reward, _, _, info = env.step(action)
        deltas.append(reward)
        assert reward == pytest.approx(
            float(info["reward_components"]["delta_round"])
        )
        observation, _ = env.reset(seed=0)
    assert deltas == pytest.approx(
        [now - then for now, then in zip([entry["utility"] for entry in components], first_round)]
    )


def test_log_prob_covers_only_the_acted_group(scenario, library):
    """Otherwise the PPO ratio would include rows the environment discarded."""

    generator = torch.Generator().manual_seed(0)
    groups, width = 4, 3
    raw = torch.randn(groups, width, generator=generator)
    mask = torch.ones(groups, width, dtype=torch.bool)
    sample = torch.zeros(groups, width)
    sample[2, 0] = 1.0

    only_two = _evaluate_grouped_dirichlet(raw, mask, sample, groups, width, only_group=2)
    all_groups = _evaluate_grouped_dirichlet(raw, mask, sample, groups, width)
    assert only_two[0].item() != pytest.approx(all_groups[0].item())

    # Sampling honours the same restriction: only the acted row is populated.
    action, log_prob, _ = _sample_grouped_dirichlet(
        raw, mask, groups, width, deterministic=True, only_group=1
    )
    action = action.reshape(groups, width)
    assert float(action[1].sum()) == pytest.approx(1.0)
    for group in (0, 2, 3):
        assert float(action[group].sum()) == pytest.approx(0.0)


def test_policy_round_trip_uses_the_acted_group(scenario, library):
    """act() and evaluate_actions() must agree on the sequential observation."""

    env = _env(scenario, library)
    policy = StructuredActorCritic(env, PPOConfig())
    observation, _ = env.reset(seed=0)
    action, log_prob, _ = policy.act(observation, deterministic=True, device="cpu")
    row = np.asarray(action["model"], dtype=np.float32).reshape(
        env.n_groups, len(env.layout.models)
    )
    acted = observation["model_group"]
    assert float(row[acted].sum()) == pytest.approx(1.0, abs=1e-5)
    for group in range(env.n_groups):
        if group != acted:
            assert float(row[group].sum()) == pytest.approx(0.0, abs=1e-6)

    batched = {
        key: torch.as_tensor(np.asarray(value))[None, ...]
        if np.asarray(value).ndim
        else torch.as_tensor([value])
        for key, value in observation.items()
    }
    batched["features"] = batched["features"].float()
    replayed = {
        "deploy": torch.zeros(1, dtype=torch.long),
        "model": torch.as_tensor(np.asarray(action["model"]))[None, ...],
    }
    new_log_prob, _, _ = policy.evaluate_actions(batched, replayed)
    assert float(new_log_prob[0]) == pytest.approx(log_prob, rel=1e-4, abs=1e-4)

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet

from agent_orch.envs import AgentOrchestrationEnv
from .device import resolve_device
from .icm import ICMModule, structured_action_vector
from .rnd import PhaseRunningMoments, RNDModule


PhaseCallback = Callable[[int, str], None]
RolloutProgressCallback = Callable[[int, int], None]
OptimizationProgressCallback = Callable[[int, int, int], None]
UpdateCallback = Callable[[dict[str, float]], None]


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3.0e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    constrained: bool = True
    constraint_limit: float = 0.0
    lagrangian_learning_rate: float = 0.05
    initial_lagrange_multiplier: float = 0.0
    max_lagrange_multiplier: float = 50.0
    exploration_mode: Literal["rnd", "none", "icm"] = "rnd"
    rnd_feature_size: int = 64
    rnd_learning_rate: float = 1.0e-4
    rnd_initial_weight: float = 0.01
    rnd_final_weight: float = 0.0
    rnd_reward_clip: float = 5.0
    rnd_loss_coefficient: float = 1.0
    icm_scale: float = 0.01


class StructuredActorCritic(nn.Module):
    def __init__(self, env: AgentOrchestrationEnv, config: PPOConfig = PPOConfig()):
        super().__init__()
        self.layout = env.layout
        feature_size = env.observation_space["features"].shape[0]
        hidden = config.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(feature_size + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.deploy_head = nn.Linear(hidden, self.layout.deployment_action_size)
        self.model_head = nn.Linear(hidden, self.layout.model_action_size)
        self.value_head = nn.Linear(hidden, 1)

    def _encode(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        features = observation["features"]
        if features.ndim == 1:
            features = features.unsqueeze(0)
        action_type = observation["action_type"].long().view(-1)
        phase = torch.nn.functional.one_hot(action_type, num_classes=2).float()
        return self.encoder(torch.cat([features, phase], dim=-1))

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(self._encode(observation)).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, Any],
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> tuple[dict[str, Any], float, float]:
        obs = _observation_to_tensors(observation, device, batched=False)
        hidden = self._encode(obs)
        value = self.value_head(hidden).squeeze(-1)
        phase = int(observation["action_type"])
        action = {
            "deploy": np.zeros(
                len(self.layout.deployment_groups), dtype=np.int64
            ),
            "model": np.zeros(self.layout.model_action_size, dtype=np.float32),
        }
        if phase == AgentOrchestrationEnv.DEPLOYMENT:
            selected, log_prob, _ = _sample_variable_categoricals(
                self.deploy_head(hidden).squeeze(0),
                torch.as_tensor(
                    observation["deploy_mask"], dtype=torch.bool, device=device
                ),
                self.layout.deployment_widths,
                deterministic,
            )
            action["deploy"] = selected.cpu().numpy().astype(np.int64)
        else:
            model, model_logp, _ = _sample_grouped_dirichlet(
                self.model_head(hidden).squeeze(0),
                torch.as_tensor(observation["model_mask"], device=device),
                len(self.layout.model_groups),
                len(self.layout.models),
                deterministic,
            )
            action["model"] = model.cpu().numpy().astype(np.float32)
            log_prob = model_logp
        return action, float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        observation: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._encode(observation)
        values = self.value_head(hidden).squeeze(-1)
        phases = observation["action_type"].long().view(-1)
        log_probs = torch.zeros_like(values)
        entropies = torch.zeros_like(values)
        for index in range(hidden.shape[0]):
            if int(phases[index].item()) == AgentOrchestrationEnv.DEPLOYMENT:
                deploy_logp, deploy_entropy = _evaluate_variable_categoricals(
                    self.deploy_head(hidden[index]),
                    observation["deploy_mask"][index],
                    actions["deploy"][index],
                    self.layout.deployment_widths,
                )
                log_probs[index] = deploy_logp
                entropies[index] = deploy_entropy
                continue
            model_logp, model_entropy = _evaluate_grouped_dirichlet(
                self.model_head(hidden[index]),
                observation["model_mask"][index],
                actions["model"][index],
                len(self.layout.model_groups),
                len(self.layout.models),
            )
            log_probs[index] = model_logp
            entropies[index] = model_entropy
        return log_probs, entropies, values


def train_ppo(
    env: AgentOrchestrationEnv,
    updates: int,
    rollout_steps: int,
    seed: int = 0,
    config: PPOConfig = PPOConfig(),
    device: str = "cpu",
    on_phase: PhaseCallback | None = None,
    on_rollout_step: RolloutProgressCallback | None = None,
    on_optimization_step: OptimizationProgressCallback | None = None,
    on_update: UpdateCallback | None = None,
) -> tuple[StructuredActorCritic, list[dict[str, float]]]:
    device = resolve_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda:"):
        torch.cuda.manual_seed_all(seed)
    env.gamma = config.gamma
    policy = StructuredActorCritic(env, config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    action_vector_size = (
        2
        + env.layout.deployment_action_size
        + env.layout.model_action_size
    )
    icm = (
        ICMModule(env.observation_space["features"].shape[0], action_vector_size).to(device)
        if config.exploration_mode == "icm"
        else None
    )
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=config.learning_rate) if icm else None
    rnd = (
        RNDModule(
            env.observation_space["features"].shape[0] + 2,
            config.hidden_size,
            config.rnd_feature_size,
        ).to(device)
        if config.exploration_mode == "rnd"
        else None
    )
    rnd_optimizer = (
        torch.optim.Adam(rnd.predictor.parameters(), lr=config.rnd_learning_rate)
        if rnd is not None
        else None
    )
    rnd_moments = PhaseRunningMoments(2)
    observation, _ = env.reset(seed=seed)
    history: list[dict[str, float]] = []
    lagrange_multiplier = config.initial_lagrange_multiplier
    episode_counter = 0
    optimizer_steps_per_update = config.update_epochs * math.ceil(
        rollout_steps / config.minibatch_size
    )

    for update in range(updates):
        if on_phase is not None:
            on_phase(update, "collecting")
        observations: list[dict[str, Any]] = []
        next_observations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        constraint_costs: list[float] = []
        discounts: list[float] = []
        terminals: list[float] = []

        for rollout_step in range(1, rollout_steps + 1):
            action, log_prob, value = policy.act(observation, device=device)
            next_observation, reward, terminated, truncated, info = env.step(action)
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            constraint_steps = max(1.0, float(info.get("constraint_steps", 1.0)))
            constraint_costs.append(
                float(info.get("constraint_cost", 0.0)) / constraint_steps
            )
            discounts.append(float(info.get("discount", config.gamma)))
            terminals.append(float(terminated or truncated))
            observation = next_observation
            if terminated or truncated:
                episode_counter += 1
                observation, _ = env.reset(seed=seed + episode_counter)
            if on_rollout_step is not None:
                on_rollout_step(update, rollout_step)

        utility_rewards = list(rewards)
        phase_array = np.asarray(
            [int(obs["action_type"]) for obs in observations], dtype=np.int64
        )
        intrinsic_raw = np.zeros(len(observations), dtype=np.float32)
        intrinsic_normalized = np.zeros(len(observations), dtype=np.float32)
        current_rnd_states = None
        next_rnd_states = None
        feature_tensor = None
        next_feature_tensor = None
        action_vector_tensor = None

        if rnd is not None:
            current_rnd_states = _rnd_state_inputs(observations, device)
            next_rnd_states = _rnd_state_inputs(next_observations, device)
            intrinsic_raw = rnd.intrinsic_reward(next_rnd_states).cpu().numpy()
            intrinsic_normalized = rnd_moments.normalize(
                intrinsic_raw,
                phase_array,
                config.rnd_reward_clip,
            )
            rnd_moments.update(intrinsic_raw, phase_array)
        if icm is not None:
            feature_tensor = torch.as_tensor(
                np.stack([obs["features"] for obs in observations]),
                dtype=torch.float32,
                device=device,
            )
            next_feature_tensor = torch.as_tensor(
                np.stack([obs["features"] for obs in next_observations]),
                dtype=torch.float32,
                device=device,
            )
            action_vectors = np.stack(
                [
                    structured_action_vector(
                        action,
                        int(obs["action_type"]),
                        env.layout.deployment_widths,
                    )
                    for obs, action in zip(observations, actions)
                ]
            )
            action_vector_tensor = torch.as_tensor(action_vectors, dtype=torch.float32, device=device)
            intrinsic = icm.intrinsic_reward(
                feature_tensor, action_vector_tensor, next_feature_tensor
            )
            intrinsic_raw = intrinsic.cpu().numpy()
            intrinsic_normalized = intrinsic_raw.copy()

        lagrangian_rewards = _lagrangian_rewards(
            utility_rewards,
            constraint_costs,
            lagrange_multiplier,
            config.constraint_limit,
            config.constrained,
        )
        exploration_weight = _exploration_weight(config, update, updates)
        if config.exploration_mode == "icm":
            exploration_weight = config.icm_scale
        rewards = _combine_training_rewards(
            lagrangian_rewards, intrinsic_normalized, exploration_weight
        )
        with torch.no_grad():
            bootstrap = float(
                policy.value(_observation_to_tensors(observation, device, False)).item()
            )
        advantages, returns = _gae(
            rewards,
            values,
            discounts,
            terminals,
            bootstrap,
            config.gae_lambda,
        )
        advantages = advantages.to(device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        observations_tensor = _stack_observations(observations, device)
        actions_tensor = _stack_actions(actions, device)
        indices = np.arange(rollout_steps)
        losses = []
        icm_losses = []
        rnd_losses = []
        optimization_step = 0
        if on_phase is not None:
            on_phase(update, "optimizing")
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, config.minibatch_size):
                batch = indices[start : start + config.minibatch_size]
                batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
                obs_batch = {key: value[batch_t] for key, value in observations_tensor.items()}
                action_batch = {key: value[batch_t] for key, value in actions_tensor.items()}
                new_logp, entropy, predicted_values = policy.evaluate_actions(
                    obs_batch, action_batch
                )
                ratio = torch.exp(new_logp - old_log_probs[batch_t])
                advantage_batch = advantages[batch_t]
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                )
                actor_loss = -torch.min(ratio * advantage_batch, clipped * advantage_batch).mean()
                value_loss = torch.nn.functional.mse_loss(
                    predicted_values, returns_tensor[batch_t]
                )
                loss = (
                    actor_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy.mean()
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.item()))
                if icm is not None and icm_optimizer is not None:
                    assert feature_tensor is not None
                    assert next_feature_tensor is not None
                    assert action_vector_tensor is not None
                    feature_batch = feature_tensor[batch_t]
                    next_feature_batch = next_feature_tensor[batch_t]
                    action_vector_batch = action_vector_tensor[batch_t]
                    icm_loss = icm.loss(
                        feature_batch, action_vector_batch, next_feature_batch
                    )
                    icm_optimizer.zero_grad()
                    icm_loss.backward()
                    nn.utils.clip_grad_norm_(icm.parameters(), config.max_grad_norm)
                    icm_optimizer.step()
                    icm_losses.append(float(icm_loss.item()))
                if rnd is not None and rnd_optimizer is not None:
                    assert current_rnd_states is not None
                    rnd_loss = (
                        config.rnd_loss_coefficient
                        * rnd.loss(current_rnd_states[batch_t])
                    )
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    nn.utils.clip_grad_norm_(
                        rnd.predictor.parameters(), config.max_grad_norm
                    )
                    rnd_optimizer.step()
                    rnd_losses.append(float(rnd_loss.item()))
                optimization_step += 1
                if on_optimization_step is not None:
                    on_optimization_step(
                        update, optimization_step, optimizer_steps_per_update
                    )

        next_lagrange_multiplier = lagrange_multiplier
        if config.constrained:
            next_lagrange_multiplier = _update_lagrange_multiplier(
                lagrange_multiplier,
                float(np.mean(constraint_costs)),
                config.constraint_limit,
                config.lagrangian_learning_rate,
                config.max_lagrange_multiplier,
            )
        record = {
            "update": float(update),
            "mean_reward": float(np.mean(rewards)),
            "mean_utility": float(np.mean(utility_rewards)),
            "mean_lagrangian_reward": float(np.mean(lagrangian_rewards)),
            "mean_constraint_cost": float(np.mean(constraint_costs)),
            "lagrange_multiplier": float(lagrange_multiplier),
            "next_lagrange_multiplier": float(next_lagrange_multiplier),
            "mean_loss": float(np.mean(losses)),
            "deployment_steps": float(
                sum(
                    phase == AgentOrchestrationEnv.DEPLOYMENT
                    for phase in phase_array
                )
            ),
            "routing_steps": float(
                sum(phase == AgentOrchestrationEnv.ROUTING for phase in phase_array)
            ),
            "exploration_weight": float(exploration_weight),
            "mean_intrinsic_reward": float(np.mean(intrinsic_normalized)),
            "mean_raw_intrinsic_reward": float(np.mean(intrinsic_raw)),
            "mean_deployment_rnd_reward": _phase_mean(
                intrinsic_normalized,
                phase_array,
                AgentOrchestrationEnv.DEPLOYMENT,
            ),
            "mean_routing_rnd_reward": _phase_mean(
                intrinsic_normalized,
                phase_array,
                AgentOrchestrationEnv.ROUTING,
            ),
            "mean_rnd_loss": float(np.mean(rnd_losses)) if rnd_losses else 0.0,
            "mean_icm_loss": float(np.mean(icm_losses)) if icm_losses else 0.0,
        }
        history.append(record)
        if on_update is not None:
            on_update(record)
        lagrange_multiplier = next_lagrange_multiplier
    return policy, history


def _exploration_weight(config: PPOConfig, update: int, updates: int) -> float:
    if config.exploration_mode != "rnd":
        return 0.0
    progress = update / (updates - 1) if updates > 1 else 0.0
    return float(
        config.rnd_initial_weight
        + progress * (config.rnd_final_weight - config.rnd_initial_weight)
    )


def _lagrangian_rewards(
    utilities: list[float],
    constraint_costs: list[float],
    multiplier: float,
    constraint_limit: float,
    constrained: bool,
) -> list[float]:
    if not constrained:
        return list(utilities)
    return [
        utility - multiplier * (constraint - constraint_limit)
        for utility, constraint in zip(utilities, constraint_costs)
    ]


def _combine_training_rewards(
    lagrangian_rewards: list[float],
    intrinsic_rewards: np.ndarray,
    exploration_weight: float,
) -> list[float]:
    return [
        reward + exploration_weight * float(intrinsic_rewards[index])
        for index, reward in enumerate(lagrangian_rewards)
    ]


def _update_lagrange_multiplier(
    multiplier: float,
    mean_constraint_cost: float,
    constraint_limit: float,
    learning_rate: float,
    maximum: float,
) -> float:
    return float(
        np.clip(
            multiplier
            + learning_rate * (mean_constraint_cost - constraint_limit),
            0.0,
            maximum,
        )
    )


def _phase_mean(values: np.ndarray, phases: np.ndarray, phase: int) -> float:
    selected = values[phases == phase]
    return float(np.mean(selected)) if len(selected) else 0.0


def _concentrations(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + 0.1, 0.1, 100.0)


def _sample_variable_categoricals(
    raw: torch.Tensor,
    mask: torch.Tensor,
    widths: tuple[int, ...],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    choices: list[torch.Tensor] = []
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    offset = 0
    for width in widths:
        logits = raw[offset : offset + width]
        valid = mask[offset : offset + width].bool()
        distribution = Categorical(logits=logits.masked_fill(~valid, -1.0e9))
        choice = torch.argmax(logits.masked_fill(~valid, -1.0e9)) if deterministic else distribution.sample()
        choices.append(choice)
        log_prob = log_prob + distribution.log_prob(choice)
        entropy = entropy + distribution.entropy()
        offset += width
    return torch.stack(choices), log_prob, entropy


def _evaluate_variable_categoricals(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
    widths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    offset = 0
    for group, width in enumerate(widths):
        logits = raw[offset : offset + width]
        valid = mask[offset : offset + width].bool()
        distribution = Categorical(logits=logits.masked_fill(~valid, -1.0e9))
        choice = action[group].long()
        log_prob = log_prob + distribution.log_prob(choice)
        entropy = entropy + distribution.entropy()
        offset += width
    return log_prob, entropy


def _sample_grouped_dirichlet(
    raw: torch.Tensor,
    mask: torch.Tensor,
    groups: int,
    width: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = raw.reshape(groups, width)
    mask = mask.reshape(groups, width).bool()
    action = torch.zeros_like(raw)
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    for group in range(groups):
        indices = torch.where(mask[group])[0]
        if len(indices) == 0:
            continue
        if len(indices) == 1:
            action[group, indices[0]] = 1.0
            continue
        distribution = Dirichlet(_concentrations(raw[group, indices]))
        sample = distribution.mean if deterministic else distribution.sample()
        action[group, indices] = sample
        log_prob = log_prob + distribution.log_prob(sample)
        entropy = entropy + distribution.entropy()
    return action.reshape(-1), log_prob, entropy


def _evaluate_grouped_dirichlet(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
    groups: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.reshape(groups, width)
    mask = mask.reshape(groups, width).bool()
    action = action.reshape(groups, width)
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    for group in range(groups):
        indices = torch.where(mask[group])[0]
        if len(indices) <= 1:
            continue
        distribution = Dirichlet(_concentrations(raw[group, indices]))
        sample = torch.clamp(action[group, indices], min=1e-8)
        sample = sample / sample.sum()
        log_prob = log_prob + distribution.log_prob(sample)
        entropy = entropy + distribution.entropy()
    return log_prob, entropy


def _observation_to_tensors(
    observation: dict[str, Any], device: torch.device | str, batched: bool
) -> dict[str, torch.Tensor]:
    result = {
        "features": torch.as_tensor(observation["features"], dtype=torch.float32, device=device),
        "action_type": torch.as_tensor(observation["action_type"], dtype=torch.long, device=device),
        "deploy_mask": torch.as_tensor(observation["deploy_mask"], dtype=torch.bool, device=device),
        "model_mask": torch.as_tensor(observation["model_mask"], dtype=torch.bool, device=device),
    }
    if batched:
        return result
    return result


def _rnd_state_inputs(
    observations: list[dict[str, Any]], device: torch.device | str
) -> torch.Tensor:
    features = torch.as_tensor(
        np.stack([observation["features"] for observation in observations]),
        dtype=torch.float32,
        device=device,
    )
    phases = torch.as_tensor(
        [observation["action_type"] for observation in observations],
        dtype=torch.long,
        device=device,
    )
    phase_encoding = torch.nn.functional.one_hot(phases, num_classes=2).float()
    return torch.cat([features, phase_encoding], dim=-1)


def _stack_observations(
    observations: list[dict[str, Any]], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(
            np.stack([obs["features"] for obs in observations]),
            dtype=torch.float32,
            device=device,
        ),
        "action_type": torch.as_tensor(
            [obs["action_type"] for obs in observations], dtype=torch.long, device=device
        ),
        "deploy_mask": torch.as_tensor(
            np.stack([obs["deploy_mask"] for obs in observations]),
            dtype=torch.bool,
            device=device,
        ),
        "model_mask": torch.as_tensor(
            np.stack([obs["model_mask"] for obs in observations]),
            dtype=torch.bool,
            device=device,
        ),
    }


def _stack_actions(
    actions: list[dict[str, Any]], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {
        "deploy": torch.as_tensor(
            np.stack([action["deploy"] for action in actions]),
            dtype=torch.long,
            device=device,
        ),
        "model": torch.as_tensor(
            np.stack([action["model"] for action in actions]),
            dtype=torch.float32,
            device=device,
        ),
    }


def _gae(
    rewards: list[float],
    values: list[float],
    discounts: list[float],
    terminals: list[float],
    bootstrap: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, list[float]]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_value = bootstrap
    next_advantage = 0.0
    for index in reversed(range(len(rewards))):
        continuation = 1.0 - terminals[index]
        discount = discounts[index] * continuation
        delta = rewards[index] + discount * next_value - values[index]
        next_advantage = delta + discount * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = (advantages + np.asarray(values, dtype=np.float32)).tolist()
    return torch.as_tensor(advantages), returns

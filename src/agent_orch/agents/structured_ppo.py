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
    deployment_gamma: float = 1.0
    deployment_gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3.0e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    constrained: bool = True
    constraint_limits: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    lagrangian_learning_rates: tuple[float, float, float, float] = (
        0.05,
        0.05,
        0.05,
        0.05,
    )
    initial_lagrange_multipliers: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    max_lagrange_multipliers: tuple[float, float, float, float] = (
        50.0,
        50.0,
        50.0,
        50.0,
    )
    routing_loss_weight: float = 1.0
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
        self.deployment_value_head = nn.Linear(hidden, 1)
        self.routing_value_head = nn.Linear(hidden, 1)

    def _encode(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        features = observation["features"]
        if features.ndim == 1:
            features = features.unsqueeze(0)
        action_type = observation["action_type"].long().view(-1)
        phase = torch.nn.functional.one_hot(action_type, num_classes=2).float()
        return self.encoder(torch.cat([features, phase], dim=-1))

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self._encode(observation)
        phases = observation["action_type"].long().view(-1)
        deployment = self.deployment_value_head(hidden).squeeze(-1)
        routing = self.routing_value_head(hidden).squeeze(-1)
        return torch.where(
            phases == AgentOrchestrationEnv.DEPLOYMENT, deployment, routing
        )

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, Any],
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> tuple[dict[str, Any], float, float]:
        obs = _observation_to_tensors(observation, device, batched=False)
        hidden = self._encode(obs)
        value = self.value(obs)
        phase = int(observation["action_type"])
        action = {
            "deploy": 0,
            "model": np.zeros(self.layout.model_action_size, dtype=np.float32),
        }
        if phase == AgentOrchestrationEnv.DEPLOYMENT:
            selected, log_prob, _ = _sample_categorical(
                self.deploy_head(hidden).squeeze(0),
                torch.as_tensor(
                    observation["deploy_mask"], dtype=torch.bool, device=device
                ),
                deterministic,
            )
            action["deploy"] = int(selected.item())
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
        phases = observation["action_type"].long().view(-1)
        deployment_values = self.deployment_value_head(hidden).squeeze(-1)
        routing_values = self.routing_value_head(hidden).squeeze(-1)
        values = torch.where(
            phases == AgentOrchestrationEnv.DEPLOYMENT,
            deployment_values,
            routing_values,
        )
        log_probs = torch.zeros_like(values)
        entropies = torch.zeros_like(values)
        for index in range(hidden.shape[0]):
            if int(phases[index].item()) == AgentOrchestrationEnv.DEPLOYMENT:
                deploy_logp, deploy_entropy = _evaluate_categorical(
                    self.deploy_head(hidden[index]),
                    observation["deploy_mask"][index],
                    actions["deploy"][index],
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
    _validate_constraint_config(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda:"):
        torch.cuda.manual_seed_all(seed)
    env.gamma = config.gamma
    env.deployment_gamma = config.deployment_gamma
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
            env.observation_space["features"].shape[0],
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
    rnd_moments = PhaseRunningMoments(1)
    observation, _ = env.reset(seed=seed)
    history: list[dict[str, float]] = []
    lagrange_multipliers = np.asarray(
        config.initial_lagrange_multipliers, dtype=np.float64
    )
    constraint_limits = np.asarray(config.constraint_limits, dtype=np.float64)
    episode_counter = 0
    pending_deployment: list[dict[str, Any]] = []

    for update in range(updates):
        if on_phase is not None:
            on_phase(update, "collecting")
        deployment_records: list[dict[str, Any]] = []
        routing_records: list[dict[str, Any]] = []
        rollout_step = 0
        # A slow deployment trajectory is completed only after the ensuing
        # physical deployment period has been evaluated.  Extending collection
        # to that boundary keeps every shared slow reward on-policy.
        while (
            rollout_step < rollout_steps
            or bool(pending_deployment)
            or not (deployment_records or routing_records)
        ):
            remaining_budget = rollout_steps - rollout_step
            minimum_period_steps = (
                env.scenario.simulation.deployment_period_slots + 1
            )
            if (
                rollout_step > 0
                and not pending_deployment
                and int(observation["action_type"])
                == AgentOrchestrationEnv.DEPLOYMENT
                and remaining_budget < minimum_period_steps
            ):
                break
            rollout_step += 1
            phase = int(observation["action_type"])
            action, log_prob, value = policy.act(observation, device=device)
            next_observation, reward, terminated, truncated, info = env.step(action)
            record = {
                "observation": observation,
                "next_observation": next_observation,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": float(reward),
                "external_reward": float(reward),
                "terminal": bool(terminated or truncated),
            }
            if phase == AgentOrchestrationEnv.DEPLOYMENT:
                pending_deployment.append(record)
            else:
                constraint_vector = np.asarray(
                    info.get("constraint_vector", [0.0, 0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                constrained_reward = _constrained_utility(
                    float(reward),
                    constraint_vector,
                    lagrange_multipliers,
                    constraint_limits,
                    config.constrained,
                )
                record["reward"] = constrained_reward
                record["external_reward"] = constrained_reward
                record["utility"] = float(reward)
                record["constraint_vector"] = constraint_vector
                record["terminal"] = bool(
                    terminated
                    or truncated
                    or "deployment_period_summary" in info
                )
                routing_records.append(record)

            summary = info.get("deployment_period_summary")
            if summary is not None and pending_deployment:
                actual = _constrained_utility(
                    float(summary["actual_utility"]),
                    np.asarray(summary["actual_constraints"], dtype=np.float64),
                    lagrange_multipliers,
                    constraint_limits,
                    config.constrained,
                )
                baseline = _constrained_utility(
                    float(summary["baseline_utility"]),
                    np.asarray(summary["baseline_constraints"], dtype=np.float64),
                    lagrange_multipliers,
                    constraint_limits,
                    config.constrained,
                )
                shared_reward = (actual - baseline) / max(
                    1, int(summary["deployment_steps"])
                )
                for index, pending in enumerate(pending_deployment):
                    pending["reward"] = shared_reward
                    pending["external_reward"] = shared_reward
                    pending["terminal"] = index == len(pending_deployment) - 1
                    deployment_records.append(pending)
                pending_deployment = []

            observation = next_observation
            if terminated or truncated:
                episode_counter += 1
                observation, _ = env.reset(seed=seed + episode_counter)
            if on_rollout_step is not None and rollout_step <= rollout_steps:
                on_rollout_step(update, rollout_step)

        exploration_weight = _exploration_weight(config, update, updates)
        intrinsic_raw = np.zeros(len(deployment_records), dtype=np.float32)
        intrinsic_normalized = np.zeros(len(deployment_records), dtype=np.float32)
        current_rnd_states = None
        if rnd is not None and deployment_records:
            current_rnd_states = _rnd_state_inputs(
                [record["observation"] for record in deployment_records], device
            )
            next_rnd_states = _rnd_state_inputs(
                [record["next_observation"] for record in deployment_records], device
            )
            intrinsic_raw = rnd.intrinsic_reward(next_rnd_states).cpu().numpy()
            intrinsic_normalized = rnd_moments.scale_by_std(
                intrinsic_raw,
                np.zeros(len(intrinsic_raw), dtype=np.int64),
                config.rnd_reward_clip,
            )
            rnd_moments.update(
                intrinsic_raw, np.zeros(len(intrinsic_raw), dtype=np.int64)
            )
            for record, intrinsic_reward in zip(
                deployment_records, intrinsic_normalized
            ):
                record["reward"] += exploration_weight * float(intrinsic_reward)

        records = deployment_records + routing_records
        feature_tensor = None
        next_feature_tensor = None
        action_vector_tensor = None
        icm_raw = np.zeros(len(records), dtype=np.float32)
        if icm is not None and records:
            feature_tensor = torch.as_tensor(
                np.stack([record["observation"]["features"] for record in records]),
                dtype=torch.float32,
                device=device,
            )
            next_feature_tensor = torch.as_tensor(
                np.stack(
                    [record["next_observation"]["features"] for record in records]
                ),
                dtype=torch.float32,
                device=device,
            )
            action_vectors = np.stack(
                [
                    structured_action_vector(
                        record["action"],
                        int(record["observation"]["action_type"]),
                        env.layout.deployment_action_size,
                    )
                    for record in records
                ]
            )
            action_vector_tensor = torch.as_tensor(
                action_vectors, dtype=torch.float32, device=device
            )
            icm_raw = icm.intrinsic_reward(
                feature_tensor, action_vector_tensor, next_feature_tensor
            ).cpu().numpy()
            for record, intrinsic_reward in zip(records, icm_raw):
                record["reward"] += config.icm_scale * float(intrinsic_reward)

        deployment_advantages, deployment_returns = _phase_gae(
            deployment_records,
            config.deployment_gamma,
            config.deployment_gae_lambda,
            0.0,
        )
        routing_bootstrap = 0.0
        if (
            routing_records
            and not routing_records[-1]["terminal"]
            and int(observation["action_type"]) == AgentOrchestrationEnv.ROUTING
        ):
            with torch.no_grad():
                routing_bootstrap = float(
                    policy.value(
                        _observation_to_tensors(observation, device, False)
                    ).item()
                )
        routing_advantages, routing_returns = _phase_gae(
            routing_records,
            config.gamma,
            config.gae_lambda,
            routing_bootstrap,
        )
        deployment_advantages = _normalize_advantages(deployment_advantages)
        routing_advantages = _normalize_advantages(routing_advantages)
        advantages = torch.cat(
            [deployment_advantages, routing_advantages], dim=0
        ).to(device)
        returns_tensor = torch.as_tensor(
            deployment_returns + routing_returns,
            dtype=torch.float32,
            device=device,
        )
        old_log_probs = torch.as_tensor(
            [record["log_prob"] for record in records],
            dtype=torch.float32,
            device=device,
        )
        observations_tensor = _stack_observations(
            [record["observation"] for record in records], device
        )
        actions_tensor = _stack_actions(
            [record["action"] for record in records], device
        )
        phase_tensor = observations_tensor["action_type"]
        indices = np.arange(len(records))
        losses = []
        icm_losses = []
        rnd_losses = []
        optimization_step = 0
        optimizer_steps_per_update = config.update_epochs * math.ceil(
            len(records) / config.minibatch_size
        )
        if on_phase is not None:
            on_phase(update, "optimizing")
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(records), config.minibatch_size):
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
                surrogate = torch.min(
                    ratio * advantage_batch, clipped * advantage_batch
                )
                batch_phases = phase_tensor[batch_t]
                deployment_mask = batch_phases == AgentOrchestrationEnv.DEPLOYMENT
                routing_mask = batch_phases == AgentOrchestrationEnv.ROUTING
                actor_loss = -_masked_mean(surrogate, deployment_mask)
                actor_loss -= config.routing_loss_weight * _masked_mean(
                    surrogate, routing_mask
                )
                value_loss = _masked_mse(
                    predicted_values, returns_tensor[batch_t], deployment_mask
                )
                value_loss += _masked_mse(
                    predicted_values, returns_tensor[batch_t], routing_mask
                )
                entropy_term = _masked_mean(entropy, deployment_mask)
                entropy_term += config.routing_loss_weight * _masked_mean(
                    entropy, routing_mask
                )
                loss = (
                    actor_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy_term
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
                if (
                    rnd is not None
                    and rnd_optimizer is not None
                    and current_rnd_states is not None
                ):
                    deployment_indices = batch_t[
                        batch_t < len(deployment_records)
                    ]
                    if len(deployment_indices) > 0:
                        rnd_loss = (
                            config.rnd_loss_coefficient
                            * rnd.loss(current_rnd_states[deployment_indices])
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

        constraint_rows = [
            record["constraint_vector"] for record in routing_records
        ]
        mean_constraints = (
            np.mean(np.stack(constraint_rows), axis=0)
            if constraint_rows
            else np.zeros(4, dtype=np.float64)
        )
        next_lagrange_multipliers = lagrange_multipliers.copy()
        if config.constrained:
            next_lagrange_multipliers = _update_lagrange_multipliers(
                lagrange_multipliers,
                mean_constraints,
                constraint_limits,
                np.asarray(config.lagrangian_learning_rates, dtype=np.float64),
                np.asarray(config.max_lagrange_multipliers, dtype=np.float64),
            )
        training_rewards = [record["reward"] for record in records]
        external_rewards = [record["external_reward"] for record in records]
        utility_rewards = [float(record["utility"]) for record in routing_records]
        record = {
            "update": float(update),
            "mean_reward": float(np.mean(training_rewards)),
            "mean_utility": float(np.mean(utility_rewards)) if utility_rewards else 0.0,
            "mean_lagrangian_reward": float(np.mean(external_rewards)),
            "mean_constraint_cost": float(np.sum(mean_constraints)),
            "lagrange_multiplier": float(np.sum(lagrange_multipliers)),
            "next_lagrange_multiplier": float(
                np.sum(next_lagrange_multipliers)
            ),
            "mean_loss": float(np.mean(losses)),
            "deployment_steps": float(len(deployment_records)),
            "routing_steps": float(len(routing_records)),
            "exploration_weight": float(exploration_weight),
            "mean_intrinsic_reward": float(np.mean(intrinsic_normalized))
            if len(intrinsic_normalized)
            else 0.0,
            "mean_raw_intrinsic_reward": float(np.mean(intrinsic_raw))
            if len(intrinsic_raw)
            else 0.0,
            "mean_deployment_rnd_reward": _phase_mean(
                intrinsic_normalized,
                np.zeros(len(intrinsic_normalized), dtype=np.int64),
                0,
            ),
            "mean_routing_rnd_reward": 0.0,
            "mean_rnd_loss": float(np.mean(rnd_losses)) if rnd_losses else 0.0,
            "mean_icm_loss": float(np.mean(icm_losses)) if icm_losses else 0.0,
        }
        for index, name in enumerate(env.CONSTRAINT_NAMES):
            record[f"mean_constraint_{name}"] = float(mean_constraints[index])
            record[f"lagrange_{name}"] = float(lagrange_multipliers[index])
            record[f"next_lagrange_{name}"] = float(
                next_lagrange_multipliers[index]
            )
        history.append(record)
        if on_update is not None:
            on_update(record)
        lagrange_multipliers = next_lagrange_multipliers
    return policy, history


def _exploration_weight(config: PPOConfig, update: int, updates: int) -> float:
    if config.exploration_mode != "rnd":
        return 0.0
    progress = update / (updates - 1) if updates > 1 else 0.0
    return float(
        config.rnd_initial_weight
        + progress * (config.rnd_final_weight - config.rnd_initial_weight)
    )


def _validate_constraint_config(config: PPOConfig) -> None:
    fields = (
        config.constraint_limits,
        config.lagrangian_learning_rates,
        config.initial_lagrange_multipliers,
        config.max_lagrange_multipliers,
    )
    if any(len(values) != 4 for values in fields):
        raise ValueError("The LLM, KV, tool, and link constraint vectors need four values")


def _constrained_utility(
    utility: float,
    constraints: np.ndarray,
    multipliers: np.ndarray,
    limits: np.ndarray,
    constrained: bool,
) -> float:
    if not constrained:
        return float(utility)
    return float(utility - np.dot(multipliers, constraints - limits))


def _update_lagrange_multipliers(
    multipliers: np.ndarray,
    mean_constraints: np.ndarray,
    limits: np.ndarray,
    learning_rates: np.ndarray,
    maxima: np.ndarray,
) -> np.ndarray:
    return np.clip(
        multipliers + learning_rates * (mean_constraints - limits),
        0.0,
        maxima,
    )


def _phase_gae(
    records: list[dict[str, Any]],
    gamma: float,
    gae_lambda: float,
    bootstrap: float,
) -> tuple[torch.Tensor, list[float]]:
    if not records:
        return torch.empty(0, dtype=torch.float32), []
    return _gae(
        rewards=[float(record["reward"]) for record in records],
        values=[float(record["value"]) for record in records],
        discounts=[gamma] * len(records),
        terminals=[float(record["terminal"]) for record in records],
        bootstrap=bootstrap,
        gae_lambda=gae_lambda,
    )


def _normalize_advantages(values: torch.Tensor) -> torch.Tensor:
    if len(values) <= 1:
        return values
    return (values - values.mean()) / (values.std(unbiased=False) + 1.0e-8)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_tensor(0.0)


def _masked_mse(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if bool(mask.any()):
        return torch.nn.functional.mse_loss(predicted[mask], target[mask])
    return predicted.new_tensor(0.0)


def _phase_mean(values: np.ndarray, phases: np.ndarray, phase: int) -> float:
    selected = values[phases == phase]
    return float(np.mean(selected)) if len(selected) else 0.0


def _concentrations(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + 0.1, 0.1, 100.0)


def _sample_categorical(
    raw: torch.Tensor,
    mask: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError("The sequential deployment step has no feasible target")
    logits = raw.masked_fill(~valid, -1.0e9)
    distribution = Categorical(logits=logits)
    choice = torch.argmax(logits) if deterministic else distribution.sample()
    return choice, distribution.log_prob(choice), distribution.entropy()


def _evaluate_categorical(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    distribution = Categorical(logits=raw.masked_fill(~valid, -1.0e9))
    choice = action.long()
    return distribution.log_prob(choice), distribution.entropy()


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
    return torch.as_tensor(
        np.stack([observation["features"] for observation in observations]),
        dtype=torch.float32,
        device=device,
    )


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

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
CheckpointCallback = Callable[[dict[str, Any]], None]


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
    constraint_limits: tuple[float, float] = (0.0, 0.0)
    lagrangian_learning_rates: tuple[float, float] = (
        0.05,
        0.05,
    )
    initial_lagrange_multipliers: tuple[float, float] = (
        0.0,
        0.0,
    )
    max_lagrange_multipliers: tuple[float, float] = (
        50.0,
        50.0,
    )
    shaping_coefficient: float = 1.0
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
            nn.Linear(feature_size + 3, hidden),
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
        phase = torch.nn.functional.one_hot(action_type, num_classes=3).float()
        return self.encoder(torch.cat([features, phase], dim=-1))

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self._encode(observation)
        phases = observation["action_type"].long().view(-1)
        deployment = self.deployment_value_head(hidden).squeeze(-1)
        routing = self.routing_value_head(hidden).squeeze(-1)
        return torch.where(
            phases < AgentOrchestrationEnv.COMPOSITION, deployment, routing
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
        if phase < AgentOrchestrationEnv.COMPOSITION:
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
            phases < AgentOrchestrationEnv.COMPOSITION,
            deployment_values,
            routing_values,
        )
        log_probs = torch.zeros_like(values)
        entropies = torch.zeros_like(values)
        for index in range(hidden.shape[0]):
            if int(phases[index].item()) < AgentOrchestrationEnv.COMPOSITION:
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
    resume_state: dict[str, Any] | None = None,
    on_checkpoint: CheckpointCallback | None = None,
) -> tuple[StructuredActorCritic, list[dict[str, float]]]:
    device = resolve_device(device)
    _validate_constraint_config(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    env.gamma = config.gamma
    policy = StructuredActorCritic(env, config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rnd = (
        RNDModule(
            env.observation_space["features"].shape[0] + 3,
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
    action_vector_size = 2 + env.layout.deployment_action_size + env.layout.model_action_size
    icm = (
        ICMModule(env.observation_space["features"].shape[0], action_vector_size).to(device)
        if config.exploration_mode == "icm" else None
    )
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=config.learning_rate) if icm is not None else None
    rnd_moments = PhaseRunningMoments(2)
    history: list[dict[str, float]] = []
    lagrange_multipliers = np.asarray(config.initial_lagrange_multipliers, dtype=np.float64)
    constraint_limits = np.asarray(config.constraint_limits, dtype=np.float64)
    episode_counter = 0
    start_update = 0

    if resume_state is not None:
        policy.load_state_dict(resume_state["policy_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if rnd is not None and resume_state.get("rnd_state_dict") is not None:
            rnd.load_state_dict(resume_state["rnd_state_dict"])
            assert rnd_optimizer is not None
            rnd_optimizer.load_state_dict(resume_state["rnd_optimizer_state_dict"])
        if icm is not None and resume_state.get("icm_state_dict") is not None:
            icm.load_state_dict(resume_state["icm_state_dict"])
            assert icm_optimizer is not None
            icm_optimizer.load_state_dict(resume_state["icm_optimizer_state_dict"])
        moments = resume_state.get("rnd_moments")
        if moments is not None:
            rnd_moments.count = np.asarray(moments["count"], dtype=np.float64)
            rnd_moments.mean = np.asarray(moments["mean"], dtype=np.float64)
            rnd_moments.m2 = np.asarray(moments["m2"], dtype=np.float64)
        history = [dict(record) for record in resume_state.get("history", [])]
        lagrange_multipliers = np.asarray(
            resume_state.get("lagrange_multipliers", lagrange_multipliers),
            dtype=np.float64,
        )
        episode_counter = int(resume_state.get("episode_counter", 0))
        start_update = int(resume_state.get("next_update", 0))
        if start_update < 0 or start_update > updates:
            raise ValueError("The PPO checkpoint has an invalid next_update value")

    observation, _ = env.reset(seed=seed + episode_counter)
    if resume_state is not None:
        if "python_random_state" in resume_state:
            random.setstate(resume_state["python_random_state"])
        if "numpy_random_state" in resume_state:
            np.random.set_state(resume_state["numpy_random_state"])
        if "torch_random_state" in resume_state:
            torch.set_rng_state(resume_state["torch_random_state"].cpu())
        if (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_state["cuda_random_state"]]
            )

    for update in range(start_update, updates):
        if on_phase is not None:
            on_phase(update, "collecting")
        records: list[dict[str, Any]] = []
        period_complete = False
        while len(records) < max(1, rollout_steps) or not period_complete:
            phase = int(observation["action_type"])
            action, log_prob, value = policy.act(observation, device=device)
            next_observation, raw_reward, terminated, truncated, info = env.step(action)
            is_composition = phase == AgentOrchestrationEnv.COMPOSITION
            constraint_vector = np.asarray(
                info.get("constraint_vector", [0.0, 0.0]), dtype=np.float64
            )
            utility = float(info.get("utility", 0.0)) if is_composition else 0.0
            if is_composition:
                reward = _constrained_utility(
                    utility, constraint_vector, lagrange_multipliers,
                    constraint_limits, config.constrained
                )
            else:
                reward = config.shaping_coefficient * float(raw_reward)
            records.append({
                "observation": observation,
                "next_observation": next_observation,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "external_reward": reward,
                "utility": utility,
                "constraint_vector": constraint_vector,
                "terminal": bool(terminated or truncated),
                "discount": float(info.get("discount", config.gamma)),
                "phase": phase,
                "policy_action_active": not bool(info.get("policy_action_ignored", False)),
            })
            observation = next_observation
            period_complete = is_composition
            if terminated or truncated:
                episode_counter += 1
                observation, _ = env.reset(seed=seed + episode_counter)
            if on_rollout_step is not None:
                on_rollout_step(update, len(records))
            if is_composition and len(records) >= max(1, rollout_steps):
                break

        exploration_weight = _exploration_weight(config, update, updates)
        intrinsic_raw = np.zeros(len(records), dtype=np.float32)
        intrinsic_normalized = np.zeros(len(records), dtype=np.float32)
        rnd_states = None
        deployment_indices = [i for i, r in enumerate(records) if r["phase"] < AgentOrchestrationEnv.COMPOSITION]
        if rnd is not None and deployment_indices:
            states = [records[i]["next_observation"] for i in deployment_indices]
            rnd_states = _rnd_state_inputs(states, device)
            raw = rnd.intrinsic_reward(rnd_states).detach().cpu().numpy()
            phase_ids = np.asarray([records[i]["phase"] for i in deployment_indices], dtype=np.int64)
            intrinsic_raw[deployment_indices] = raw
            intrinsic_normalized[deployment_indices] = rnd_moments.scale_by_std(
                raw, phase_ids, config.rnd_reward_clip
            )
            rnd_moments.update(raw, phase_ids)
            for index, value_intrinsic in zip(deployment_indices, intrinsic_normalized[deployment_indices]):
                records[index]["reward"] += exploration_weight * float(value_intrinsic)

        icm_losses: list[float] = []
        icm_raw = np.zeros(len(records), dtype=np.float32)
        current_icm_states = None
        next_icm_states = None
        icm_actions = None
        if icm is not None:
            current_icm_states = torch.as_tensor(
                np.stack([r["observation"]["features"] for r in records]),
                dtype=torch.float32, device=device
            )
            next_icm_states = torch.as_tensor(
                np.stack([r["next_observation"]["features"] for r in records]),
                dtype=torch.float32, device=device
            )
            icm_actions = torch.as_tensor(
                np.stack([
                    structured_action_vector(r["action"], r["phase"], env.layout.deployment_action_size)
                    for r in records
                ]), dtype=torch.float32, device=device
            )
            icm_raw = icm.intrinsic_reward(current_icm_states, icm_actions, next_icm_states).detach().cpu().numpy()
            for r, intrinsic in zip(records, icm_raw):
                r["reward"] += config.icm_scale * float(intrinsic)

        values = [float(r["value"]) for r in records]
        bootstrap = 0.0
        if records and not records[-1]["terminal"]:
            with torch.no_grad():
                bootstrap = float(
                    policy.value(_observation_to_tensors(observation, device, batched=False)).item()
                )
        advantages, returns = _gae(
            [float(r["reward"]) for r in records], values,
            [float(r["discount"]) for r in records],
            [float(r["terminal"]) for r in records], bootstrap, config.gae_lambda
        )
        advantages = _normalize_advantages(advantages)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantages = advantages.to(device)
        old_log_probs = torch.as_tensor([r["log_prob"] for r in records], dtype=torch.float32, device=device)
        observations_tensor = _stack_observations([r["observation"] for r in records], device)
        actions_tensor = _stack_actions([r["action"] for r in records], device)
        phase_tensor = observations_tensor["action_type"]
        indices = np.arange(len(records))
        losses: list[float] = []
        rnd_losses: list[float] = []
        optimization_step = 0
        optimizer_steps_per_update = config.update_epochs * math.ceil(len(records) / config.minibatch_size)
        if on_phase is not None:
            on_phase(update, "optimizing")
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(records), config.minibatch_size):
                batch = indices[start:start + config.minibatch_size]
                batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
                obs_batch = {key: value[batch_t] for key, value in observations_tensor.items()}
                action_batch = {key: value[batch_t] for key, value in actions_tensor.items()}
                new_logp, entropy, predicted_values = policy.evaluate_actions(obs_batch, action_batch)
                ratio = torch.exp(new_logp - old_log_probs[batch_t])
                adv_batch = advantages[batch_t]
                clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                surrogate = torch.min(ratio * adv_batch, clipped * adv_batch)
                deployment_mask = phase_tensor[batch_t] < AgentOrchestrationEnv.COMPOSITION
                composition_mask = ~deployment_mask
                policy_mask = torch.as_tensor(
                    [records[index]["policy_action_active"] for index in batch],
                    dtype=torch.bool,
                    device=device,
                )
                policy_deployment_mask = deployment_mask & policy_mask
                policy_composition_mask = composition_mask & policy_mask
                actor_loss = -_masked_mean(surrogate, policy_deployment_mask) - _masked_mean(surrogate, policy_composition_mask)
                value_loss = _masked_mse(predicted_values, returns_tensor[batch_t], deployment_mask) + _masked_mse(predicted_values, returns_tensor[batch_t], composition_mask)
                entropy_term = _masked_mean(entropy, deployment_mask) + _masked_mean(entropy, composition_mask)
                loss = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy_term
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.item()))
                if icm is not None and icm_optimizer is not None:
                    assert current_icm_states is not None and next_icm_states is not None and icm_actions is not None
                    icm_loss = icm.loss(current_icm_states[batch_t], icm_actions[batch_t], next_icm_states[batch_t])
                    icm_optimizer.zero_grad()
                    icm_loss.backward()
                    nn.utils.clip_grad_norm_(icm.parameters(), config.max_grad_norm)
                    icm_optimizer.step()
                    icm_losses.append(float(icm_loss.item()))
                if rnd is not None and rnd_optimizer is not None and rnd_states is not None:
                    local = [pos for pos, original in enumerate(deployment_indices) if original in batch]
                    if local:
                        rnd_loss = config.rnd_loss_coefficient * rnd.loss(rnd_states[local])
                        rnd_optimizer.zero_grad()
                        rnd_loss.backward()
                        nn.utils.clip_grad_norm_(rnd.predictor.parameters(), config.max_grad_norm)
                        rnd_optimizer.step()
                        rnd_losses.append(float(rnd_loss.item()))
                optimization_step += 1
                if on_optimization_step is not None:
                    on_optimization_step(update, optimization_step, optimizer_steps_per_update)

        composition_records = [r for r in records if r["phase"] == AgentOrchestrationEnv.COMPOSITION]
        mean_constraints = (
            np.mean(np.stack([r["constraint_vector"] for r in composition_records]), axis=0)
            if composition_records else np.zeros(2, dtype=np.float64)
        )
        next_lagrange = lagrange_multipliers.copy()
        if config.constrained:
            next_lagrange = _update_lagrange_multipliers(
                lagrange_multipliers, mean_constraints, constraint_limits,
                np.asarray(config.lagrangian_learning_rates, dtype=np.float64),
                np.asarray(config.max_lagrange_multipliers, dtype=np.float64)
            )
        record = {
            "update": float(update),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "mean_utility": float(np.mean([r["utility"] for r in composition_records])) if composition_records else 0.0,
            "mean_lagrangian_reward": float(np.mean([r["external_reward"] for r in records])),
            "mean_constraint_cost": float(np.sum(mean_constraints)),
            "lagrange_multiplier": float(np.sum(lagrange_multipliers)),
            "next_lagrange_multiplier": float(np.sum(next_lagrange)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "deployment_steps": float(sum(r["phase"] < AgentOrchestrationEnv.COMPOSITION for r in records)),
            "routing_steps": float(len(composition_records)),
            "exploration_weight": float(exploration_weight),
            "mean_intrinsic_reward": float(np.mean(intrinsic_normalized)) if deployment_indices else 0.0,
            "mean_raw_intrinsic_reward": float(np.mean(intrinsic_raw)) if deployment_indices else 0.0,
            "mean_deployment_rnd_reward": float(np.mean(intrinsic_normalized[deployment_indices])) if deployment_indices else 0.0,
            "mean_routing_rnd_reward": 0.0,
            "mean_rnd_loss": float(np.mean(rnd_losses)) if rnd_losses else 0.0,
            "mean_icm_loss": float(np.mean(icm_losses)) if icm_losses else 0.0,
            "mean_icm_intrinsic_reward": float(np.mean(icm_raw)) if len(icm_raw) else 0.0,
        }
        for index, name in enumerate(env.CONSTRAINT_NAMES):
            record[f"mean_constraint_{name}"] = float(mean_constraints[index])
            record[f"lagrange_{name}"] = float(lagrange_multipliers[index])
            record[f"next_lagrange_{name}"] = float(next_lagrange[index])
        history.append(record)
        if on_update is not None:
            on_update(record)
        lagrange_multipliers = next_lagrange
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "next_update": update + 1,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rnd_state_dict": rnd.state_dict() if rnd is not None else None,
                    "rnd_optimizer_state_dict": (
                        rnd_optimizer.state_dict() if rnd_optimizer is not None else None
                    ),
                    "icm_state_dict": icm.state_dict() if icm is not None else None,
                    "icm_optimizer_state_dict": (
                        icm_optimizer.state_dict() if icm_optimizer is not None else None
                    ),
                    "rnd_moments": {
                        "count": rnd_moments.count.copy(),
                        "mean": rnd_moments.mean.copy(),
                        "m2": rnd_moments.m2.copy(),
                    },
                    "lagrange_multipliers": lagrange_multipliers.copy(),
                    "episode_counter": episode_counter,
                    "history": list(history),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": (
                        torch.cuda.get_rng_state_all()
                        if str(device).startswith("cuda") and torch.cuda.is_available()
                        else None
                    ),
                }
            )
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
    if any(len(values) != 2 for values in fields):
        raise ValueError("The LLM and stateless-service constraint vectors need two values")


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
    features = np.stack([observation["features"] for observation in observations])
    phases = np.asarray([int(observation["action_type"]) for observation in observations])
    phase_one_hot = np.zeros((len(observations), 3), dtype=np.float32)
    phase_one_hot[np.arange(len(observations)), np.clip(phases, 0, 2)] = 1.0
    normalized = np.concatenate((features / 10.0, phase_one_hot), axis=1)
    return torch.as_tensor(
        np.clip(normalized, -1.0, 1.0),
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

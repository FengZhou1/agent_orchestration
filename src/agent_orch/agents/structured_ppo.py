from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet

from agent_orch.envs import AgentOrchestrationEnv
from .icm import ICMModule, structured_action_vector


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
        self.deploy_head = nn.Linear(hidden, 2)
        self.model_head = nn.Linear(hidden, self.layout.model_action_size)
        self.llm_head = nn.Linear(hidden, self.layout.llm_action_size)
        self.tool_head = nn.Linear(hidden, self.layout.tool_action_size)
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
            "deploy": 0,
            "model": np.zeros(self.layout.model_action_size, dtype=np.float32),
            "llm": np.zeros(self.layout.llm_action_size, dtype=np.float32),
            "tool": np.zeros(self.layout.tool_action_size, dtype=np.float32),
        }
        if phase == AgentOrchestrationEnv.DEPLOYMENT:
            logits = self.deploy_head(hidden).squeeze(0)
            mask = torch.as_tensor(observation["deploy_mask"], dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask, -1.0e9)
            distribution = Categorical(logits=logits)
            selected = torch.argmax(logits) if deterministic else distribution.sample()
            action["deploy"] = int(selected.item())
            log_prob = distribution.log_prob(selected)
        else:
            model, model_logp, _ = _sample_grouped_dirichlet(
                self.model_head(hidden).squeeze(0),
                torch.as_tensor(observation["model_mask"], device=device),
                len(self.layout.model_groups),
                len(self.layout.models),
                deterministic,
            )
            llm, llm_logp, _ = _sample_grouped_dirichlet(
                self.llm_head(hidden).squeeze(0),
                torch.as_tensor(observation["llm_mask"], device=device),
                len(self.layout.llm_groups),
                len(self.layout.candidates),
                deterministic,
            )
            tool, tool_logp, _ = _sample_grouped_dirichlet(
                self.tool_head(hidden).squeeze(0),
                torch.as_tensor(observation["tool_mask"], device=device),
                len(self.layout.tool_groups),
                len(self.layout.servers),
                deterministic,
            )
            action["model"] = model.cpu().numpy().astype(np.float32)
            action["llm"] = llm.cpu().numpy().astype(np.float32)
            action["tool"] = tool.cpu().numpy().astype(np.float32)
            log_prob = model_logp + llm_logp + tool_logp
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
                logits = self.deploy_head(hidden[index])
                mask = observation["deploy_mask"][index].bool()
                distribution = Categorical(logits=logits.masked_fill(~mask, -1.0e9))
                selected = actions["deploy"][index].long()
                log_probs[index] = distribution.log_prob(selected)
                entropies[index] = distribution.entropy()
                continue
            model_logp, model_entropy = _evaluate_grouped_dirichlet(
                self.model_head(hidden[index]),
                observation["model_mask"][index],
                actions["model"][index],
                len(self.layout.model_groups),
                len(self.layout.models),
            )
            llm_logp, llm_entropy = _evaluate_grouped_dirichlet(
                self.llm_head(hidden[index]),
                observation["llm_mask"][index],
                actions["llm"][index],
                len(self.layout.llm_groups),
                len(self.layout.candidates),
            )
            tool_logp, tool_entropy = _evaluate_grouped_dirichlet(
                self.tool_head(hidden[index]),
                observation["tool_mask"][index],
                actions["tool"][index],
                len(self.layout.tool_groups),
                len(self.layout.servers),
            )
            log_probs[index] = model_logp + llm_logp + tool_logp
            entropies[index] = model_entropy + llm_entropy + tool_entropy
        return log_probs, entropies, values


def train_ppo(
    env: AgentOrchestrationEnv,
    updates: int,
    rollout_steps: int,
    seed: int = 0,
    config: PPOConfig = PPOConfig(),
    device: str = "cpu",
    use_icm: bool = False,
    icm_scale: float = 0.01,
) -> tuple[StructuredActorCritic, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = StructuredActorCritic(env, config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    action_vector_size = (
        4
        + env.layout.model_action_size
        + env.layout.llm_action_size
        + env.layout.tool_action_size
    )
    icm = (
        ICMModule(env.observation_space["features"].shape[0], action_vector_size).to(device)
        if use_icm
        else None
    )
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=config.learning_rate) if icm else None
    observation, _ = env.reset(seed=seed)
    history: list[dict[str, float]] = []

    for update in range(updates):
        observations: list[dict[str, Any]] = []
        next_observations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        discounts: list[float] = []
        terminals: list[float] = []

        for _ in range(rollout_steps):
            action, log_prob, value = policy.act(observation, device=device)
            next_observation, reward, terminated, truncated, info = env.step(action)
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            discounts.append(float(info.get("discount", config.gamma)))
            terminals.append(float(terminated or truncated))
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + update + 1)

        intrinsic_mean = 0.0
        action_vectors = np.stack(
            [
                structured_action_vector(action, int(obs["action_type"]))
                for obs, action in zip(observations, actions)
            ]
        )
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
            action_vector_tensor = torch.as_tensor(
                action_vectors, dtype=torch.float32, device=device
            )
            intrinsic = icm.intrinsic_reward(
                feature_tensor, action_vector_tensor, next_feature_tensor
            )
            intrinsic_mean = float(intrinsic.mean().item())
            rewards = [
                reward + icm_scale * float(intrinsic[index].item())
                for index, reward in enumerate(rewards)
            ]
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        observations_tensor = _stack_observations(observations, device)
        actions_tensor = _stack_actions(actions, device)
        indices = np.arange(rollout_steps)
        losses = []
        icm_losses = []
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
        history.append(
            {
                "update": float(update),
                "mean_reward": float(np.mean(rewards)),
                "mean_loss": float(np.mean(losses)),
                "routing_steps": float(
                    sum(int(obs["action_type"] == AgentOrchestrationEnv.ROUTING) for obs in observations)
                ),
                "mean_intrinsic_reward": intrinsic_mean,
                "mean_icm_loss": float(np.mean(icm_losses)) if icm_losses else 0.0,
            }
        )
    return policy, history


def _concentrations(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + 0.1, 0.1, 100.0)


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
        "llm_mask": torch.as_tensor(observation["llm_mask"], dtype=torch.bool, device=device),
        "tool_mask": torch.as_tensor(observation["tool_mask"], dtype=torch.bool, device=device),
    }
    if batched:
        return result
    return result


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
        "llm_mask": torch.as_tensor(
            np.stack([obs["llm_mask"] for obs in observations]),
            dtype=torch.bool,
            device=device,
        ),
        "tool_mask": torch.as_tensor(
            np.stack([obs["tool_mask"] for obs in observations]),
            dtype=torch.bool,
            device=device,
        ),
    }


def _stack_actions(
    actions: list[dict[str, Any]], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {
        "deploy": torch.as_tensor(
            [action["deploy"] for action in actions], dtype=torch.long, device=device
        ),
        "model": torch.as_tensor(
            np.stack([action["model"] for action in actions]),
            dtype=torch.float32,
            device=device,
        ),
        "llm": torch.as_tensor(
            np.stack([action["llm"] for action in actions]),
            dtype=torch.float32,
            device=device,
        ),
        "tool": torch.as_tensor(
            np.stack([action["tool"] for action in actions]),
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

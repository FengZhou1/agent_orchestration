"""The structured actor-critic: one deployment branch and one composition branch."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from agent_orch.envs import AgentOrchestrationEnv
from .config import PPOConfig
from .distributions import (
    _evaluate_categorical,
    _evaluate_grouped_dirichlet,
    _sample_categorical,
    _sample_grouped_dirichlet,
)
from .rollout import _observation_to_tensors


class StructuredActorCritic(nn.Module):
    def __init__(self, env: AgentOrchestrationEnv, config: PPOConfig = PPOConfig()):
        super().__init__()
        self.config = config
        self.layout = env.layout
        feature_size = env.observation_space["features"].shape[0]
        hidden = config.hidden_size
        self.deployment_encoder = nn.Sequential(
            nn.Linear(feature_size + 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.composition_encoder = nn.Sequential(
            nn.Linear(feature_size + 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.deploy_head = nn.Linear(hidden, self.layout.deployment_action_size)
        self.model_head = nn.Linear(
            hidden,
            len(self.layout.models)
            if config.shared_composition_head
            else self.layout.model_action_size,
        )
        self.deployment_value_head = nn.Linear(hidden, 1)
        self.routing_value_head = nn.Linear(hidden, 1)

    def _encoded_phases(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = observation["features"]
        if features.ndim == 1:
            features = features.unsqueeze(0)
        action_type = observation["action_type"].long().view(-1)
        phase = torch.nn.functional.one_hot(action_type, num_classes=3).float()
        inputs = torch.cat([features, phase], dim=-1)
        return (
            self.deployment_encoder(inputs),
            self.composition_encoder(inputs),
            action_type,
        )

    def set_training_phase(
        self, phase: Literal["joint", "deployment", "composition"]
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if phase == "deployment":
            modules = (
                self.composition_encoder,
                self.model_head,
                self.routing_value_head,
            )
        elif phase == "composition":
            modules = (
                self.deployment_encoder,
                self.deploy_head,
                self.deployment_value_head,
            )
        else:
            return
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _composition_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.model_head(hidden)
        if not self.config.shared_composition_head:
            return logits
        if logits.ndim == 1:
            return logits.repeat(len(self.layout.model_groups))
        return logits.unsqueeze(1).expand(
            -1, len(self.layout.model_groups), -1
        ).reshape(hidden.shape[0], self.layout.model_action_size)

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        deployment_hidden, composition_hidden, phases = self._encoded_phases(
            observation
        )
        deployment = self.deployment_value_head(deployment_hidden).squeeze(-1)
        routing = self.routing_value_head(composition_hidden).squeeze(-1)
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
        phase = int(observation["action_type"])
        features = obs["features"]
        if features.ndim == 1:
            features = features.unsqueeze(0)
        phase_one_hot = torch.nn.functional.one_hot(
            obs["action_type"].long().view(-1), num_classes=3
        ).float()
        inputs = torch.cat([features, phase_one_hot], dim=-1)
        action = {
            "deploy": 0,
            "model": np.zeros(self.layout.model_action_size, dtype=np.float32),
        }
        if phase < AgentOrchestrationEnv.COMPOSITION:
            deployment_hidden = self.deployment_encoder(inputs)
            value = self.deployment_value_head(deployment_hidden).squeeze(-1)
            selected, log_prob, _ = _sample_categorical(
                self.deploy_head(deployment_hidden).squeeze(0),
                torch.as_tensor(
                    observation["deploy_mask"], dtype=torch.bool, device=device
                ),
                deterministic,
            )
            action["deploy"] = int(selected.item())
        else:
            composition_hidden = self.composition_encoder(inputs)
            value = self.routing_value_head(composition_hidden).squeeze(-1)
            model, model_logp, _ = _sample_grouped_dirichlet(
                self._composition_logits(composition_hidden).squeeze(0),
                torch.as_tensor(observation["model_mask"], device=device),
                len(self.layout.model_groups),
                len(self.layout.models),
                deterministic,
                concentration_min=self.config.composition_concentration_min,
            )
            action["model"] = model.cpu().numpy().astype(np.float32)
            log_prob = model_logp
        return action, float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        observation: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deployment_hidden, composition_hidden, phases = self._encoded_phases(
            observation
        )
        deployment_values = self.deployment_value_head(deployment_hidden).squeeze(-1)
        routing_values = self.routing_value_head(composition_hidden).squeeze(-1)
        values = torch.where(
            phases < AgentOrchestrationEnv.COMPOSITION,
            deployment_values,
            routing_values,
        )
        log_probs = torch.zeros_like(values)
        entropies = torch.zeros_like(values)
        for index in range(deployment_hidden.shape[0]):
            if int(phases[index].item()) < AgentOrchestrationEnv.COMPOSITION:
                deploy_logp, deploy_entropy = _evaluate_categorical(
                    self.deploy_head(deployment_hidden[index]),
                    observation["deploy_mask"][index],
                    actions["deploy"][index],
                )
                log_probs[index] = deploy_logp
                entropies[index] = deploy_entropy
                continue
            model_logp, model_entropy = _evaluate_grouped_dirichlet(
                self._composition_logits(composition_hidden[index].unsqueeze(0)).squeeze(0),
                observation["model_mask"][index],
                actions["model"][index],
                len(self.layout.model_groups),
                len(self.layout.models),
                concentration_min=self.config.composition_concentration_min,
            )
            log_probs[index] = model_logp
            entropies[index] = model_entropy
        return log_probs, entropies, values


__all__ = ["StructuredActorCritic"]

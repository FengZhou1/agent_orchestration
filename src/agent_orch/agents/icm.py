from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ICMModule(nn.Module):
    """Forward/inverse dynamics comparator for the structured action vector."""

    def __init__(self, feature_size: int, action_size: int, latent_size: int = 64):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(latent_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def intrinsic_reward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        next_features: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            current = self.feature_encoder(features)
            target = self.feature_encoder(next_features)
            predicted = self.forward_model(torch.cat([current, actions], dim=-1))
            return 0.5 * torch.mean((predicted - target) ** 2, dim=-1)

    def loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        next_features: torch.Tensor,
        forward_weight: float = 0.2,
    ) -> torch.Tensor:
        current = self.feature_encoder(features)
        target = self.feature_encoder(next_features)
        predicted_next = self.forward_model(torch.cat([current, actions], dim=-1))
        predicted_action = self.inverse_model(torch.cat([current, target], dim=-1))
        forward_loss = torch.nn.functional.mse_loss(predicted_next, target.detach())
        inverse_loss = torch.nn.functional.mse_loss(predicted_action, actions)
        return forward_weight * forward_loss + (1.0 - forward_weight) * inverse_loss


def structured_action_vector(
    action: dict,
    phase: int,
    deployment_widths: tuple[int, ...],
) -> np.ndarray:
    phase_one_hot = np.asarray([float(phase == 0), float(phase == 1)], dtype=np.float32)
    deployment = np.asarray(action["deploy"], dtype=np.int64)
    encoded: list[float] = []
    for choice, width in zip(deployment, deployment_widths):
        encoded.extend(float(choice == value) for value in range(width))
    return np.concatenate(
        [
            phase_one_hot,
            np.asarray(encoded, dtype=np.float32),
            np.asarray(action["model"], dtype=np.float32),
        ]
    )

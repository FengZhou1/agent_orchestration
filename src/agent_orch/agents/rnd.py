from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RNDModule(nn.Module):
    """Random Network Distillation with a frozen target network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        feature_size: int = 64,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.target = _RNDNetwork(input_size, hidden_size, feature_size)
        self.predictor = _RNDNetwork(input_size, hidden_size, feature_size)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def intrinsic_reward(self, states: torch.Tensor) -> torch.Tensor:
        error = (self.predictor(states) - self.target(states)).pow(2)
        return error.mean(dim=-1)

    def loss(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(states)
        return (self.predictor(states) - target).pow(2).mean()


class PhaseRunningMoments:
    """Independent scalar running moments for deployment and routing rewards."""

    def __init__(self, phases: int = 2) -> None:
        self.count = np.zeros(phases, dtype=np.float64)
        self.mean = np.zeros(phases, dtype=np.float64)
        self.m2 = np.zeros(phases, dtype=np.float64)

    def normalize(
        self,
        values: np.ndarray,
        phases: np.ndarray,
        clip_max: float,
        epsilon: float = 1.0e-8,
    ) -> np.ndarray:
        normalized = np.zeros_like(values, dtype=np.float64)
        for phase in range(len(self.count)):
            selected = phases == phase
            if not np.any(selected):
                continue
            variance = (
                self.m2[phase] / max(self.count[phase], 1.0)
                if self.count[phase] > 1.0
                else 1.0
            )
            normalized[selected] = (
                values[selected] - self.mean[phase]
            ) / (np.sqrt(max(variance, 0.0)) + epsilon)
        return np.clip(normalized, 0.0, clip_max).astype(np.float32)

    def scale_by_std(
        self,
        values: np.ndarray,
        phases: np.ndarray,
        clip_max: float,
        epsilon: float = 1.0e-8,
    ) -> np.ndarray:
        """Scale non-negative RND errors by the historical phase standard deviation."""
        normalized = np.zeros_like(values, dtype=np.float64)
        for phase in range(len(self.count)):
            selected = phases == phase
            if not np.any(selected):
                continue
            variance = (
                self.m2[phase] / max(self.count[phase], 1.0)
                if self.count[phase] > 1.0
                else 1.0
            )
            normalized[selected] = values[selected] / (
                np.sqrt(max(variance, 0.0)) + epsilon
            )
        return np.clip(normalized, 0.0, clip_max).astype(np.float32)

    def update(self, values: np.ndarray, phases: np.ndarray) -> None:
        for value, phase in zip(values.astype(float), phases.astype(int)):
            self.count[phase] += 1.0
            delta = value - self.mean[phase]
            self.mean[phase] += delta / self.count[phase]
            delta_after = value - self.mean[phase]
            self.m2[phase] += delta * delta_after


class _RNDNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, feature_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

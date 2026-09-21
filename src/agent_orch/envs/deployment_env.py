"""Deployment-only environment: composition is held uniform at evaluation points."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseOrchestrationEnv


class DeploymentOnlyEnv(BaseOrchestrationEnv):
    """Sequential deployment with a uniform composition at evaluation points."""

    def step(self, action: dict[str, Any]):
        policy_action_ignored = False
        if self.phase == self.COMPOSITION:
            action = dict(action)
            action["model"] = np.ones(self.layout.model_action_size, dtype=np.float32)
            policy_action_ignored = True
        observation, reward, terminated, truncated, info = super().step(action)
        if policy_action_ignored:
            info = dict(info)
            info["policy_action_ignored"] = True
        return observation, reward, terminated, truncated, info


__all__ = ["DeploymentOnlyEnv"]

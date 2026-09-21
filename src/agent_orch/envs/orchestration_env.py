"""Joint environment: sequential deployment followed by composition."""

from __future__ import annotations

from .base import BaseOrchestrationEnv


class AgentOrchestrationEnv(BaseOrchestrationEnv):
    """Sequential deployment across both resource pools, then composition."""


__all__ = ["AgentOrchestrationEnv"]

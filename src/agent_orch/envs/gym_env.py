"""Backwards-compatible import surface for the orchestration environments.

The implementations now live in ``envs.base`` plus one module per environment
variant.  This module re-exports the historical names so existing scripts,
tests and checkpoints keep resolving.
"""

from __future__ import annotations

from .base import BaseOrchestrationEnv, resolve_deployment_library
from .composition_env import CompositionLibraryEnv
from .deployment_env import DeploymentOnlyEnv
from .layout import StructuredActionLayout
from .orchestration_env import AgentOrchestrationEnv

# ``RoutingOnlyEnv`` was the historical name for composition-only training.
RoutingOnlyEnv = CompositionLibraryEnv

__all__ = [
    "AgentOrchestrationEnv",
    "BaseOrchestrationEnv",
    "CompositionLibraryEnv",
    "DeploymentOnlyEnv",
    "RoutingOnlyEnv",
    "StructuredActionLayout",
    "resolve_deployment_library",
]

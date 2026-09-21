"""Compatibility surface for the structured PPO trainer.

The implementation now lives in :mod:`agent_orch.agents.config`,
``distributions``, ``rollout``, ``ppo``, ``networks`` and ``training``; this
module re-exports the historical names so existing scripts, tests and
checkpoints keep resolving.
"""

from __future__ import annotations

from .config import PPOConfig
from .distributions import (
    _concentrations,
    _evaluate_categorical,
    _evaluate_grouped_dirichlet,
    _masked_mean,
    _masked_mse,
    _normalize_advantages,
    _normalize_phase_advantages,
    _phase_mean,
    _sample_categorical,
    _sample_grouped_dirichlet,
)
from .networks import StructuredActorCritic
from .ppo import (
    _constrained_utility,
    _exploration_weight,
    _gae,
    _phase_gae,
    _update_lagrange_multipliers,
    _validate_constraint_config,
    build_update_record,
    optimize_ppo,
)
from .rollout import (
    CheckpointCallback,
    EpisodeCallback,
    OptimizationProgressCallback,
    PhaseCallback,
    RolloutBatch,
    RolloutProgressCallback,
    UpdateCallback,
    _observation_to_tensors,
    _rnd_state_inputs,
    _stack_actions,
    _stack_observations,
    collect_rollout,
)
from .training import train_ppo

__all__ = [
    "CheckpointCallback",
    "EpisodeCallback",
    "OptimizationProgressCallback",
    "PPOConfig",
    "PhaseCallback",
    "RolloutBatch",
    "RolloutProgressCallback",
    "StructuredActorCritic",
    "UpdateCallback",
    "build_update_record",
    "collect_rollout",
    "optimize_ppo",
    "train_ppo",
    "_concentrations",
    "_constrained_utility",
    "_evaluate_categorical",
    "_evaluate_grouped_dirichlet",
    "_exploration_weight",
    "_gae",
    "_masked_mean",
    "_masked_mse",
    "_normalize_advantages",
    "_normalize_phase_advantages",
    "_observation_to_tensors",
    "_phase_gae",
    "_phase_mean",
    "_rnd_state_inputs",
    "_sample_categorical",
    "_sample_grouped_dirichlet",
    "_stack_actions",
    "_stack_observations",
    "_update_lagrange_multipliers",
    "_validate_constraint_config",
]

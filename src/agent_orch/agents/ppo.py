"""Advantage estimation, constrained-objective bookkeeping and the PPO step."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from agent_orch.envs import AgentOrchestrationEnv
from .config import PPOConfig
from .distributions import (
    _masked_mean,
    _masked_mse,
    _normalize_grouped_advantages,
    _normalize_phase_advantages,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .icm import ICMModule
    from .networks import StructuredActorCritic
    from .rnd import RNDModule
    from .rollout import (
        OptimizationProgressCallback,
        PhaseCallback,
        RolloutBatch,
    )


def _exploration_weight(config: PPOConfig, update: int, updates: int) -> float:
    if config.exploration_mode != "rnd":
        return 0.0
    progress = update / (updates - 1) if updates > 1 else 0.0
    return float(
        config.rnd_initial_weight
        + progress * (config.rnd_final_weight - config.rnd_initial_weight)
    )


def _validate_constraint_config(config: PPOConfig) -> None:
    lengths = {
        len(config.constraint_limits),
        len(config.lagrangian_learning_rates),
        len(config.initial_lagrange_multipliers),
        len(config.max_lagrange_multipliers),
    }
    if len(lengths) != 1:
        raise ValueError(
            "The constraint vectors need one value per constraint; got lengths "
            f"{sorted(lengths)}"
        )


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
    """DEPRECATED: unused, kept for reference.

    Superseded by the per-transition discounts and phase-specific lambdas that
    ``optimize_ppo`` feeds to :func:`_gae`; external scripts may still import it.
    """

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


def _gae(
    rewards: list[float],
    values: list[float],
    discounts: list[float],
    terminals: list[float],
    bootstrap: float,
    gae_lambda: float | list[float] | np.ndarray,
) -> tuple[torch.Tensor, list[float]]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_value = bootstrap
    next_advantage = 0.0
    if np.isscalar(gae_lambda):
        lambdas = np.full(len(rewards), float(gae_lambda), dtype=np.float32)
    else:
        lambdas = np.asarray(gae_lambda, dtype=np.float32)
        if len(lambdas) != len(rewards):
            raise ValueError("GAE lambda must be scalar or match the rollout length")
    for index in reversed(range(len(rewards))):
        continuation = 1.0 - terminals[index]
        discount = discounts[index] * continuation
        delta = rewards[index] + discount * next_value - values[index]
        next_advantage = delta + discount * float(lambdas[index]) * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = (advantages + np.asarray(values, dtype=np.float32)).tolist()
    return torch.as_tensor(advantages), returns


def optimize_ppo(
    policy: "StructuredActorCritic",
    batch: "RolloutBatch",
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    device: torch.device | str,
    *,
    update: int,
    bootstrap_value: float,
    trainable_parameters: Sequence[nn.Parameter],
    rnd: "RNDModule | None" = None,
    rnd_optimizer: torch.optim.Optimizer | None = None,
    icm: "ICMModule | None" = None,
    icm_optimizer: torch.optim.Optimizer | None = None,
    on_phase: "PhaseCallback | None" = None,
    on_optimization_step: "OptimizationProgressCallback | None" = None,
) -> tuple[list[float], float]:
    """Run the clipped-objective epochs over one batch.

    Returns the per-minibatch losses and the last minibatch's entropy term (a
    diagnostic only).  The auxiliary RND/ICM losses and the optimization timer
    are written back onto ``batch``, where the record builder reads them.
    """

    from .rollout import _stack_actions, _stack_observations

    records = batch.records
    deployment_indices = batch.deployment_indices
    rnd_states = batch.rnd_states
    current_icm_states = batch.current_icm_states
    next_icm_states = batch.next_icm_states
    icm_actions = batch.icm_actions

    values = [float(r["value"]) for r in records]
    gae_lambdas = [
        config.deployment_gae_lambda
        if r["phase"] < AgentOrchestrationEnv.COMPOSITION
        else config.gae_lambda
        for r in records
    ]
    advantages, returns = _gae(
        [float(r["reward"]) for r in records],
        values,
        [float(r["discount"]) for r in records],
        [float(r["terminal"]) for r in records],
        bootstrap_value,
        gae_lambdas,
    )
    is_composition = np.asarray(
        [int(r["phase"] == AgentOrchestrationEnv.COMPOSITION) for r in records],
        dtype=np.int64,
    )
    if config.composition_group_relative_advantages and config.training_phase == "composition":
        # Each episode is one fixed deployment context.  Normalising across the
        # whole rollout would make the advantage mostly a context indicator, which
        # the policy cannot act on, and the gradient would chase sampling luck.
        episodes = np.asarray([int(r.get("episode", 0)) for r in records], dtype=np.int64)
        advantages = _normalize_grouped_advantages(
            advantages, episodes, mode=config.composition_group_relative_mode
        )
    else:
        advantages = _normalize_phase_advantages(advantages, is_composition)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages = advantages.to(device)
    old_log_probs = torch.as_tensor(
        [r["log_prob"] for r in records], dtype=torch.float32, device=device
    )
    observations_tensor = _stack_observations(
        [r["observation"] for r in records], device
    )
    actions_tensor = _stack_actions([r["action"] for r in records], device)
    phase_tensor = observations_tensor["action_type"]
    indices = np.arange(len(records))
    losses: list[float] = []
    rnd_losses: list[float] = []
    icm_losses: list[float] = []
    last_entropy = 0.0
    optimization_step = 0
    optimizer_steps_per_update = config.update_epochs * math.ceil(
        len(records) / config.minibatch_size
    )
    group_old_all = None
    group_adv_all = None
    group_active_all = None
    if config.factorized_credit:
        group_old_all, group_adv_all, group_active_all = _factorized_group_targets(
            policy, records, device
        )
    if on_phase is not None:
        on_phase(update, "optimizing")
    optimization_started = time.perf_counter()
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(records), config.minibatch_size):
            minibatch = indices[start : start + config.minibatch_size]
            batch_t = torch.as_tensor(minibatch, dtype=torch.long, device=device)
            obs_batch = {
                key: value[batch_t] for key, value in observations_tensor.items()
            }
            action_batch = {
                key: value[batch_t] for key, value in actions_tensor.items()
            }
            new_logp, entropy, predicted_values = policy.evaluate_actions(
                obs_batch, action_batch
            )
            ratio = torch.exp(new_logp - old_log_probs[batch_t])
            adv_batch = advantages[batch_t]
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            )
            surrogate = torch.min(ratio * adv_batch, clipped * adv_batch)
            deployment_mask = (
                phase_tensor[batch_t] < AgentOrchestrationEnv.COMPOSITION
            )
            composition_mask = ~deployment_mask
            policy_mask = torch.as_tensor(
                [records[index]["policy_action_active"] for index in minibatch],
                dtype=torch.bool,
                device=device,
            )
            policy_deployment_mask = deployment_mask & policy_mask
            policy_composition_mask = composition_mask & policy_mask
            if group_old_all is not None:
                # Composition groups are credited with their own application's
                # reward, so the actor loss for them is a per-group surrogate
                # rather than one scalar shared by every group.
                g_old = group_old_all[batch_t]
                g_adv = group_adv_all[batch_t]
                g_active = group_active_all[batch_t] & composition_mask.unsqueeze(1)
                g_new = policy.group_log_probs_from_tensors(obs_batch, action_batch)
                g_ratio = torch.exp(g_new - g_old)
                g_clipped = torch.clamp(
                    g_ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                )
                g_surrogate = torch.min(g_ratio * g_adv, g_clipped * g_adv)
                g_count = g_active.sum().clamp_min(1)
                composition_actor_loss = -(
                    g_surrogate * g_active.float()
                ).sum() / g_count
            else:
                composition_actor_loss = -_masked_mean(
                    surrogate, policy_composition_mask
                )
            actor_loss = (
                -_masked_mean(surrogate, policy_deployment_mask)
                + composition_actor_loss
            )
            value_loss = _masked_mse(
                predicted_values, returns_tensor[batch_t], policy_deployment_mask
            ) + _masked_mse(
                predicted_values, returns_tensor[batch_t], policy_composition_mask
            )
            deployment_entropy = _masked_mean(entropy, deployment_mask)
            composition_entropy = _masked_mean(entropy, composition_mask)
            entropy_term = (
                config.entropy_coefficient * deployment_entropy
                + config.composition_entropy_coefficient * composition_entropy
            )
            loss = actor_loss + config.value_coefficient * value_loss - entropy_term
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            last_entropy = float(entropy_term.item())
            if icm is not None and icm_optimizer is not None:
                assert (
                    current_icm_states is not None
                    and next_icm_states is not None
                    and icm_actions is not None
                )
                icm_loss = icm.loss(
                    current_icm_states[batch_t],
                    icm_actions[batch_t],
                    next_icm_states[batch_t],
                )
                icm_optimizer.zero_grad()
                icm_loss.backward()
                nn.utils.clip_grad_norm_(icm.parameters(), config.max_grad_norm)
                icm_optimizer.step()
                icm_losses.append(float(icm_loss.item()))
            if rnd is not None and rnd_optimizer is not None and rnd_states is not None:
                local = [
                    pos
                    for pos, original in enumerate(deployment_indices)
                    if original in minibatch
                ]
                if local:
                    rnd_loss = config.rnd_loss_coefficient * rnd.loss(
                        rnd_states[local]
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
        # PPO's standard safeguard: stop reusing a rollout once the policy has
        # moved away from the one that generated it.  Without it a small rollout
        # is recycled for every epoch and the policy keeps travelling after the
        # data has stopped supporting the direction -- the observed pattern is an
        # early peak followed by a slow decline.
        if config.target_kl is not None:
            with torch.no_grad():
                _new_logp, _, _ = policy.evaluate_actions(
                    observations_tensor, actions_tensor
                )
                approx_kl = float((old_log_probs - _new_logp).mean())
            batch.last_approx_kl = approx_kl
            if approx_kl > config.target_kl:
                break

    batch.optimization_time_s = time.perf_counter() - optimization_started
    batch.rnd_losses = rnd_losses
    batch.icm_losses = icm_losses
    return losses, last_entropy


def _factorized_group_targets(
    policy, records: list[dict], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group old log-probs, advantages and activity masks, aligned to records.

    Each composition group is credited with its own application's utility, minus
    the mean of that utility over the episode and the application.  Subtracting
    that mean removes the level the deployment fixes and leaves the effect of the
    composition decision, which is what the group's policy should be graded on.
    Rows that are not composition transitions, or groups with fewer than two
    active models, stay masked out.
    """

    groups = len(policy.layout.model_groups)
    size = len(records)
    old_logp = torch.zeros(size, groups, device=device)
    advantages = torch.zeros(size, groups, device=device)
    active = torch.zeros(size, groups, dtype=torch.bool, device=device)

    composition_rows = [
        index
        for index, record in enumerate(records)
        if record.get("group_log_prob") is not None
    ]
    if not composition_rows:
        return old_logp, advantages, active

    buckets: dict[tuple, list[float]] = {}
    for index in composition_rows:
        record = records[index]
        for group, (app_id, _ingress) in enumerate(policy.layout.model_groups):
            buckets.setdefault((record["episode"], app_id), []).append(
                float((record["app_utility"] or {}).get(app_id, 0.0))
            )
    baselines = {key: sum(values) / len(values) for key, values in buckets.items()}

    for index in composition_rows:
        record = records[index]
        old_logp[index] = torch.as_tensor(
            record["group_log_prob"], dtype=torch.float32, device=device
        )
        for group, (app_id, _ingress) in enumerate(policy.layout.model_groups):
            value = float((record["app_utility"] or {}).get(app_id, 0.0))
            advantages[index, group] = value - baselines[(record["episode"], app_id)]
        mask = np.asarray(record["observation"]["model_mask"]).reshape(groups, -1)
        active[index] = torch.as_tensor(
            mask.sum(axis=1) >= 2, dtype=torch.bool, device=device
        )
    return old_logp, advantages, active


def build_update_record(
    *,
    update: int,
    batch: "RolloutBatch",
    config: PPOConfig,
    env: AgentOrchestrationEnv,
    losses: list[float],
    lagrange_multipliers: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Assemble one update's history record and the next multiplier vector.

    The keys are the contract read by ``run_rl_matrix.py``, the summariser and
    the plotting scripts, so they are fixed; the per-constraint keys follow
    ``env.constraint_names``.
    """

    records = batch.records
    deployment_indices = batch.deployment_indices
    composition_records = batch.composition_records()
    # A sequential model-selection slot has many composition transitions, but
    # only its final transition contains the physical metrics and violations.
    # Dual ascent must see one absolute constraint vector per physical slot,
    # not that vector diluted by the preceding zero-valued substeps.
    slot_records = [
        record for record in composition_records
        if record.get("period_complete", True)
    ]
    constraint_limits = np.asarray(config.constraint_limits, dtype=np.float64)
    mean_constraints = (
        np.mean(np.stack([r["constraint_vector"] for r in slot_records]), axis=0)
        if slot_records
        else np.zeros_like(constraint_limits)
    )
    next_lagrange = lagrange_multipliers.copy()
    if config.constrained:
        next_lagrange = _update_lagrange_multipliers(
            lagrange_multipliers,
            mean_constraints,
            constraint_limits,
            np.asarray(config.lagrangian_learning_rates, dtype=np.float64),
            np.asarray(config.max_lagrange_multipliers, dtype=np.float64),
        )
    record: dict[str, float] = {
        "update": float(update),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "mean_utility": float(np.mean([r["utility"] for r in slot_records]))
        if slot_records
        else 0.0,
        "mean_learning_utility": float(
            np.mean([r["learning_utility"] for r in slot_records])
        )
        if slot_records
        else 0.0,
        "mean_lagrangian_reward": float(
            np.mean([r["external_reward"] for r in records])
        ),
        "mean_period_return": float(
            np.sum([r["reward"] for r in records]) / max(len(slot_records), 1)
        ),
        "mean_external_period_return": float(
            np.sum([r["external_reward"] for r in records])
            / max(len(slot_records), 1)
        ),
        "mean_composition_lagrangian_reward": float(
            np.mean([r["external_reward"] for r in slot_records])
        )
        if slot_records
        else 0.0,
        "mean_deployment_shaping_reward": float(
            np.mean(
                [
                    r["external_reward"]
                    for r in records
                    if r["phase"] < AgentOrchestrationEnv.COMPOSITION
                ]
            )
        )
        if deployment_indices
        else 0.0,
        "deployment_shaping_return_per_period": float(
            np.sum(
                [
                    r["external_reward"]
                    for r in records
                    if r["phase"] < AgentOrchestrationEnv.COMPOSITION
                ]
            )
            / max(len(slot_records), 1)
        ),
        "mean_constraint_cost": float(np.sum(mean_constraints)),
        "lagrange_multiplier": float(np.sum(lagrange_multipliers)),
        "next_lagrange_multiplier": float(np.sum(next_lagrange)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "deployment_steps": float(
            sum(r["phase"] < AgentOrchestrationEnv.COMPOSITION for r in records)
        ),
        "routing_steps": float(len(composition_records)),
        "transition_steps": float(len(records)),
        "collection_time_s": float(batch.collection_time_s),
        "optimization_time_s": float(batch.optimization_time_s),
        "exploration_weight": float(batch.exploration_weight),
        "last_approx_kl": float(getattr(batch, "last_approx_kl", 0.0)),
        # Same accounting as the training objective: the cumulative quantities
        # over the timeline, which is what a time-varying load reward is about.
        "episode_cumulative_utility": float(slot_records[-1]["episode_utility_sum"])
        if slot_records
        else 0.0,
        "episode_cumulative_cost": float(slot_records[-1]["episode_cost_sum"])
        if slot_records
        else 0.0,
        "episode_cumulative_latency": float(
            slot_records[-1]["episode_latency_sum"]
        )
        if slot_records
        else 0.0,
        "episode_length": float(slot_records[-1]["episode_slot"])
        if slot_records
        else 0.0,
        "trace_offset": float(slot_records[-1]["trace_offset"])
        if slot_records
        else 0.0,
        "mean_intrinsic_reward": float(np.mean(batch.intrinsic_normalized))
        if deployment_indices
        else 0.0,
        "mean_raw_intrinsic_reward": float(np.mean(batch.intrinsic_raw))
        if deployment_indices
        else 0.0,
        "mean_deployment_rnd_reward": float(
            np.mean(batch.intrinsic_normalized[deployment_indices])
        )
        if deployment_indices
        else 0.0,
        "mean_routing_rnd_reward": 0.0,
        "mean_rnd_loss": float(np.mean(batch.rnd_losses)) if batch.rnd_losses else 0.0,
        "mean_icm_loss": float(np.mean(batch.icm_losses)) if batch.icm_losses else 0.0,
        "mean_icm_intrinsic_reward": float(np.mean(batch.icm_raw))
        if len(batch.icm_raw)
        else 0.0,
    }
    constraint_names = getattr(
        env, "constraint_names", getattr(env, "CONSTRAINT_NAMES", ())
    )
    for index, name in enumerate(constraint_names):
        record[f"mean_constraint_{name}"] = float(mean_constraints[index])
        record[f"lagrange_{name}"] = float(lagrange_multipliers[index])
        record[f"next_lagrange_{name}"] = float(next_lagrange[index])
    return record, next_lagrange


__all__ = [
    "_constrained_utility",
    "_exploration_weight",
    "_gae",
    "_phase_gae",
    "_update_lagrange_multipliers",
    "_validate_constraint_config",
    "build_update_record",
    "optimize_ppo",
]

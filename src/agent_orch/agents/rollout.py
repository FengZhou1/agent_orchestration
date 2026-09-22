"""Rollout collection for the structured PPO trainer.

The batch collected here carries both the transitions themselves and the
per-update diagnostics (intrinsic rewards, timers, episode seeds) that the
record builder and the run logger need, so the trainer only wires modules
together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from agent_orch.envs import AgentOrchestrationEnv
from .config import PPOConfig
from .icm import structured_action_vector
from .ppo import _constrained_utility

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .icm import ICMModule
    from .networks import StructuredActorCritic
    from .rnd import PhaseRunningMoments, RNDModule


PhaseCallback = Callable[[int, str], None]
RolloutProgressCallback = Callable[[int, int], None]
OptimizationProgressCallback = Callable[[int, int, int], None]
UpdateCallback = Callable[[dict[str, float]], None]
CheckpointCallback = Callable[[dict[str, Any]], None]
EpisodeCallback = Callable[[int, dict[str, Any]], None]


@dataclass
class RolloutBatch:
    """One PPO update's transitions plus the diagnostics collected with them."""

    records: list[dict[str, Any]]
    composition_indices: list[int]
    deployment_indices: list[int]
    completed_periods: int
    episode_seeds: list[int]
    final_observation: dict[str, Any]
    episode_counter: int
    exploration_weight: float
    collection_time_s: float
    intrinsic_raw: np.ndarray
    intrinsic_normalized: np.ndarray
    icm_raw: np.ndarray
    rnd_states: torch.Tensor | None = None
    current_icm_states: torch.Tensor | None = None
    next_icm_states: torch.Tensor | None = None
    icm_actions: torch.Tensor | None = None
    model_share_samples: list[dict[Any, float]] = field(default_factory=list)
    rnd_losses: list[float] = field(default_factory=list)
    icm_losses: list[float] = field(default_factory=list)
    optimization_time_s: float = 0.0

    def __len__(self) -> int:
        return len(self.records)

    def phases(self) -> np.ndarray:
        """The environment phase of every transition, in collection order."""

        return np.asarray(
            [int(record["phase"]) for record in self.records], dtype=np.int64
        )

    def composition_records(self) -> list[dict[str, Any]]:
        return [self.records[index] for index in self.composition_indices]

    def deployment_records(self) -> list[dict[str, Any]]:
        """Deployment transitions the policy actually acted on."""

        return [self.records[index] for index in self.deployment_indices]


def collect_rollout(
    policy: "StructuredActorCritic",
    env: AgentOrchestrationEnv,
    *,
    update: int,
    rollout_steps: int,
    rollout_periods: int | None,
    config: PPOConfig,
    device: torch.device | str,
    exploration_weight: float,
    lagrange_multipliers: np.ndarray,
    observation: dict[str, Any],
    seed: int,
    episode_counter: int,
    rnd: "RNDModule | None" = None,
    icm: "ICMModule | None" = None,
    rnd_moments: "PhaseRunningMoments | None" = None,
    on_rollout_step: RolloutProgressCallback | None = None,
    on_episode: EpisodeCallback | None = None,
) -> RolloutBatch:
    """Collect one update's transitions, stopping on periods or transitions.

    ``observation`` and ``episode_counter`` continue the episode the caller is
    already in; both are advanced here across episode boundaries and returned on
    the batch, so the trainer keeps owning the seed bookkeeping it checkpoints.
    ``model_share`` is not part of the environment's ``info``; it is read from
    ``env.last_routing`` after each composition step, so the sample list is empty
    for environments that expose neither.
    """

    constraint_limits = np.asarray(config.constraint_limits, dtype=np.float64)
    records: list[dict[str, Any]] = []
    episode_seeds: list[int] = []
    model_share_samples: list[dict[Any, float]] = []
    completed_periods = 0
    target_periods = (
        max(1, int(rollout_periods)) if rollout_periods is not None else None
    )
    collection_started = time.perf_counter()
    while True:
        phase = int(observation["action_type"])
        active_phase = (
            config.training_phase == "joint"
            or (
                config.training_phase == "deployment"
                and phase < AgentOrchestrationEnv.COMPOSITION
            )
            or (
                config.training_phase == "composition"
                and phase == AgentOrchestrationEnv.COMPOSITION
            )
        )
        action, log_prob, value = policy.act(
            observation,
            deterministic=not active_phase,
            device=device,
        )
        next_observation, raw_reward, terminated, truncated, info = env.step(action)
        is_composition = phase == AgentOrchestrationEnv.COMPOSITION
        constraint_vector = np.asarray(
            info.get("constraint_vector", [0.0, 0.0]), dtype=np.float64
        )
        utility = float(info.get("utility", 0.0)) if is_composition else 0.0
        learning_utility = (
            float(info.get("learning_utility", utility)) if is_composition else 0.0
        )
        if is_composition:
            reward = _constrained_utility(
                learning_utility,
                constraint_vector,
                lagrange_multipliers,
                constraint_limits,
                config.constrained,
            )
            share = getattr(getattr(env, "last_routing", None), "model_share", None)
            if share:
                model_share_samples.append(dict(share))
        else:
            reward = (
                config.shaping_coefficient * float(raw_reward)
                if active_phase
                else 0.0
            )
        group_log_prob = None
        app_utility = None
        if config.factorized_credit and is_composition:
            # Credit each (application, ingress) group with its own application's
            # reward, so the gradient is not one scalar shared by all of them.
            with torch.no_grad():
                group_log_prob = (
                    policy.group_log_probs(observation, action).cpu().numpy().copy()
                )
            raw_app_utility = info.get("app_utility") or {}
            if raw_app_utility:
                app_utility = dict(raw_app_utility)
        records.append(
            {
                "observation": observation,
                "next_observation": next_observation,
                "action": action,
                "log_prob": log_prob,
                "group_log_prob": group_log_prob,
                "app_utility": app_utility,
                "value": value,
                "reward": reward,
                "external_reward": reward,
                "utility": utility,
                "learning_utility": learning_utility,
                "episode_utility_sum": float(info.get("episode_utility_sum", 0.0)),
                "episode_cost_sum": float(info.get("episode_cost_sum", 0.0)),
                "episode_latency_sum": float(info.get("episode_latency_sum", 0.0)),
                "episode_slot": int(info.get("episode_slot", 0)),
                "trace_offset": int(info.get("trace_offset", 0)),
                "constraint_vector": constraint_vector,
                "terminal": bool(terminated or truncated),
                "discount": float(
                    config.composition_gamma
                    if (
                        is_composition
                        and config.training_phase == "composition"
                        and config.composition_gamma is not None
                    )
                    else info.get("discount", config.gamma)
                ),
                "phase": phase,
                "episode": episode_counter,
                "policy_action_active": active_phase
                and not bool(info.get("policy_action_ignored", False)),
            }
        )
        observation = next_observation
        if is_composition:
            completed_periods += 1
        if terminated or truncated:
            episode_counter += 1
            observation, reset_info = env.reset(seed=seed + episode_counter)
            episode_seeds.append(seed + episode_counter)
            if on_episode is not None:
                on_episode(
                    episode_counter,
                    {
                        "seed": seed + episode_counter,
                        "deployment_stratum": reset_info.get("deployment_stratum"),
                        "deployment_active_models": reset_info.get(
                            "deployment_active_models"
                        ),
                        "fixed_deployment_index": reset_info.get(
                            "fixed_deployment_index"
                        ),
                    },
                )
        if on_rollout_step is not None:
            on_rollout_step(
                update,
                completed_periods if target_periods is not None else len(records),
            )
        if target_periods is not None and completed_periods >= target_periods:
            break
        if (
            target_periods is None
            and is_composition
            and len(records) >= max(1, rollout_steps)
        ):
            break

    collection_time_s = time.perf_counter() - collection_started

    intrinsic_raw = np.zeros(len(records), dtype=np.float32)
    intrinsic_normalized = np.zeros(len(records), dtype=np.float32)
    rnd_states = None
    deployment_indices = [
        i
        for i, r in enumerate(records)
        if r["phase"] < AgentOrchestrationEnv.COMPOSITION
        and r["policy_action_active"]
    ]
    if rnd is not None and deployment_indices:
        assert rnd_moments is not None
        states = [records[i]["next_observation"] for i in deployment_indices]
        rnd_states = _rnd_state_inputs(states, device)
        raw = rnd.intrinsic_reward(rnd_states).detach().cpu().numpy()
        phase_ids = np.asarray(
            [records[i]["phase"] for i in deployment_indices], dtype=np.int64
        )
        intrinsic_raw[deployment_indices] = raw
        intrinsic_normalized[deployment_indices] = rnd_moments.scale_by_std(
            raw, phase_ids, config.rnd_reward_clip
        )
        rnd_moments.update(raw, phase_ids)
        for index, value_intrinsic in zip(
            deployment_indices, intrinsic_normalized[deployment_indices]
        ):
            records[index]["reward"] += exploration_weight * float(value_intrinsic)

    icm_raw = np.zeros(len(records), dtype=np.float32)
    current_icm_states = None
    next_icm_states = None
    icm_actions = None
    if icm is not None:
        current_icm_states = torch.as_tensor(
            np.stack([r["observation"]["features"] for r in records]),
            dtype=torch.float32,
            device=device,
        )
        next_icm_states = torch.as_tensor(
            np.stack([r["next_observation"]["features"] for r in records]),
            dtype=torch.float32,
            device=device,
        )
        icm_actions = torch.as_tensor(
            np.stack(
                [
                    structured_action_vector(
                        r["action"], r["phase"], env.layout.deployment_action_size
                    )
                    for r in records
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        icm_raw = (
            icm.intrinsic_reward(current_icm_states, icm_actions, next_icm_states)
            .detach()
            .cpu()
            .numpy()
        )
        for r, intrinsic in zip(records, icm_raw):
            r["reward"] += config.icm_scale * float(intrinsic)

    composition_indices = [
        i for i, r in enumerate(records) if r["phase"] == AgentOrchestrationEnv.COMPOSITION
    ]
    return RolloutBatch(
        records=records,
        composition_indices=composition_indices,
        deployment_indices=deployment_indices,
        completed_periods=completed_periods,
        episode_seeds=episode_seeds,
        final_observation=observation,
        episode_counter=episode_counter,
        exploration_weight=exploration_weight,
        collection_time_s=collection_time_s,
        intrinsic_raw=intrinsic_raw,
        intrinsic_normalized=intrinsic_normalized,
        icm_raw=icm_raw,
        rnd_states=rnd_states,
        current_icm_states=current_icm_states,
        next_icm_states=next_icm_states,
        icm_actions=icm_actions,
        model_share_samples=model_share_samples,
    )


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


__all__ = [
    "CheckpointCallback",
    "EpisodeCallback",
    "OptimizationProgressCallback",
    "PhaseCallback",
    "RolloutBatch",
    "RolloutProgressCallback",
    "UpdateCallback",
    "collect_rollout",
    "_observation_to_tensors",
    "_rnd_state_inputs",
    "_stack_actions",
    "_stack_observations",
]

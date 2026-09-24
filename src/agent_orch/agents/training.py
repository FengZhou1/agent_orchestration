"""The structured PPO training loop: collect a rollout, optimize, record, checkpoint."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from agent_orch.envs import AgentOrchestrationEnv
from .config import PPOConfig
from .device import resolve_device
from .icm import ICMModule
from .networks import StructuredActorCritic
from .ppo import (
    _exploration_weight,
    _validate_constraint_config,
    build_update_record,
    optimize_ppo,
)
from .rnd import PhaseRunningMoments, RNDModule
from .rollout import (
    CheckpointCallback,
    OptimizationProgressCallback,
    PhaseCallback,
    RolloutProgressCallback,
    UpdateCallback,
    _observation_to_tensors,
    collect_rollout,
)


def train_ppo(
    env: AgentOrchestrationEnv,
    updates: int,
    rollout_steps: int,
    seed: int = 0,
    config: PPOConfig = PPOConfig(),
    device: str = "cpu",
    rollout_periods: int | None = None,
    on_phase: PhaseCallback | None = None,
    on_rollout_step: RolloutProgressCallback | None = None,
    on_optimization_step: OptimizationProgressCallback | None = None,
    on_update: UpdateCallback | None = None,
    resume_state: dict[str, Any] | None = None,
    initial_policy_state_dict: dict[str, Any] | None = None,
    on_checkpoint: CheckpointCallback | None = None,
    run_logger: Any | None = None,
    phase_schedule: tuple[tuple[str, int], ...] | None = None,
) -> tuple[StructuredActorCritic, list[dict[str, float]]]:
    """Train the structured policy and return it with its per-update history.

    ``run_logger`` is used structurally: any object exposing ``log_update`` and
    ``log_episode`` is accepted, so no telemetry backend is imported here.
    """

    device = resolve_device(device)
    config = config.for_constraint_count(len(env.constraint_names))
    _validate_constraint_config(config)
    if phase_schedule is not None:
        if not phase_schedule or any(
            phase not in ("composition", "deployment", "joint") or count <= 0
            for phase, count in phase_schedule
        ) or sum(count for _, count in phase_schedule) != updates:
            raise ValueError("phase_schedule must contain positive phases totaling updates")

    def phase_at(update_index: int) -> str:
        if phase_schedule is None:
            return config.training_phase
        remainder = update_index
        for phase, count in phase_schedule:
            if remainder < count:
                return phase
            remainder -= count
        return phase_schedule[-1][0]
    log_update = getattr(run_logger, "log_update", None)
    log_episode = getattr(run_logger, "log_episode", None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    env.gamma = config.gamma
    policy = StructuredActorCritic(env, config).to(device)
    if initial_policy_state_dict is not None and resume_state is None:
        policy.load_state_dict(initial_policy_state_dict)
    resumed_next_update = int(resume_state.get("next_update", 0)) if resume_state else 0
    optimizer_phase = phase_at(max(0, resumed_next_update - 1))
    policy.set_training_phase(optimizer_phase)
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer_learning_rate = (
        config.composition_learning_rate
        if optimizer_phase == "composition"
        else config.learning_rate
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=optimizer_learning_rate)
    rnd = (
        RNDModule(
            env.observation_space["features"].shape[0] + 3,
            config.hidden_size,
            config.rnd_feature_size,
        ).to(device)
        if config.exploration_mode == "rnd"
        else None
    )
    rnd_optimizer = (
        torch.optim.Adam(rnd.predictor.parameters(), lr=config.rnd_learning_rate)
        if rnd is not None
        else None
    )
    action_vector_size = (
        2 + env.action_space["deploy"].n + env.layout.model_action_size
    )
    icm = (
        ICMModule(env.observation_space["features"].shape[0], action_vector_size).to(
            device
        )
        if config.exploration_mode == "icm"
        else None
    )
    icm_optimizer = (
        torch.optim.Adam(icm.parameters(), lr=config.learning_rate)
        if icm is not None
        else None
    )
    rnd_moments = PhaseRunningMoments(2)
    history: list[dict[str, float]] = []
    lagrange_multipliers = np.asarray(
        config.initial_lagrange_multipliers, dtype=np.float64
    )
    episode_counter = 0
    start_update = 0

    if resume_state is not None:
        if resume_state.get("phase_schedule") != phase_schedule:
            raise ValueError("The checkpoint belongs to a different PPO phase schedule")
        if resume_state.get("environment_reward_state") is not None:
            env.restore_reward_state(resume_state["environment_reward_state"])
        policy.load_state_dict(resume_state["policy_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if rnd is not None and resume_state.get("rnd_state_dict") is not None:
            rnd.load_state_dict(resume_state["rnd_state_dict"])
            assert rnd_optimizer is not None
            rnd_optimizer.load_state_dict(resume_state["rnd_optimizer_state_dict"])
        if icm is not None and resume_state.get("icm_state_dict") is not None:
            icm.load_state_dict(resume_state["icm_state_dict"])
            assert icm_optimizer is not None
            icm_optimizer.load_state_dict(resume_state["icm_optimizer_state_dict"])
        moments = resume_state.get("rnd_moments")
        if moments is not None:
            rnd_moments.count = np.asarray(moments["count"], dtype=np.float64)
            rnd_moments.mean = np.asarray(moments["mean"], dtype=np.float64)
            rnd_moments.m2 = np.asarray(moments["m2"], dtype=np.float64)
        history = [dict(record) for record in resume_state.get("history", [])]
        lagrange_multipliers = np.asarray(
            resume_state.get("lagrange_multipliers", lagrange_multipliers),
            dtype=np.float64,
        )
        episode_counter = int(resume_state.get("episode_counter", 0))
        start_update = int(resume_state.get("next_update", 0))
        if start_update < 0 or start_update > updates:
            raise ValueError("The PPO checkpoint has an invalid next_update value")

    if hasattr(env, "training_phase"):
        env.training_phase = phase_at(start_update if start_update < updates else max(0, updates - 1))
    observation, _ = env.reset(seed=seed + episode_counter)
    if resume_state is not None:
        if "python_random_state" in resume_state:
            random.setstate(resume_state["python_random_state"])
        if "numpy_random_state" in resume_state:
            np.random.set_state(resume_state["numpy_random_state"])
        if "torch_random_state" in resume_state:
            torch.set_rng_state(resume_state["torch_random_state"].cpu())
        if (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_state["cuda_random_state"]]
            )

    for update in range(start_update, updates):
        phase = phase_at(update)
        if phase != optimizer_phase:
            policy.set_training_phase(phase)
            trainable_parameters = [
                parameter for parameter in policy.parameters() if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=(config.composition_learning_rate if phase == "composition" else config.learning_rate),
            )
            optimizer_phase = phase
            if hasattr(env, "training_phase"):
                env.training_phase = phase
                observation, _ = env.reset(seed=seed + episode_counter)
        stage_config = replace(config, training_phase=phase)
        if on_phase is not None:
            on_phase(update, "collecting")

        def report_episode(episode_index: int, info: dict[str, Any]) -> None:
            if log_episode is not None:
                log_episode(update, episode_index, info)

        batch = collect_rollout(
            policy,
            env,
            update=update,
            rollout_steps=rollout_steps,
            rollout_periods=rollout_periods,
            config=stage_config,
            device=device,
            exploration_weight=_exploration_weight(config, update, updates),
            lagrange_multipliers=lagrange_multipliers,
            observation=observation,
            seed=seed,
            episode_counter=episode_counter,
            rnd=rnd,
            icm=icm,
            rnd_moments=rnd_moments,
            on_rollout_step=on_rollout_step,
            on_episode=report_episode if log_episode is not None else None,
        )
        observation = batch.final_observation
        episode_counter = batch.episode_counter
        bootstrap_value = 0.0
        if batch.records and not batch.records[-1]["terminal"]:
            with torch.no_grad():
                bootstrap_value = float(
                    policy.value(
                        _observation_to_tensors(observation, device, batched=False)
                    ).item()
                )
        losses, _entropy = optimize_ppo(
            policy,
            batch,
            optimizer,
            stage_config,
            device,
            update=update,
            bootstrap_value=bootstrap_value,
            trainable_parameters=trainable_parameters,
            rnd=rnd,
            rnd_optimizer=rnd_optimizer,
            icm=icm,
            icm_optimizer=icm_optimizer,
            on_phase=on_phase,
            on_optimization_step=on_optimization_step,
        )
        record, next_lagrange = build_update_record(
            update=update,
            batch=batch,
            config=stage_config,
            env=env,
            losses=losses,
            lagrange_multipliers=lagrange_multipliers,
        )
        history.append(record)
        record["training_phase"] = phase
        if on_update is not None:
            on_update(record)
        lagrange_multipliers = next_lagrange
        if log_update is not None:
            log_update(update, record)
        log_action = getattr(run_logger, "log_action_distribution", None)
        if log_action is not None and getattr(batch, "model_share_samples", None):
            log_action(update, batch.model_share_samples[-1])
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "next_update": update + 1,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_training_phase": optimizer_phase,
                    "phase_schedule": phase_schedule,
                    "rnd_state_dict": rnd.state_dict() if rnd is not None else None,
                    "rnd_optimizer_state_dict": (
                        rnd_optimizer.state_dict() if rnd_optimizer is not None else None
                    ),
                    "icm_state_dict": icm.state_dict() if icm is not None else None,
                    "icm_optimizer_state_dict": (
                        icm_optimizer.state_dict() if icm_optimizer is not None else None
                    ),
                    "rnd_moments": {
                        "count": rnd_moments.count.copy(),
                        "mean": rnd_moments.mean.copy(),
                        "m2": rnd_moments.m2.copy(),
                    },
                    "lagrange_multipliers": lagrange_multipliers.copy(),
                    "episode_counter": episode_counter,
                    "environment_reward_state": (
                        env.reward_state() if hasattr(env, "reward_state") else None
                    ),
                    "history": list(history),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": (
                        torch.cuda.get_rng_state_all()
                        if str(device).startswith("cuda") and torch.cuda.is_available()
                        else None
                    ),
                }
            )
    return policy, history


__all__ = ["train_ppo"]

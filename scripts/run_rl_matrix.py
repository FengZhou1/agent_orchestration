from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
from statistics import mean
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from agent_orch.agents import (
    PPOConfig,
    StructuredActorCritic,
    TrainingProgressReporter,
    device_metadata,
    resolve_device,
    train_ppo,
)
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv
from agent_orch.metrics import summarize_slot_metrics
from agent_orch.objective import PROFILES, ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.telemetry import build_run_logger
from agent_orch.workload import ArrivalTrace


ENVIRONMENTS = {
    "joint": AgentOrchestrationEnv,
    "deploy": DeploymentOnlyEnv,
    "route": RoutingOnlyEnv,
}

DEFAULT_SEEDS = "0"
DEFAULT_MODES = "joint"
DEFAULT_VARIANTS = "rnd"

_LEGACY_RUN_SPEC_DEFAULTS = {
    "rollout_periods": 0,
    "train_mapping_samples": 4096,
    "eval_mapping_samples": 4096,
    "update_epochs": 10,
    "minibatch_size": 256,
    "validation_interval": 5,
    "validation_periods": 20,
    "validation_warmup_periods": 5,
    "validation_mapping_samples": 128,
    "training_phase": "joint",
    "initial_policy": None,
}


def _replace_with_retry(temporary: Path, target: Path) -> None:
    """Atomically publish an artifact despite transient Windows reader locks."""

    last_error: PermissionError | None = None
    for attempt in range(12):
        try:
            temporary.replace(target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.025 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        _replace_with_retry(temporary, path)
    except PermissionError:
        # JSON status files may be inspected while the experiment is running.
        # An in-place fallback preserves progress on Windows when a reader holds
        # the destination open across the retry window.
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    serializable = frame.copy()
    for column in serializable.columns:
        values = serializable[column].dropna()
        if any(isinstance(value, (dict, list, tuple)) for value in values):
            serializable[column] = serializable[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    serializable.to_parquet(temporary, index=False)
    _replace_with_retry(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.pt")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"rl:{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arrival_trace(scenario, slots: int, rate_scale: float, args=None, realization: int = 0):
    """Build the arrival trace.

    A stationary intensity leaves the deployment with one constant optimum, so a
    deployment policy has nothing to react to; the bursty pattern alternates a low
    and a high level and is what makes the deployment decision time-varying.
    """

    pattern = getattr(args, "arrival_pattern", "bursty") if args is not None else "bursty"
    if pattern == "stationary":
        return ArrivalTrace.stationary_poisson_intensity(
            scenario, slots, rate_scale=rate_scale
        )
    if pattern == "mix":
        # Every (application, ingress) group drawn independently, so the *mix*
        # moves and not just the total.  With the total alone the composition
        # optimum barely moves, and a policy trained on it is load-blind: it emits
        # identical shares at 0.4x and 2.0x.  Values are constant within a block of
        # slots, the granularity a deployment decision can act at.
        return ArrivalTrace.randomized_mix_intensity(
            scenario,
            slots,
            base_scale=rate_scale,
            seed=int(getattr(args, "mix_seed", 0) or 0) + realization,
            block=max(1, int(getattr(args, "mix_block", 4) or 4)),
            per_group_sigma=float(
                getattr(args, "mix_sigma", None)
                if getattr(args, "mix_sigma", None) is not None
                else 0.6
            ),
        )
    # A scenario may carry its own burst definition (the arrival-burst stress
    # variant does); explicit CLI values win over it.
    scenario_burst = dict(scenario.metadata.get("arrival_burst", {}) or {})
    low_fraction = float(
        getattr(args, "burst_low_fraction", None)
        if getattr(args, "burst_low_fraction", None) is not None
        else scenario_burst.get("low_fraction", 0.4)
    )
    # ``realization`` makes training, validation and evaluation independent
    # realisations of the same bursty process, so validation measures adaptation
    # to a different trace instead of repeating one fixed operating point.
    return ArrivalTrace.gaussian_burst_intensity(
        scenario,
        slots,
        base_scale=rate_scale * low_fraction,
        burst_scale=rate_scale * (1.0 - low_fraction),
        period=int(getattr(args, "burst_period", None) or scenario_burst.get("period_slots", 60)),
        sigma=float(getattr(args, "burst_sigma", None) or scenario_burst.get("sigma_slots", 8.0)),
        phase=float(getattr(args, "burst_phase", None) or scenario_burst.get("phase", 0.25)),
        seed=int(getattr(args, "seed", 0)) * 101 + realization,
        jitter=float(getattr(args, "burst_jitter", None) or scenario_burst.get("jitter", 0.05)),
    )


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _run_specs_match(saved: dict, requested: dict) -> bool:
    normalized = {**_LEGACY_RUN_SPEC_DEFAULTS, **saved}
    return normalized == requested


def _evaluate(
    env: AgentOrchestrationEnv,
    policy: StructuredActorCritic,
    seed: int,
    device: str = "cpu",
    on_step=None,
) -> tuple[list[dict], list[float]]:
    records: list[dict] = []
    decision_times: list[float] = []
    if isinstance(env, RoutingOnlyEnv) and len(env._fixed_deployment_catalog) > 1:
        context_count = min(3, len(env._fixed_deployment_catalog))
        deployment_indices = np.linspace(
            0,
            len(env._fixed_deployment_catalog) - 1,
            context_count,
            dtype=int,
        ).tolist()
    else:
        deployment_indices = [None]
    for context_offset, deployment_index in enumerate(deployment_indices):
        options = (
            {"fixed_deployment_index": deployment_index}
            if deployment_index is not None
            else None
        )
        observation, _ = env.reset(seed=seed + context_offset, options=options)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            started = time.perf_counter()
            action, _, _ = policy.act(observation, deterministic=True, device=device)
            decision_times.append(time.perf_counter() - started)
            observation, _, terminated, truncated, info = env.step(action)
            if "metrics" in info:
                metric = asdict(info["metrics"])
                metric["_validation_context"] = context_offset
                records.append(metric)
            for metrics in info.get("interval_metrics", []):
                metric = asdict(metrics)
                metric["_validation_context"] = context_offset
                records.append(metric)
            if on_step is not None:
                on_step(len(records))
    return records, decision_times


def _summary(
    run_id: str,
    records: list[dict],
    history: list[dict],
    train_wall_time_s: float,
    decision_times_s: list[float],
    selection: dict | None = None,
) -> dict:
    if not records:
        raise RuntimeError(f"Evaluation for {run_id} produced no physical-slot metrics")
    physical_metrics = summarize_slot_metrics(records)
    result = {
        "run_id": run_id,
        "evaluation_slots": len(records),
        **physical_metrics,
        "final_transition_reward": history[-1]["mean_reward"],
        "final_training_utility": history[-1]["mean_utility"],
        "best_training_utility": max(item["mean_utility"] for item in history),
        "final_period_return": history[-1]["mean_period_return"],
        "train_wall_time_s": train_wall_time_s,
        "mean_decision_time_ms": 1_000.0 * mean(decision_times_s),
        "p95_decision_time_ms": 1_000.0 * float(np.quantile(decision_times_s, 0.95)),
    }
    if selection:
        result.update(selection)
    return result


def _mean_evaluation_utility(
    env: AgentOrchestrationEnv,
    records: list[dict],
    warmup_periods: int = 0,
) -> float:
    if not records:
        return float("-inf")
    utilities: list[float] = []
    # The validation trace is a stationary Poisson intensity trace, so the
    # current simulator rates are the same rates used for every validation
    # period.  Keep checkpoint scoring aligned with the environment reward
    # instead of calling the removed incremental-delta interface.
    arrival_rates = env.simulator.current_arrival_rates()
    for record in records:
        utility, _ = env._normalized_utility(
            record["cost"],
            record["mean_latency_s"],
            record["slo_attainment"],
            record["quality"],
            record.get("app_latency_s", {}),
            arrival_rates,
        )
        utilities.append(utility)
    context_ids = [record.get("_validation_context", 0) for record in records]
    scored: list[float] = []
    for context_id in dict.fromkeys(context_ids):
        context_utilities = [
            utility
            for utility, record_context in zip(utilities, context_ids)
            if record_context == context_id
        ]
        scored.extend(context_utilities[max(0, int(warmup_periods)) :])
    return float(np.mean(scored)) if scored else float("-inf")


def _combinations(modes: list[str], variants: list[str]) -> list[tuple[str, str]]:
    if variants == ["auto"]:
        combinations: list[tuple[str, str]] = []
        for mode in modes:
            if mode == "joint":
                combinations.extend(
                    [
                        (mode, "rnd"),
                        (mode, "no-rnd"),
                        (mode, "unconstrained-rnd"),
                    ]
                )
            else:
                combinations.append((mode, "no-rnd"))
        return combinations
    combinations = [(mode, variant) for mode in modes for variant in variants]
    invalid = [
        (mode, variant)
        for mode, variant in combinations
        if mode != "joint" and variant != "no-rnd"
    ]
    if invalid:
        raise ValueError(
            "Deployment-only and routing-only ablations use the no-rnd variant: "
            f"{invalid}"
        )
    return combinations


def _resolve_training_library(args, scenario):
    """Load the deployment library and select the requested split.

    Composition training must not see the deployments the gate scores, so the
    split is applied here and the resulting library is handed to every
    environment the run builds.
    """

    from agent_orch.deployment import DeploymentLibrary

    path = (
        Path(args.deployment_library)
        if args.deployment_library
        else DeploymentLibrary.default_path(scenario.id)
    )
    if not path.exists():
        if args.deployment_library:
            raise FileNotFoundError(
                f"Deployment library {path} not found; run scripts/build_deployment_library.py"
            )
        return None
    library = DeploymentLibrary.load(path)
    if args.library_split == "all":
        return library
    train_library, test_library = library.train_test_split(
        test_fraction=args.library_test_fraction,
        seed=args.library_split_seed,
        stratify=True,
    )
    return train_library if args.library_split == "train" else test_library


def _variant_config(variant: str) -> tuple[PPOConfig, bool]:
    # Potential shaping is on by default: with it off every deployment sub-step
    # carries zero immediate reward, so all ~105 of them share one advantage and
    # the policy cannot tell which decision helped. The "potential" variant stays
    # as an explicit name for the same setting.
    if variant == "rnd":
        return PPOConfig(constrained=True, exploration_mode="rnd"), True
    if variant == "no-rnd":
        return PPOConfig(constrained=True, exploration_mode="none"), True
    if variant == "unconstrained-rnd":
        return PPOConfig(constrained=False, exploration_mode="rnd"), True
    if variant == "icm":
        return PPOConfig(constrained=True, exploration_mode="icm"), True
    if variant == "potential":
        return PPOConfig(constrained=True, exploration_mode="none"), True
    raise ValueError(f"Unknown RL variant: {variant}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the structural PPO and reward-component matrix."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help="default: rnd; use auto only when running the full ablation matrix",
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument(
        "--rollout-periods",
        type=int,
        default=16,
        help=(
            "complete orchestration periods collected per PPO update; set to 0 "
            "to retain the legacy transition-count budget"
        ),
    )
    parser.add_argument(
        "--rollout-contexts",
        type=int,
        default=2,
        help=(
            "fixed deployment contexts collected per PPO update in route mode; "
            "each context uses rollout-periods"
        ),
    )
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument(
        "--arrival-pattern",
        default="bursty",
        choices=["stationary", "bursty", "mix"],
        help=(
            "stationary keeps one constant optimum; bursty alternates a low and a "
            "high intensity; mix draws every (application, ingress) group "
            "independently so the load *composition* moves, which is what gives a "
            "composition policy something to condition on"
        ),
    )
    parser.add_argument(
        "--mix-seed",
        type=int,
        default=0,
        help="seed of the mix realisation; train / validation / holdout families use disjoint seeds",
    )
    parser.add_argument(
        "--mix-block",
        type=int,
        default=4,
        help="slots a mix vector is held for; align with --deployment-periods",
    )
    parser.add_argument(
        "--mix-sigma", type=float, default=None, help="per-group lognormal sigma (default 0.6)"
    )
    parser.add_argument("--burst-period", type=int, default=None, help="default: scenario metadata, else 60")
    parser.add_argument("--burst-sigma", type=float, default=None, help="default: scenario metadata, else 8.0")
    parser.add_argument("--burst-phase", type=float, default=None, help="default: scenario metadata, else 0.25")
    parser.add_argument("--burst-jitter", type=float, default=None, help="default: scenario metadata, else 0.05")
    parser.add_argument("--burst-low-fraction", type=float, default=None, help="default: scenario metadata, else 0.4")
    parser.add_argument(
        "--deployment-periods",
        type=int,
        default=1,
        help="T^dep: the deployment may only change every N orchestration periods",
    )
    parser.add_argument(
        "--composition-group-features",
        action="store_true",
        help=(
            "feed the composition head the features of the group it is deciding "
            "for (SLO type and thresholds, per-model quality, token demand, "
            "arrival rate). With a shared hidden vector the per-group decision "
            "can only differ through separate rows of one weight matrix, so the "
            "policy has to learn the head-index-to-observation association"
        ),
    )
    parser.add_argument(
        "--factorized-credit",
        action="store_true",
        help=(
            "credit each (application, ingress) composition group with its own "
            "application's utility instead of one scalar shared by all of them. "
            "The objective is a sum over applications and the cost term is "
            "action-independent, so the decomposition is unbiased for the "
            "policy gradient while giving each sample ~20 rewards instead of one"
        ),
    )
    parser.add_argument(
        "--composition-fixed-concentration",
        type=float,
        default=None,
        help=(
            "pin the Dirichlet's total concentration so the action's mean is "
            "softmax(logits) and its spread is constant. The composition is "
            "scored at its mean, so a free concentration is a nuisance knob that "
            "lets the policy raise the sampled reward without improving the mean"
        ),
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help=(
            "stop reusing a rollout once the policy's mean KL from the sampling "
            "policy exceeds this value (PPO's standard safeguard)"
        ),
    )
    parser.add_argument(
        "--lagrangian-learning-rate",
        type=float,
        default=None,
        help=(
            "dual ascent step for every constraint. The default of 0.05 makes "
            "the multiplier grow by lr*c per update; when a stage cannot remove "
            "the violation it diverges linearly and the penalty then dwarfs the "
            "utility it is supposed to bound"
        ),
    )
    parser.add_argument(
        "--max-lagrange-multiplier",
        type=float,
        default=None,
        help="cap on every multiplier, so an unsatisfiable constraint cannot dominate",
    )
    parser.add_argument(
        "--baseline-periods",
        type=int,
        default=None,
        help=(
            "periods the uniform control variate is averaged over; 0 matches the "
            "episode length. The system reaches its fixed point within a few "
            "periods, so a short window is equivalent and much cheaper"
        ),
    )
    parser.add_argument(
        "--baseline-warmup",
        type=int,
        default=None,
        help="periods discarded before averaging the control variate",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help=(
            "drop the Lagrangian penalty and optimise the raw utility. A stage "
            "whose deployment is exogenous cannot remove a deployment-level "
            "utilization violation, so the dual variable grows until the penalty "
            "dwarfs the objective"
        ),
    )
    parser.add_argument(
        "--composition-group-relative",
        action="store_true",
        help=(
            "normalise composition advantages within each episode. A rollout spans "
            "several exogenous deployment contexts, so batch-wide normalisation "
            "encodes mostly 'which context am I in' rather than what the "
            "composition decision changed"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override PPO Adam learning rate",
    )
    parser.add_argument(
        "--composition-learning-rate",
        type=float,
        default=None,
        help="override the composition-stage PPO Adam learning rate",
    )
    parser.add_argument(
        "--composition-concentration-min",
        type=float,
        default=None,
        help="minimum Dirichlet concentration for composition actions",
    )
    parser.add_argument(
        "--composition-entropy-coefficient",
        type=float,
        default=None,
        help="entropy coefficient used by the composition action head",
    )
    parser.add_argument(
        "--composition-gamma",
        type=float,
        default=None,
        help="discount used only by composition training (default 0 for route mode)",
    )
    parser.add_argument(
        "--composition-deployment-index",
        type=int,
        default=None,
        help="fix RoutingOnlyEnv to one catalog deployment for diagnostic runs",
    )
    parser.add_argument(
        "--shared-composition-head",
        action="store_true",
        help="use one shared model-composition head for all application groups",
    )
    parser.add_argument(
        "--train-mapping-samples",
        type=int,
        default=128,
        help="physical execution mappings sampled per pattern flow during training",
    )
    parser.add_argument(
        "--eval-mapping-samples",
        type=int,
        default=512,
        help="physical execution mappings sampled per pattern flow during evaluation",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=5,
        help="PPO updates between deterministic policy-selection evaluations",
    )
    parser.add_argument(
        "--validation-periods",
        type=int,
        default=20,
        help="scored orchestration periods in each policy-selection evaluation",
    )
    parser.add_argument(
        "--validation-warmup-periods",
        type=int,
        default=5,
        help="unscored warm-up periods before each policy-selection evaluation",
    )
    parser.add_argument(
        "--validation-mapping-samples",
        type=int,
        default=128,
        help="physical mappings sampled during policy-selection evaluation",
    )
    parser.add_argument("--train-slots", type=int, default=600)
    parser.add_argument("--eval-slots", type=int, default=600)
    parser.add_argument("--arrival-scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="training device: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--status-interval-steps",
        type=int,
        default=32,
        help="work steps between training_status.json updates",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the terminal progress bar; live files are still written",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume PPO updates from checkpoints and skip completed runs",
    )
    parser.add_argument(
        "--training-phase",
        choices=("auto", "joint", "deployment", "composition"),
        default="auto",
        help=(
            "actor branch updated by PPO; auto maps joint/deploy/route modes to "
            "joint/deployment/composition"
        ),
    )
    parser.add_argument(
        "--initial-policy",
        default=None,
        help="policy.pt or policy_final.pt used to initialize a curriculum stage",
    )
    parser.add_argument("--output", default="results/rl_matrix")
    parser.add_argument(
        "--objective-profile",
        default="slo_constrained",
        choices=list(PROFILES),
        help=(
            "slo_constrained keeps SLO attainment as a constraint and optimises "
            "quality/cost/latency; legacy reproduces the four-term equal-weight sum"
        ),
    )
    parser.add_argument("--attainment-target", type=float, default=0.9)
    parser.add_argument(
        "--no-attainment-constraint",
        action="store_true",
        help=(
            "drop the attainment constraint. Required when the deployment is "
            "exogenous (composition-only training): attainment is bounded by the "
            "deployment, so the dual variable would chase an unremovable violation"
        ),
    )
    parser.add_argument(
        "--deployment-library",
        default=None,
        help="path to a deployment library JSON; defaults to data/processed/<scenario>",
    )
    parser.add_argument(
        "--library-sampler",
        default="cycle",
        choices=["cycle", "uniform", "fixed"],
        help="how composition training draws deployment contexts from the library",
    )
    parser.add_argument(
        "--library-split",
        default="train",
        choices=["all", "train", "test"],
        help=(
            "which side of the stratified deployment-library split to train on; "
            "the gate is scored on 'test' so the two must not overlap"
        ),
    )
    parser.add_argument("--library-test-fraction", type=float, default=0.25)
    parser.add_argument("--library-split-seed", type=int, default=2026)
    parser.add_argument(
        "--potential-cost-weight",
        type=float,
        default=0.05,
        help="cost share of the deployment potential shaping term (0 disables it)",
    )
    parser.add_argument(
        "--telemetry",
        default="both",
        choices=["none", "tensorboard", "swanlab", "both"],
        help="which experiment trackers to write",
    )
    parser.add_argument("--swanlab-project", default="agent-orch")
    parser.add_argument(
        "--swanlab-online",
        action="store_true",
        help="upload to SwanLab instead of writing an offline run",
    )
    args = parser.parse_args()

    if args.rollout_periods < 0:
        raise ValueError("--rollout-periods must be non-negative")
    if args.rollout_contexts <= 0:
        raise ValueError("--rollout-contexts must be positive")
    if args.update_epochs <= 0 or args.minibatch_size <= 0:
        raise ValueError("PPO update epochs and minibatch size must be positive")
    if (
        args.train_mapping_samples <= 0
        or args.eval_mapping_samples <= 0
        or args.validation_mapping_samples <= 0
    ):
        raise ValueError("Physical mapping sample counts must be positive")
    if (
        args.validation_interval <= 0
        or args.validation_periods <= 0
        or args.validation_warmup_periods < 0
    ):
        raise ValueError("Validation interval and periods must be positive")

    objective_spec = (
        ObjectiveSpec.slo_constrained(
            None if args.no_attainment_constraint else args.attainment_target
        )
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    print(
        "Objective profile: %s (constraints: %s)"
        % (objective_spec.profile, ", ".join(objective_spec.constraint_names)),
        flush=True,
    )
    # Validation and evaluation must score with the same objective as training,
    # otherwise checkpoint selection optimises something the run does not report.
    scoring_kwargs: dict[str, object] = {
        "objective": objective_spec,
        "deployment_periods": args.deployment_periods,
    }
    library_kwargs: dict[str, object] = {}

    initial_policy_path = (
        Path(args.initial_policy).resolve() if args.initial_policy else None
    )
    initial_policy_state = None
    if initial_policy_path is not None:
        if not initial_policy_path.exists():
            raise FileNotFoundError(initial_policy_path)
        initial_payload = torch.load(
            initial_policy_path, map_location="cpu", weights_only=False
        )
        initial_policy_state = initial_payload.get(
            "policy_state_dict", initial_payload
        )

    modes = _parse_csv(args.modes)
    variants = _parse_csv(args.variants)
    unknown_modes = set(modes) - set(ENVIRONMENTS)
    unknown_variants = set(variants) - {
        "auto",
        "rnd",
        "no-rnd",
        "unconstrained-rnd",
        "potential",
        "icm",
    }
    if unknown_modes or unknown_variants:
        raise ValueError(
            f"Unknown modes={sorted(unknown_modes)} variants={sorted(unknown_variants)}"
        )

    scenario_path = Path(args.scenario).resolve()
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16]
    scenario = ScenarioLoader.load(scenario_path)
    training_library = _resolve_training_library(args, scenario)
    if training_library is not None:
        scoring_kwargs["deployment_library"] = training_library
        print(
            "Deployment library: %d entries (split=%s)"
            % (len(training_library), args.library_split),
            flush=True,
        )
    elif args.deployment_library:
        scoring_kwargs["deployment_library_path"] = args.deployment_library
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger(output / "rl_experiment.log")
    resolved_device = resolve_device(args.device)
    hardware = device_metadata(args.device, resolved_device)
    combinations = _combinations(modes, variants)
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    print(
        f"RL plan: seeds={_parse_csv(args.seeds)}, "
        f"combinations={combinations}, device={resolved_device}",
        flush=True,
    )
    logger.info(
        "RL plan seeds=%s combinations=%s device=%s", seeds, combinations, resolved_device
    )
    slot_records: list[dict] = []
    run_summaries: list[dict] = []
    total_runs = len(seeds) * len(combinations)
    completed_runs = 0
    matrix_started = time.perf_counter()
    status_path = output / "matrix_status.json"
    matrix_bar = tqdm(
        total=total_runs,
        desc="RL matrix",
        unit="run",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    manifest = {
        "implementation_revision": 4,
        "scenario": str(scenario_path),
        "scenario_hash": scenario_hash,
        "seeds": seeds,
        "modes": modes,
        "variants": variants,
        "combinations": combinations,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "rollout_periods": args.rollout_periods,
        "rollout_contexts": args.rollout_contexts,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "train_mapping_samples": args.train_mapping_samples,
        "eval_mapping_samples": args.eval_mapping_samples,
        "validation_interval": args.validation_interval,
        "validation_periods": args.validation_periods,
        "validation_warmup_periods": args.validation_warmup_periods,
        "validation_mapping_samples": args.validation_mapping_samples,
        "train_slots": args.train_slots,
        "eval_slots": args.eval_slots,
        "arrival_process": "stationary_poisson_intensity",
        "arrival_scale": args.arrival_scale,
                "arrival_pattern": args.arrival_pattern,
                "burst_period": args.burst_period,
                "burst_sigma": args.burst_sigma,
                "burst_phase": args.burst_phase,
                "burst_jitter": args.burst_jitter,
                "burst_low_fraction": args.burst_low_fraction,
                "mix_seed": args.mix_seed,
                "mix_block": args.mix_block,
                "mix_sigma": args.mix_sigma,
                "deployment_periods": args.deployment_periods,
        "device": hardware,
        "status_interval_steps": args.status_interval_steps,
        "resume_enabled": args.resume,
        "training_phase": args.training_phase,
        "initial_policy": str(initial_policy_path) if initial_policy_path else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    _atomic_json(output / "manifest.json", manifest)

    def write_matrix_status(
        status: str, current: str | None = None, error: str | None = None
    ) -> None:
        elapsed = time.perf_counter() - matrix_started
        rate = completed_runs / elapsed if elapsed > 0.0 else 0.0
        payload = {
            "status": status,
            "current_run": current,
            "completed_runs": completed_runs,
            "total_runs": total_runs,
            "progress_percent": 100.0 * completed_runs / max(total_runs, 1),
            "elapsed_time_s": elapsed,
            "eta_seconds": (total_runs - completed_runs) / rate if rate > 0 else None,
            "updated_at_utc": _timestamp(),
        }
        if error is not None:
            payload["error"] = error
        _atomic_json(status_path, payload)

    def write_aggregates() -> None:
        if slot_records:
            _atomic_parquet(pd.DataFrame(slot_records), output / "slot_metrics.parquet")
        if run_summaries:
            _atomic_parquet(pd.DataFrame(run_summaries), output / "run_summary.parquet")

    write_matrix_status("running")
    try:
        for seed, mode, variant in (
            (seed, mode, variant)
            for seed in seeds
            for mode, variant in combinations
        ):
            config, potential_shaping = _variant_config(variant)
            training_phase = (
                {"joint": "joint", "deploy": "deployment", "route": "composition"}[
                    mode
                ]
                if args.training_phase == "auto"
                else args.training_phase
            )
            config = replace(
                config,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                training_phase=training_phase,
                **{
                    key: value
                    for key, value in {
                        "learning_rate": args.learning_rate,
                        "composition_learning_rate": args.composition_learning_rate,
                        "composition_concentration_min": args.composition_concentration_min,
                        "composition_entropy_coefficient": args.composition_entropy_coefficient,
                        "composition_gamma": args.composition_gamma,
                        "constrained": False if args.unconstrained else None,
                        "composition_fixed_concentration": (
                            args.composition_fixed_concentration
                        ),
                        "target_kl": args.target_kl,
                        "factorized_credit": args.factorized_credit,
                        "composition_group_features": args.composition_group_features,
                        "factorized_credit": (
                            True if args.factorized_credit else None
                        ),
                        "composition_group_features": (
                            True if args.composition_group_features else None
                        ),
                        "lagrangian_learning_rates": (
                            (args.lagrangian_learning_rate,) * 2
                            if args.lagrangian_learning_rate is not None
                            else None
                        ),
                        "max_lagrange_multipliers": (
                            (args.max_lagrange_multiplier,) * 2
                            if args.max_lagrange_multiplier is not None
                            else None
                        ),
                        "composition_group_relative_advantages": (
                            True if args.composition_group_relative else None
                        ),
                        "shared_composition_head": (
                            True if args.shared_composition_head else None
                        ),
                    }.items()
                    if value is not None
                },
            )
            run_id = (
                f"{scenario.id}-{mode}-{variant}-{training_phase}-s{seed}-"
                f"{scenario_hash}"
            )
            run_dir = output / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run_spec = {
                "implementation_revision": 4,
                "scenario_hash": scenario_hash,
                "seed": seed,
                "mode": mode,
                "variant": variant,
                "updates": args.updates,
                "rollout_steps": args.rollout_steps,
                "rollout_periods": args.rollout_periods,
                "rollout_contexts": args.rollout_contexts,
                "update_epochs": args.update_epochs,
                "minibatch_size": args.minibatch_size,
                "learning_rate": args.learning_rate,
                "composition_learning_rate": args.composition_learning_rate,
                "composition_concentration_min": args.composition_concentration_min,
                "composition_entropy_coefficient": args.composition_entropy_coefficient,
                "composition_gamma": args.composition_gamma,
                "composition_group_relative": args.composition_group_relative,
                "unconstrained": args.unconstrained,
                "lagrangian_learning_rate": args.lagrangian_learning_rate,
                "max_lagrange_multiplier": args.max_lagrange_multiplier,
                "composition_fixed_concentration": args.composition_fixed_concentration,
                "target_kl": args.target_kl,
                "shared_composition_head": args.shared_composition_head,
                "train_mapping_samples": args.train_mapping_samples,
                "eval_mapping_samples": args.eval_mapping_samples,
                "validation_interval": args.validation_interval,
                "validation_periods": args.validation_periods,
                "validation_warmup_periods": args.validation_warmup_periods,
                "validation_mapping_samples": args.validation_mapping_samples,
                "train_slots": args.train_slots,
                "eval_slots": args.eval_slots,
                "arrival_scale": args.arrival_scale,
                "training_phase": training_phase,
                "initial_policy": (
                    str(initial_policy_path) if initial_policy_path else None
                ),
            }
            complete_path = run_dir / "run_complete.json"
            run_slots_path = run_dir / "evaluation_slots.parquet"
            run_summary_path = run_dir / "run_summary.parquet"
            if args.resume and complete_path.exists():
                completed = json.loads(complete_path.read_text(encoding="utf-8"))
                if not _run_specs_match(completed.get("run_spec", {}), run_spec):
                    raise ValueError(
                        f"Completed run {run_id} does not match the requested configuration"
                    )
                if not run_slots_path.exists() or not run_summary_path.exists():
                    raise ValueError(f"Completed run {run_id} lacks incremental artifacts")
                slot_records.extend(pd.read_parquet(run_slots_path).to_dict("records"))
                run_summaries.extend(pd.read_parquet(run_summary_path).to_dict("records"))
                completed_runs += 1
                matrix_bar.update(1)
                logger.info("resumed completed run %s", run_id)
                write_aggregates()
                write_matrix_status("running", run_id)
                continue
            env_class = ENVIRONMENTS[mode]
            train_trace = _arrival_trace(
                scenario, args.train_slots, args.arrival_scale, args, realization=0
            )
            composition_kwargs: dict[str, object] = {}
            if mode == "route":
                composition_kwargs["sampler_mode"] = args.library_sampler
                if args.baseline_periods is not None:
                    composition_kwargs["baseline_periods"] = args.baseline_periods
                if args.baseline_warmup is not None:
                    composition_kwargs["baseline_warmup"] = args.baseline_warmup
                if args.composition_deployment_index is not None:
                    composition_kwargs["fixed_deployment_index"] = (
                        args.composition_deployment_index
                    )
            train_env = env_class(
                scenario,
                # One episode is one timeline of ``--train-slots`` slots for every
                # mode.  Route training used to cap the episode at the rollout
                # length, which made an episode a container for drawing a batch of
                # contexts rather than a stretch of time; the deployment context is
                # now a property of the episode (stratified sampler) alongside the
                # trace realisation, not a replacement for the time axis.
                max_slots=args.train_slots,
                potential_shaping=potential_shaping,
                seed=seed,
                arrival_trace=train_trace,
                mapping_samples=args.train_mapping_samples,
                objective=objective_spec,
                deployment_library_path=args.deployment_library,
                potential_cost_weight=args.potential_cost_weight,
                deployment_periods=args.deployment_periods,
                trace_offset_span=len(train_trace.rates),
                **composition_kwargs,
            )
            print(f"Starting training run {run_id}", flush=True)
            logger.info("starting training run %s", run_id)
            checkpoint_path = run_dir / "checkpoint.pt"
            best_policy_path = run_dir / "policy_best.pt"
            # Checkpoint selection by the 3-context validation window is noisy and
            # does not track the held-out gate, so a second "best" is kept by the
            # smoothed training objective.  Every PPO run so far peaks well before
            # its final update, and without this the peak cannot be recovered.
            best_train_policy_path = run_dir / "policy_best_train.pt"
            best_train_utility = float("-inf")
            best_train_update = -1
            if not args.resume:
                checkpoint_path.unlink(missing_ok=True)
                best_policy_path.unlink(missing_ok=True)
                best_train_policy_path.unlink(missing_ok=True)
            resume_state = None
            previous_train_wall_time_s = 0.0
            if args.resume and checkpoint_path.exists():
                checkpoint = torch.load(
                    checkpoint_path, map_location=resolved_device, weights_only=False
                )
                if not _run_specs_match(checkpoint.get("run_spec", {}), run_spec):
                    raise ValueError(
                        f"Checkpoint for {run_id} does not match the requested configuration"
                    )
                resume_state = checkpoint["training_state"]
                previous_train_wall_time_s = float(
                    checkpoint.get("train_wall_time_s", 0.0)
                )
                logger.info(
                    "resuming %s from PPO update %s",
                    run_id,
                    resume_state.get("next_update", 0),
                )
            best_validation_utility = float("-inf")
            best_validation_update = -1
            best_validation_violation_fraction = float("inf")
            best_validation_mean_violations = float("inf")
            best_validation_key = (float("-inf"),) * 3
            if args.resume and best_policy_path.exists():
                best_payload = torch.load(
                    best_policy_path,
                    map_location=resolved_device,
                    weights_only=False,
                )
                best_validation_utility = float(
                    best_payload.get("validation_utility", float("-inf"))
                )
                best_validation_update = int(best_payload.get("update", -1))
                best_validation_violation_fraction = float(
                    best_payload.get("violation_slot_fraction", float("inf"))
                )
                best_validation_mean_violations = float(
                    best_payload.get("mean_violations", float("inf"))
                )
                best_validation_key = (
                    -best_validation_violation_fraction,
                    -best_validation_mean_violations,
                    best_validation_utility,
                )

            validation_seed = seed + 5_000
            validation_total_periods = (
                args.validation_warmup_periods + args.validation_periods
            )
            validation_trace = _arrival_trace(
                scenario, validation_total_periods, args.arrival_scale, args, realization=1
            )
            validation_env = env_class(
                scenario,
                max_slots=validation_total_periods,
                potential_shaping=False,
                seed=validation_seed,
                arrival_trace=validation_trace,
                mapping_samples=args.validation_mapping_samples,
                **scoring_kwargs,
                **(
                    {
                        "fixed_deployment_index": args.composition_deployment_index,
                        "sampler_mode": args.library_sampler,
                    }
                    if mode == "route" and args.composition_deployment_index is not None
                    else ({"sampler_mode": args.library_sampler} if mode == "route" else {})
                ),
            )
            validation_policy = StructuredActorCritic(validation_env, config).to(
                resolved_device
            )
            validation_history_path = run_dir / "validation_history.jsonl"
            if resume_state is None:
                validation_history_path.unlink(missing_ok=True)
            train_started = time.perf_counter()
            initial_update = int(resume_state.get("next_update", 0)) if resume_state else 0
            history_stream = run_dir / "training_history.jsonl"
            if resume_state is not None and history_stream.exists():
                lines = history_stream.read_text(encoding="utf-8").splitlines()
                history_stream.write_text(
                    "\n".join(lines[:initial_update])
                    + ("\n" if initial_update else ""),
                    encoding="utf-8",
                )
            reporter = TrainingProgressReporter(
                run_id=run_id,
                output_dir=run_dir,
                updates=args.updates,
                rollout_steps=(
                    args.rollout_periods
                    * (
                        args.rollout_contexts
                        if mode == "route" and args.rollout_periods > 0
                        else 1
                    )
                    if args.rollout_periods > 0
                    else args.rollout_steps
                ),
                rollout_unit=(
                    "orchestration_period"
                    if args.rollout_periods > 0
                    else "transition"
                ),
                update_epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
                device=resolved_device,
                status_interval_steps=(
                    1 if args.rollout_periods > 0 else args.status_interval_steps
                ),
                show_progress=not args.no_progress,
                initial_update=initial_update,
                append_history=bool(resume_state),
            )
            run_logger = build_run_logger(
                args.telemetry,
                output_dir=run_dir,
                # ``run_id`` encodes scenario/mode/variant/phase/seed, which are
                # identical across every ablation of the same experiment, so it
                # cannot name the run on its own.  The output directory is what
                # actually distinguishes an experiment, so it leads the name.
                run_name=f"{output.name} | {mode}-{variant}-{training_phase}-s{seed}",
                config={
                    **run_spec,
                    "objective": objective_spec.to_dict(),
                    "scenario_id": scenario.id,
                    "constraint_names": list(train_env.constraint_names),
                    "deployment_library": (
                        str(args.deployment_library)
                        if args.deployment_library
                        else (
                            str(
                                Path("data/processed")
                                / f"deployment_library_{scenario.id}.json"
                            )
                            if mode == "route"
                            else None
                        )
                    ),
                    "library_strata": (
                        train_env.stratum_counts() if mode == "route" else {}
                    ),
                },
                progress=None,
                project=args.swanlab_project,
                offline=not args.swanlab_online,
            )
            run_logger.log_config(
                {
                    "run_spec": run_spec,
                    "objective": objective_spec.to_dict(),
                    "scenario": scenario.id,
                    "constraint_names": list(train_env.constraint_names),
                }
            )

            def _smoothed_train_utility(history: list, window: int = 20) -> float:
                values = [
                    float(row["mean_learning_utility"])
                    for row in history[-window:]
                    if "mean_learning_utility" in row
                ]
                return float(mean(values)) if values else float("-inf")

            def save_checkpoint(training_state: dict) -> None:
                nonlocal best_validation_utility, best_validation_update
                nonlocal best_validation_violation_fraction
                nonlocal best_validation_mean_violations, best_validation_key
                nonlocal best_train_utility, best_train_update
                _atomic_torch(
                    checkpoint_path,
                    {
                        "run_spec": run_spec,
                        "training_state": training_state,
                        "train_wall_time_s": (
                            previous_train_wall_time_s
                            + time.perf_counter()
                            - train_started
                        ),
                        "updated_at_utc": _timestamp(),
                    },
                )
                smoothed = _smoothed_train_utility(training_state.get("history", []))
                if smoothed > best_train_utility:
                    best_train_utility = smoothed
                    best_train_update = int(training_state["next_update"])
                    _atomic_torch(
                        best_train_policy_path,
                        {
                            "policy_state_dict": {
                                key: value.detach().cpu()
                                for key, value in training_state["policy_state_dict"].items()
                            },
                            "smoothed_training_utility": smoothed,
                            "update": best_train_update,
                            "seed": seed,
                            "selection": "best_smoothed_training_utility",
                            "run_spec": run_spec,
                        },
                    )
                completed_update = int(training_state["next_update"])
                should_validate = (
                    completed_update % args.validation_interval == 0
                    or completed_update == args.updates
                )
                if not should_validate:
                    return
                validation_policy.load_state_dict(training_state["policy_state_dict"])
                validation_policy.eval()
                validation_records, _ = _evaluate(
                    validation_env,
                    validation_policy,
                    validation_seed,
                    resolved_device,
                )
                scored_validation_records = [
                    record
                    for context_id in dict.fromkeys(
                        record.get("_validation_context", 0)
                        for record in validation_records
                    )
                    for record in [
                        item
                        for item in validation_records
                        if item.get("_validation_context", 0) == context_id
                    ][args.validation_warmup_periods :]
                ]
                validation_utility = _mean_evaluation_utility(
                    validation_env,
                    validation_records,
                    args.validation_warmup_periods,
                )
                validation_mean_violations = float(
                    np.mean(
                        [record.get("violations", 0) for record in scored_validation_records]
                    )
                )
                validation_violation_fraction = float(
                    np.mean(
                        [
                            record.get("violations", 0) > 0
                            for record in scored_validation_records
                        ]
                    )
                )
                validation_key = (
                    -validation_violation_fraction,
                    -validation_mean_violations,
                    validation_utility,
                )
                with validation_history_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "update": completed_update,
                                "mean_utility": validation_utility,
                                "periods": len(scored_validation_records),
                                "warmup_periods": args.validation_warmup_periods,
                                "mean_violations": validation_mean_violations,
                                "violation_slot_fraction": validation_violation_fraction,
                            }
                        )
                        + "\n"
                    )
                run_logger.log_validation(
                    completed_update,
                    {
                        "mean_utility": validation_utility,
                        "mean_violations": validation_mean_violations,
                        "violation_slot_fraction": validation_violation_fraction,
                    },
                    contexts=len(
                        dict.fromkeys(
                            record.get("_validation_context", 0)
                            for record in validation_records
                        )
                    ),
                )
                if validation_key > best_validation_key:
                    best_validation_utility = validation_utility
                    best_validation_update = completed_update
                    best_validation_violation_fraction = (
                        validation_violation_fraction
                    )
                    best_validation_mean_violations = validation_mean_violations
                    best_validation_key = validation_key
                    _atomic_torch(
                        best_policy_path,
                        {
                            "policy_state_dict": {
                                key: value.detach().cpu()
                                for key, value in training_state[
                                    "policy_state_dict"
                                ].items()
                            },
                            "validation_utility": validation_utility,
                            "violation_slot_fraction": validation_violation_fraction,
                            "mean_violations": validation_mean_violations,
                            "update": completed_update,
                            "seed": seed,
                            "run_spec": run_spec,
                        },
                    )
                logger.info(
                    "validated %s at update %d: violations=%.3f/%.3f utility=%.6f "
                    "best=%.3f/%.3f/%.6f@%d",
                    run_id,
                    completed_update,
                    validation_violation_fraction,
                    validation_mean_violations,
                    validation_utility,
                    best_validation_violation_fraction,
                    best_validation_mean_violations,
                    best_validation_utility,
                    best_validation_update,
                )

            with reporter:
                policy, history = train_ppo(
                    train_env,
                    updates=args.updates,
                    rollout_steps=args.rollout_steps,
                    # Steps collected per update; independent of the episode
                    # length now that an episode spans the whole timeline.
                    rollout_periods=(
                        args.rollout_periods if args.rollout_periods > 0 else None
                    ),
                    seed=seed,
                    config=config,
                    device=resolved_device,
                    on_phase=reporter.on_phase,
                    on_rollout_step=reporter.on_rollout_step,
                    on_optimization_step=reporter.on_optimization_step,
                    on_update=reporter.on_update,
                    resume_state=resume_state,
                    initial_policy_state_dict=initial_policy_state,
                    on_checkpoint=save_checkpoint,
                    run_logger=run_logger,
                )
            train_wall_time_s = (
                previous_train_wall_time_s + time.perf_counter() - train_started
            )
            run_logger.finish()
            final_policy_payload = {
                "policy_state_dict": {
                    key: value.detach().cpu()
                    for key, value in policy.state_dict().items()
                },
                "ppo_config": asdict(config),
                "layout_signature": {
                    "models": train_env.layout.models,
                    "candidates": train_env.layout.candidates,
                    "servers": train_env.layout.servers,
                    "deployment_targets": train_env.layout.deployment_targets,
                    "model_groups": train_env.layout.model_groups,
                },
                "seed": seed,
                "device": hardware,
                "selection": "final",
            }
            _atomic_torch(
                run_dir / "policy_final.pt",
                final_policy_payload,
            )
            selected_policy = "final"
            if best_policy_path.exists():
                best_payload = torch.load(
                    best_policy_path,
                    map_location=resolved_device,
                    weights_only=False,
                )
                policy.load_state_dict(best_payload["policy_state_dict"])
                selected_policy = "best_validation"
            _atomic_torch(
                run_dir / "policy.pt",
                {
                    "policy_state_dict": {
                        key: value.detach().cpu()
                        for key, value in policy.state_dict().items()
                    },
                    "ppo_config": asdict(config),
                    "layout_signature": {
                        "models": train_env.layout.models,
                        "candidates": train_env.layout.candidates,
                        "servers": train_env.layout.servers,
                        "deployment_targets": train_env.layout.deployment_targets,
                        "model_groups": train_env.layout.model_groups,
                    },
                    "seed": seed,
                    "device": hardware,
                    "selection": selected_policy,
                    "selected_update": best_validation_update,
                    "validation_utility": best_validation_utility,
                    "validation_violation_slot_fraction": (
                        best_validation_violation_fraction
                    ),
                    "validation_mean_violations": best_validation_mean_violations,
                },
            )
            (run_dir / "training_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            eval_trace = _arrival_trace(
                scenario,
                args.eval_slots,
                args.arrival_scale,
                args,
                realization=2,
            )
            eval_env = env_class(
                scenario,
                max_slots=args.eval_slots,
                potential_shaping=False,
                seed=seed + 10_000,
                arrival_trace=eval_trace,
                mapping_samples=args.eval_mapping_samples,
                **scoring_kwargs,
            )
            eval_bar = tqdm(
                total=args.eval_slots,
                desc=f"{run_id} | evaluating",
                unit="slot",
                dynamic_ncols=True,
                disable=args.no_progress,
            )
            evaluated_slots = 0

            def on_eval_step(value: int) -> None:
                nonlocal evaluated_slots
                target = min(value, args.eval_slots)
                eval_bar.update(max(0, target - evaluated_slots))
                evaluated_slots = target

            records, decision_times = _evaluate(
                eval_env, policy, seed + 10_000, resolved_device, on_eval_step
            )
            eval_bar.close()
            decorated_records = [
                {
                    "run_id": run_id,
                    "scenario": scenario.id,
                    "mode": mode,
                    "variant": variant,
                    "seed": seed,
                    **record,
                }
                for record in records
            ]
            run_summary = {
                    "scenario": scenario.id,
                    "mode": mode,
                    "variant": variant,
                    "seed": seed,
                    **_summary(
                        run_id,
                        records,
                        history,
                        train_wall_time_s,
                        decision_times,
                        {
                            "selected_policy": selected_policy,
                            "selected_update": best_validation_update,
                            "selected_validation_utility": best_validation_utility,
                            "selected_validation_violation_slot_fraction": (
                                best_validation_violation_fraction
                            ),
                            "selected_validation_mean_violations": (
                                best_validation_mean_violations
                            ),
                            "state_cost_scale": eval_env.cost_reference,
                            "state_latency_scale": eval_env.latency_reference,
                        },
                    ),
            }
            _atomic_parquet(pd.DataFrame(decorated_records), run_slots_path)
            _atomic_parquet(pd.DataFrame([run_summary]), run_summary_path)
            _atomic_json(
                complete_path,
                {"run_spec": run_spec, "completed_at_utc": _timestamp()},
            )
            slot_records.extend(decorated_records)
            run_summaries.append(run_summary)
            completed_runs += 1
            matrix_bar.update(1)
            write_aggregates()
            write_matrix_status("running", run_id)
            logger.info("completed run %s", run_id)
        write_matrix_status("completed")
        logger.info("completed RL matrix in %.3fs", time.perf_counter() - matrix_started)
    except BaseException as exc:
        logger.exception("RL matrix failed")
        write_matrix_status("failed", error=repr(exc))
        raise
    finally:
        matrix_bar.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

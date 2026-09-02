from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean
import time

import numpy as np
import pandas as pd
import torch

from agent_orch.agents import (
    PPOConfig,
    StructuredActorCritic,
    TrainingProgressReporter,
    device_metadata,
    resolve_device,
    train_ppo,
)
from agent_orch.backends import ProfileBackend
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv
from agent_orch.metrics import summarize_slot_metrics
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace


ENVIRONMENTS = {
    "joint": AgentOrchestrationEnv,
    "deploy": DeploymentOnlyEnv,
    "route": RoutingOnlyEnv,
}

DEFAULT_SEEDS = "0"
DEFAULT_MODES = "joint"
DEFAULT_VARIANTS = "rnd"


def _arrival_trace(scenario, slots: int, rate_scale: float):
    return ArrivalTrace.stationary_poisson_intensity(
        scenario, slots, rate_scale=rate_scale
    )


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _evaluate(
    env: AgentOrchestrationEnv,
    policy: StructuredActorCritic,
    seed: int,
    device: str = "cpu",
) -> tuple[list[dict], list[float]]:
    observation, _ = env.reset(seed=seed)
    records: list[dict] = []
    decision_times: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        started = time.perf_counter()
        action, _, _ = policy.act(observation, deterministic=True, device=device)
        decision_times.append(time.perf_counter() - started)
        observation, _, terminated, truncated, info = env.step(action)
        if "metrics" in info:
            records.append(asdict(info["metrics"]))
        records.extend(asdict(metrics) for metrics in info.get("interval_metrics", []))
    return records, decision_times


def _summary(
    run_id: str,
    records: list[dict],
    history: list[dict],
    train_wall_time_s: float,
    decision_times_s: list[float],
) -> dict:
    if not records:
        raise RuntimeError(f"Evaluation for {run_id} produced no physical-slot metrics")
    physical_metrics = summarize_slot_metrics(records)
    return {
        "run_id": run_id,
        "evaluation_slots": len(records),
        **physical_metrics,
        "final_training_reward": history[-1]["mean_reward"],
        "train_wall_time_s": train_wall_time_s,
        "mean_decision_time_ms": 1_000.0 * mean(decision_times_s),
        "p95_decision_time_ms": 1_000.0 * float(np.quantile(decision_times_s, 0.95)),
    }


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


def _variant_config(variant: str) -> tuple[PPOConfig, bool]:
    if variant == "rnd":
        return PPOConfig(constrained=True, exploration_mode="rnd"), False
    if variant == "no-rnd":
        return PPOConfig(constrained=True, exploration_mode="none"), False
    if variant == "unconstrained-rnd":
        return PPOConfig(constrained=False, exploration_mode="rnd"), False
    if variant == "icm":
        return PPOConfig(constrained=True, exploration_mode="icm"), False
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
    parser.add_argument("--train-slots", type=int, default=600)
    parser.add_argument("--eval-slots", type=int, default=600)
    parser.add_argument("--profile", help="LLMServingSim/vLLM performance table CSV")
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
    parser.add_argument("--output", default="results/rl_matrix")
    args = parser.parse_args()

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
    profile = ProfileBackend.from_csv(args.profile) if args.profile else None
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(args.device)
    hardware = device_metadata(args.device, resolved_device)
    print(
        f"RL plan: seeds={_parse_csv(args.seeds)}, "
        f"combinations={_combinations(modes, variants)}, device={resolved_device}",
        flush=True,
    )
    slot_records: list[dict] = []
    run_summaries: list[dict] = []

    for seed in (int(value) for value in _parse_csv(args.seeds)):
        for mode, variant in _combinations(modes, variants):
            config, potential_shaping = _variant_config(variant)
            run_id = f"{scenario.id}-{mode}-{variant}-s{seed}-{scenario_hash}"
            run_dir = output / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            env_class = ENVIRONMENTS[mode]
            train_trace = _arrival_trace(
                scenario, args.train_slots, args.arrival_scale
            )
            train_env = env_class(
                scenario,
                max_slots=args.train_slots,
                potential_shaping=potential_shaping,
                seed=seed,
                arrival_trace=train_trace,
                llm_profile_backend=profile,
            )
            print(f"Starting training run {run_id}", flush=True)
            train_started = time.perf_counter()
            reporter = TrainingProgressReporter(
                run_id=run_id,
                output_dir=run_dir,
                updates=args.updates,
                rollout_steps=args.rollout_steps,
                update_epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
                device=resolved_device,
                status_interval_steps=args.status_interval_steps,
                show_progress=not args.no_progress,
            )
            with reporter:
                policy, history = train_ppo(
                    train_env,
                    updates=args.updates,
                    rollout_steps=args.rollout_steps,
                    seed=seed,
                    config=config,
                    device=resolved_device,
                    on_phase=reporter.on_phase,
                    on_rollout_step=reporter.on_rollout_step,
                    on_optimization_step=reporter.on_optimization_step,
                    on_update=reporter.on_update,
                )
            train_wall_time_s = time.perf_counter() - train_started
            torch.save(
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
                        "deployment_widths": train_env.layout.deployment_widths,
                        "model_groups": train_env.layout.model_groups,
                    },
                    "seed": seed,
                    "device": hardware,
                },
                run_dir / "policy.pt",
            )
            (run_dir / "training_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            eval_trace = _arrival_trace(
                scenario,
                args.eval_slots,
                args.arrival_scale,
            )
            eval_env = env_class(
                scenario,
                max_slots=args.eval_slots,
                potential_shaping=False,
                seed=seed + 10_000,
                arrival_trace=eval_trace,
                llm_profile_backend=profile,
            )
            records, decision_times = _evaluate(
                eval_env, policy, seed + 10_000, resolved_device
            )
            for record in records:
                slot_records.append(
                    {
                        "run_id": run_id,
                        "scenario": scenario.id,
                        "mode": mode,
                        "variant": variant,
                        "seed": seed,
                        **record,
                    }
                )
            run_summaries.append(
                {
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
                    ),
                }
            )

    pd.DataFrame(slot_records).to_parquet(output / "slot_metrics.parquet", index=False)
    pd.DataFrame(run_summaries).to_parquet(output / "run_summary.parquet", index=False)
    manifest = {
        "scenario": str(scenario_path),
        "scenario_hash": scenario_hash,
        "seeds": [int(value) for value in _parse_csv(args.seeds)],
        "modes": modes,
        "variants": variants,
        "combinations": _combinations(modes, variants),
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "train_slots": args.train_slots,
        "eval_slots": args.eval_slots,
        "arrival_process": "stationary_poisson_intensity",
        "arrival_scale": args.arrival_scale,
        "profile": str(Path(args.profile).resolve()) if args.profile else None,
        "device": hardware,
        "status_interval_steps": args.status_interval_steps,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

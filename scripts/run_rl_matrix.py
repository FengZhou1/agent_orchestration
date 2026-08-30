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

from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv
from agent_orch.schema.loader import ScenarioLoader


ENVIRONMENTS = {
    "joint": AgentOrchestrationEnv,
    "deploy": DeploymentOnlyEnv,
    "route": RoutingOnlyEnv,
}


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _evaluate(
    env: AgentOrchestrationEnv,
    policy: StructuredActorCritic,
    seed: int,
) -> tuple[list[dict], list[float]]:
    observation, _ = env.reset(seed=seed)
    records: list[dict] = []
    decision_times: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        started = time.perf_counter()
        action, _, _ = policy.act(observation, deterministic=True)
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
    return {
        "run_id": run_id,
        "evaluation_slots": len(records),
        "mean_cost": mean(row["cost"] for row in records),
        "mean_latency_s": mean(row["mean_latency_s"] for row in records),
        "mean_goodput_rps": mean(row["goodput_rps"] for row in records),
        "mean_quality": mean(row["quality"] for row in records),
        "mean_slo_attainment": mean(row["slo_attainment"] for row in records),
        "mean_violations": mean(row["violations"] for row in records),
        "final_training_reward": history[-1]["mean_reward"],
        "train_wall_time_s": train_wall_time_s,
        "mean_decision_time_ms": 1_000.0 * mean(decision_times_s),
        "p95_decision_time_ms": 1_000.0 * float(np.quantile(decision_times_s, 0.95)),
    }


def _combinations(modes: list[str], variants: list[str]) -> list[tuple[str, str]]:
    if variants == ["auto"]:
        return [
            (mode, variant)
            for mode in modes
            for variant in (
                ("vanilla", "potential", "icm") if mode == "joint" else ("vanilla",)
            )
        ]
    return [(mode, variant) for mode in modes for variant in variants]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the structural PPO and reward-component matrix."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--modes", default="joint,deploy,route")
    parser.add_argument(
        "--variants",
        default="auto",
        help="auto runs reward ablations for joint PPO and vanilla for split modes",
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--train-slots", type=int, default=600)
    parser.add_argument("--eval-slots", type=int, default=600)
    parser.add_argument("--output", default="results/rl_matrix")
    args = parser.parse_args()

    modes = _parse_csv(args.modes)
    variants = _parse_csv(args.variants)
    unknown_modes = set(modes) - set(ENVIRONMENTS)
    unknown_variants = set(variants) - {"auto", "vanilla", "potential", "icm"}
    if unknown_modes or unknown_variants:
        raise ValueError(
            f"Unknown modes={sorted(unknown_modes)} variants={sorted(unknown_variants)}"
        )

    scenario_path = Path(args.scenario).resolve()
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16]
    scenario = ScenarioLoader.load(scenario_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    slot_records: list[dict] = []
    run_summaries: list[dict] = []

    for seed in (int(value) for value in _parse_csv(args.seeds)):
        for mode, variant in _combinations(modes, variants):
            run_id = f"{scenario.id}-{mode}-{variant}-s{seed}-{scenario_hash}"
            run_dir = output / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            env_class = ENVIRONMENTS[mode]
            train_env = env_class(
                scenario,
                max_slots=args.train_slots,
                potential_shaping=variant == "potential",
                seed=seed,
            )
            train_started = time.perf_counter()
            policy, history = train_ppo(
                train_env,
                updates=args.updates,
                rollout_steps=args.rollout_steps,
                seed=seed,
                config=PPOConfig(),
                use_icm=variant == "icm",
            )
            train_wall_time_s = time.perf_counter() - train_started
            torch.save(policy.state_dict(), run_dir / "policy.pt")
            (run_dir / "training_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            eval_env = env_class(
                scenario,
                max_slots=args.eval_slots,
                potential_shaping=False,
                seed=seed + 10_000,
            )
            records, decision_times = _evaluate(eval_env, policy, seed + 10_000)
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

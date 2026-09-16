from __future__ import annotations

import argparse
from dataclasses import asdict
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


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
    on_step=None,
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
        if on_step is not None:
            on_step(len(records))
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
        return PPOConfig(constrained=True, exploration_mode="rnd"), True
    if variant == "no-rnd":
        return PPOConfig(constrained=True, exploration_mode="none"), True
    if variant == "unconstrained-rnd":
        return PPOConfig(constrained=False, exploration_mode="rnd"), True
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
        "scenario": str(scenario_path),
        "scenario_hash": scenario_hash,
        "seeds": seeds,
        "modes": modes,
        "variants": variants,
        "combinations": combinations,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "train_slots": args.train_slots,
        "eval_slots": args.eval_slots,
        "arrival_process": "stationary_poisson_intensity",
        "arrival_scale": args.arrival_scale,
        "device": hardware,
        "status_interval_steps": args.status_interval_steps,
        "resume_enabled": args.resume,
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
            run_id = f"{scenario.id}-{mode}-{variant}-s{seed}-{scenario_hash}"
            run_dir = output / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run_spec = {
                "scenario_hash": scenario_hash,
                "seed": seed,
                "mode": mode,
                "variant": variant,
                "updates": args.updates,
                "rollout_steps": args.rollout_steps,
                "train_slots": args.train_slots,
                "eval_slots": args.eval_slots,
                "arrival_scale": args.arrival_scale,
            }
            complete_path = run_dir / "run_complete.json"
            run_slots_path = run_dir / "evaluation_slots.parquet"
            run_summary_path = run_dir / "run_summary.parquet"
            if args.resume and complete_path.exists():
                completed = json.loads(complete_path.read_text(encoding="utf-8"))
                if completed.get("run_spec") != run_spec:
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
                scenario, args.train_slots, args.arrival_scale
            )
            train_env = env_class(
                scenario,
                max_slots=args.train_slots,
                potential_shaping=potential_shaping,
                seed=seed,
                arrival_trace=train_trace,
            )
            print(f"Starting training run {run_id}", flush=True)
            logger.info("starting training run %s", run_id)
            checkpoint_path = run_dir / "checkpoint.pt"
            resume_state = None
            previous_train_wall_time_s = 0.0
            if args.resume and checkpoint_path.exists():
                checkpoint = torch.load(
                    checkpoint_path, map_location=resolved_device, weights_only=False
                )
                if checkpoint.get("run_spec") != run_spec:
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
                rollout_steps=args.rollout_steps,
                update_epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
                device=resolved_device,
                status_interval_steps=args.status_interval_steps,
                show_progress=not args.no_progress,
                initial_update=initial_update,
                append_history=bool(resume_state),
            )

            def save_checkpoint(training_state: dict) -> None:
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
                    resume_state=resume_state,
                    on_checkpoint=save_checkpoint,
                )
            train_wall_time_s = (
                previous_train_wall_time_s + time.perf_counter() - train_started
            )
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
                        "deployment_targets": train_env.layout.deployment_targets,
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

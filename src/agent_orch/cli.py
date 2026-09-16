from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from agent_orch.agents import (
    PPOConfig,
    TrainingProgressReporter,
    device_metadata,
    resolve_device,
    train_ppo,
)
from agent_orch.action_decoder import ActionDecoder
from agent_orch.baselines import make_policy
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv
from agent_orch.metrics import summarize_slot_metrics
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator.core import Simulator, metrics_to_dict
from agent_orch.workload import ArrivalTrace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-orch-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--scenario", required=True)
    run.add_argument(
        "--policy",
        choices=["random", "static", "equal", "least_load", "greedy"],
        default="greedy",
    )
    run.add_argument("--slots", type=int, default=10)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--output", default="results")
    run.add_argument("--arrival-scale", type=float, default=1.0)
    train = subparsers.add_parser("train")
    train.add_argument("--scenario", required=True)
    train.add_argument("--updates", type=int, default=10)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument(
        "--periods", "--max-slots", dest="periods", type=int, default=600,
        help="number of macro orchestration periods"
    )
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--potential-shaping", dest="potential_shaping", action="store_true")
    train.add_argument("--no-potential-shaping", dest="potential_shaping", action="store_false")
    train.set_defaults(potential_shaping=True)
    train.add_argument(
        "--exploration", choices=["rnd", "none"], default="rnd"
    )
    train.add_argument("--unconstrained", action="store_true")
    train.add_argument("--mode", choices=["joint", "deploy", "route"], default="joint")
    train.add_argument("--output", default="checkpoints")
    train.add_argument("--arrival-scale", type=float, default=1.0)
    train.add_argument(
        "--device",
        default="auto",
        help="training device: auto, cpu, cuda, or cuda:<index>",
    )
    train.add_argument("--status-interval-steps", type=int, default=32)
    train.add_argument("--no-progress", action="store_true")
    return parser


def _load_arrivals(
    scenario,
    slots: int,
    rate_scale: float,
) -> ArrivalTrace:
    return ArrivalTrace.stationary_poisson_intensity(
        scenario, slots, rate_scale=rate_scale
    )


def _run(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    simulator = Simulator(scenario)
    simulator.set_arrival_trace(
        _load_arrivals(scenario, args.slots, args.arrival_scale)
    )
    simulator.reset(args.seed)
    policy = make_policy(args.policy, scenario, args.seed)
    deployment = policy.deployment()
    decoder = ActionDecoder(scenario)
    decoder.validate_deployment(deployment)

    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16]
    run_id = f"{scenario.id}-{args.policy}-s{args.seed}-{scenario_hash}"
    output_dir = Path(args.output).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    records = []
    with metrics_path.open("w", encoding="utf-8") as handle:
        for _ in range(args.slots):
            routing = policy.routing(deployment, simulator.last_metrics)
            decoder.validate_routing(deployment, routing)
            transition = simulator.step(deployment, routing)
            record = metrics_to_dict(transition.metrics)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "run_id": run_id,
        "scenario": str(scenario_path),
        "scenario_hash": scenario_hash,
        "policy": args.policy,
        "seed": args.seed,
        "slots": args.slots,
        "arrival_process": "stationary_poisson_intensity",
        "arrival_scale": args.arrival_scale,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        **summarize_slot_metrics(records),
        "output": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return _run(args)
    if args.command == "train":
        scenario = ScenarioLoader.load(args.scenario)
        arrival_trace = _load_arrivals(
            scenario,
            args.periods,
            args.arrival_scale,
        )
        env_class = {
            "joint": AgentOrchestrationEnv,
            "deploy": DeploymentOnlyEnv,
            "route": RoutingOnlyEnv,
        }[args.mode]
        env = env_class(
            scenario,
            max_slots=args.periods,
            potential_shaping=args.potential_shaping,
            seed=args.seed,
            arrival_trace=arrival_trace,
        )
        config = PPOConfig(
            constrained=not args.unconstrained,
            exploration_mode=args.exploration,
        )
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        base = "unconstrained" if args.unconstrained else "constrained"
        suffix = f"{base}-shaped-{args.exploration}" if args.potential_shaping else f"{base}-{args.exploration}"
        run_id = f"ppo-{scenario.id}-{args.mode}-{suffix}-s{args.seed}"
        resolved_device = resolve_device(args.device)
        hardware = device_metadata(args.device, resolved_device)
        print(f"Starting {run_id} on {resolved_device}", flush=True)
        reporter = TrainingProgressReporter(
            run_id=run_id,
            output_dir=output,
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            update_epochs=config.update_epochs,
            minibatch_size=config.minibatch_size,
            device=resolved_device,
            status_interval_steps=args.status_interval_steps,
            show_progress=not args.no_progress,
            status_filename=f"{run_id}.training_status.json",
            history_filename=f"{run_id}.training_history.jsonl",
        )
        with reporter:
            policy, history = train_ppo(
                env,
                updates=args.updates,
                rollout_steps=args.rollout_steps,
                seed=args.seed,
                config=config,
                device=resolved_device,
                on_phase=reporter.on_phase,
                on_rollout_step=reporter.on_rollout_step,
                on_optimization_step=reporter.on_optimization_step,
                on_update=reporter.on_update,
            )
        checkpoint = output / f"{run_id}.pt"
        torch.save(
            {
                "policy_state_dict": {
                    key: value.detach().cpu()
                    for key, value in policy.state_dict().items()
                },
                "ppo_config": asdict(config),
                "layout_signature": {
                    "models": env.layout.models,
                    "candidates": env.layout.candidates,
                    "servers": env.layout.servers,
                    "deployment_targets": env.layout.deployment_targets,
                    "model_groups": env.layout.model_groups,
                },
                "seed": args.seed,
                "device": hardware,
            },
            checkpoint,
        )
        (output / f"{run_id}.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"checkpoint": str(checkpoint), "history": history}, indent=2))
        return 0
    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

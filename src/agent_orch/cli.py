from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from agent_orch.agents import PPOConfig, train_ppo
from agent_orch.action_decoder import ActionDecoder
from agent_orch.backends import ProfileBackend
from agent_orch.baselines import make_policy
from agent_orch.envs import AgentOrchestrationEnv, DeploymentOnlyEnv, RoutingOnlyEnv
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
    run.add_argument("--trace")
    run.add_argument("--profile", help="LLMServingSim/vLLM performance table CSV")
    run.add_argument(
        "--arrival-mode",
        choices=["trace", "nhpp", "poisson", "synthetic-stress"],
        default="trace",
    )
    run.add_argument("--synthetic-bursty", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--scenario", required=True)
    train.add_argument("--updates", type=int, default=10)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument("--max-slots", type=int, default=600)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--potential-shaping", action="store_true")
    train.add_argument("--icm", action="store_true")
    train.add_argument("--unconstrained", action="store_true")
    train.add_argument("--mode", choices=["joint", "deploy", "route"], default="joint")
    train.add_argument("--output", default="checkpoints")
    train.add_argument("--trace")
    train.add_argument("--profile", help="LLMServingSim/vLLM performance table CSV")
    train.add_argument(
        "--arrival-mode",
        choices=["trace", "nhpp", "poisson", "synthetic-stress"],
        default="trace",
    )
    return parser


def _load_arrivals(
    scenario,
    trace_path: str | None,
    mode: str,
    slots: int,
    seed: int,
) -> ArrivalTrace | None:
    if mode == "synthetic-stress":
        return ArrivalTrace.synthetic_bursty(scenario, slots, seed)
    if trace_path is None:
        if mode != "trace":
            raise ValueError(f"Arrival mode {mode} requires --trace")
        return None
    source = ArrivalTrace.from_csv(trace_path)
    if mode == "trace":
        return source
    if mode == "nhpp":
        return ArrivalTrace.nhpp_control(scenario, source, slots, seed)
    if mode == "poisson":
        return ArrivalTrace.homogeneous_poisson(scenario, source, slots, seed)
    raise ValueError(f"Unknown arrival mode {mode}")


def _run(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    profile = ProfileBackend.from_csv(args.profile) if args.profile else None
    simulator = Simulator(scenario, llm_profile_backend=profile)
    arrival_mode = "synthetic-stress" if args.synthetic_bursty else args.arrival_mode
    simulator.set_arrival_trace(
        _load_arrivals(scenario, args.trace, arrival_mode, args.slots, args.seed)
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
        "arrival_mode": arrival_mode,
        "trace": str(Path(args.trace).resolve()) if args.trace else None,
        "profile": str(Path(args.profile).resolve()) if args.profile else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        "mean_cost": mean(row["cost"] for row in records),
        "mean_latency_s": mean(row["mean_latency_s"] for row in records),
        "mean_goodput_rps": mean(row["goodput_rps"] for row in records),
        "mean_quality": mean(row["quality"] for row in records),
        "mean_slo_attainment": mean(row["slo_attainment"] for row in records),
        "output": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return _run(args)
    if args.command == "train":
        if args.potential_shaping and args.icm:
            raise ValueError("Use either potential shaping or ICM in one run, not both")
        scenario = ScenarioLoader.load(args.scenario)
        profile = ProfileBackend.from_csv(args.profile) if args.profile else None
        arrival_trace = _load_arrivals(
            scenario,
            args.trace,
            args.arrival_mode,
            args.max_slots,
            args.seed,
        )
        env_class = {
            "joint": AgentOrchestrationEnv,
            "deploy": DeploymentOnlyEnv,
            "route": RoutingOnlyEnv,
        }[args.mode]
        env = env_class(
            scenario,
            max_slots=args.max_slots,
            potential_shaping=args.potential_shaping,
            seed=args.seed,
            arrival_trace=arrival_trace,
            llm_profile_backend=profile,
        )
        policy, history = train_ppo(
            env,
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            config=PPOConfig(constrained=not args.unconstrained),
            use_icm=args.icm,
        )
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        base = "unconstrained" if args.unconstrained else "constrained"
        suffix = (
            f"{base}-potential"
            if args.potential_shaping
            else (f"{base}-icm" if args.icm else base)
        )
        checkpoint = output / f"ppo-{scenario.id}-{args.mode}-{suffix}-s{args.seed}.pt"
        torch.save(policy.state_dict(), checkpoint)
        (output / f"ppo-{scenario.id}-{args.mode}-{suffix}-s{args.seed}.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"checkpoint": str(checkpoint), "history": history}, indent=2))
        return 0
    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

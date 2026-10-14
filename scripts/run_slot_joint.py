"""Train the slot-only, sequential deployment/model-routing experiment.

The sampled trajectory is written before training and replayed from slot zero
in every episode. Evaluation replays that exact saved training trajectory.
Old stage_a/formal_v* results and checkpoints are incompatible with this run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from agent_orch.agents import StructuredActorCritic, TrainingProgressReporter, train_ppo
from agent_orch.baselines import make_policy
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.deployment.warmup import (DEFAULT_CONFIG, FixedWarmupDeployments,
    VERSION as WARMUP_PROTOCOL, WarmupSpec, coverage_report, generate_warmup_deployments)
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec
from agent_orch.parameters import (
    DEFAULT_SETTINGS, audit_scenario, load_settings, make_ppo_config,
    require_consistent, require_formal_ready, settings_receipt,
)
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import SlotTrajectory, SlotVariationSpec


PROTOCOL = "slot_joint_v7"
PHYSICAL_ROUTING = "capacity_proportional"
EVALUATION_PROTOCOL = "training_trajectory_replay"


def environment_options(manifest: dict) -> dict:
    """Reject obsolete protocols instead of rescoring them under different rules."""
    protocol = manifest.get("protocol")
    if (protocol == PROTOCOL and manifest.get("physical_routing") == PHYSICAL_ROUTING
            and manifest.get("evaluation_protocol") == EVALUATION_PROTOCOL):
        return {"routing_method": PHYSICAL_ROUTING, "replay_frozen_deployment_steps": False}
    raise ValueError("Unknown or inconsistent slot-joint routing protocol")


def load_training_trajectory(run: Path, manifest: dict) -> SlotTrajectory:
    """Load and integrity-check the one permitted policy-evaluation trajectory."""
    environment_options(manifest)
    trajectory = SlotTrajectory.from_dict(json.loads(
        (run / "train_trajectory.json").read_text(encoding="utf-8")
    ))
    if (trajectory.digest() != manifest["trajectory_digest"]
            or len(trajectory) != manifest["slots"]
            or trajectory.seed != manifest["trajectory_seed"]):
        raise ValueError("Saved training trajectory differs from its manifest")
    return trajectory


def checkpoint_trajectory_digest(checkpoint):
    """Earlier v7 callbacks placed the digest inside environment_reward_state."""
    return checkpoint.get("trajectory_digest") or checkpoint.get("environment_reward_state", {}).get("trajectory_digest")


def load_warmup_deployments(run, manifest, scenario, trajectory):
    digest = manifest.get("warmup_deployment_digest")
    if digest is None:
        if manifest.get("warmup_protocol") or (Path(run) / "warmup_deployments.json").exists():
            raise ValueError("Warm-up provenance is missing from the manifest")
        # Historical runs retain their historical context sampler when read.
        return None
    if manifest.get("warmup_protocol") != WARMUP_PROTOCOL:
        raise ValueError("Unknown warm-up context protocol in manifest")
    bank = FixedWarmupDeployments.load(Path(run) / "warmup_deployments.json", scenario, trajectory)
    if bank.digest() != digest:
        raise ValueError("Saved warm-up bank differs from training manifest")
    return bank


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _evaluate(
    scenario, trajectory: SlotTrajectory, policy: StructuredActorCritic,
    mapping_samples: int, device: str, objective_spec: ObjectiveSpec,
    evaluation_phase: str = "joint",
    routing_method: str = PHYSICAL_ROUTING,
    replay_frozen_deployment_steps: bool = False,
    warmup_deployments: FixedWarmupDeployments | None = None,
) -> list[dict[str, object]]:
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=mapping_samples,
        seed=trajectory.seed, objective=objective_spec,
        routing_method=routing_method,
        replay_frozen_deployment_steps=replay_frozen_deployment_steps,
        warmup_deployments=warmup_deployments,
    )
    env.training_phase = evaluation_phase
    observation, _ = env.reset(seed=trajectory.seed)
    rows = []
    done = False
    while not done:
        action, _, _ = policy.act(observation, deterministic=True, device=device)
        observation, _, done, _, info = env.step(action)
        if not info.get("period_complete"):
            continue
        metrics = info["metrics"]
        rows.append({
            "slot": info["physical_slot"],
            "cost": metrics.cost,
            "mean_latency_s": metrics.mean_latency_s,
            "goodput_rps": metrics.goodput_rps,
            "quality": metrics.quality,
            "slo_attainment": metrics.slo_attainment,
            "total_arrival_rps": metrics.total_arrival_rps,
            "completed_rps": metrics.completed_rps,
            "failed_rps": metrics.failed_rps,
            "app_latency_s": metrics.app_latency_s,
            "app_completed_rps": metrics.app_completed_rps,
            "violations": metrics.violations,
            "utility": info["utility"],
            "constraint_vector": info["constraint_vector"],
            "deployment_steps": info["deployment_steps"],
        })
    return rows


def _evaluate_baselines(
    scenario, trajectory: SlotTrajectory, mapping_samples: int,
    objective_spec, references,
    routing_method: str = PHYSICAL_ROUTING,
) -> dict[str, object]:
    if routing_method != PHYSICAL_ROUTING:
        raise ValueError("Only capacity_proportional physical routing is supported")
    trajectory.validate_fixed_workload()
    results = {}
    for name in ("greedy", "equal", "least_load", "random"):
        previous = None
        last_metrics = None
        rows = []
        for slot in range(len(trajectory)):
            current = trajectory.scenario_at(slot, scenario)
            simulator = Simulator(current, max_mapping_samples=mapping_samples)
            simulator.slot = slot
            if previous is not None:
                simulator.previous_deployment = previous
            simulator.last_metrics = last_metrics
            baseline = make_policy(name, current, seed=trajectory.seed + slot)
            deployment = baseline.deployment()
            routing = baseline.routing(deployment, last_metrics)
            routing = CapacityProportionalRouter(current).route(
                deployment, routing.model_share, arrival_rates=simulator.current_arrival_rates()
            )
            metrics = simulator.step(deployment, routing).metrics
            value = ObjectiveEvaluator(current, objective_spec, references).evaluate(
                metrics,
                {
                    (app.id, ingress): rate
                    for app in current.applications.values()
                    for ingress, rate in app.ingress_rates.items()
                },
            )
            rows.append({
                "slot": slot,
                "utility": float(value.utility),
                "cost": float(metrics.cost),
                "mean_latency_s": metrics.mean_latency_s,
                "completed_rps": metrics.completed_rps,
                "failed_rps": metrics.failed_rps,
                "total_arrival_rps": metrics.total_arrival_rps,
                "app_latency_s": metrics.app_latency_s,
                "app_completed_rps": metrics.app_completed_rps,
                "goodput_rps": float(metrics.goodput_rps),
                "quality": float(metrics.quality),
                "slo_attainment": float(metrics.slo_attainment),
                "violations": int(metrics.violations),
                "constraint_vector": list(value.constraints),
            })
            previous = deployment.copy()
            last_metrics = metrics
        results[name] = {
            "mean_utility": float(np.mean([row["utility"] for row in rows])),
            "total_utility": float(sum(row["utility"] for row in rows)),
            "slots": rows,
        }
    return results


def _evaluate_routing_baselines(
    scenario, trajectory: SlotTrajectory, mapping_samples: int,
    objective_spec: ObjectiveSpec,
    routing_method: str = PHYSICAL_ROUTING,
    replay_frozen_deployment_steps: bool = False,
    warmup_deployments: FixedWarmupDeployments | None = None,
) -> dict[str, object]:
    """Compare model shares under exactly the routing warm-up placements."""
    results = {}
    for name in ("greedy", "equal", "least_load", "random"):
        env = SlotSequentialJointEnv(
            scenario, trajectory, mapping_samples=mapping_samples,
            seed=trajectory.seed, objective=objective_spec,
            routing_method=routing_method,
            replay_frozen_deployment_steps=replay_frozen_deployment_steps,
            warmup_deployments=warmup_deployments,
        )
        env.training_phase = "composition"
        observation, _ = env.reset(seed=trajectory.seed)
        rows = []
        done = False
        proposal_slot = -1
        proposal = None
        while not done:
            action = {"deploy": 0, "model": np.zeros(env.layout.model_action_size, dtype=np.float32)}
            if observation["action_type"] == env.COMPOSITION:
                app_id, ingress = env.layout.model_groups[observation["model_group"]]
                if proposal_slot != env._period_index:
                    baseline = make_policy(name, env.scenario, seed=trajectory.seed + env._period_index)
                    proposal = baseline.routing(env.current_deployment, env.simulator.last_metrics)
                    proposal_slot = env._period_index
                assert proposal is not None
                width = len(env.layout.models)
                start = observation["model_group"] * width
                for index, model in enumerate(env.layout.models):
                    action["model"][start + index] = proposal.model_share.get((app_id, ingress, model), 0.0)
            observation, _, done, _, info = env.step(action)
            if info.get("period_complete"):
                metrics = info["metrics"]
                rows.append({
                    "slot": info["physical_slot"],
                    "utility": float(info["utility"]),
                    "cost": metrics.cost,
                    "mean_latency_s": metrics.mean_latency_s,
                    "goodput_rps": metrics.goodput_rps,
                    "quality": metrics.quality,
                    "total_arrival_rps": metrics.total_arrival_rps,
                    "completed_rps": metrics.completed_rps,
                    "failed_rps": metrics.failed_rps,
                    "app_latency_s": metrics.app_latency_s,
                    "app_completed_rps": metrics.app_completed_rps,
                    "violations": metrics.violations,
                    "slo_attainment": float(metrics.slo_attainment),
                    "constraint_vector": list(info["constraint_vector"]),
                })
        results[name] = {
            "mean_utility": float(np.mean([row["utility"] for row in rows])),
            "total_utility": float(sum(row["utility"] for row in rows)),
            "slots": rows,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    parser.add_argument("--smoke", action="store_true",
                        help="Explicit non-paper diagnostic; permits toy or declared runtime overrides")
    parser.add_argument("--formal", action="store_true",
                        help="Require the parameter register to be author-frozen")
    parser.add_argument("--output", required=True)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--updates", type=int, required=True,
                        help="Explicit training budget; no unconfirmed formal episode default")
    parser.add_argument("--route-updates", type=int, default=None,
                        help="Model-selection warm-up updates; defaults to registered phase ratio")
    parser.add_argument("--deploy-updates", type=int, default=None,
                        help="Deployment-only updates; defaults to registered phase ratio")
    parser.add_argument("--seed", type=int, default=0, help="PPO random seed")
    parser.add_argument("--trajectory-seed", type=int, default=101)
    parser.add_argument("--warmup-bundle", type=Path,
                        help="Reuse both saved trajectory and deployment contexts, independent of PPO seed")
    parser.add_argument("--warmup-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--warmup-seed", type=int, default=20261014)
    parser.add_argument("--variation", help="JSON file with SlotVariationSpec fields")
    parser.add_argument("--mapping-samples", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--exploration", choices=("none",), default="none")
    parser.add_argument("--shared-model-head", action="store_true",
                        help="Share one model-selection head across (application, ingress) groups")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-bar", action="store_true", help="Optional dynamic bar in addition to per-update lines")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.formal:
        if args.smoke:
            raise ValueError("Formal and smoke modes are mutually exclusive")
        require_formal_ready(settings)
    args.slots = args.slots if args.slots is not None else settings["training"]["slots"]
    args.mapping_samples = (args.mapping_samples if args.mapping_samples is not None
                            else settings["training"]["mapping_samples"])
    args.update_epochs = (args.update_epochs if args.update_epochs is not None
                          else settings["ppo"]["update_epochs"])
    args.minibatch_size = (args.minibatch_size if args.minibatch_size is not None
                           else settings["ppo"]["minibatch_size"])
    args.hidden_size = args.hidden_size if args.hidden_size is not None else settings["ppo"]["hidden_size"]
    args.hidden_layers = args.hidden_layers if args.hidden_layers is not None else settings["ppo"]["hidden_layers"]
    if not args.smoke and any(getattr(args, key) != settings["ppo"][key] for key in
                             ("update_epochs", "minibatch_size", "hidden_size", "hidden_layers")):
        raise ValueError("Runtime PPO overrides require explicit --smoke or an updated settings profile")
    if args.shared_model_head and not settings["ppo"]["shared_composition_head"] and not args.smoke:
        raise ValueError("Model-head override requires --smoke or an updated settings profile")
    config = make_ppo_config(settings, training_phase="joint", exploration_mode="none",
        update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
        shared_composition_head=(args.shared_model_head or settings["ppo"]["shared_composition_head"]),
        composition_group_relative_advantages=False, factorized_credit=False)
    if args.slots <= 0 or args.updates <= 0 or args.mapping_samples <= 0:
        raise ValueError("slots, updates and mapping-samples must be positive")
    ratio = settings["training"]["phase_ratio"]
    route_updates = args.route_updates if args.route_updates is not None else (
        max(1, int(args.updates * ratio[0] / sum(ratio))) if args.updates >= 3 else 0)
    deploy_updates = args.deploy_updates if args.deploy_updates is not None else (
        max(1, int(args.updates * ratio[1] / sum(ratio))) if args.updates >= 3 else 0)
    joint_updates = args.updates - route_updates - deploy_updates
    if route_updates < 0 or deploy_updates < 0 or joint_updates < 0:
        raise ValueError("Stage update counts must be nonnegative and total no more than updates")
    phase_schedule = tuple((phase, count) for phase, count in (
        ("composition", route_updates), ("deployment", deploy_updates), ("joint", joint_updates)
    ) if count > 0)
    manifest_phase_schedule = [list(item) for item in phase_schedule]
    training_settings = {
        "mapping_samples": args.mapping_samples,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "exploration": args.exploration,
        "phase_schedule": manifest_phase_schedule,
        "shared_model_head": config.shared_composition_head,
        "ppo_config": config.to_dict(),
        "parameter_receipt": settings_receipt(settings),
        "smoke": args.smoke,
    }
    # JSON reload turns dataclass tuples into lists. Compare the same canonical
    # representation on resume instead of rejecting every fresh checkpoint.
    training_settings = json.loads(json.dumps(training_settings))
    output = Path(args.output)
    scenario_path = Path(args.scenario)
    scenario = ScenarioLoader.load(scenario_path)
    parameter_audit = audit_scenario(scenario, settings)
    if not args.smoke:
        require_consistent(parameter_audit)
        if scenario.metadata.get("parameter_receipt") != settings_receipt(settings):
            raise ValueError("Scenario provenance differs from the current parameter profile; regenerate "
                             "using the saved pool, not the old infrastructure template")
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    variation = (
        SlotVariationSpec(**json.loads(Path(args.variation).read_text(encoding="utf-8")))
        if args.variation else SlotVariationSpec(**settings["variation"])
    )
    variation.validate()
    # All four model objectives remain visible while SLO attainment is also an
    # explicit constraint.  Weights/target are placeholders pending calibration.
    objective_spec = ObjectiveSpec(**settings["objective"])
    warmup_spec = WarmupSpec.load(args.warmup_config)
    if not args.smoke and asdict(variation) != asdict(SlotVariationSpec(**settings["variation"])):
        raise ValueError("Trajectory overrides require --smoke or an updated settings profile")
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "train_trajectory.json"
    checkpoint_path = output / "checkpoint.pt"
    manifest_path = output / "manifest.json"
    if args.resume:
        if not trajectory_path.exists() or not checkpoint_path.exists() or not manifest_path.exists():
            raise FileNotFoundError("Resume requires trajectory, manifest and checkpoint")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL:
            raise ValueError(f"Resume requires {PROTOCOL}; use a new output directory")
        environment_options(manifest)
        trajectory = load_training_trajectory(output, manifest)
        warmup = load_warmup_deployments(output, manifest, scenario, trajectory)
        if route_updates and warmup is None:
            raise ValueError("Old warm-up runs cannot resume with the new contexts; choose a new directory")
        if warmup is not None and (warmup.seed != args.warmup_seed or
                json.loads(json.dumps(warmup.spec)) != json.loads(json.dumps(asdict(warmup_spec)))):
            raise ValueError("Warm-up generation settings differ from checkpoint")
        if args.warmup_bundle:
            requested = FixedWarmupDeployments.load(args.warmup_bundle / "warmup_deployments.json", scenario, trajectory)
            if warmup is None or requested.digest() != warmup.digest():
                raise ValueError("Requested warm-up bundle differs from saved run")
        if manifest["scenario_hash"] != scenario_hash or manifest["trajectory_digest"] != trajectory.digest():
            raise ValueError("Scenario or trajectory changed since the checkpoint")
        if len(trajectory) != args.slots or trajectory.seed != args.trajectory_seed:
            raise ValueError("Requested slot count or trajectory seed differs from checkpoint")
        if manifest["variation"] != json.loads(json.dumps(asdict(variation))) or manifest["ppo_seed"] != args.seed:
            raise ValueError("Variation or PPO seed differs from checkpoint")
        if manifest["training_settings"] != training_settings:
            raise ValueError("Training settings differ from checkpoint")
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (resume_state.get("protocol") != PROTOCOL
                or resume_state.get("physical_routing") != PHYSICAL_ROUTING):
            raise ValueError("Checkpoint does not belong to the capacity-routing protocol")
        if resume_state.get("warmup_deployment_digest") != (warmup.digest() if warmup else None):
            raise ValueError("Checkpoint uses different warm-up deployments")
    else:
        if any(path.exists() for path in (trajectory_path, checkpoint_path, manifest_path)):
            raise FileExistsError("Output already contains a slot-joint run; choose a new directory or --resume")
        if args.warmup_bundle:
            trajectory = SlotTrajectory.from_dict(json.loads((args.warmup_bundle / "train_trajectory.json").read_text(encoding="utf-8")))
            if (len(trajectory) != args.slots or trajectory.seed != args.trajectory_seed
                    or json.loads(json.dumps(asdict(trajectory.variation))) != json.loads(json.dumps(asdict(variation)))):
                raise ValueError("Warm-up bundle slot count/seed/variation differs from requested training")
            warmup = FixedWarmupDeployments.load(args.warmup_bundle / "warmup_deployments.json", scenario, trajectory)
            if warmup.seed != args.warmup_seed or json.loads(json.dumps(warmup.spec)) != json.loads(json.dumps(asdict(warmup_spec))):
                raise ValueError("Warm-up bundle settings differ from request")
        else:
            trajectory = SlotTrajectory.sample(scenario, args.slots, args.trajectory_seed, variation)
            warmup = None
            if route_updates:
                print("[init] generating fixed warm-up deployments (no PPO score filtering)", flush=True)
                warmup = generate_warmup_deployments(scenario, trajectory, args.warmup_seed, warmup_spec,
                    lambda e: print(f"[warmup] slot={e['slot']+1}/{len(trajectory)} models={len(e['active_models'])} "
                        f"LLM={e['llm_instances']} capacity_risk={e['capacity_risk']}", flush=True))
        _write_json(trajectory_path, trajectory.to_dict())
        if warmup:
            warmup.save(output / "warmup_deployments.json")
            _write_json(output / "warmup_coverage.json", coverage_report(warmup))
        resume_state = None
        _write_json(manifest_path, {
            "protocol": PROTOCOL,
            "scenario": str(scenario_path.resolve()),
            "scenario_hash": scenario_hash,
            "trajectory_digest": trajectory.digest(),
            "warmup_deployment_digest": warmup.digest() if warmup else None,
            "warmup_protocol": WARMUP_PROTOCOL if warmup else None,
            "trajectory_seed": trajectory.seed,
            "slots": len(trajectory),
            "variation": asdict(variation),
            "ppo_seed": args.seed,
            "objective": objective_spec.to_dict(),
            "training_settings": training_settings,
            "ppo_config": config.to_dict(),
            "parameter_receipt": settings_receipt(settings),
            "parameter_status": "smoke" if args.smoke else settings["status"],
            "pending_parameters": settings["pending"],
            "reward": "same-slot previous-episode four-component difference minus absolute constraint penalty",
            "constraint_accounting": "one absolute constraint vector per completed physical slot",
            "first_episode_reference": "deterministic greedy policy on train trajectory",
            "slo": "fixed per application",
            "model_action": "one Dirichlet simplex per (application, ingress) substep",
            "deployment_action": "one binary action per LLM candidate; one masked replica count per (tool, server) pool",
            "training": ", ".join(phase for phase, _ in phase_schedule),
            "physical_routing": PHYSICAL_ROUTING,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "frozen_deployment_substeps": 0,
        })
        _write_json(output / "parameter_audit.json", parameter_audit)
    print(f"[init] building environment: {len(scenario.servers)} servers, "
          f"{len(scenario.applications)} applications, {len(trajectory)} slot(s)", flush=True)
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=args.mapping_samples,
        seed=args.seed, objective=objective_spec,
        warmup_deployments=warmup,
    )

    def checkpoint(state):
        state = {**state, "protocol": PROTOCOL, "physical_routing": PHYSICAL_ROUTING,
                 "evaluation_protocol": EVALUATION_PROTOCOL,
                 "scenario_hash": scenario_hash, "trajectory_digest": trajectory.digest(),
                 "warmup_deployment_digest": warmup.digest() if warmup else None,
                 "ppo_config": config.to_dict()}
        state["parameter_receipt"] = settings_receipt(settings)
        temporary = checkpoint_path.with_suffix(".tmp.pt")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        _write_json(output / "history.json", state["history"])

    def decision_progress():
        return {"training_phase": env.training_phase, "slot_index": env._period_index,
                "deployment_target_index": env._deployment_target_index,
                "deployment_targets": len(env._deployment_targets),
                "model_group_index": env._model_group,
                "model_groups": len(env.layout.model_groups)}

    with TrainingProgressReporter(
        run_id=output.name, output_dir=output, updates=args.updates,
        rollout_steps=len(trajectory), rollout_unit="slot",
        update_epochs=config.update_epochs, minibatch_size=config.minibatch_size,
        device=args.device, initial_update=int(resume_state.get("next_update", 0)) if resume_state else 0,
        append_history=args.resume, print_updates=True, details=decision_progress,
        show_progress=args.progress_bar,
    ) as progress:
        progress.on_phase(int(resume_state.get("next_update", 0)) if resume_state else 0,
                          "initializing_policy_and_fixed_baseline")
        env.on_evaluation_phase = progress.on_evaluation_phase
        policy, history = train_ppo(
            env, updates=args.updates, rollout_steps=1,
            rollout_periods=len(trajectory), seed=args.seed, config=config,
            device=args.device, resume_state=resume_state, on_checkpoint=checkpoint,
            phase_schedule=phase_schedule, on_phase=progress.on_phase,
            on_rollout_step=progress.on_rollout_step,
            on_optimization_step=progress.on_optimization_step, on_update=progress.on_update,
        )
    env.on_evaluation_phase = None
    torch.save(policy.state_dict(), output / "policy.pt")
    _write_json(output / "history.json", history)
    _write_json(output / "fixed_baseline.json", env.baseline_report())
    evaluation_trace = load_training_trajectory(output, json.loads(
        manifest_path.read_text(encoding="utf-8")
    ))
    policy.eval()
    print("[evaluation] replaying the saved training trajectory", flush=True)
    rows = _evaluate(
        scenario, evaluation_trace, policy, args.mapping_samples,
        args.device, objective_spec,
        evaluation_phase=("composition" if phase_schedule[-1][0] == "composition" else "joint"),
        warmup_deployments=warmup,
    )
    _write_json(output / "evaluation.json", {
        "protocol": PROTOCOL, "physical_routing": PHYSICAL_ROUTING,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "trajectory_digest": evaluation_trace.digest(),
        "warmup_deployment_digest": warmup.digest() if warmup else None,
        "evaluation_phase": ("composition" if phase_schedule[-1][0] == "composition" else "joint"),
        "slots": rows,
        "mean_utility": float(np.mean([row["utility"] for row in rows])),
        "total_utility": float(sum(row["utility"] for row in rows)),
    })
    print("[evaluation] evaluating fixed baselines on the same trajectory", flush=True)
    baselines = _evaluate_baselines(
        scenario, evaluation_trace, args.mapping_samples,
        env.objective.spec, env.objective.references,
    )
    _write_json(output / "baselines.json", {
        "protocol": PROTOCOL, "physical_routing": PHYSICAL_ROUTING,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "trajectory_digest": evaluation_trace.digest(),
        "policies": baselines,
    })
    if phase_schedule[-1][0] == "composition":
        _write_json(output / "routing_baselines_same_placement.json", {
            "protocol": PROTOCOL, "physical_routing": PHYSICAL_ROUTING,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "trajectory_digest": evaluation_trace.digest(),
            "warmup_deployment_digest": warmup.digest() if warmup else None,
            "policies": _evaluate_routing_baselines(
                scenario, evaluation_trace, args.mapping_samples, objective_spec,
                warmup_deployments=warmup,
            ),
        })
    print(f"Completed {args.updates} PPO updates; trajectory {trajectory.digest()[:12]}; output {output}")


if __name__ == "__main__":
    main()

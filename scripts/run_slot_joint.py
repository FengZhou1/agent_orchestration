"""Train the slot-only, sequential deployment/model-routing experiment.

The sampled trajectory is written before training and replayed from slot zero
in every episode.  A distinct sampled trajectory is used only for evaluation.
Old stage_a/formal_v* results and checkpoints are incompatible with this run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from agent_orch.agents import PPOConfig, StructuredActorCritic, train_ppo
from agent_orch.baselines import make_policy
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import SlotTrajectory, SlotVariationSpec


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _evaluate(
    scenario, trajectory: SlotTrajectory, policy: StructuredActorCritic,
    mapping_samples: int, device: str, objective_spec: ObjectiveSpec,
    evaluation_phase: str = "joint",
) -> list[dict[str, object]]:
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=mapping_samples,
        seed=trajectory.seed, objective=objective_spec,
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
            "violations": metrics.violations,
            "utility": info["utility"],
            "constraint_vector": info["constraint_vector"],
            "deployment_steps": info["deployment_steps"],
        })
    return rows


def _evaluate_baselines(
    scenario, trajectory: SlotTrajectory, mapping_samples: int,
    objective_spec, references,
) -> dict[str, object]:
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
                "mean_latency_s": float(metrics.mean_latency_s),
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
) -> dict[str, object]:
    """Compare model shares under exactly the routing warm-up placements."""
    results = {}
    for name in ("greedy", "equal", "least_load", "random"):
        env = SlotSequentialJointEnv(
            scenario, trajectory, mapping_samples=mapping_samples,
            seed=trajectory.seed, objective=objective_spec,
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
    parser.add_argument("--scenario", default="configs/toy.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--route-updates", type=int, default=None,
                        help="Model-selection warm-up updates; default one quarter")
    parser.add_argument("--deploy-updates", type=int, default=None,
                        help="Deployment-only updates with frozen model selection; default one quarter")
    parser.add_argument("--seed", type=int, default=0, help="PPO random seed")
    parser.add_argument("--trajectory-seed", type=int, default=101)
    parser.add_argument("--variation", help="JSON file with SlotVariationSpec fields")
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--exploration", choices=("none", "rnd", "icm"), default="none")
    parser.add_argument("--shared-model-head", action="store_true",
                        help="Share one model-selection head across (application, ingress) groups")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.slots <= 0 or args.updates <= 0 or args.mapping_samples <= 0:
        raise ValueError("slots, updates and mapping-samples must be positive")
    route_updates = args.route_updates if args.route_updates is not None else (max(1, args.updates // 4) if args.updates >= 3 else 0)
    deploy_updates = args.deploy_updates if args.deploy_updates is not None else (max(1, args.updates // 4) if args.updates >= 3 else 0)
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
    }
    if args.shared_model_head:
        training_settings["shared_model_head"] = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scenario_path = Path(args.scenario)
    scenario = ScenarioLoader.load(scenario_path)
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    variation = (
        SlotVariationSpec(**json.loads(Path(args.variation).read_text(encoding="utf-8")))
        if args.variation else SlotVariationSpec()
    )
    variation.validate()
    # All four model objectives remain visible while SLO attainment is also an
    # explicit constraint.  Weights/target are placeholders pending calibration.
    objective_spec = replace(
        ObjectiveSpec.slo_constrained(attainment_target=0.9),
        quality_weight=0.25, goodput_weight=0.25,
        network_utilization_target=0.9,
    )
    trajectory_path = output / "train_trajectory.json"
    checkpoint_path = output / "checkpoint.pt"
    manifest_path = output / "manifest.json"
    if args.resume:
        if not trajectory_path.exists() or not checkpoint_path.exists() or not manifest_path.exists():
            raise FileNotFoundError("Resume requires trajectory, manifest and checkpoint")
        trajectory = SlotTrajectory.from_dict(
            json.loads(trajectory_path.read_text(encoding="utf-8"))
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != "slot_joint_v4":
            raise ValueError("Checkpoint uses an incompatible deployment-action protocol")
        if manifest["scenario_hash"] != scenario_hash or manifest["trajectory_digest"] != trajectory.digest():
            raise ValueError("Scenario or trajectory changed since the checkpoint")
        if len(trajectory) != args.slots or trajectory.seed != args.trajectory_seed:
            raise ValueError("Requested slot count or trajectory seed differs from checkpoint")
        if manifest["variation"] != json.loads(json.dumps(asdict(variation))) or manifest["ppo_seed"] != args.seed:
            raise ValueError("Variation or PPO seed differs from checkpoint")
        if manifest["training_settings"] != training_settings:
            raise ValueError("Training settings differ from checkpoint")
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        if any(path.exists() for path in (trajectory_path, checkpoint_path, manifest_path)):
            raise FileExistsError("Output already contains a slot-joint run; choose a new directory or --resume")
        trajectory = SlotTrajectory.sample(scenario, args.slots, args.trajectory_seed, variation)
        _write_json(trajectory_path, trajectory.to_dict())
        resume_state = None
        _write_json(manifest_path, {
            "protocol": "slot_joint_v4",
            "scenario": str(scenario_path.resolve()),
            "scenario_hash": scenario_hash,
            "trajectory_digest": trajectory.digest(),
            "trajectory_seed": trajectory.seed,
            "slots": len(trajectory),
            "variation": asdict(variation),
            "ppo_seed": args.seed,
            "objective": objective_spec.to_dict(),
            "training_settings": training_settings,
            "reward": "same-slot previous-episode four-component difference minus absolute constraint penalty",
            "constraint_accounting": "one absolute constraint vector per completed physical slot",
            "first_episode_reference": "deterministic greedy policy on train trajectory",
            "slo": "fixed per application",
            "model_action": "one Dirichlet simplex per (application, ingress) substep",
            "deployment_action": "one binary action per LLM candidate; one masked replica count per (tool, server) pool",
            "training": ", ".join(phase for phase, _ in phase_schedule),
            "physical_routing": "deterministic softmin after all model substeps",
        })
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=args.mapping_samples,
        seed=args.seed, objective=objective_spec,
    )
    config = replace(
        PPOConfig(), training_phase="joint", exploration_mode=args.exploration,
        update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
        gae_lambda=1.0, deployment_gae_lambda=1.0,
        shaping_coefficient=0.0, composition_group_relative_advantages=False,
        factorized_credit=False, shared_composition_head=args.shared_model_head,
    )

    def checkpoint(state):
        temporary = checkpoint_path.with_suffix(".tmp.pt")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        _write_json(output / "history.json", state["history"])

    policy, history = train_ppo(
        env, updates=args.updates, rollout_steps=1,
        rollout_periods=len(trajectory), seed=args.seed, config=config,
        device=args.device, resume_state=resume_state, on_checkpoint=checkpoint,
        phase_schedule=phase_schedule,
    )
    torch.save(policy.state_dict(), output / "policy.pt")
    _write_json(output / "history.json", history)
    _write_json(output / "fixed_baseline.json", env.baseline_report())
    evaluation_trace = SlotTrajectory.sample(
        scenario, args.slots, args.trajectory_seed + 1, variation
    )
    _write_json(output / "evaluation_trajectory.json", evaluation_trace.to_dict())
    policy.eval()
    rows = _evaluate(
        scenario, evaluation_trace, policy, args.mapping_samples,
        args.device, objective_spec,
        evaluation_phase=("composition" if phase_schedule[-1][0] == "composition" else "joint"),
    )
    _write_json(output / "evaluation.json", {
        "trajectory_digest": evaluation_trace.digest(),
        "evaluation_phase": ("composition" if phase_schedule[-1][0] == "composition" else "joint"),
        "slots": rows,
        "mean_utility": float(np.mean([row["utility"] for row in rows])),
        "total_utility": float(sum(row["utility"] for row in rows)),
    })
    baselines = _evaluate_baselines(
        scenario, evaluation_trace, args.mapping_samples,
        env.objective.spec, env.objective.references,
    )
    _write_json(output / "baselines.json", {
        "trajectory_digest": evaluation_trace.digest(),
        "policies": baselines,
    })
    if phase_schedule[-1][0] == "composition":
        _write_json(output / "routing_baselines_same_placement.json", {
            "trajectory_digest": evaluation_trace.digest(),
            "policies": _evaluate_routing_baselines(
                scenario, evaluation_trace, args.mapping_samples, objective_spec
            ),
        })
    print(f"Completed {args.updates} PPO updates; trajectory {trajectory.digest()[:12]}; output {output}")


if __name__ == "__main__":
    main()

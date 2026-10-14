"""Evaluate a slot-joint checkpoint by replaying its saved training trajectory.

This is a diagnostic read of an atomic checkpoint; it does not modify the
training process or its reward baseline. Snapshot scores are not used to update
the policy.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from agent_orch.agents import StructuredActorCritic
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.parameters import ppo_config_from_manifest

from run_slot_joint import (
    _evaluate, _evaluate_baselines, _evaluate_routing_baselines,
    _write_json, environment_options, load_training_trajectory, EVALUATION_PROTOCOL,
    checkpoint_trajectory_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--mapping-samples", type=int,
                        help="Evaluation fidelity; defaults to the training value")
    parser.add_argument("--policy-snapshot", type=Path,
                        help="Evaluate saved snapshot weights instead of the latest checkpoint")
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    options = environment_options(manifest)
    scenario_path = Path(manifest["scenario"])
    if hashlib.sha256(scenario_path.read_bytes()).hexdigest() != manifest["scenario_hash"]:
        raise ValueError("Scenario changed since training")
    scenario = ScenarioLoader.load(scenario_path)
    trajectory = load_training_trajectory(args.run, manifest)
    spec_fields = {field.name for field in fields(ObjectiveSpec)}
    objective = ObjectiveSpec(**{
        key: value for key, value in manifest["objective"].items()
        if key in spec_fields
    })
    mapping_samples = (
        args.mapping_samples
        if args.mapping_samples is not None else
        manifest["training_settings"]["mapping_samples"]
    )
    if mapping_samples <= 0:
        raise ValueError("mapping-samples must be positive")
    checkpoint_source = args.policy_snapshot or (args.run / "checkpoint.pt")
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol", manifest["protocol"]) != manifest["protocol"]:
        raise ValueError("Checkpoint and run use different routing protocols")
    if checkpoint.get("physical_routing") != options["routing_method"]:
        raise ValueError("Capacity-routing snapshot is missing its routing provenance")
    if checkpoint_trajectory_digest(checkpoint) != trajectory.digest():
        raise ValueError("Policy snapshot belongs to a different training trajectory")
    if (checkpoint.get("scenario_hash") is not None
            and checkpoint["scenario_hash"] != manifest["scenario_hash"]):
        raise ValueError("Policy snapshot belongs to a different scenario")
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=mapping_samples, objective=objective, **options
    )
    policy = StructuredActorCritic(
        env, ppo_config_from_manifest(manifest)
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    evaluation_phase = ("composition" if manifest["training_settings"]["phase_schedule"][-1][0]
                        == "composition" else "joint")
    rows = _evaluate(
        scenario, trajectory, policy, mapping_samples, "cpu", objective,
        evaluation_phase=evaluation_phase, **options,
    )
    update = int(checkpoint["next_update"])
    report = {
        "protocol": manifest["protocol"],
        "physical_routing": options["routing_method"],
        "evaluation_phase": evaluation_phase,
        "checkpoint_next_update": update,
        "checkpoint_source": str(checkpoint_source),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "trajectory_seed": trajectory.seed,
        "policy_source_trajectory_digest": checkpoint_trajectory_digest(checkpoint),
        "mapping_samples": mapping_samples,
        "trajectory_digest": trajectory.digest(),
        "mean_utility": float(np.mean([row["utility"] for row in rows])),
        "mean_slo_attainment": (sum(row["goodput_rps"] for row in rows)
                                / max(sum(row["total_arrival_rps"] for row in rows), 1e-12)),
        "mean_constraint_vector": np.mean(
            [row["constraint_vector"] for row in rows], axis=0
        ).tolist(),
        "slots": rows,
    }
    if args.baselines:
        if evaluation_phase == "composition":
            report["routing_baselines_same_placement"] = _evaluate_routing_baselines(
                scenario, trajectory, mapping_samples, objective, **options
            )
        else:
            report["baselines"] = _evaluate_baselines(
                scenario, trajectory, mapping_samples, objective, env.objective.references,
                routing_method=options["routing_method"],
            )
    output = args.run / (
        f"snapshot_update_{update:03d}_samples_{mapping_samples}.json"
    )
    policy_output = output.with_suffix(".pt")
    if output.exists() or policy_output.exists():
        raise FileExistsError(output)
    temporary_policy_output = policy_output.with_suffix(".tmp.pt")
    torch.save({
        "next_update": update,
        "protocol": manifest["protocol"], "physical_routing": options["routing_method"],
        "policy_state_dict": checkpoint["policy_state_dict"],
        "trajectory_digest": trajectory.digest(),
        "scenario_hash": manifest["scenario_hash"],
        "mapping_samples": mapping_samples,
    }, temporary_policy_output)
    temporary_policy_output.replace(policy_output)
    report["policy_snapshot"] = policy_output.name
    _write_json(output, report)
    print(f"update={update} mean_utility={report['mean_utility']:.6f} output={output}")


if __name__ == "__main__":
    main()

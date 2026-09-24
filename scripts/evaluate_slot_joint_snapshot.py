"""Evaluate a running slot-joint checkpoint on the fixed held-out trajectory.

This is a diagnostic read of an atomic checkpoint; it does not modify the
training process or its reward baseline. Snapshot scores are not used to update
the policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory, SlotVariationSpec

from run_slot_joint import _evaluate, _evaluate_routing_baselines, _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--mapping-samples", type=int,
                        help="Evaluation fidelity; defaults to the training value")
    parser.add_argument("--policy-snapshot", type=Path,
                        help="Evaluate saved snapshot weights instead of the latest checkpoint")
    parser.add_argument("--evaluation-seed", type=int,
                        help="Held-out trajectory seed; defaults to training seed plus one")
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") not in ("slot_joint_v3", "slot_joint_v4"):
        raise ValueError("Snapshot evaluator requires slot_joint_v3 or slot_joint_v4")
    scenario = ScenarioLoader.load(manifest["scenario"])
    variation = SlotVariationSpec(**manifest["variation"])
    evaluation_seed = (
        args.evaluation_seed
        if args.evaluation_seed is not None else manifest["trajectory_seed"] + 1
    )
    trajectory = SlotTrajectory.sample(
        scenario, manifest["slots"], evaluation_seed, variation
    )
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
    if (checkpoint.get("scenario_hash") is not None
            and checkpoint["scenario_hash"] != manifest["scenario_hash"]):
        raise ValueError("Policy snapshot belongs to a different scenario")
    env = SlotSequentialJointEnv(
        scenario, trajectory, mapping_samples=mapping_samples, objective=objective
    )
    policy = StructuredActorCritic(
        env, PPOConfig(shared_composition_head=bool(
            manifest["training_settings"].get("shared_model_head", False)
        ))
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    rows = _evaluate(
        scenario, trajectory, policy, mapping_samples, "cpu", objective,
        evaluation_phase="composition",
    )
    update = int(checkpoint["next_update"])
    report = {
        "protocol": manifest["protocol"],
        "checkpoint_next_update": update,
        "checkpoint_source": str(checkpoint_source),
        "evaluation_seed": evaluation_seed,
        "policy_source_trajectory_digest": checkpoint.get("trajectory_digest"),
        "mapping_samples": mapping_samples,
        "trajectory_digest": trajectory.digest(),
        "mean_utility": float(np.mean([row["utility"] for row in rows])),
        "mean_slo_attainment": float(np.mean([row["slo_attainment"] for row in rows])),
        "mean_constraint_vector": np.mean(
            [row["constraint_vector"] for row in rows], axis=0
        ).tolist(),
        "slots": rows,
    }
    if args.baselines:
        report["routing_baselines_same_placement"] = _evaluate_routing_baselines(
            scenario, trajectory, mapping_samples, objective
        )
    seed_suffix = (
        f"_seed_{evaluation_seed}" if args.evaluation_seed is not None else ""
    )
    output = args.run / (
        f"snapshot_update_{update:03d}_samples_{mapping_samples}{seed_suffix}.json"
    )
    policy_output = output.with_suffix(".pt")
    if output.exists() or policy_output.exists():
        raise FileExistsError(output)
    temporary_policy_output = policy_output.with_suffix(".tmp.pt")
    torch.save({
        "next_update": update,
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

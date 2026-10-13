"""Evaluate routing heuristics under the fixed composition warm-up placements.

This reads a slot-joint run manifest but does not load or modify its policy.
The resulting baselines are comparable across policy snapshots that share the
same scenario, trajectory seed, variation, and mapping-sample count.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader

from run_slot_joint import (
    _evaluate_routing_baselines, _write_json, environment_options,
    load_training_trajectory, EVALUATION_PROTOCOL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mapping-samples", type=int, required=True)
    args = parser.parse_args()
    if args.mapping_samples <= 0:
        raise ValueError("mapping-samples must be positive")

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
    output = args.run / (
        "routing_baselines_training_replay"
        f"_samples_{args.mapping_samples}.json"
    )
    if output.exists():
        raise FileExistsError(output)
    results = _evaluate_routing_baselines(
        scenario, trajectory, args.mapping_samples, objective, **options
    )
    _write_json(output, {
        "protocol": manifest["protocol"],
        "physical_routing": options["routing_method"],
        "scenario_hash": manifest["scenario_hash"],
        "trajectory_digest": trajectory.digest(),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "trajectory_seed": trajectory.seed,
        "mapping_samples": args.mapping_samples,
        "policies": results,
    })
    print(f"routing baselines written to {output}")


if __name__ == "__main__":
    main()

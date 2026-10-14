"""Generate a frozen multi-slot trajectory and constructive warm-up deployments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from agent_orch.deployment.warmup import (DEFAULT_CONFIG, WarmupSpec,
    generate_warmup_deployments, coverage_report)
from agent_orch.parameters import DEFAULT_SETTINGS, audit_scenario, load_settings, require_consistent, settings_receipt
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory, SlotVariationSpec


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def print_slot(entry):
    print(f"[warmup] slot={entry['slot'] + 1} models={'+'.join(entry['active_models'])} "
          f"LLM={entry['llm_instances']} tools={entry['tool_replicas']} "
          f"GPU={entry['gpu_used']:.0f}/{entry['available_gpu']} "
          f"capacity_risk={entry['capacity_risk']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--trajectory", type=Path, help="Reuse a saved trajectory without resampling")
    parser.add_argument("--slots", type=int)
    parser.add_argument("--trajectory-seed", type=int, default=101)
    parser.add_argument("--deployment-seed", type=int, default=20261014)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    scenario = ScenarioLoader.load(args.scenario)
    audit = audit_scenario(scenario, settings)
    if not args.smoke:
        require_consistent(audit)
        if scenario.metadata.get("parameter_receipt") != settings_receipt(settings):
            raise ValueError("Regenerate scenario with current parameter provenance")
    if args.output.exists():
        raise FileExistsError("Choose a new bundle directory; no overwrites")
    spec = WarmupSpec.load(args.config)
    trajectory = (SlotTrajectory.from_dict(json.loads(args.trajectory.read_text(encoding="utf-8")))
        if args.trajectory else SlotTrajectory.sample(scenario,
            args.slots if args.slots is not None else settings["training"]["slots"],
            args.trajectory_seed, SlotVariationSpec(**settings["variation"])))
    if args.trajectory and args.slots is not None and len(trajectory) != args.slots:
        raise ValueError("Saved trajectory has a different slot count")
    trajectory.validate_against(scenario)
    args.output.mkdir(parents=True)
    write_json(args.output / "train_trajectory.json", trajectory.to_dict())
    write_json(args.output / "parameter_audit.json", audit)
    print(f"[warmup] generating {len(trajectory)} fixed contexts; no PPO or workflow score filtering", flush=True)
    bank = generate_warmup_deployments(scenario, trajectory, args.deployment_seed, spec, print_slot)
    bank.save(args.output / "warmup_deployments.json")
    coverage = coverage_report(bank)
    write_json(args.output / "coverage.json", coverage)
    write_json(args.output / "manifest.json", {"status": "generated_preexperiment_not_trained",
        "scenario": str(args.scenario.resolve()),
        "scenario_file_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "trajectory_digest": trajectory.digest(), "warmup_deployment_digest": bank.digest(),
        "trajectory_seed": trajectory.seed, "deployment_seed": args.deployment_seed,
        "slots": len(trajectory), "spec": asdict(spec), "parameter_receipt": settings_receipt(settings),
        "purpose": "model-selection PPO warm-up contexts only; no teacher/model-choice targets",
        "capacity_risk_slots": coverage["capacity_risk_slots"]})
    print(f"[warmup] complete: {coverage['unique_physical_deployments']}/{len(trajectory)} unique; "
          f"risk slots={coverage['capacity_risk_slots']}; output={args.output}", flush=True)


if __name__ == "__main__":
    main()

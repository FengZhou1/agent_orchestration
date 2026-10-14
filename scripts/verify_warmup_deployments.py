"""Verify a saved warm-up bundle, optionally regenerating or measuring workflows.

Diagnostic routing probes are never used as labels or to select deployments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from agent_orch.deployment.warmup import (FixedWarmupDeployments, WarmupSpec,
    generate_warmup_deployments, physical_signature)
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import SlotTrajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--workflow-slots", default="", help="Zero-based comma-separated slots; empty skips expensive workflows")
    parser.add_argument("--mapping-samples", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.mapping_samples < 1: raise ValueError("mapping-samples must be positive")
    scenario = ScenarioLoader.load(args.scenario)
    trajectory = SlotTrajectory.from_dict(json.loads((args.bundle / "train_trajectory.json").read_text(encoding="utf-8")))
    bank = FixedWarmupDeployments.load(args.bundle / "warmup_deployments.json", scenario, trajectory)
    original = (bank.scenario_hash, bank.trajectory_digest, bank.digest())
    signatures = [physical_signature(trajectory.scenario_at(t, scenario), bank.deployment_at(t)) for t in range(len(trajectory))]
    config_exposure = Counter()
    multi_replica_slots = []
    for t in range(len(trajectory)):
        groups = Counter()
        for c, active in bank.deployment_at(t).llm_active.items():
            if active:
                candidate = scenario.candidates[c]
                config_exposure[candidate.config] += 1
                groups[(candidate.model, candidate.config, candidate.server)] += 1
        if any(v > 1 for v in groups.values()): multi_replica_slots.append(t)
    regenerated = None
    if args.regenerate:
        print("[verify] regenerating all contexts from the saved inputs and rules", flush=True)
        second = generate_warmup_deployments(scenario, trajectory, bank.seed, WarmupSpec(**bank.spec),
            lambda e: print(f"[verify] regenerated slot {e['slot']+1}/{len(trajectory)}", flush=True))
        regenerated = second.digest() == bank.digest()
        if not regenerated: raise AssertionError("Regeneration did not reproduce the original digest")
    rows = []
    for slot in [int(v) for v in args.workflow_slots.split(",") if v.strip()]:
        if not 0 <= slot < len(trajectory): raise ValueError("Invalid workflow slot")
        current = trajectory.scenario_at(slot, scenario)
        deployment = bank.deployment_at(slot)
        for probe in bank.entries[slot]["probes"]:
            simulator = Simulator(current, max_mapping_samples=args.mapping_samples)
            simulator.slot = slot
            if slot:
                simulator.previous_deployment = bank.deployment_at(slot - 1)
            shares = {(a.id, g, m): x for a in current.applications.values()
                      for g in a.ingress_rates for m, x in probe["model_shares"].items()}
            routing = CapacityProportionalRouter(current).route(deployment, shares)
            metrics = simulator.step(deployment, routing).metrics
            row = {"slot": slot, "mixture": probe["mixture"], "model_shares": probe["model_shares"],
                "slo_attainment": metrics.slo_attainment, "completed_rps": metrics.completed_rps,
                "failed_rps": metrics.failed_rps, "goodput_rps": metrics.goodput_rps,
                "mean_latency_s": metrics.mean_latency_s, "quality": metrics.quality,
                "cost": metrics.cost}
            rows.append(row)
            print(f"[verify] slot={slot} mixture={probe['mixture']} "
                  f"SLO={metrics.slo_attainment:.4f} latency={metrics.mean_latency_s} "
                  f"failed_rps={metrics.failed_rps:.6g}", flush=True)
    bank.validate(scenario, trajectory)
    if original != (bank.scenario_hash, bank.trajectory_digest, bank.digest()):
        raise AssertionError("Diagnostic probes mutated the frozen inputs")
    args.output.write_text(json.dumps({"status": "verified", "slots": len(trajectory),
        "warmup_deployment_digest": bank.digest(), "trajectory_digest": trajectory.digest(),
        "resources_and_positive_workload_coverage": "pass_all_slots",
        "unique_physical_deployments": len(set(signatures)),
        "deterministic_regeneration": regenerated, "config_instance_counts": dict(config_exposure),
        "same_configuration_multi_replica_slots": multi_replica_slots,
        "workflow_mapping_samples": args.mapping_samples if rows else None,
        "workflow_probe_results": rows, "inputs_unchanged": True,
        "limits": "Fixed mixture diagnostics only; no trained-policy or generalization result."},
        indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"[verify] written {args.output}", flush=True)


if __name__ == "__main__":
    main()

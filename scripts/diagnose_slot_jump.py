"""Locate the discrete step in the per-slot objective (Stage A, step A0.1).

A constant arrival trace and a fixed composition should give a constant per-slot
utility.  Instead the utility sits on an exact plateau and then steps once.  The
prime suspect is the lagged utilisation feedback feeding a *hard* feasibility
filter in the router: when a candidate's projected utilisation reaches 1.0 it is
dropped from the candidate set entirely, so its whole share is reassigned and the
next candidate can be pushed over the edge in turn.

This dumps, per slot: the utility, how many candidate instances the router
dropped, which ones, and the utilisation the router was reading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import build_composition_env, dense_composition_action
from agent_orch.objective import ObjectiveSpec
from agent_orch.routing.physical import PhysicalRouter
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

REPO_ROOT = Path(__file__).resolve().parents[1]
ARRIVAL_SCALE = 7.165234375


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument(
        "--reference",
        default="data/processed/composition_reference_agent-abilene-20_env_test.json",
    )
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--slots", type=int, default=24)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--output", default="results/stage_a/slot_jump_diagnosis.json")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(REPO_ROOT / args.scenario)
    objective = ObjectiveSpec.slo_constrained(0.9)
    library = DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))
    _, test_library = library.train_test_split(
        test_fraction=0.25, seed=2026, stratify=True
    )
    payload = json.loads((REPO_ROOT / args.reference).read_text(encoding="utf-8"))
    records = {int(k): v for k, v in payload["entries"].items()}
    entry = test_library.entries[args.position]
    share = {tuple(k.split("|")): float(v) for k, v in records[entry.index]["model_share"].items()}

    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, args.slots, rate_scale=ARRIVAL_SCALE
    )
    env = build_composition_env(
        scenario,
        objective,
        trace,
        test_library,
        position=args.position,
        periods=args.slots,
        mapping_samples=args.mapping_samples,
        seed=0,
    )

    # Record what the router saw and what it dropped, per slot.
    calls: list[dict] = []
    original = PhysicalRouter._llm_conditional_probabilities

    def wrapper(
        self,
        deployment,
        app,
        ingress,
        node,
        model,
        previous_metrics,
        model_share,
        arrival_rates,
    ):
        active = [
            key
            for key, flag in deployment.llm_active.items()
            if flag and scenario.candidates[key].model == model
        ]
        result = original(
            self,
            deployment,
            app,
            ingress,
            node,
            model,
            previous_metrics,
            model_share,
            arrival_rates,
        )
        calls.append(
            {
                "app": app.id,
                "ingress": ingress,
                "model": model,
                "active": sorted(active),
                "kept": sorted(result),
                "dropped": sorted(set(active) - set(result)),
                "probs": {k: round(float(v), 5) for k, v in result.items()},
                "utilization": {
                    key: round(
                        float(
                            previous_metrics.llm_utilization.get(key, 0.0)
                            if previous_metrics is not None
                            else 0.0
                        ),
                        5,
                    )
                    for key in sorted(active)
                },
            }
        )
        return result

    PhysicalRouter._llm_conditional_probabilities = wrapper
    rows = []
    action = dense_composition_action(env, share)
    observation, _ = env.reset(seed=0)
    for slot in range(args.slots):
        calls.clear()
        observation, _, terminated, truncated, info = env.step(
            {"deploy": 0, "model": action.copy()}
        )
        metrics = info["metrics"]
        dropped_calls = [call for call in calls if call["dropped"]]
        backend = env.simulator.backend
        instances = getattr(backend, "_last_llm_instance_performance", {}) or {}
        instance_rows = {
            key: {
                "arrival_rps": round(float(value.arrival_rate_rps), 8),
                "concurrency": round(float(value.active_concurrency), 5),
                "capacity_rps": round(float(value.throughput_capacity_rps), 8),
                "service_s": round(float(value.mean_service_s), 5),
                "utilization": round(float(value.utilization), 6),
                "resident_cap": int(value.resident_capacity),
                "kv_slack": round(float(value.kv_slack), 4),
            }
            for key, value in sorted(instances.items())
            if float(value.arrival_rate_rps) > 0
        }
        rows.append(
            {
                "slot": slot,
                "utility": round(float(info["utility"]), 6)
                if info.get("period_complete")
                else None,
                "router_calls": len(calls),
                "calls_with_drops": len(dropped_calls),
                "dropped_candidates": sorted(
                    {key for call in dropped_calls for key in call["dropped"]}
                ),
                "llm_utilization": {
                    k: round(float(v), 5)
                    for k, v in sorted(metrics.llm_utilization.items())
                    if float(v) > 0
                },
                "tool_utilization_max": round(
                    max([float(v) for v in metrics.tool_utilization.values()] or [0.0]), 5
                ),
                "slo_attainment": round(float(metrics.slo_attainment), 5),
                "mean_latency_s": round(float(metrics.mean_latency_s), 4),
                "llm_instances": instance_rows,
            }
        )
        if terminated or truncated:
            break
    PhysicalRouter._llm_conditional_probabilities = original

    print(f"deployment {entry.index} ({entry.stratum}), composition = reference, {len(rows)} slots")
    print()
    header = (
        f"{'slot':>4} {'utility':>9} {'calls':>6} {'w/drop':>7} {'llm_util (max)':>15} "
        f"{'tool_util':>10} {'attain':>7} {'lat':>9}  dropped"
    )
    print(header)
    for row in rows:
        utilizations = list(row["llm_utilization"].values())
        print(
            f"{row['slot']:>4} {str(row['utility']):>9} {row['router_calls']:>6} "
            f"{row['calls_with_drops']:>7} {(max(utilizations) if utilizations else 0.0):>15.5f} "
            f"{row['tool_utilization_max']:>10.5f} {row['slo_attainment']:>7.4f} "
            f"{row['mean_latency_s']:>9.3f}  {','.join(row['dropped_candidates'])[:40]}"
        )
    detail_slots = {0, 5, 14, 15, 16}
    print()
    print("per-instance detail (arrival rps / capacity rps / utilization):")
    for row in rows:
        if row["slot"] not in detail_slots:
            continue
        print(f"  slot {row['slot']:>2}: utility={row['utility']}")
        for key, values in row["llm_instances"].items():
            print(
                f"      {key:<28} arrival={values['arrival_rps']:.8f} "
                f"cap={values['capacity_rps']:.8f} util={values['utilization']:.6f} "
                f"conc={values['concurrency']:.5f} svc={values['service_s']:.5f} "
                f"res={values['resident_cap']} kv_slack={values['kv_slack']:.3f}"
            )

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "deployment_index": entry.index,
                "stratum": entry.stratum,
                "rows": rows,
                "router_calls_last_slot": calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

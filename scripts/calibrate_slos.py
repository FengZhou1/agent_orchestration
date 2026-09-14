from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from agent_orch.baselines import make_policy
from agent_orch.capacity import estimate_reference_capacity
from agent_orch.performance import AnalyticalBackend
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("A non-empty sample and a valid quantile are required")
    order = np.argsort(np.asarray(values, dtype=float))
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    total = float(sorted_weights.sum())
    if total <= 0.0:
        raise ValueError("SLO calibration requires positive traffic weights")
    cumulative = np.cumsum(sorted_weights) / total
    return float(sorted_values[min(np.searchsorted(cumulative, quantile), len(values) - 1)])


def collect_reference_metrics(
    scenario,
    trace: ArrivalTrace,
    slots: int,
    policy_name: str,
    seed: int,
) -> dict[str, list[dict[str, float]]]:
    simulator = Simulator(scenario)
    simulator.set_arrival_trace(trace)
    simulator.reset(seed)
    policy = make_policy(policy_name, scenario, seed)
    deployment = policy.deployment()
    records = {app_id: [] for app_id in scenario.applications}
    for _ in range(slots):
        routing = policy.routing(deployment, simulator.last_metrics)
        metrics = simulator.step(deployment, routing).metrics
        for key, values in metrics.diagnostics["flow_metrics"].items():
            app_id = key.split(":", 1)[0]
            if float(values["weight_rps"]) > 0.0:
                records[app_id].append(values)
    return records


def calibrated_slos(
    records: dict[str, list[dict[str, float]]],
    ttft_multiplier: float,
    tbt_multiplier: float,
    deadline_multiplier: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    slos, stages = {}, {}
    for app_id, samples in records.items():
        if not samples:
            raise ValueError(f"No positive-rate calibration samples for {app_id}")
        weights = [float(sample["weight_rps"]) for sample in samples]
        slos[app_id] = {
            "ttft_s": ttft_multiplier * weighted_quantile(
                [float(sample["ttft_s"]) for sample in samples], weights, 0.95
            ),
            "tbt_s": tbt_multiplier * weighted_quantile(
                [float(sample["tbt_s"]) for sample in samples], weights, 0.95
            ),
            "deadline_s": deadline_multiplier * weighted_quantile(
                [float(sample["e2e_s"]) for sample in samples], weights, 0.95
            ),
        }
        stage_names = sorted(
            {
                key
                for sample in samples
                for key in sample
                if key.startswith("stage:") and key.endswith("_s")
            }
        )
        stages[app_id] = {}
        for metric in stage_names:
            values, metric_weights = [], []
            for sample in samples:
                if metric in sample:
                    values.append(float(sample[metric]))
                    metric_weights.append(float(sample["weight_rps"]))
            node_id = metric.removeprefix("stage:").removesuffix("_s")
            stages[app_id][node_id] = deadline_multiplier * weighted_quantile(
                values, metric_weights, 0.95
            )
    return slos, stages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", default="greedy")
    parser.add_argument(
        "--low-load-fraction",
        type=float,
        default=0.20,
        help="Fraction of the reference stable capacity used for calibration",
    )
    parser.add_argument("--slots", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ttft-multiplier", type=float, default=1.5)
    parser.add_argument("--tbt-multiplier", type=float, default=1.25)
    parser.add_argument("--deadline-multiplier", type=float, default=1.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = ScenarioLoader.load(scenario_path)
    backend = AnalyticalBackend(scenario)
    policy = make_policy(args.policy, scenario, args.seed)
    deployment = policy.deployment()
    routing = policy.routing(deployment)
    reference = estimate_reference_capacity(scenario, deployment, routing, backend)
    low_load_scale = reference.arrival_scale * args.low_load_fraction

    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, args.slots, rate_scale=low_load_scale
    )
    records = collect_reference_metrics(
        scenario, trace, args.slots, args.policy, args.seed
    )
    slos, stages = calibrated_slos(
        records, args.ttft_multiplier, args.tbt_multiplier, args.deadline_multiplier
    )
    for app in raw["applications"]:
        values = slos[app["id"]]
        kind = app["slo"]["type"]
        if kind == "lat":
            app["slo"] = {
                "type": kind, "ttft_s": values["ttft_s"], "tbt_s": values["tbt_s"],
            }
        elif kind == "ddl":
            app["slo"] = {"type": kind, "deadline_s": values["deadline_s"]}
        else:
            app["slo"] = {"type": kind, **values}
            for node in app["nodes"]:
                if node["id"] in stages[app["id"]]:
                    node["stage_deadline_s"] = stages[app["id"]][node["id"]]
    raw.setdefault("metadata", {})["slo_status"] = "frozen low-load P95 calibration"
    raw["metadata"]["slo_calibration"] = {
        "policy": args.policy,
        "slots": args.slots,
        "seed": args.seed,
        "low_load_fraction": args.low_load_fraction,
        "low_load_rate_scale": low_load_scale,
        "reference_capacity_rps": reference.stable_capacity_rps,
        "reference_arrival_scale": reference.arrival_scale,
        "limiting_resource": reference.limiting_resource,
        "ttft_multiplier": args.ttft_multiplier,
        "tbt_multiplier": args.tbt_multiplier,
        "deadline_multiplier": args.deadline_multiplier,
        "arrival_process": "stationary_intensity",
        "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    ScenarioLoader.load(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
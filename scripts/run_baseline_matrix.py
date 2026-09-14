from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from agent_orch.baselines import make_policy
from agent_orch.metrics import summarize_slot_metrics
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def _parquet_record(metrics) -> dict:
    record = asdict(metrics)
    return {
        key: (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
        for key, value in record.items()
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_levels(path: str) -> list[tuple[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (str(level["name"]), float(level["rate_scale"]))
        for level in payload["levels"]
    ]


def _run_scale(scenario, seeds, policies, slots, rate_scale) -> list[dict]:
    records = []
    for seed in seeds:
        trace = ArrivalTrace.stationary_poisson_intensity(
            scenario, slots, rate_scale=rate_scale
        )
        for policy_name in policies:
            policy = make_policy(policy_name, scenario, seed)
            deployment = policy.deployment()
            simulator = Simulator(scenario)
            simulator.set_arrival_trace(trace)
            simulator.reset(seed)
            for _ in range(slots):
                routing = policy.routing(deployment, simulator.last_metrics)
                metrics = simulator.step(deployment, routing).metrics
                records.append(
                    {
                        "scenario": scenario.id,
                        "policy": policy_name,
                        "seed": seed,
                        "arrival_scale": rate_scale,
                        **_parquet_record(metrics),
                    }
                )
    return records


def _write(records, output: Path, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output, index=False)
    group_columns = ["scenario", "policy", "seed", "arrival_scale"]
    summary_rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                **summarize_slot_metrics(group.to_dict("records")),
            }
        )
    pd.DataFrame(summary_rows).to_parquet(
        output.with_name(f"{output.stem}.summary.parquet"), index=False
    )
    output.with_name(f"{output.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slots", type=int, default=600)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--policies", default="static,equal,least_load,random,greedy")
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the scenario arrival intensities",
    )
    parser.add_argument(
        "--load-levels",
        help="Load-level JSON emitted by calibrate_load_levels.py",
    )
    parser.add_argument("--output", default="results/baseline_matrix.parquet")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    seeds = [int(value) for value in _parse_csv_list(args.seeds)]
    policies = _parse_csv_list(args.policies)
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16]
    levels = _load_levels(args.load_levels) if args.load_levels else [("", args.arrival_scale)]

    for name, rate_scale in levels:
        records = _run_scale(scenario, seeds, policies, args.slots, rate_scale)
        if args.load_levels:
            output = Path(args.output) / f"baseline_load_{name}.parquet"
        else:
            output = Path(args.output)
        manifest = {
            "scenario": str(scenario_path),
            "scenario_hash": scenario_hash,
            "slots": args.slots,
            "seeds": seeds,
            "policies": policies,
            "arrival_process": "stationary_intensity",
            "arrival_scale": rate_scale,
            "load_level": name or None,
            "load_levels_source": (
                str(Path(args.load_levels).resolve()) if args.load_levels else None
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        _write(records, output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
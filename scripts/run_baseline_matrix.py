from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from agent_orch.backends import ProfileBackend
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slots", type=int, default=600)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--policies", default="static,equal,least_load,random,greedy")
    parser.add_argument("--profile", help="LLMServingSim/vLLM performance table CSV")
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the stationary Poisson rates in the scenario",
    )
    parser.add_argument("--output", default="results/baseline_matrix.parquet")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    profile = ProfileBackend.from_csv(args.profile) if args.profile else None
    records = []
    for seed in (int(value) for value in args.seeds.split(",")):
        trace = ArrivalTrace.stationary_poisson_intensity(
            scenario,
            args.slots,
            rate_scale=args.arrival_scale,
        )
        for policy_name in args.policies.split(","):
            policy = make_policy(policy_name, scenario, seed)
            deployment = policy.deployment()
            simulator = Simulator(scenario, llm_profile_backend=profile)
            simulator.set_arrival_trace(trace)
            simulator.reset(seed)
            for _ in range(args.slots):
                routing = policy.routing(deployment, simulator.last_metrics)
                metrics = simulator.step(deployment, routing).metrics
                records.append(
                    {
                        "scenario": scenario.id,
                        "policy": policy_name,
                        "seed": seed,
                        "arrival_scale": args.arrival_scale,
                        **_parquet_record(metrics),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output, index=False)
    summary_rows = []
    group_columns = ["scenario", "policy", "seed", "arrival_scale"]
    for keys, group in frame.groupby(group_columns, sort=False):
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                **summarize_slot_metrics(group.to_dict("records")),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(
        output.with_name(f"{output.stem}.summary.parquet"), index=False
    )
    scenario_path = Path(args.scenario).resolve()
    manifest = {
        "scenario": str(scenario_path),
        "scenario_hash": hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16],
        "slots": args.slots,
        "seeds": [int(value) for value in args.seeds.split(",")],
        "policies": args.policies.split(","),
        "arrival_process": "stationary_poisson_intensity",
        "arrival_scale": args.arrival_scale,
        "profile": str(Path(args.profile).resolve()) if args.profile else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    output.with_name(f"{output.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slots", type=int, default=600)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--policies", default="static,equal,least_load,random,greedy")
    parser.add_argument("--bursty", action="store_true")
    parser.add_argument("--output", default="results/baseline_matrix.parquet")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    records = []
    for seed in (int(value) for value in args.seeds.split(",")):
        trace = (
            ArrivalTrace.synthetic_bursty(scenario, args.slots, seed)
            if args.bursty
            else None
        )
        for policy_name in args.policies.split(","):
            policy = make_policy(policy_name, scenario, seed)
            deployment = policy.deployment()
            simulator = Simulator(scenario)
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
                        **asdict(metrics),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output, index=False)
    summary = (
        frame.groupby(["scenario", "policy", "seed"], as_index=False)
        .agg(
            mean_cost=("cost", "mean"),
            mean_latency_s=("mean_latency_s", "mean"),
            mean_goodput_rps=("goodput_rps", "mean"),
            mean_quality=("quality", "mean"),
            mean_slo_attainment=("slo_attainment", "mean"),
            mean_violations=("violations", "mean"),
        )
    )
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
        "bursty": args.bursty,
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

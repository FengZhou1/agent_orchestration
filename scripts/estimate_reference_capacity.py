from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from agent_orch.baselines import make_policy
from agent_orch.capacity import estimate_reference_capacity
from agent_orch.performance import AnalyticalBackend
from agent_orch.schema.loader import ScenarioLoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", default="greedy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/processed/reference_capacity.json")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    backend = AnalyticalBackend(scenario)
    policy = make_policy(args.policy, scenario, args.seed)
    deployment = policy.deployment()
    routing = policy.routing(deployment)
    estimate = estimate_reference_capacity(
        scenario, deployment, routing, backend
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(estimate),
        "scenario": str(Path(args.scenario).resolve()),
        "policy": args.policy,
        "seed": args.seed,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

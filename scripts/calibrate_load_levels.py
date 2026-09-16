from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from agent_orch.baselines import make_policy
from agent_orch.capacity import estimate_reference_capacity
from agent_orch.performance import AnalyticalBackend
from agent_orch.schema.loader import ScenarioLoader


LOAD_LEVELS = (
    ("low", 0.40),
    ("medium", 0.65),
    ("high", 0.85),
    ("overload", 1.05),
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", default="greedy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/processed/load_levels.json")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    backend = AnalyticalBackend(scenario)
    policy = make_policy(args.policy, scenario, args.seed)
    deployment = policy.deployment()
    routing = policy.routing(deployment)
    estimate = estimate_reference_capacity(scenario, deployment, routing, backend)

    levels = [
        {
            "name": name,
            "target_load": target,
            "rate_scale": estimate.arrival_scale * target,
            "total_arrival_rps": estimate.stable_capacity_rps * target,
        }
        for name, target in LOAD_LEVELS
    ]
    payload = {
        "scenario": _portable_path(scenario_path),
        "scenario_hash": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "policy": args.policy,
        "seed": args.seed,
        "arrival_process": "stationary_intensity",
        "reference_capacity": asdict(estimate),
        "levels": levels,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def set_composition(
    scenario: dict,
    short_app: str,
    long_app: str,
    long_fraction: float,
    total_rate_rps: float,
) -> dict:
    if not 0.0 <= long_fraction <= 1.0:
        raise ValueError("long_fraction must be in [0, 1]")
    applications = {app["id"]: app for app in scenario["applications"]}
    if short_app not in applications or long_app not in applications:
        raise KeyError("Both composition applications must exist in the scenario")
    result = yaml.safe_load(yaml.safe_dump(scenario, sort_keys=False))
    copied = {app["id"]: app for app in result["applications"]}
    _scale_ingress(copied[short_app], total_rate_rps * (1.0 - long_fraction))
    _scale_ingress(copied[long_app], total_rate_rps * long_fraction)
    suffix = f"long-{long_fraction:.3f}".replace(".", "p")
    result["id"] = f"{scenario['id']}-{suffix}"
    return result


def _scale_ingress(application: dict, target_rate: float) -> None:
    ingress = application["ingress_rates"]
    original = sum(float(value) for value in ingress.values())
    if original <= 0.0:
        uniform = target_rate / len(ingress)
        for node in ingress:
            ingress[node] = uniform
        return
    for node, value in ingress.items():
        ingress[node] = target_rate * float(value) / original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--short-app", required=True)
    parser.add_argument("--long-app", required=True)
    parser.add_argument("--long-fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--total-rate-rps", type=float)
    parser.add_argument("--output", default="configs/generated/composition")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    applications = {app["id"]: app for app in scenario["applications"]}
    if args.total_rate_rps is None:
        total_rate = sum(applications[args.short_app]["ingress_rates"].values())
        total_rate += sum(applications[args.long_app]["ingress_rates"].values())
    else:
        total_rate = args.total_rate_rps
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for raw_fraction in args.long_fractions.split(","):
        fraction = float(raw_fraction)
        derived = set_composition(
            scenario, args.short_app, args.long_app, fraction, total_rate
        )
        path = output / f"long_fraction_{fraction:.3f}.yaml"
        path.write_text(
            yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        generated.append({"long_fraction": fraction, "scenario": str(path)})
    manifest = {
        "source": str(scenario_path),
        "source_hash": hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16],
        "short_app": args.short_app,
        "long_app": args.long_app,
        "total_rate_rps": total_rate,
        "generated": generated,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

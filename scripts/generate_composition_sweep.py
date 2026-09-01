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


def set_family_composition(
    scenario: dict,
    dominant_family: str,
    total_rate_rps: float,
) -> dict:
    families = sorted({app.get("family", "unspecified") for app in scenario["applications"]})
    if dominant_family not in families or len(families) != 4:
        raise ValueError("Family sweeps require exactly four application families")
    others = [family for family in families if family != dominant_family]
    shares = {
        dominant_family: 0.60,
        others[0]: 0.20,
        others[1]: 0.10,
        others[2]: 0.10,
    }
    result = yaml.safe_load(yaml.safe_dump(scenario, sort_keys=False))
    by_family = {
        family: [app for app in result["applications"] if app.get("family") == family]
        for family in families
    }
    for family, applications in by_family.items():
        original = sum(sum(app["ingress_rates"].values()) for app in applications)
        target = total_rate_rps * shares[family]
        for application in applications:
            app_rate = sum(application["ingress_rates"].values())
            _scale_ingress(
                application,
                target * app_rate / original if original > 0.0 else target / len(applications),
            )
    result["id"] = f"{scenario['id']}-{dominant_family}-dominant"
    result.setdefault("metadata", {})["family_composition"] = shares
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--short-app")
    parser.add_argument("--long-app")
    parser.add_argument("--family-sweep", action="store_true")
    parser.add_argument("--long-fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--total-rate-rps", type=float)
    parser.add_argument("--output", default="configs/generated/composition")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    applications = {app["id"]: app for app in scenario["applications"]}
    if args.family_sweep:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        total_rate = args.total_rate_rps or sum(
            sum(app["ingress_rates"].values()) for app in scenario["applications"]
        )
        generated = []
        families = sorted({app.get("family", "unspecified") for app in scenario["applications"]})
        for family in families:
            derived = set_family_composition(scenario, family, total_rate)
            path = output / f"dominant_{family}.yaml"
            path.write_text(
                yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            generated.append({"dominant_family": family, "scenario": str(path)})
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "source": str(scenario_path),
                    "source_hash": hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16],
                    "total_rate_rps": total_rate,
                    "generated": generated,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(output)
        return 0
    if not args.short_app or not args.long_app:
        parser.error("--short-app and --long-app are required unless --family-sweep is used")
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

"""Audit how much each objective term actually discriminates decisions.

A term whose value barely moves between the policies, deployments and load
levels under comparison carries no gradient, whatever its nominal weight is.
This script measures that spread directly and flags dead terms, so a
mis-calibrated objective is caught before any training budget is spent.

Usage::

    python scripts/audit_objective_sensitivity.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --load-levels data/processed/load_levels.json \\
        --output results/objective_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_orch.baselines import make_policy
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.base import resolve_deployment_library
from agent_orch.objective import (
    ObjectiveAudit,
    ObjectiveEvaluator,
    ObjectiveSpec,
    ReferenceScales,
)
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace

DEFAULT_POLICIES = ("static", "equal", "least_load", "random", "greedy")
DEFAULT_PROFILES = ("legacy", "slo_constrained")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_levels(path: Path | None) -> list[tuple[str, float]]:
    if path is None or not path.exists():
        return [("default", 1.0)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    levels = payload.get("load_levels") or payload.get("levels") or []
    parsed: list[tuple[str, float]] = []
    for item in levels:
        name = str(item.get("name", "unnamed"))
        scale = item.get("rate_scale")
        if scale is None:
            continue
        parsed.append((name, float(scale)))
    return parsed or [("default", 1.0)]


def _run_policy(
    scenario,
    evaluator: ObjectiveEvaluator,
    policy_name: str,
    rate_scale: float,
    periods: int,
    warmup: int,
    mapping_samples: int,
    seed: int,
) -> list:
    trace = ArrivalTrace.stationary_poisson_intensity(scenario, periods, rate_scale=rate_scale)
    policy = make_policy(policy_name, scenario, seed)
    deployment = policy.deployment()
    simulator = Simulator(scenario, max_mapping_samples=mapping_samples)
    simulator.set_arrival_trace(trace)
    simulator.reset(seed)
    values = []
    for period in range(periods):
        routing = policy.routing(deployment, simulator.last_metrics)
        metrics = simulator.step(deployment, routing).metrics
        if period >= warmup:
            values.append(evaluator.evaluate(metrics, simulator.current_arrival_rates()))
    return values


def _run_library_deployments(
    scenario,
    evaluator: ObjectiveEvaluator,
    library: DeploymentLibrary,
    rate_scale: float,
    periods: int,
    warmup: int,
    mapping_samples: int,
    seed: int,
    limit: int,
) -> dict[str, list]:
    """Evaluate library deployments under a uniform composition.

    These are the contexts the composition policy is trained on, so their spread
    is exactly the signal available to it.
    """

    trace = ArrivalTrace.stationary_poisson_intensity(scenario, periods, rate_scale=rate_scale)
    grouped: dict[str, list] = {}
    for entry in library.entries[:limit]:
        deployment = entry.to_deployment()
        simulator = Simulator(scenario, max_mapping_samples=mapping_samples)
        simulator.set_arrival_trace(trace)
        simulator.reset(seed)
        arrival_rates = simulator.current_arrival_rates()
        uniform = {
            (app.id, ingress, model): 1.0 / max(1, len(scenario.models))
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
            for model in scenario.models
        }
        from agent_orch.routing import PhysicalRouter

        router = PhysicalRouter(scenario)
        routing = router.route(deployment, uniform, None, arrival_rates)
        values = []
        for period in range(periods):
            metrics = simulator.step(deployment, routing).metrics
            if period >= warmup:
                values.append(evaluator.evaluate(metrics, simulator.current_arrival_rates()))
        grouped.setdefault(entry.stratum, []).extend(values)
    return grouped


def _write_observations(path: Path, audit: ObjectiveAudit) -> None:
    rows = []
    for label, values in audit._utility.by_label.items():  # noqa: SLF001 - audit payload
        for value in values:
            rows.append({"label": label, "utility": value})
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "utility"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--load-levels", default="data/processed/load_levels.json")
    parser.add_argument("--deployment-library", default=None)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--periods", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--library-deployments", type=int, default=0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output", default="results/objective_audit")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    load_levels = _load_levels(
        Path(args.load_levels) if args.load_levels else None
    )
    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    profiles = [name.strip() for name in args.profiles.split(",") if name.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    library = resolve_deployment_library(
        scenario,
        library_path=args.deployment_library,
        auto=False,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "generated_at_utc": _timestamp(),
        "scenario": scenario.id,
        "scenario_path": args.scenario,
        "periods": args.periods,
        "warmup": args.warmup,
        "mapping_samples": args.mapping_samples,
        "load_levels": {name: scale for name, scale in load_levels},
        "policies": policies,
        "seeds": seeds,
        "profiles": {},
    }

    markdown: list[str] = [
        f"# Objective sensitivity audit — {scenario.id}",
        "",
        f"Generated {report['generated_at_utc']} · periods={args.periods} · "
        f"warmup={args.warmup} · mapping_samples={args.mapping_samples}",
        "",
    ]

    if library is not None:
        coverage = {
            "n_entries": len(library),
            "strata": library.strata_counts(),
            "n_active_model_sets": len(library.active_model_sets(scenario)),
        }
        report["deployment_library"] = coverage
        markdown.extend(
            [
                "## Deployment library",
                "",
                f"- entries: {coverage['n_entries']}",
                f"- strata: {len(coverage['strata'])}",
                f"- distinct active-model sets: {coverage['n_active_model_sets']}",
                "",
            ]
        )
    else:
        report["deployment_library"] = None
        markdown.extend(
            [
                "## Deployment library",
                "",
                "not found — cost bounds fall back to the theoretical range, which is "
                "far wider than any reachable deployment and flattens the cost term.",
                "",
            ]
        )

    for profile in profiles:
        if profile == "slo_constrained":
            spec = ObjectiveSpec.slo_constrained()
        else:
            spec = ObjectiveSpec.legacy()
        references = ReferenceScales.from_scenario(scenario, spec, library)
        evaluator = ObjectiveEvaluator(scenario, spec, references)
        audit = ObjectiveAudit(spec, references)

        for level_name, rate_scale in load_levels:
            for policy_name in policies:
                for seed in seeds:
                    values = _run_policy(
                        scenario,
                        evaluator,
                        policy_name,
                        rate_scale,
                        args.periods,
                        args.warmup,
                        args.mapping_samples,
                        seed,
                    )
                    audit.observe_many(f"{level_name}/{policy_name}", values)

        if library is not None and args.library_deployments > 0:
            grouped = _run_library_deployments(
                scenario,
                evaluator,
                library,
                load_levels[-1][1],
                args.periods,
                args.warmup,
                args.mapping_samples,
                seeds[0],
                args.library_deployments,
            )
            for stratum, values in grouped.items():
                audit.observe_many(f"library/{stratum}", values)

        report["profiles"][profile] = {
            "spec": spec.to_dict(),
            "references": references.to_dict(),
            "audit": audit.to_dict(),
        }
        markdown.append(f"## Profile `{profile}`")
        markdown.append("")
        markdown.append(
            f"- cost bounds: `{references.cost_source}` "
            f"[{references.cost_min:.6f}, {references.cost_max:.6f}]"
        )
        markdown.append(f"- latency reference: {references.latency_reference:.4f} s")
        markdown.append("")
        markdown.append(audit.to_markdown(title=f"`{profile}` term sensitivity"))
        markdown.append("")
        markdown.append("Label means (higher is better):")
        markdown.append("")
        markdown.append("| label | mean utility | n |")
        markdown.append("|---|---|---|")
        for label, mean in sorted(
            audit.label_means().items(), key=lambda item: -item[1]
        ):
            count = len(audit._utility.by_label[label])  # noqa: SLF001
            markdown.append(f"| {label} | {mean:+.5f} | {count} |")
        markdown.append("")

    (output_dir / "objective_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "objective_audit.md").write_text("\n".join(markdown), encoding="utf-8")
    for profile, payload in report["profiles"].items():  # type: ignore[union-attr]
        rows = payload["audit"]["terms"] + payload["audit"]["constraints"]
        with (output_dir / f"terms_{profile}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n".join(markdown))
    print(f"\nwrote {output_dir / 'objective_audit.md'}")
    print(f"wrote {output_dir / 'objective_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

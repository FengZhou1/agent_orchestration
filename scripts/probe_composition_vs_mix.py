"""Does the optimal composition move when the load *mix* moves? (Stage A, A1.4)

The decisive measurement for the rest of Stage A, and for whether Stage B has
anything to demonstrate.

Two measurements, in the same run:

* **curve** -- for a fixed deployment, sweep one group's share while the rest of
  the composition is held at an incumbent, under several load mixes, and record
  the argmax.  If the optimal share for a group is the same under every mix, the
  composition sub-problem is a point and there is no function to learn; if it
  moves, the reference has to be built as a function of the load (A2).
* **ranking** -- does the ordering of deployments by utility change across mixes?
  If it does not, deployment stays a static choice.

Load vectors come from :meth:`ArrivalTrace.randomized_mix_intensity`, which draws
each (application, ingress) group independently: scaling every group together would
only move the operating point along one direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import CompositionEnvScorer
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec, ReferenceScales
from agent_orch.routing.composition import CompositionSolver
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCALE = 7.165234375
# Share of the swept group's traffic given to --preferred-model; the remainder is
# split evenly over the other active models, so every point on the axis is a valid
# distribution.
SHARE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _shortlist(library: DeploymentLibrary, count: int) -> list[int]:
    """A deterministic, capability-weighted spread over the library."""

    entries = library.entries
    ordered = sorted(
        range(len(entries)),
        key=lambda position: (-entries[position].n_models, -entries[position].n_llm, position),
    )
    chosen: list[int] = []
    for stride in (1, 3, 7):
        for position in ordered[::stride]:
            if position not in chosen:
                chosen.append(position)
            if len(chosen) >= count:
                return sorted(chosen)
    return sorted(chosen)


def _mix(scenario, args, index: int) -> ArrivalTrace:
    return ArrivalTrace.randomized_mix_intensity(
        scenario,
        args.periods,
        base_scale=BASE_SCALE,
        seed=args.seed + index,
        block=args.block,
    )


def _scorer(scenario, evaluator, objective, trace, library, position, args):
    return CompositionEnvScorer(
        scenario,
        evaluator,
        objective,
        trace,
        library,
        position=position,
        periods=args.periods,
        warmup=args.warmup,
        mapping_samples=args.mapping_samples,
        seed=0,
    )


def _group_rates(scenario) -> list[tuple[str, str]]:
    return [
        (app.id, ingress)
        for app in scenario.applications.values()
        for ingress in app.ingress_rates
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--deployments", type=int, default=6)
    parser.add_argument("--mixes", type=int, default=3)
    parser.add_argument("--groups", type=int, default=6, help="groups swept per deployment")
    parser.add_argument(
        "--preferred-model",
        default="qwen3-32b",
        help="the sweep varies this model's share within each swept group",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--output", default="results/stage_a/composition_vs_mix.json")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(REPO_ROOT / args.scenario)
    objective = ObjectiveSpec.slo_constrained(0.9)
    library = DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))
    references = ReferenceScales.from_scenario(scenario, objective, library)
    evaluator = ObjectiveEvaluator(scenario, objective, references)
    positions = _shortlist(library, args.deployments)
    groups = _group_rates(scenario)
    traces = [_mix(scenario, args, index) for index in range(args.mixes)]
    print(f"library {len(library)} deployments; probing {len(positions)}: {positions}")
    print(f"mixes: {args.mixes} (seeds {args.seed}..{args.seed + args.mixes - 1}), "
          f"{len(groups)} groups, sweeping {args.groups} per deployment")
    for index, trace in enumerate(traces):
        totals = [sum(trace.at(slot, scenario).values()) for slot in range(args.periods)]
        print(f"  mix {index}: total arrival {min(totals):.5f}-{max(totals):.5f}")

    solver = CompositionSolver(
        scenario,
        evaluator,
        mapping_samples=args.mapping_samples,
        seed=0,
        protocol_periods=args.periods,
        protocol_warmup=args.warmup,
    )

    records: list[dict] = []
    for position in positions:
        entry = library.entries[position]
        deployment = entry.to_deployment()
        for mix_index, trace in enumerate(traces):
            solver.scorer = _scorer(
                scenario, evaluator, objective, trace, library, position, args
            )
            rates = trace.at(0, scenario)
            solution = solver.solve(deployment, rates, budget=args.budget, sweeps=1)
            incumbent = dict(solution.model_share)
            records.append(
                {
                    "position": position,
                    "index": entry.index,
                    "stratum": entry.stratum,
                    "mix": mix_index,
                    "utility": float(solution.utility),
                    "source": solution.source,
                    "share": {
                        f"{k[0]}|{k[1]}|{k[2]}": float(v) for k, v in incumbent.items()
                    },
                }
            )
            print(
                f"\n  position {position:>3} ({entry.stratum}), mix {mix_index}: "
                f"incumbent {solution.utility:+.5f} ({solution.source})"
            )

            # Sweep the groups that carry the most traffic: does their optimal share
            # depend on the mix?
            by_rate = sorted(
                groups,
                key=lambda group: -sum(
                    rate for (app_id, ingress), rate in rates.items()
                    if (app_id, ingress) == group
                ),
            )[: args.groups]
            active = [model for model in scenario.models]
            for app_id, ingress in by_rate:
                best_share, best_utility = None, -np.inf
                curve = []
                others = [model for model in active if model != args.preferred_model]
                for candidate_share in SHARE_GRID:
                    trial = dict(incumbent)
                    remainder = (
                        (1.0 - candidate_share) / len(others) if others else 0.0
                    )
                    for model in active:
                        trial[(app_id, ingress, model)] = (
                            candidate_share if model == args.preferred_model else remainder
                        )
                    utility = solver.evaluate_share(deployment, trial, rates)
                    curve.append((candidate_share, float(utility)))
                    if utility > best_utility:
                        best_utility, best_share = float(utility), candidate_share
                if best_share is None:
                    continue
                records.append(
                    {
                        "position": position,
                        "index": entry.index,
                        "mix": mix_index,
                        "group": f"{app_id}|{ingress}",
                        "curve": curve,
                        "argmax_share": best_share,
                        "argmax_utility": best_utility,
                        "incumbent_share": float(
                            incumbent.get((app_id, ingress, args.preferred_model), 0.0)
                        ),
                    }
                )

    # Do the per-group argmaxes move with the mix?
    moved, total = 0, 0
    per_group: dict[str, dict[int, float]] = {}
    for record in records:
        if "argmax_share" not in record:
            continue
        key = f"{record['position']}|{record['group']}"
        per_group.setdefault(key, {})[record["mix"]] = record["argmax_share"]
    spread: list[float] = []
    for key, by_mix in per_group.items():
        if len(by_mix) < 2:
            continue
        total += 1
        values = list(by_mix.values())
        spread.append(max(values) - min(values))
        if max(values) - min(values) > 1.0e-9:
            moved += 1

    incumbent_records = [record for record in records if "source" in record]
    rankings: dict[int, list[int]] = {}
    for mix_index in range(args.mixes):
        ordered = sorted(
            [record for record in incumbent_records if record["mix"] == mix_index],
            key=lambda record: -record["utility"],
        )
        rankings[mix_index] = [record["index"] for record in ordered]
    best_per_mix = {str(mix): ranking[0] for mix, ranking in rankings.items()}
    overlap = []
    for i in range(args.mixes):
        for j in range(i + 1, args.mixes):
            overlap.append(len(set(rankings[i][:3]) & set(rankings[j][:3])) / 3.0)

    incumbent_spread = []
    for position in positions:
        utilities = [
            record["utility"]
            for record in incumbent_records
            if record["position"] == position
        ]
        if len(utilities) > 1:
            incumbent_spread.append(max(utilities) - min(utilities))

    report = {
        "deployments_probed": positions,
        "mixes": args.mixes,
        "seed": args.seed,
        "budget": args.budget,
        "protocol_periods": args.periods,
        "protocol_warmup": args.warmup,
        "block": args.block,
        "groups_swept_per_deployment": args.groups,
        "group_argmax_moved": moved,
        "group_argmax_total": total,
        "group_argmax_share_spread_mean": float(np.mean(spread)) if spread else 0.0,
        "group_argmax_share_spread_max": float(np.max(spread)) if spread else 0.0,
        "best_deployment_by_mix": best_per_mix,
        "distinct_best_deployments": sorted(set(best_per_mix.values())),
        "mean_top3_overlap": float(np.mean(overlap)) if overlap else 1.0,
        "incumbent_utility_spread_mean": float(np.mean(incumbent_spread)) if incumbent_spread else 0.0,
        "records": records,
    }
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(
        f"per-group optimal share moved across mixes in {moved}/{total} (group, deployment) cases; "
        f"mean spread {report['group_argmax_share_spread_mean']:.3f}, "
        f"max {report['group_argmax_share_spread_max']:.3f}"
    )
    print(f"best deployment per mix: {best_per_mix}  (distinct: {report['distinct_best_deployments']})")
    print(f"mean top-3 overlap between mixes: {report['mean_top3_overlap']:.2f}")
    print()
    if total and moved / total >= 0.5:
        print("VERDICT-COMPOSITION: the optimal composition depends on the mix, so the")
        print("reference must be a function of the load (A2) and the policy has a real")
        print("target to learn.")
    else:
        print("VERDICT-COMPOSITION: the optimal share per group barely moves with the mix.")
        print("Randomising the load would add context variance without a learnable target;")
        print("re-examine the objective before building the reference family.")
    if len(report["distinct_best_deployments"]) > 1:
        print("VERDICT-DEPLOYMENT: the best deployment changes across mixes, so a")
        print("time-varying deployment decision has value to demonstrate.")
    else:
        print("VERDICT-DEPLOYMENT: the best deployment is the same across mixes; the")
        print("deployment sub-problem stays static on this axis.")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

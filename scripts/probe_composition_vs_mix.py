"""Does the optimal composition move when the load *mix* moves? (Stage A, A1.4)

The decisive measurement for the rest of Stage A, and for whether Stage B has
anything to demonstrate.  Two questions, both answered on the same runs:

1. For a fixed deployment, how far does the optimal share vector move between
   different load mixes?  If it barely moves, the reference is a point rather than
   a function, distillation is memorising a table, and no amount of randomising
   the load gives a policy anything to learn.
2. Does the *ranking* of deployments change across mixes?  If it does not,
   deployment is a static choice and "dynamic deployment reconfiguration" has no
   value to show.

Load vectors come from :meth:`ArrivalTrace.randomized_mix_intensity`, which draws
each (application, ingress) group independently -- scaling every group together
would only move the operating point along one direction.
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def _shortlist(library: DeploymentLibrary, count: int) -> list[int]:
    """A deterministic spread over the library, weighted to capable deployments."""

    entries = library.entries
    by_capability = sorted(
        range(len(entries)),
        key=lambda position: (-entries[position].n_models, -entries[position].n_llm, position),
    )
    chosen: list[int] = []
    for stride in (1, 3, 7):
        for position in by_capability[::stride]:
            if position not in chosen:
                chosen.append(position)
            if len(chosen) >= count:
                return sorted(chosen)
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--deployments", type=int, default=12)
    parser.add_argument("--mixes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budget", type=int, default=12)
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
    print(f"library {len(library)} deployments, probing {len(positions)}: {positions}")

    solver = CompositionSolver(
        scenario,
        evaluator,
        mapping_samples=args.mapping_samples,
        seed=0,
        protocol_periods=args.periods,
        protocol_warmup=args.warmup,
    )

    records: list[dict] = []
    for mix_index in range(args.mixes):
        trace = ArrivalTrace.randomized_mix_intensity(
            scenario,
            args.periods,
            base_scale=BASE_SCALE,
            seed=args.seed + mix_index,
            block=args.block,
        )
        totals = [sum(trace.at(slot, scenario).values()) for slot in range(args.periods)]
        print(f"\nmix {mix_index}: total arrival {min(totals):.5f}-{max(totals):.5f}")
        for position in positions:
            entry = library.entries[position]
            solver.scorer = CompositionEnvScorer(
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
            solution = solver.solve(
                entry.to_deployment(), trace.at(0, scenario), budget=args.budget, sweeps=1
            )
            records.append(
                {
                    "mix": mix_index,
                    "position": position,
                    "index": entry.index,
                    "stratum": entry.stratum,
                    "n_models": entry.n_models,
                    "utility": float(solution.utility),
                    "share": {f"{k[0]}|{k[1]}|{k[2]}": float(v) for k, v in solution.model_share.items()},
                }
            )
            print(
                f"  {entry.index:>4} {entry.stratum:<28} n_models={entry.n_models} "
                f"utility={solution.utility:+.5f} source={solution.source}"
            )

    # 1. How far does the optimum move across mixes, per deployment?
    movement: list[float] = []
    per_deployment: list[dict] = []
    for position in positions:
        shares = [
            record["share"] for record in records if record["position"] == position
        ]
        if len(shares) < 2:
            continue
        keys = sorted(shares[0])
        distances = []
        for i in range(len(shares)):
            for j in range(i + 1, len(shares)):
                distances.append(
                    0.5 * sum(abs(shares[i][k] - shares[j][k]) for k in keys)
                )
        mean_distance = float(np.mean(distances)) if distances else 0.0
        movement.append(mean_distance)
        utilities = [
            record["utility"] for record in records if record["position"] == position
        ]
        per_deployment.append(
            {
                "position": position,
                "index": records[positions.index(position)]["index"],
                "stratum": records[positions.index(position)]["stratum"],
                "mean_share_move": mean_distance,
                "utility_by_mix": utilities,
                "utility_spread": float(max(utilities) - min(utilities)),
            }
        )

    # 2. Does the deployment ranking change across mixes?
    rankings = {}
    overlap = []
    for mix_index in range(args.mixes):
        ordered = sorted(
            [record for record in records if record["mix"] == mix_index],
            key=lambda record: -record["utility"],
        )
        rankings[mix_index] = [record["index"] for record in ordered]
    for i in range(args.mixes):
        for j in range(i + 1, args.mixes):
            top_three_i = set(rankings[i][:3])
            top_three_j = set(rankings[j][:3])
            overlap.append(len(top_three_i & top_three_j) / 3.0)

    best_per_mix = {mix: rankings[mix][0] for mix in rankings}
    distinct_best = sorted(set(best_per_mix.values()))

    report = {
        "deployments_probed": positions,
        "mixes": args.mixes,
        "budget": args.budget,
        "protocol_periods": args.periods,
        "protocol_warmup": args.warmup,
        "block": args.block,
        "base_scale": BASE_SCALE,
        "mean_share_move_across_mixes": float(np.mean(movement)) if movement else 0.0,
        "min_share_move": float(np.min(movement)) if movement else 0.0,
        "max_share_move": float(np.max(movement)) if movement else 0.0,
        "best_deployment_by_mix": {str(k): v for k, v in best_per_mix.items()},
        "distinct_best_deployments": distinct_best,
        "mean_top3_overlap": float(np.mean(overlap)) if overlap else 1.0,
        "per_deployment": per_deployment,
        "records": records,
    }
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"optimal-share movement across mixes: mean L1(0.5) = {report['mean_share_move_across_mixes']:.4f} "
          f"(min {report['min_share_move']:.4f}, max {report['max_share_move']:.4f})")
    print(f"best deployment per mix: {report['best_deployment_by_mix']}")
    print(f"distinct best deployments: {distinct_best}")
    print(f"mean top-3 overlap between mixes: {report['mean_top3_overlap']:.2f}")
    print()
    if report["mean_share_move_across_mixes"] < 0.02:
        print("VERDICT: the composition optimum barely moves with the mix.  The reference")
        print("is a point, not a function: randomising the load adds context variance")
        print("without adding a learnable target.  Re-examine the objective.")
    else:
        print("VERDICT: the composition optimum moves with the mix, so a policy that")
        print("conditions on the load has something to learn.  Build the reference as a")
        print("function of the load (A2).")
    if len(distinct_best) > 1:
        print(f"DEPLOYMENT: the best deployment changes across mixes ({len(distinct_best)} distinct),")
        print("so a time-varying deployment decision has value to demonstrate.")
    else:
        print("DEPLOYMENT: the best deployment is the same across mixes; the deployment")
        print("sub-problem stays static and Stage B needs a different variation axis.")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

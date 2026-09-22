"""Is there an action-discriminating signal in the composition problem? (A4, leg 1)

The A4 diagnosis distinguishes three reasons a policy can fail: no signal, a
parameterisation ceiling, or an optimisation failure.  This measures the first.

For the policy's advantage estimate to carry information about *which action is
better*, the utility has to vary more across actions within a context than it does
across contexts.  If the spread across contexts dominates, the advantage mostly
encodes "which deployment/load am I in" -- something the policy cannot act on.

Reports both spreads over the training load family, with the ratio between them.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--deployments", default="2,3,14,23")
    parser.add_argument("--mixes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--periods", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--output", default="results/stage_a/advantage_signal.json")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(REPO_ROOT / args.scenario)
    objective = ObjectiveSpec.slo_constrained(0.9)
    library = DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))
    references = ReferenceScales.from_scenario(scenario, objective, library)
    evaluator = ObjectiveEvaluator(scenario, objective, references)
    positions = [int(value) for value in args.deployments.split(",") if value.strip()]
    solver = CompositionSolver(
        scenario,
        evaluator,
        mapping_samples=args.mapping_samples,
        protocol_periods=args.periods,
        protocol_warmup=args.warmup,
    )

    table: dict[tuple[int, int], dict[str, float]] = {}
    for mix in range(args.mixes):
        trace = ArrivalTrace.randomized_mix_intensity(
            scenario,
            args.periods,
            base_scale=BASE_SCALE,
            seed=args.seed + mix,
            block=args.periods,
        )
        rates = trace.at(0, scenario)
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
            utilities = {
                candidate.name: solver.evaluate_share(
                    entry.to_deployment(), candidate.share, rates
                )
                for candidate in solver.canonical_candidates(entry.to_deployment())
            }
            table[(position, mix)] = utilities

    names = sorted(next(iter(table.values())))
    within: list[float] = []
    for utilities in table.values():
        values = [utilities[name] for name in names]
        within.append(float(np.std(values)))
    between: list[float] = []
    for name in names:
        values = [utilities[name] for utilities in table.values()]
        between.append(float(np.std(values)))

    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    print(f"{'context':>12} " + " ".join(f"{n:>16}" for n in names))
    for (position, mix), utilities in sorted(table.items()):
        print(f"{f'{position}/{mix}':>12} " + " ".join(f"{utilities[n]:>16.5f}" for n in names))
    print()
    print(f"action spread within a context (mean std over candidates): {within_mean:.5f}")
    print(f"context spread for a fixed action    (mean std over contexts): {between_mean:.5f}")
    print(f"ratio within/between: {within_mean / between_mean if between_mean else float('inf'):.3f}")
    verdict = within_mean > between_mean
    print(
        "VERDICT: the signal about *which action is better* dominates the context spread."
        if verdict
        else "VERDICT: the context spread dominates; the advantage mostly encodes which "
        "context the policy is in."
    )

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "within_context_std": within_mean,
                "between_context_std": between_mean,
                "ratio": within_mean / between_mean if between_mean else None,
                "table": {
                    f"{position}/{mix}": utilities
                    for (position, mix), utilities in sorted(table.items())
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

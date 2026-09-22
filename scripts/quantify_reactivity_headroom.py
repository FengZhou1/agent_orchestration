"""Quantify how much a deployment policy could gain from reacting to the load.

Builds a utility table ``U[deployment][load level]`` by evaluating every
deployment in the library under a constant arrival intensity, then compares:

* **best fixed** -- one deployment held for the whole run, scored against the load
  distribution the workload actually presents;
* **best reactive** -- the best deployment for each load level, chosen with
  perfect foresight and no switching cost, which is an upper bound on any policy.

The gap is the headroom a deployment policy could in principle capture.  If it is
near zero the deployment sub-problem is a static optimisation and reinforcement
learning buys nothing, however well it is tuned -- which is why this is measured
before any training.

Usage::

    python scripts/quantify_reactivity_headroom.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --composition-policy results/stage_a/pretrain/policy_pretrained.pt \\
        --load-levels 0.4,0.7,1.0 --periods 14
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

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import build_composition_env
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace


def _evaluate(
    scenario,
    library,
    policy,
    rate_scale: float,
    periods: int,
    warmup: int,
    mapping_samples: int,
    seed: int,
) -> list[float]:
    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, periods, rate_scale=rate_scale
    )
    utilities: list[float] = []
    for position in range(len(library.entries)):
        # The whole library goes in and the deployment is pinned, exactly as during
        # training and in the gate.  Handing the policy a one-entry subset would
        # change the deployment-count features it observes and make it act on a
        # context it never saw.
        env = build_composition_env(
            scenario,
            policy["objective_spec"],
            trace,
            library,
            position=position,
            periods=periods,
            mapping_samples=mapping_samples,
            seed=seed,
        )
        observation, _ = env.reset(seed=seed)
        for slot in range(periods):
            action, _, _ = policy["policy"].act(observation, deterministic=True, device="cpu")
            observation, _, terminated, truncated, info = env.step(action)
            if info.get("period_complete") and slot >= warmup:
                utilities.append(float(info["utility"]))
            if terminated or truncated:
                break
    return utilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--composition-policy", required=True)
    parser.add_argument("--deployment-library", default=None)
    parser.add_argument(
        "--load-levels",
        default="0.4,0.7,1.0",
        help="arrival scales to sweep, as fractions of --arrival-scale",
    )
    parser.add_argument("--arrival-scale", type=float, default=7.165234375)
    parser.add_argument(
        "--load-weights",
        default=None,
        help=(
            "relative time spent at each load level; defaults to uniform, which is "
            "what a bursty trace with a symmetric burst approximates"
        ),
    )
    parser.add_argument("--periods", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--mapping-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective-profile", default="slo_constrained")
    parser.add_argument("--no-attainment-constraint", action="store_true")
    parser.add_argument("--output", default="results/stage_b/reactivity_headroom")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    library = DeploymentLibrary.load(
        args.deployment_library or DeploymentLibrary.default_path(scenario.id)
    )
    spec = (
        ObjectiveSpec.slo_constrained(
            None if args.no_attainment_constraint else 0.9
        )
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    fractions = [float(value) for value in args.load_levels.split(",") if value.strip()]
    weights = (
        [float(value) for value in args.load_weights.split(",") if value.strip()]
        if args.load_weights
        else [1.0 / len(fractions)] * len(fractions)
    )
    if len(weights) != len(fractions):
        raise ValueError("--load-weights must have one entry per load level")
    total_weight = sum(weights)
    weights = [value / total_weight for value in weights]

    # Only built to give the actor its observation and action shapes.
    probe_env = build_composition_env(
        scenario,
        spec,
        ArrivalTrace.stationary_poisson_intensity(
            scenario, 1, rate_scale=args.arrival_scale
        ),
        library,
        position=0,
        periods=1,
        mapping_samples=8,
        seed=args.seed,
    )
    payload = torch_load(Path(args.composition_policy))
    actor = StructuredActorCritic(probe_env, PPOConfig())
    actor.load_state_dict(payload.get("policy_state_dict", payload))
    actor.eval()
    policy = {"policy": actor, "objective_spec": spec}

    table: dict[int, list[float]] = {}
    load_means: list[float] = []
    for fraction in fractions:
        values = _evaluate(
            scenario,
            library,
            policy,
            rate_scale=args.arrival_scale * fraction,
            periods=args.periods,
            warmup=args.warmup,
            mapping_samples=args.mapping_samples,
            seed=args.seed,
        )
        per_entry = np.asarray(values).reshape(len(library.entries), args.periods - args.warmup)
        table[int(round(fraction * 1000))] = per_entry.mean(axis=1).tolist()
        load_means.append(float(per_entry.mean()))
        print(
            "load %.2f x baseline: mean utility %.5f, best deployment %.5f, spread %.5f"
            % (
                fraction,
                float(per_entry.mean()),
                float(per_entry.mean(axis=1).max()),
                float(per_entry.mean(axis=1).max() - per_entry.mean(axis=1).min()),
            ),
            flush=True,
        )

    keys = list(table)
    utilities = np.asarray([table[key] for key in keys])  # (levels, deployments)
    weights_array = np.asarray(weights)[:, None]
    per_deployment = (utilities * weights_array).sum(axis=0)  # (deployments,)
    best_fixed_index = int(per_deployment.argmax())
    best_fixed = float(per_deployment[best_fixed_index])
    # Perfect foresight, no switching cost: an upper bound on any policy.
    per_level_best = utilities.max(axis=1)
    best_reactive = float((per_level_best * np.asarray(weights)).sum())
    headroom = best_reactive - best_fixed

    rows = []
    for position, entry in enumerate(library.entries):
        rows.append(
            {
                "index": entry.index,
                "stratum": entry.stratum,
                "n_models": entry.n_models,
                "cost_per_slot": entry.cost_per_slot,
                **{f"utility_at_{key}": round(utilities[i, position], 6) for i, key in enumerate(keys)},
                "weighted_utility": round(float(per_deployment[position]), 6),
            }
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "utility_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_per_level = [
        {
            "load_fraction": fractions[i],
            "best_index": int(utilities[i].argmax()),
            "best_stratum": library.entries[int(utilities[i].argmax())].stratum,
            "best_utility": float(utilities[i].max()),
        }
        for i in range(len(keys))
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.id,
        "deployment_library": str(args.deployment_library or DeploymentLibrary.default_path(scenario.id)),
        "composition_policy": str(args.composition_policy),
        "load_fractions": fractions,
        "load_weights": weights,
        "best_fixed": {
            "index": library.entries[best_fixed_index].index,
            "stratum": library.entries[best_fixed_index].stratum,
            "utility": best_fixed,
        },
        "best_reactive_upper_bound": best_reactive,
        "headroom": headroom,
        "relative_headroom": headroom / max(abs(best_fixed), 1.0e-12),
        "best_per_level": best_per_level,
        "mean_utility_per_level": load_means,
    }
    (output_dir / "reactivity_headroom.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    lines = [
        f"# Reactivity headroom — {scenario.id}",
        "",
        f"library {report['deployment_library']} ({len(library)} deployments), "
        f"composition policy `{args.composition_policy}`",
        f"load levels (x baseline {args.arrival_scale:g}): {fractions}, weights {[round(w,3) for w in weights]}",
        "",
        "| load fraction | mean utility | best deployment utility | spread over library | best deployment |",
        "|---|---|---|---|---|",
    ]
    for index, fraction in enumerate(fractions):
        lines.append(
            "| %.2f | %+.5f | %+.5f | %.5f | %s |"
            % (
                fraction,
                load_means[index],
                best_per_level[index]["best_utility"],
                float(utilities[index].max() - utilities[index].min()),
                best_per_level[index]["best_stratum"],
            )
        )
    lines.extend(
        [
            "",
            f"- best fixed deployment: `{report['best_fixed']['stratum']}` at {best_fixed:+.5f}",
            f"- best reactive (perfect foresight, no switching cost): {best_reactive:+.5f}",
            f"- **headroom: {headroom:+.5f}** ({100 * report['relative_headroom']:.1f}% of best fixed)",
            "",
            (
                "A near-zero headroom means the deployment sub-problem is a static "
                "optimisation under this load distribution and a deployment policy "
                "cannot beat the best fixed deployment by reacting."
                if headroom < 0.005
                else "The load distribution leaves real headroom for a reactive deployment policy."
            ),
            "",
        ]
    )
    (output_dir / "reactivity_headroom.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())

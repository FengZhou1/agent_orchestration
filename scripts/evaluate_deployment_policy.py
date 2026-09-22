"""Score one deployment under the frozen composition policy.

Stage B decides the deployment while the composition policy answers the
per-cycle composition question.  The two have to be measured together, so this
evaluates an arbitrary deployment with a fixed composition policy and an
identical arrival trace for every candidate, which is what makes a deployment's
score comparable across the baselines and the learned policy.

The deployment is wrapped in a one-entry library so the environment machinery
(observation, objective, router) is exactly the one training uses.

Usage::

    python scripts/evaluate_deployment_policy.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --composition-policy results/stage_a/pretrain/policy_pretrained.pt \\
        --deployments static,equal,least_load,random,greedy,initial \\
        --seeds 0,1,2 --periods 40 --output results/stage_b/deployment_bar
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
from agent_orch.baselines import make_policy
from agent_orch.capacity import CapacityPlanner
from agent_orch.deployment import DeploymentEntry, DeploymentLibrary, deployment_signature
from agent_orch.envs import CompositionLibraryEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

BASELINE_DEPLOYMENTS = ("static", "equal", "least_load", "random", "greedy")


def _entry_for(scenario, deployment, index: int, label: str) -> DeploymentEntry:
    planner = CapacityPlanner(scenario)
    slot_seconds = scenario.simulation.slot_seconds
    active = [cid for cid, on in deployment.llm_active.items() if on]
    models = {scenario.candidates[cid].model for cid in active}
    total_gpu = sum(scenario.llm_configs[scenario.candidates[cid].config].gpu_count for cid in active)
    steady = sum(
        scenario.llm_configs[scenario.candidates[cid].config].running_cost_per_slot * slot_seconds
        for cid in active
    ) + sum(
        replicas * scenario.tools[tool_id].running_cost_per_slot * period
        for (tool_id, _server), replicas in deployment.tool_replicas.items()
    )
    if not planner.deployment_feasible(deployment):
        raise ValueError(f"{label} is not a feasible deployment")
    return DeploymentEntry(
        index=index,
        stratum=f"deployment:{label}",
        llm_active=dict(deployment.llm_active),
        tool_replicas=dict(deployment.tool_replicas),
        n_models=len(models),
        n_llm=len(active),
        n_tool_replicas=int(sum(deployment.tool_replicas.values())),
        total_gpu=int(total_gpu),
        cost_per_slot=float(steady),
        signature=deployment_signature(deployment.llm_active, deployment.tool_replicas),
    )


def _single_entry_library(scenario, deployment, label: str, index: int) -> DeploymentLibrary:
    entry = _entry_for(scenario, deployment, index, label)
    return DeploymentLibrary(
        scenario_id=scenario.id,
        scenario_hash=DeploymentLibrary.scenario_hash_of(scenario),
        entries=(entry,),
        metadata={"role": "single_deployment_probe", "label": label},
    )


def _resolve_deployment(scenario, spec: str):
    """Return ``(label, DeploymentDecision)`` for a baseline name or a checkpoint."""

    if spec in BASELINE_DEPLOYMENTS:
        return spec, make_policy(spec, scenario, 0).deployment()
    if spec == "initial":
        return "initial", CapacityPlanner(scenario).initial_deployment()
    payload_path = Path(spec)
    if not payload_path.exists():
        raise FileNotFoundError(
            f"{spec} is neither a baseline ({', '.join(BASELINE_DEPLOYMENTS)}, initial) "
            "nor an existing checkpoint"
        )
    raise ValueError(
        "checkpoint deployments are evaluated by training-time rollout; pass a baseline "
        "name or 'initial' here"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--composition-policy", required=True)
    parser.add_argument("--deployments", default=",".join(BASELINE_DEPLOYMENTS) + ",initial")
    parser.add_argument(
        "--library-scan",
        default=None,
        help=(
            "evaluate every deployment in this library file. With a stationary "
            "arrival trace the best fixed deployment is the global optimum, so this "
            "is the bar a deployment policy has to clear"
        ),
    )
    parser.add_argument("--library-split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--library-limit", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--arrival-scale", type=float, default=7.165234375)
    parser.add_argument("--objective-profile", default="slo_constrained")
    parser.add_argument("--no-attainment-constraint", action="store_true")
    parser.add_argument("--mapping-samples", type=int, default=64)
    parser.add_argument("--deployment-periods", type=int, default=1)
    parser.add_argument("--arrival-pattern", default="stationary", choices=["stationary", "bursty"])
    parser.add_argument("--burst-period", type=int, default=60)
    parser.add_argument("--burst-duty", type=float, default=0.5)
    parser.add_argument("--burst-low-fraction", type=float, default=0.5)
    parser.add_argument("--output", default="results/stage_b/deployment_bar")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    spec = (
        ObjectiveSpec.slo_constrained(
            None if args.no_attainment_constraint else 0.9
        )
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    labels = [name.strip() for name in args.deployments.split(",") if name.strip()]

    if args.library_scan:
        full = DeploymentLibrary.load(args.library_scan)
        if args.library_split != "all":
            train_library, test_library = full.train_test_split()
            full = train_library if args.library_split == "train" else test_library
        entries = list(full.entries)
        if args.library_limit > 0:
            entries = entries[: args.library_limit]
        probes = [(entry.stratum, entry.to_deployment(), full.subset([entry.index])) for entry in entries]
    else:
        probes = []
        for label in labels:
            resolved_label, deployment = _resolve_deployment(scenario, label)
            probes.append(
                (resolved_label, deployment, _single_entry_library(scenario, deployment, resolved_label, 0))
            )

    rows: list[dict] = []
    for probe_index, (resolved_label, deployment, library) in enumerate(probes):
        label = resolved_label if args.library_scan else resolved_label
        if args.arrival_pattern == "bursty":
            trace = ArrivalTrace.bursty_intensity(
                scenario,
                args.periods,
                low_scale=args.arrival_scale * args.burst_low_fraction,
                high_scale=args.arrival_scale,
                period=args.burst_period,
                duty=args.burst_duty,
            )
        else:
            trace = ArrivalTrace.stationary_poisson_intensity(
                scenario, args.periods, rate_scale=args.arrival_scale
            )
        env = CompositionLibraryEnv(
            scenario,
            max_slots=args.periods,
            seed=seeds[0],
            arrival_trace=trace,
            mapping_samples=args.mapping_samples,
            deployment_periods=args.deployment_periods,
            objective=spec,
            deployment_library=library,
            # Pure evaluation: the control variate only matters for the training
            # reward, and computing it here would re-run the episode a second time.
            use_uniform_baseline=False,
        )
        payload = torch_load(Path(args.composition_policy))
        policy = StructuredActorCritic(env, PPOConfig())
        policy.load_state_dict(payload.get("policy_state_dict", payload))
        policy.eval()

        for seed in seeds:
            observation, _ = env.reset(
                seed=seed, options={"fixed_deployment_index": 0}
            )
            utilities: list[float] = []
            attainment: list[float] = []
            costs: list[float] = []
            latencies: list[float] = []
            violations = 0
            for period in range(args.periods):
                action, _, _ = policy.act(observation, deterministic=True, device="cpu")
                observation, _, terminated, truncated, info = env.step(action)
                if info.get("period_complete"):
                    metrics = info["metrics"]
                    violations += int(metrics.violations)
                    if period >= args.warmup:
                        utilities.append(float(info["utility"]))
                        attainment.append(float(metrics.slo_attainment))
                        costs.append(float(metrics.cost))
                        latencies.append(float(metrics.mean_latency_s))
                if terminated or truncated:
                    break
            rows.append(
                {
                    "deployment": ("%03d %s" % (probe_index, label)) if args.library_scan else label,
                    "seed": seed,
                    "n_llm": int(sum(deployment.llm_active.values())),
                    "n_models": len(
                        {scenario.candidates[cid].model for cid, on in deployment.llm_active.items() if on}
                    ),
                    "cost_per_slot": float(
                        (library.entries[0].cost_per_slot if args.library_scan else 0.0)
                        or _entry_for(scenario, deployment, 0, label).cost_per_slot
                    ),
                    "mean_utility": float(np.mean(utilities)) if utilities else float("nan"),
                    "mean_slo_attainment": float(np.mean(attainment)) if attainment else float("nan"),
                    "mean_cost": float(np.mean(costs)) if costs else float("nan"),
                    "mean_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
                    "violations": violations,
                }
            )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "deployment_bar.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = summary.setdefault(row["deployment"], {"n": 0, "utility": 0.0, "attainment": 0.0, "cost": 0.0, "latency": 0.0, "violations": 0.0})
        bucket["n"] += 1
        bucket["utility"] += row["mean_utility"]
        bucket["attainment"] += row["mean_slo_attainment"]
        bucket["cost"] += row["mean_cost"]
        bucket["latency"] += row["mean_latency_s"]
        bucket["violations"] += row["violations"]
    for bucket in summary.values():
        count = max(1, int(bucket["n"]))
        for key in ("utility", "attainment", "cost", "latency", "violations"):
            bucket[key] /= count

    lines = [
        f"# Deployment bar under the frozen composition policy — {scenario.id}",
        "",
        f"composition policy `{args.composition_policy}`; {len(seeds)} seed(s), "
        f"{args.periods} periods ({args.warmup} warmup), arrival_scale {args.arrival_scale:.6f}",
        "",
        "| deployment | LLM inst. | models | steady cost | utility | SLO attain. | cost | latency s | violations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    first = {row["deployment"]: row for row in rows}
    for label, bucket in sorted(summary.items(), key=lambda item: -item[1]["utility"]):
        sample = first[label]
        lines.append(
            f"| {label} | {sample['n_llm']} | {sample['n_models']} | "
            f"{sample['cost_per_slot']:.4f} | {bucket['utility']:+.5f} | "
            f"{bucket['attainment']:.4f} | {bucket['cost']:.4f} | "
            f"{bucket['latency']:.1f} | {bucket['violations']:.1f} |"
        )
    best = max(summary.items(), key=lambda item: item[1]["utility"])
    lines.extend(
        [
            "",
            f"Best fixed deployment: `{best[0]}` at {best[1]['utility']:+.5f}. "
            "A deployment policy has to beat this to be worth training.",
            "",
        ]
    )
    (output_dir / "deployment_bar.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "deployment_bar.json").write_text(
        json.dumps(
            {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary, "rows": rows},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())

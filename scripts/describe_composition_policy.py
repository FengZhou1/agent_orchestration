"""Describe what a composition policy actually does, aggregated over the library.

The per-update ``action/model_share`` telemetry records only the last context of
each rollout, so it shows one deployment's composition rather than the policy's
behaviour.  This script runs the policy deterministically over many deployments
and aggregates the composition per application, grouped by which models were
active, which is the view needed to explain what the policy learned.

Usage::

    python scripts/describe_composition_policy.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --policy results/stage_a/composition_fixed/<run>/policy_best.pt \\
        --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs import CompositionLibraryEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--deployment-library", default=None)
    parser.add_argument("--objective-profile", default="slo_constrained")
    parser.add_argument("--attainment-target", type=float, default=0.9)
    parser.add_argument("--split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--arrival-scale", type=float, default=None)
    parser.add_argument("--mapping-samples", type=int, default=8)
    parser.add_argument("--only-full-model-set", action="store_true")
    parser.add_argument("--output", default="results/stage_a/composition_description")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    library = DeploymentLibrary.load(
        args.deployment_library or DeploymentLibrary.default_path(scenario.id)
    )
    train_library, test_library = library.train_test_split(
        test_fraction=args.test_fraction, seed=args.split_seed, stratify=True
    )
    selected = {"all": library, "train": train_library, "test": test_library}[args.split]

    arrival_scale = args.arrival_scale
    if arrival_scale is None:
        reference_path = (
            Path("data/processed")
            / f"composition_reference_{scenario.id}_steady.json"
        )
        if not reference_path.exists():
            reference_path = Path("data/processed") / f"composition_reference_{scenario.id}.json"
        arrival_scale = float(json.loads(reference_path.read_text(encoding="utf-8"))["arrival_scale"])

    spec = (
        ObjectiveSpec.slo_constrained(args.attainment_target)
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    trace = ArrivalTrace.stationary_poisson_intensity(scenario, 1, rate_scale=arrival_scale)
    env = CompositionLibraryEnv(
        scenario,
        max_slots=1,
        seed=0,
        arrival_trace=trace,
        mapping_samples=args.mapping_samples,
        objective=spec,
        deployment_library=selected,
    )
    policy = StructuredActorCritic(env, PPOConfig())
    state = torch_load(Path(args.policy))
    policy.load_state_dict(state.get("policy_state_dict", state))
    policy.eval()

    per_app: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_stratum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    contexts = 0
    for position in range(len(selected.entries)):
        entry = selected.entries[position]
        if args.only_full_model_set and entry.n_models != len(scenario.models):
            continue
        if entry.n_models < 2:
            continue
        observation, _ = env.reset(seed=0, options={"fixed_deployment_index": position})
        active = env.active_models()
        if len(active) < 2:
            continue
        action, _, _ = policy.act(observation, deterministic=True, device="cpu")
        dense = np.asarray(action["model"], dtype=float).reshape(
            len(env.layout.model_groups), len(env.layout.models)
        )
        for group_index, (app_id, _ingress) in enumerate(env.layout.model_groups):
            for model_index, model in enumerate(env.layout.models):
                value = float(dense[group_index, model_index])
                per_app[app_id][model].append(value)
                per_stratum[entry.stratum][model].append(value)
        contexts += 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(per_app, scenario, "app")
    stratum_rows = _rows(per_stratum, scenario, "stratum")

    with (output_dir / "composition_by_app.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "composition_by_stratum.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in stratum_rows for k in row}))
        writer.writeheader()
        writer.writerows(stratum_rows)

    models = list(scenario.models)
    lines = [
        f"# Composition policy — {scenario.id}",
        "",
        f"policy `{args.policy}` over {contexts} deployment contexts "
        f"(split `{args.split}`, arrival_scale {arrival_scale:.6f})",
        "",
        "Multi-model contexts only; values are the mean deterministic share per model.",
        "",
        "| app | " + " | ".join(model.replace("qwen3-", "") for model in models) + " |",
        "|" + "---|" * (len(models) + 1),
    ]
    for row in rows:
        lines.append(
            "| {app} | ".format(**row)
            + " | ".join("%.3f" % row.get(f"share_{model}", 0.0) for model in models)
            + " |"
        )
    lines.extend(["", "## By stratum", "", "| stratum | n | " + " | ".join(
        model.replace("qwen3-", "") for model in models
    ) + " |", "|" + "---|" * (len(models) + 2)])
    for row in stratum_rows:
        lines.append(
            "| {stratum} | {contexts} | ".format(**row)
            + " | ".join("%.3f" % row.get(f"share_{model}", 0.0) for model in models)
            + " |"
        )
    (output_dir / "composition_description.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {output_dir / 'composition_description.md'}")
    return 0


def _rows(grouped, scenario, key_name: str) -> list[dict]:
    rows: list[dict] = []
    for key, shares in sorted(grouped.items()):
        row: dict[str, object] = {key_name: key, "contexts": len(next(iter(shares.values())))}
        for model in scenario.models:
            values = shares.get(model, [])
            row[f"share_{model}"] = float(np.mean(values)) if values else 0.0
        rows.append(row)
    return rows


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())

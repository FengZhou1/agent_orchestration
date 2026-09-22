"""Can the composition head express the teacher's optima? (Stage A, A3.3 acceptance)

The head's action is a Dirichlet *mean*: concentrations are clamped to
``[floor, max(1000, 500*floor)]``, so a group with ``k`` active models cannot put
more than ``cap / (cap + (k-1)*floor)`` on one model -- 0.997 at four models, and
0.9709 under the old bound of 100.

The teacher's optima are nearly one-hot, so this measures the loss directly: clip
each target group to the reachable range, redistribute the excess over the other
coordinates (renormalising instead would put the clipped mass straight back), and
rescore.  The drop from the reference to the projected reference is what the
parameterisation -- not the policy -- is responsible for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import build_composition_env, score_composition
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCALE = 7.165234375


def _ceiling(k: int, floor: float, cap: float) -> float:
    return cap / (cap + (k - 1) * floor) if k > 1 else 1.0


def project(share: dict, floor: float, cap: float) -> dict:
    """Nearest reachable share: cap each coordinate, spread the residual."""

    by_group: dict[tuple[str, str], dict[str, float]] = {}
    for key, value in share.items():
        app_id, ingress, model = key.split("|")
        by_group.setdefault((app_id, ingress), {})[model] = float(value)
    out = dict(share)
    for (app_id, ingress), weights in by_group.items():
        limit = _ceiling(len(weights), floor, cap)
        clipped = {model: min(value, limit) for model, value in weights.items()}
        residual = 1.0 - sum(clipped.values())
        if residual > 0:
            headroom = {
                model: limit - value for model, value in clipped.items() if limit - value > 0
            }
            total_headroom = sum(headroom.values())
            if total_headroom > 0:
                for model, room in headroom.items():
                    clipped[model] += residual * room / total_headroom
        for model, value in clipped.items():
            out[f"{app_id}|{ingress}|{model}"] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument(
        "--reference", default="data/processed/composition_reference_agent-abilene-20_v5_test.json"
    )
    parser.add_argument("--floor", type=float, default=1.0)
    parser.add_argument(
        "--cap",
        type=float,
        default=1.0e6,
        help="must match distributions._concentrations, or this measures a stale head",
    )
    parser.add_argument("--periods", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--arrival-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--output", default="results/stage_a/reachable_set.json")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(REPO_ROOT / args.scenario)
    objective = ObjectiveSpec.slo_constrained(0.9)
    library = DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))
    _, test_library = library.train_test_split(test_fraction=0.25, seed=2026, stratify=True)
    payload = json.loads((REPO_ROOT / args.reference).read_text(encoding="utf-8"))
    records = {int(k): v for k, v in payload["entries"].items()}
    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, args.periods, rate_scale=args.arrival_scale
    )

    reference_values: list[float] = []
    projected_values: list[float] = []
    outside = 0
    groups = 0
    for position, entry in enumerate(test_library.entries):
        record = records.get(entry.index)
        if record is None:
            continue
        share = record["model_share"]
        # The stored keys are "app|ingress|model" strings; the action layout is keyed
        # by tuples.  A mismatch silently scores the uniform fallback, so convert.
        share_by_tuple = {tuple(key.split("|")): float(value) for key, value in share.items()}
        env = build_composition_env(
            scenario,
            objective,
            trace,
            test_library,
            position=position,
            periods=args.periods,
            mapping_samples=args.mapping_samples,
            seed=0,
        )
        reference_values.append(
            score_composition(env, share_by_tuple, periods=args.periods, warmup=args.warmup)
        )
        projected = project(share, args.floor, args.cap)
        projected_by_tuple = {
            tuple(key.split("|")): float(value) for key, value in projected.items()
        }
        env = build_composition_env(
            scenario,
            objective,
            trace,
            test_library,
            position=position,
            periods=args.periods,
            mapping_samples=args.mapping_samples,
            seed=0,
        )
        projected_values.append(
            score_composition(env, projected_by_tuple, periods=args.periods, warmup=args.warmup)
        )
        by_group: dict[tuple[str, str], dict[str, float]] = {}
        for key, value in share.items():
            app_id, ingress, model = key.split("|")
            by_group.setdefault((app_id, ingress), {})[model] = float(value)
        for weights in by_group.values():
            groups += 1
            if max(weights.values()) > _ceiling(len(weights), args.floor, args.cap):
                outside += 1

    reference_mean = float(np.mean(reference_values))
    projected_mean = float(np.mean(projected_values))
    print(f"groups above the reachable ceiling: {outside}/{groups}")
    print(f"ceiling at 4 models: {_ceiling(4, args.floor, args.cap):.4f}")
    print(f"reference          mean = {reference_mean:+.5f}")
    print(f"reference projected mean = {projected_mean:+.5f}")
    print(f"loss owned by the parameterisation = {reference_mean - projected_mean:+.5f}")
    if reference_mean - projected_mean < 1.0e-4:
        print("VERDICT: the head can express the teacher's optima; no A3.3 change needed.")
    else:
        print("VERDICT: the head cannot express the teacher's optima.  Change the action")
        print("parameterisation (A3.3) or the reference is unreachable by construction.")

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "reference": str(args.reference),
                "floor": args.floor,
                "cap": args.cap,
                "periods": args.periods,
                "warmup": args.warmup,
                "groups_above_ceiling": outside,
                "groups": groups,
                "reference_mean": reference_mean,
                "projected_mean": projected_mean,
                "parameterisation_loss": reference_mean - projected_mean,
                "rows": [
                    {"reference": reference, "projected": projected}
                    for reference, projected in zip(reference_values, projected_values)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

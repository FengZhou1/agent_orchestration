"""Gate G_A: does the composition policy actually solve the inner problem?

Two questions decide whether a deployment policy can be trained on top of the
composition policy at all:

1. Does the learned composition beat every closed-form heuristic on held-out
   deployments, stratum by stratum?
2. Does it track the inner solver's ranking of deployments?  If the policy is
   not a good approximation of the deployment-to-composition response function,
   an outer deployment policy is optimising against a lower level that answers
   the wrong question.

Everything is measured under one protocol (the steady multi-period protocol the
composition policy is trained with) so the learned policy, the heuristics and
the solver reference are directly comparable.

Usage::

    python scripts/evaluate_composition_response.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --policy results/stage_a/<run>/policy_best.pt \\
        --reference data/processed/composition_reference_agent-abilene-20.json \\
        --output results/stage_a/gate
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
from agent_orch.data.provenance import StaleArtifactError, assert_same_library
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs import CompositionLibraryEnv
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec, ReferenceScales
from agent_orch.routing.composition import CompositionSolver
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

# Heuristics the learned policy has to beat.  ``uniform`` is included because it
# is the composition the deployment-only environment silently applies.
HEURISTIC_NAMES = (
    "uniform",
    "quality_greedy",
    "cost_greedy",
    "latency_greedy",
    "quality_softmax",
    "quality_uniform_mix",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation, tie-aware, no SciPy dependency."""

    if left.size < 2:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks_out = np.empty(values.size, dtype=float)
        start = 0
        for index in range(1, values.size + 1):
            if index == values.size or sorted_values[index] != sorted_values[start]:
                ranks_out[order[start:index]] = 0.5 * (start + index - 1)
                start = index
        return ranks_out

    a = ranks(left)
    b = ranks(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((a * b).sum() / denominator)


def _parse_float_list(values) -> list[str]:
    return [chunk for chunk in str(values).split(",") if chunk.strip()]


def _steady_utility(
    env,
    action_fn,
    periods: int,
    warmup: int,
) -> float:
    """Mean objective over the post-warmup periods of one episode."""

    observation, _ = env.reset(seed=env._seed)  # noqa: SLF001 - env owns its seed
    utilities: list[float] = []
    for period in range(periods):
        action = action_fn(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if period >= warmup and info.get("period_complete"):
            utilities.append(float(info["utility"]))
        if terminated or truncated:
            break
    if not utilities:
        return float("nan")
    return float(np.mean(utilities))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", required=True, help="composition policy checkpoint")
    parser.add_argument("--reference", required=True, help="composition reference JSON")
    parser.add_argument("--deployment-library", default=None)
    parser.add_argument("--objective-profile", default="slo_constrained")
    parser.add_argument("--attainment-target", type=float, default=0.9)
    parser.add_argument(
        "--no-attainment-constraint",
        action="store_true",
        help=(
            "must match how the policy was trained: the constraint vector is part "
            "of the observation, so a different constraint count changes the "
            "feature width and the checkpoint will not load"
        ),
    )
    parser.add_argument("--split", default="test", choices=["all", "train", "test"])
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 solves every entry")
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--rho-threshold", type=float, default=0.8)
    parser.add_argument(
        "--min-lift-capture",
        type=float,
        default=0.8,
        help="fraction of the solver's lift over uniform that the policy must capture",
    )
    parser.add_argument(
        "--stratum-tolerance-ratio",
        type=float,
        default=0.1,
        help=(
            "allowed per-stratum shortfall against the best heuristic, as a "
            "fraction of the achievable lift over uniform"
        ),
    )
    parser.add_argument("--solver-budget", type=int, default=32)
    parser.add_argument(
        "--allow-stale-reference",
        action="store_true",
        help="score a reference whose recorded deployment-library hash differs (unsafe)",
    )
    parser.add_argument("--output", default="results/stage_a/gate")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    spec = (
        ObjectiveSpec.slo_constrained(
            None if args.no_attainment_constraint else args.attainment_target
        )
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    library = DeploymentLibrary.load(
        args.deployment_library or DeploymentLibrary.default_path(scenario.id)
    )
    train_library, test_library = library.train_test_split(
        test_fraction=args.test_fraction, seed=args.split_seed, stratify=True
    )
    split_library = {"all": library, "train": train_library, "test": test_library}[args.split]

    references = ReferenceScales.from_scenario(scenario, spec, library)
    evaluator = ObjectiveEvaluator(scenario, spec, references)
    payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if not args.allow_stale_reference:
        try:
            assert_same_library(payload, args.deployment_library or DeploymentLibrary.default_path(scenario.id))
        except StaleArtifactError as error:
            raise SystemExit(f"{args.reference}: {error}") from error
    reference_entries = {int(key): value for key, value in payload["entries"].items()}
    arrival_scale = float(payload["arrival_scale"])
    # The reference is the optimum of *its own* scoring protocol, so the policy has
    # to be scored the same way.  A cold-start reference against a steady rollout
    # measures the protocol gap rather than the composition gap.
    reference_protocol_periods = max(1, int(payload.get("protocol_periods", 1)))
    reference_protocol_warmup = max(0, int(payload.get("protocol_warmup", 0)))

    checkpoint = Path(args.policy)
    state = torch_load(checkpoint)
    policy_state = state.get("policy_state_dict", state)

    periods = args.periods + args.warmup
    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, periods, rate_scale=arrival_scale
    )
    env = CompositionLibraryEnv(
        scenario,
        max_slots=periods,
        seed=0,
        arrival_trace=trace,
        mapping_samples=args.mapping_samples,
        objective=spec,
        library=split_library,
    )
    policy = StructuredActorCritic(env, PPOConfig())
    policy.load_state_dict(policy_state)
    policy.eval()

    solver = CompositionSolver(
        scenario, evaluator, mapping_samples=args.mapping_samples
    )
    arrival_rates = trace.at(0, scenario)

    selected = list(range(len(split_library.entries)))
    if args.limit > 0:
        selected = selected[: args.limit]

    rows: list[dict[str, object]] = []
    for position in selected:
        entry = split_library.entries[position]
        reference = reference_entries.get(entry.index)

        def act(observation):
            action, _, _ = policy.act(observation, deterministic=True, device="cpu")
            return action

        policy_utility = _steady_utility(
            _fixed_env(env, position, scenario, trace, args, spec), act, periods, args.warmup
        )
        # The solver comparison runs under the reference's own protocol.
        policy_cold = _protocol_utility(
            _fixed_env(env, position, scenario, trace, args, spec),
            act,
            reference_protocol_periods,
            reference_protocol_warmup,
        )

        heuristic_utilities: dict[str, float] = {}
        heuristic_cold: dict[str, float] = {}
        for name, share in _heuristic_shares(scenario, solver, env, position).items():
            heuristic_cold[name] = _protocol_utility(
                _fixed_env(env, position, scenario, trace, args, spec),
                _constant_action(env, share),
                reference_protocol_periods,
                reference_protocol_warmup,
            )
            heuristic_utilities[name] = _steady_utility(
                _fixed_env(env, position, scenario, trace, args, spec),
                _constant_action(env, share),
                periods,
                args.warmup,
            )

        reference_utility = float("nan")
        reference_cold = float("nan")
        if reference is not None and reference.get("model_share"):
            share = _share_from_json(scenario, reference["model_share"])
            reference_utility = _steady_utility(
                _fixed_env(env, position, scenario, trace, args, spec),
                _constant_action(env, share),
                periods,
                args.warmup,
            )
            # Always recompute from the stored composition rather than trusting
            # the stored utility.  The utility scale depends on the objective's
            # normalisation references, so a stored value from an earlier premise
            # is not comparable with a policy scored now: it silently mixes two
            # scales and makes both rho and the captured lift meaningless.
            reference_cold = _protocol_utility(
                _fixed_env(env, position, scenario, trace, args, spec),
                _constant_action(env, share),
                reference_protocol_periods,
                reference_protocol_warmup,
            )

        best_heuristic = max(heuristic_utilities.values()) if heuristic_utilities else float("nan")
        rows.append(
            {
                "position": position,
                "source_index": entry.index,
                "stratum": entry.stratum,
                "n_models": entry.n_models,
                "policy": policy_utility,
                "policy_cold": policy_cold,
                "best_heuristic": best_heuristic,
                "best_heuristic_name": (
                    max(heuristic_utilities, key=heuristic_utilities.get)
                    if heuristic_utilities
                    else ""
                ),
                "reference": reference_utility,
                "reference_cold": reference_cold,
                **{f"h_{name}": value for name, value in heuristic_utilities.items()},
                **{f"hc_{name}": value for name, value in heuristic_cold.items()},
            }
        )

    policy_values = np.array([row["policy"] for row in rows], dtype=float)
    best_values = np.array([row["best_heuristic"] for row in rows], dtype=float)
    reference_values = np.array([row["reference"] for row in rows], dtype=float)
    uniform_values = np.array(
        [row.get("h_uniform", float("nan")) for row in rows], dtype=float
    )
    policy_cold_values = np.array([row["policy_cold"] for row in rows], dtype=float)
    reference_cold_values = np.array([row["reference_cold"] for row in rows], dtype=float)
    uniform_cold_values = np.array(
        [row.get("hc_uniform", float("nan")) for row in rows], dtype=float
    )

    # Solver comparison: like-for-like, both sides scored cold.
    solver_comparable = (
        np.isfinite(policy_cold_values)
        & np.isfinite(reference_cold_values)
        & np.isfinite(uniform_cold_values)
    )
    rho = _spearman(policy_cold_values[solver_comparable], reference_cold_values[solver_comparable])
    mean_policy_cold = (
        float(np.mean(policy_cold_values[solver_comparable])) if solver_comparable.any() else float("nan")
    )
    mean_reference_cold = (
        float(np.mean(reference_cold_values[solver_comparable])) if solver_comparable.any() else float("nan")
    )
    mean_uniform_cold = (
        float(np.mean(uniform_cold_values[solver_comparable])) if solver_comparable.any() else float("nan")
    )
    achievable_lift = mean_reference_cold - mean_uniform_cold
    captured_lift = mean_policy_cold - mean_uniform_cold
    lift_capture = (
        captured_lift / achievable_lift if abs(achievable_lift) > 1.0e-12 else float("nan")
    )
    gap = mean_policy_cold - mean_reference_cold

    # Heuristic comparison: steady protocol, the one training optimises.
    steady_comparable = np.isfinite(policy_values) & np.isfinite(best_values) & np.isfinite(uniform_values)
    mean_policy = float(np.mean(policy_values[steady_comparable])) if steady_comparable.any() else float("nan")
    mean_reference = (
        float(np.mean(reference_values[steady_comparable])) if steady_comparable.any() else float("nan")
    )
    mean_uniform = (
        float(np.mean(uniform_values[steady_comparable])) if steady_comparable.any() else float("nan")
    )

    wins = int(np.sum(policy_values >= best_values - 1e-12))
    losses = [
        (row["position"], row["stratum"], float(row["policy"]), float(row["best_heuristic"]))
        for row in rows
        if row["policy"] < row["best_heuristic"] - 1e-12
    ]

    per_stratum: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = per_stratum.setdefault(
            row["stratum"], {"n": 0, "policy": 0.0, "best_heuristic": 0.0, "reference": 0.0}
        )
        bucket["n"] += 1
        bucket["policy"] += float(row["policy"])
        bucket["best_heuristic"] += float(row["best_heuristic"])
        bucket["reference"] += float(row["reference"])
    for bucket in per_stratum.values():
        count = max(1, int(bucket["n"]))
        for key in ("policy", "best_heuristic", "reference"):
            bucket[key] /= count

    stratum_tolerance = args.stratum_tolerance_ratio * abs(achievable_lift)
    stratum_shortfalls = [
        (name, bucket["policy"] - bucket["best_heuristic"])
        for name, bucket in per_stratum.items()
        if bucket["policy"] < bucket["best_heuristic"] - stratum_tolerance - 1.0e-12
    ]
    heuristic_failures = [
        (name, int(np.sum(policy_values < np.array([row[f"h_{name}"] for row in rows]) - 1e-12)))
        for name in HEURISTIC_NAMES
        if any(f"h_{name}" in row for row in rows)
    ]

    ceiling = _dirichlet_ceiling(
        policy.config.composition_concentration_min,
        scenario,
        split_library,
        selected,
        reference_entries,
    )

    gate_rho = bool(np.isfinite(rho) and rho >= args.rho_threshold)
    gate_lift = bool(np.isfinite(lift_capture) and lift_capture >= args.min_lift_capture)
    gate_strata = len(stratum_shortfalls) == 0
    passed = gate_rho and gate_lift and gate_strata

    report = {
        "generated_at_utc": _timestamp(),
        "scenario": scenario.id,
        "policy": str(checkpoint),
        "reference": str(args.reference),
        "split": args.split,
        "reference_protocol_periods": reference_protocol_periods,
        "reference_utility_recomputed": True,
        "reference_protocol_warmup": reference_protocol_warmup,
        "n_deployments": len(rows),
        "objective": spec.to_dict(),
        "arrival_scale": arrival_scale,
        "rho": rho,
        "mean_policy": mean_policy,
        "mean_reference": mean_reference,
        "mean_uniform": mean_uniform,
        "mean_policy_cold": mean_policy_cold,
        "mean_reference_cold": mean_reference_cold,
        "mean_uniform_cold": mean_uniform_cold,
        "achievable_lift": achievable_lift,
        "captured_lift": captured_lift,
        "lift_capture": lift_capture,
        "mean_gap_to_reference": gap,
        "heuristic_wins": wins,
        "heuristic_losses": losses,
        "heuristic_failure_counts": heuristic_failures,
        "stratum_shortfalls": stratum_shortfalls,
        "stratum_tolerance": stratum_tolerance,
        "dirichlet_ceiling": ceiling,
        "gates": {
            "rho": gate_rho,
            "lift_capture": gate_lift,
            "per_stratum": gate_strata,
        },
        "passed": passed,
        "rows": rows,
        "per_stratum": per_stratum,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "gate_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Stage A gate — {scenario.id}",
        "",
        f"policy `{checkpoint}` on split `{args.split}` ({len(rows)} deployments), "
        f"arrival_scale {arrival_scale:.6f}, objective `{spec.profile}`",
        "",
        "## Decision",
        "",
        "| test | value | requirement | verdict |",
        "|---|---|---|---|",
        f"| rank tracking (Spearman rho vs solver) | {rho:.4f} | >= {args.rho_threshold} | "
        f"{'PASS' if gate_rho else 'FAIL'} |",
        f"| lift captured over uniform | {lift_capture:.1%} | >= {args.min_lift_capture:.0%} | "
        f"{'PASS' if gate_lift else 'FAIL'} |",
        f"| strata not below best heuristic | {len(per_stratum) - len(stratum_shortfalls)}"
        f"/{len(per_stratum)} | all (tol {args.stratum_tolerance_ratio:.0%} of lift = "
        f"{stratum_tolerance:+.4f}) | "
        f"{'PASS' if gate_strata else 'FAIL'} |",
        "",
        "## Levels",
        "",
        f"Solver comparison (reference protocol: {reference_protocol_periods} periods, "
        f"warmup {reference_protocol_warmup}; like-for-like):",
        "",
        "| level | value |",
        "|---|---|",
        f"| policy | {mean_policy_cold:+.5f} |",
        f"| solver reference | {mean_reference_cold:+.5f} |",
        f"| uniform composition | {mean_uniform_cold:+.5f} |",
        f"| achievable lift over uniform | {achievable_lift:+.5f} |",
        f"| captured lift | {captured_lift:+.5f} |",
        f"| mean gap to solver | {gap:+.5f} |",
        "",
        "Heuristic comparison (steady protocol, what training optimises):",
        "",
        "| level | value |",
        "|---|---|",
        f"| policy | {mean_policy:+.5f} |",
        f"| best heuristic per deployment | {float(np.mean(best_values[steady_comparable])):+.5f} |",
        f"| reference composition under steady protocol | {mean_reference:+.5f} |",
        f"| uniform composition | {mean_uniform:+.5f} |",
        f"| deployments strictly beating every heuristic | {wins}/{len(rows)} |",
        "",
        "## Parameterisation limit",
        "",
        "The Dirichlet head clamps concentrations to "
        f"`max(100, 2*{ceiling['concentration_min']:.3f})`, so its mean share is bounded by:",
        "",
        "| active models per group | max reachable mean share |",
        "|---|---|",
    ]
    lines.extend(
        f"| {size} | {value:.4f} |"
        for size, value in ceiling["mean_ceiling_by_group_size"].items()
    )
    lines.extend(
        [
            "",
            f"Solver targets above the ceiling: {int(ceiling['groups_above_ceiling'])}"
            f"/{int(ceiling['groups_checked'])} groups"
            + (
                f" (worst excess {ceiling['worst_excess_over_ceiling']:+.4f}); "
                "this part of the gap is unreachable by construction."
                if ceiling["groups_above_ceiling"] > 0
                else "; the ceiling is not binding."
            ),
            "",
        ]
    )
    lines.extend(
        ["| heuristic | deployments where the policy loses |", "|---|---|"]
    )
    lines.extend(f"| {name} | {count} |" for name, count in heuristic_failures)
    lines.extend(["", "| stratum | n | policy | best heuristic | reference |", "|---|---|---|---|---|"])
    for name, bucket in sorted(per_stratum.items()):
        lines.append(
            f"| {name} | {int(bucket['n'])} | {bucket['policy']:+.5f} | "
            f"{bucket['best_heuristic']:+.5f} | {bucket['reference']:+.5f} |"
        )
    if stratum_shortfalls:
        lines.extend(["", "Strata below the best heuristic:", ""])
        lines.extend(f"- `{name}`: {delta:+.5f}" for name, delta in stratum_shortfalls)
    lines.extend(["", f"**GATE G_A: {'PASS' if passed else 'FAIL'}**"])
    (output_dir / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    if not passed:
        print(
            "\nGate G_A failed.  Training a deployment policy on this composition "
            "policy would optimise against a lower level that answers the wrong "
            "question; fix the composition stage first."
        )
    return 0 if passed else 1


def _protocol_utility(
    env, action_fn, periods: int, warmup: int, steady_action=None
) -> float:
    """Objective under a fixed number of periods with utilization feedback.

    ``periods == 1`` reproduces the inner solver's cold-start score; longer
    protocols average the post-warmup periods exactly as the solver does when it
    is run with ``--protocol-periods``.  ``steady_action`` lets a caller freeze a
    composition while the policy drives a different one, which is how the
    heuristic and policy comparisons share a protocol.
    """

    observation, _ = env.reset(seed=env._seed)  # noqa: SLF001 - env owns its seed
    utilities: list[float] = []
    for period in range(max(1, periods)):
        action = action_fn(observation) if steady_action is None else steady_action(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if period >= warmup and info.get("period_complete"):
            utilities.append(float(info["utility"]))
        if terminated or truncated:
            break
    if not utilities:
        return float("nan")
    return float(np.mean(utilities))


def _dirichlet_ceiling(
    concentration_min: float,
    scenario,
    library,
    positions: list[int],
    entries: dict[int, dict],
) -> dict[str, float]:
    """How close to one-hot the Dirichlet head can actually place its mean.

    ``concentrations`` clamps to ``max(100, 2 * minimum)``, so for a group with
    ``k`` active models the largest mean share is ``C_hi / (C_hi + (k - 1) C_lo)``.
    A solver target above that ceiling is unreachable by construction, and the
    resulting shortfall is a parameterisation limit, not a learning failure.
    """

    c_hi = max(100.0, 2.0 * float(concentration_min))
    c_lo = max(float(concentration_min), 1.0e-3)
    ceilings = {k: c_hi / (c_hi + (k - 1) * c_lo) for k in range(1, 8)}

    unreachable = 0
    total_groups = 0
    worst_excess = 0.0
    for position in positions:
        entry = library.entries[position]
        record = entries.get(entry.index)
        if not record or not record.get("model_share"):
            continue
        active = {
            (app_id, ingress): set()
            for app_id, ingress in (
                (app.id, ingress)
                for app in scenario.applications.values()
                for ingress in app.ingress_rates
            )
        }
        for key in record["model_share"]:
            parts = key.split("|")
            active.setdefault((parts[0], parts[1]), set()).add("|".join(parts[2:]))
        for (app_id, ingress), models in active.items():
            if not models:
                continue
            total_groups += 1
            largest = max(
                float(record["model_share"].get(f"{app_id}|{ingress}|{model}", 0.0))
                for model in models
            )
            limit = ceilings.get(len(models))
            if limit is not None and largest > limit + 1.0e-9:
                unreachable += 1
                worst_excess = max(worst_excess, largest - limit)
    return {
        "concentration_min": float(concentration_min),
        "mean_ceiling_by_group_size": {str(k): round(v, 6) for k, v in ceilings.items()},
        "groups_checked": float(total_groups),
        "groups_above_ceiling": float(unreachable),
        "worst_excess_over_ceiling": float(worst_excess),
    }


def _fixed_env(env, position: int, scenario, trace, args, spec):
    """A fresh environment pinned to one deployment position.

    The library goes in through ``deployment_library`` so the base environment
    builds the objective's normalisation scales from it.  Passing it as
    ``library`` instead would leave the cost bounds theoretical and put this
    environment's utility on a different scale than the solver's.
    """

    return CompositionLibraryEnv(
        scenario,
        max_slots=args.periods + args.warmup,
        seed=0,
        arrival_trace=trace,
        mapping_samples=args.mapping_samples,
        objective=spec,
        deployment_library=env.deployment_library,
        fixed_deployment_index=position,
    )


def _heuristic_shares(scenario, solver, env, position: int) -> dict[str, dict]:
    deployment = env.deployment_library.entries[position].to_deployment()
    return {candidate.name: candidate.share for candidate in solver.canonical_candidates(deployment)}


def _share_from_json(scenario, payload: dict[str, float]) -> dict[tuple[str, str, str], float]:
    share: dict[tuple[str, str, str], float] = {}
    for key, value in payload.items():
        parts = key.split("|")
        app, ingress, model = parts[0], parts[1], "|".join(parts[2:])
        share[(app, ingress, model)] = float(value)
    return share


def _constant_action(env, share):
    """Emit a fixed composition, expressed in the environment's action layout."""

    dense = np.zeros(
        (len(env.layout.model_groups), len(env.layout.models)), dtype=np.float32
    )
    for group_index, (app_id, ingress) in enumerate(env.layout.model_groups):
        for model_index, model in enumerate(env.layout.models):
            dense[group_index, model_index] = float(share.get((app_id, ingress, model), 0.0))

    def act(_observation):
        return {"deploy": 0, "model": dense.reshape(-1).copy()}

    return act


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())

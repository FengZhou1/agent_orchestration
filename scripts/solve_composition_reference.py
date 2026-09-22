"""Solve the inner composition problem for every deployment in a library.

The output is the "reachable optimum" reference the composition gate scores a learned
policy against, and the offline teacher used to warm-start it.  One JSON record per
deployment holds the solved model shares, their utility, how many evaluations the
search spent and which candidate produced the answer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from agent_orch.data import file_sha256
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import CompositionEnvScorer
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec, ReferenceScales
from agent_orch.routing import CompositionSolution, CompositionSolver
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace


REPO_ROOT = Path(__file__).resolve().parents[1]
# 2: candidates are scored through the composition environment rather than a raw
# simulator rollout, so a v1 entry is not the argmax of the gate's metric.
# 3: the environment kept the episode inside its arrival trace.  Before that it
# overshot by a random offset and ran 33-40% of every episode at the unscaled base
# rate, so a v2 entry is the optimum of a lower-load tail, not of this metric.
# 4: the search can leave the vertex set (a whole-composition line search toward
# uniform).  Vertices are not the optimum -- blending 25% toward uniform beats the
# best vertex by 0.0054 on deployment 2 -- so a v3 entry is a local optimum of a
# restricted move set, not the argmax of this metric.
SCHEMA_VERSION = 4
DEFAULT_LOAD_LEVEL = "high"


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_arrival_scale(
    explicit: float | None, load_levels: Path, level: str
) -> tuple[float, str]:
    """Arrival multiplier: the explicit argument, else a named load level, else 1.0."""

    if explicit is not None:
        return float(explicit), "argument"
    if load_levels.exists():
        payload = json.loads(load_levels.read_text(encoding="utf-8"))
        for item in payload.get("levels", []):
            if str(item.get("name")) == level:
                return float(item["rate_scale"]), f"{_portable_path(load_levels)}::{level}"
    return 1.0, "fallback:1.0"


def _scaled_arrival_rates(scenario, scale: float) -> dict[tuple[str, str], float]:
    return {
        (app.id, ingress): float(rate) * scale
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }


def _entry_record(solution: CompositionSolution, stratum: str) -> dict[str, Any]:
    record = {"stratum": stratum}
    record.update(solution.to_json())
    return record


def _breakdown_line(
    index: int, record: dict[str, Any], seconds: float | None, resumed: bool = False
) -> str:
    uniform = float(record["candidate_utilities"].get("uniform", float("nan")))
    best = float(record["utility"])
    timing = "resumed   " if resumed else f"{seconds:6.1f}s"
    return (
        f"[{index:>4}] {str(record['stratum']):<32} "
        f"uniform={uniform:+.5f} best={best:+.5f} lift={best - uniform:+.5f} "
        f"evals={int(record['evaluations']):>3} source={str(record['source']):<20} {timing}"
    )


def _load_resumable(
    output: Path,
    *,
    scenario_id: str,
    objective: dict[str, Any],
    arrival_scale: float,
    current: dict[str, Any],
    resume: bool,
) -> dict[str, dict[str, Any]]:
    """Entries already solved by a compatible earlier run.

    Compatibility is the identity of the *problem*: schema version, scenario,
    objective profile, arrival scale and deployment library.  A difference in budget,
    mapping samples or split still resumes (the solved entries stay valid) but is
    reported, because the file then mixes searches of different strength.
    """

    if not resume or not output.exists():
        return {}
    payload = json.loads(output.read_text(encoding="utf-8"))
    mismatches = []
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        mismatches.append("schema_version")
    if str(payload.get("scenario_id")) != scenario_id:
        mismatches.append("scenario_id")
    if dict(payload.get("objective", {})) != objective:
        mismatches.append("objective")
    # Entries are keyed by library index, so a rebuilt library makes every solved
    # entry a composition for a deployment that may no longer sit at that index.
    if str(payload.get("library_json_sha256", "")) != str(current.get("library_json_sha256", "")):
        mismatches.append("library_json_sha256")
    if payload.get("arrival_scale") is None or abs(
        float(payload["arrival_scale"]) - float(arrival_scale)
    ) > 1.0e-12:
        mismatches.append("arrival_scale")
    # A cold-start entry is not the optimum of the steady protocol, so mixing the
    # two inside one reference would silently corrupt the gate's rank check.
    if int(payload.get("protocol_periods", 1)) != int(current.get("protocol_periods", 1)):
        mismatches.append("protocol_periods")
    if int(payload.get("protocol_warmup", 0)) != int(current.get("protocol_warmup", 0)):
        mismatches.append("protocol_warmup")
    if mismatches:
        print(f"resume refused: {output.name} differs in {', '.join(mismatches)}")
        return {}
    for name, value in current.items():
        if name in payload and payload[name] != value:
            print(f"  note: resuming across a different {name} ({payload[name]} != {value})")
    entries = {
        str(key): dict(value) for key, value in dict(payload.get("entries", {})).items()
    }
    print(f"resumed {len(entries)} solved entries from {_portable_path(output)}")
    return entries


def _stratum_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["stratum"]), []).append(record)
    for stratum, items in sorted(grouped.items()):
        best = [float(item["utility"]) for item in items]
        uniform = [
            float(item["candidate_utilities"].get("uniform", float("nan"))) for item in items
        ]
        summary[stratum] = {
            "count": len(items),
            "utility_min": min(best),
            "utility_mean": statistics.fmean(best),
            "utility_max": max(best),
            "uniform_mean": statistics.fmean(uniform),
            "lift_mean": statistics.fmean(b - u for b, u in zip(best, uniform)),
            "evaluations_mean": statistics.fmean(
                float(item["evaluations"]) for item in items
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve the inner composition problem over a deployment library."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--objective-profile", default="slo_constrained", choices=("legacy", "slo_constrained")
    )
    parser.add_argument("--attainment-target", type=float, default=0.9)
    parser.add_argument(
        "--deployment-library",
        default=None,
        help="Deployment library JSON (default: data/processed/deployment_library_<id>.json).",
    )
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=None,
        help="Arrival multiplier (default: the 'high' level of --load-levels).",
    )
    parser.add_argument("--load-levels", default="data/processed/load_levels.json")
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "test"),
        help="Solve the whole library or only one side of the held-out split.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0, help="Solve the first N entries (0: all).")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument(
        "--protocol-periods",
        type=int,
        default=1,
        help=(
            "periods a candidate composition is held for before scoring; 1 is the "
            "cold-start protocol, higher values match steady-state composition training"
        ),
    )
    parser.add_argument(
        "--protocol-warmup",
        type=int,
        default=0,
        help="periods discarded before averaging the protocol score",
    )
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument(
        "--arrival-pattern",
        default="stationary",
        choices=["stationary", "mix"],
        help="mix draws each (application, ingress) rate independently",
    )
    parser.add_argument("--mix-seed", type=int, default=0)
    parser.add_argument("--mix-block", type=int, default=1)
    parser.add_argument("--mix-sigma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip entries already solved by a compatible run (default: on).",
    )
    parser.add_argument(
        "--breakdown", action="store_true", help="Print one line per solved deployment."
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    library_path = (
        Path(args.deployment_library).resolve()
        if args.deployment_library
        else DeploymentLibrary.default_path(scenario.id, REPO_ROOT / "data" / "processed")
    )
    library = DeploymentLibrary.load(library_path)
    if library.scenario_id != scenario.id:
        parser.error(
            f"library {library_path.name} belongs to scenario {library.scenario_id!r}, "
            f"not {scenario.id!r}"
        )

    spec = (
        ObjectiveSpec.slo_constrained(args.attainment_target)
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    # The normalisation scales come from the *whole* library, not from the solved
    # split, so train and test utilities live on the same scale.
    evaluator = ObjectiveEvaluator(
        scenario, spec, ReferenceScales.from_scenario(scenario, spec, library)
    )

    arrival_scale, arrival_source = _resolve_arrival_scale(
        args.arrival_scale, Path(args.load_levels), DEFAULT_LOAD_LEVEL
    )
    arrival_rates = _scaled_arrival_rates(scenario, arrival_scale)

    if args.split == "all":
        selected = library
    else:
        train, test = library.train_test_split(
            test_fraction=args.test_fraction, seed=args.split_seed
        )
        selected = train if args.split == "train" else test
    targets = list(selected.entries)
    if args.limit > 0:
        targets = targets[: args.limit]

    output = (
        Path(args.output).resolve()
        if args.output
        else (REPO_ROOT / "data" / "processed" / f"composition_reference_{scenario.id}.json")
    )
    objective_payload = spec.to_dict()
    solved = _load_resumable(
        output,
        scenario_id=scenario.id,
        objective=objective_payload,
        arrival_scale=arrival_scale,
        current={
            "budget": args.budget,
            "mapping_samples": args.mapping_samples,
            "split": args.split,
            "protocol_periods": args.protocol_periods,
            "protocol_warmup": args.protocol_warmup,
            "library_json_sha256": file_sha256(library_path),
        },
        resume=args.resume,
    )

    solver = CompositionSolver(
        scenario,
        evaluator,
        mapping_samples=args.mapping_samples,
        seed=args.seed,
        protocol_periods=args.protocol_periods,
        protocol_warmup=args.protocol_warmup,
    )
    # The reference must be the argmax of the metric it will be compared against, so
    # candidates are scored through the same environment the gate rolls out.  A
    # second scoring path disagrees with this one on individual deployments, which
    # makes the reference lose to plain heuristics for no reason the search can fix.
    # A "mix" load keeps each (application, ingress) group at its own randomly drawn
    # rate instead of scaling them together, which is the only way the composition
    # optimum can depend on the load -- and therefore the only way a policy that
    # conditions on the load has anything to learn.  Held constant across the
    # protocol's slots, so the (2, 1) protocol stays equivalent to the long one.
    if args.arrival_pattern == "mix":
        trace = ArrivalTrace.randomized_mix_intensity(
            scenario,
            max(1, args.protocol_periods),
            base_scale=arrival_scale,
            seed=args.mix_seed,
            block=max(1, args.mix_block),
            per_group_sigma=args.mix_sigma if args.mix_sigma is not None else 0.6,
        )
    else:
        trace = ArrivalTrace.stationary_poisson_intensity(
            scenario, max(1, args.protocol_periods), rate_scale=arrival_scale
        )
    load_label = (
        f"mix (seed {args.mix_seed}, block {args.mix_block})"
        if args.arrival_pattern == "mix"
        else "stationary"
    )
    positions = {entry.index: position for position, entry in enumerate(selected.entries)}
    print(f"scoring path    : composition environment, periods={args.protocol_periods} "
          f"warmup={args.protocol_warmup}")
    print(f"scenario        : {scenario.id} ({_portable_path(scenario_path)})")
    print(f"library         : {_portable_path(library_path)} ({len(library)} entries)")
    print(f"split           : {args.split} -> {len(targets)} deployments")
    print(f"objective       : {args.objective_profile} {objective_payload['constraint_names']}")
    print(f"arrival scale   : {arrival_scale:g} ({arrival_source})")
    print(f"arrival pattern : {load_label}")
    print(f"budget          : {args.budget} evaluations, sweeps={args.sweeps}")
    print("solving:")

    start = time.perf_counter()
    entry_seconds: list[float] = []
    for entry in targets:
        key = str(entry.index)
        if key in solved:
            if args.breakdown:
                print(f"  {_breakdown_line(entry.index, solved[key], None, resumed=True)}")
            continue
        began = time.perf_counter()
        solver.scorer = CompositionEnvScorer(
            scenario,
            evaluator,
            spec,
            trace,
            selected,
            position=positions[entry.index],
            periods=args.protocol_periods,
            warmup=args.protocol_warmup,
            mapping_samples=args.mapping_samples,
            seed=args.seed,
        )
        solution = solver.solve(
            entry.to_deployment(),
            arrival_rates,
            budget=args.budget,
            sweeps=args.sweeps,
        )
        elapsed = time.perf_counter() - began
        entry_seconds.append(elapsed)
        solved[key] = _entry_record(solution, entry.stratum)
        record = solved[key]
        if args.breakdown:
            print(f"  {_breakdown_line(entry.index, record, elapsed)}")
        else:
            print(f"  [{entry.index:>4}] {record['stratum']:<32} "
                  f"best={float(record['utility']):+.5f} ({elapsed:5.1f}s)")
    total_seconds = time.perf_counter() - start

    records = [solved[str(entry.index)] for entry in targets if str(entry.index) in solved]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_id": scenario.id,
                "objective": objective_payload,
                "arrival_scale": arrival_scale,
                "budget": args.budget,
                "sweeps": args.sweeps,
                "protocol_periods": args.protocol_periods,
                "protocol_warmup": args.protocol_warmup,
                "mapping_samples": args.mapping_samples,
                "split": args.split,
                "split_test_fraction": args.test_fraction,
                "split_seed": args.split_seed,
                "library": _portable_path(library_path),
                "library_scenario_hash": library.scenario_hash,
                # Entries are keyed by library index, so every consumer needs the
                # library's own digest to prove the index mapping still holds.
                "library_json_sha256": file_sha256(library_path),
                "entries": {key: solved[key] for key in sorted(solved, key=int)},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    strata = _stratum_summary(records)
    mean_seconds = statistics.fmean(entry_seconds) if entry_seconds else 0.0
    extrapolated = mean_seconds * len(targets)
    manifest = {
        "artifact_type": "composition_reference",
        "scenario": _portable_path(scenario_path),
        "scenario_id": scenario.id,
        "scenario_file_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "library": _portable_path(library_path),
        "library_scenario_hash": library.scenario_hash,
        "library_json_sha256": file_sha256(library_path),
        "objective": objective_payload,
        "arrival_scale": arrival_scale,
        "arrival_scale_source": arrival_source,
        "arrival_pattern": args.arrival_pattern,
        "mix_seed": args.mix_seed if args.arrival_pattern == "mix" else None,
        "mix_block": args.mix_block if args.arrival_pattern == "mix" else None,
        "mix_sigma": args.mix_sigma if args.arrival_pattern == "mix" else None,
        "split": args.split,
        "split_test_fraction": args.test_fraction,
        "split_seed": args.split_seed,
        "budget": args.budget,
        "sweeps": args.sweeps,
        "mapping_samples": args.mapping_samples,
        "n_deployments": len(targets),
        "n_solved_this_run": len(entry_seconds),
        "strata_counts": {
            stratum: int(summary["count"]) for stratum, summary in strata.items()
        },
        "strata_utility": strata,
        "source_counts": _source_counts(records),
        "utility_mean": statistics.fmean(float(item["utility"]) for item in records)
        if records
        else None,
        "uniform_mean": statistics.fmean(
            float(item["candidate_utilities"].get("uniform", float("nan"))) for item in records
        )
        if records
        else None,
        "evaluations_mean": statistics.fmean(
            float(item["evaluations"]) for item in records
        )
        if records
        else None,
        "seconds_per_deployment": mean_seconds,
        "seconds_total": total_seconds,
        "seconds_extrapolated_full_library": mean_seconds * len(selected),
        "reference_json": _portable_path(output),
        "reference_json_sha256": file_sha256(output),
        "generated_by": "scripts/solve_composition_reference.py",
        "command": " ".join(["python", _portable_path(Path(__file__)), *sys.argv[1:]]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("per-stratum (best vs uniform):")
    for stratum, summary in strata.items():
        print(
            f"  {stratum:<32} n={int(summary['count']):>4} "
            f"uniform={summary['uniform_mean']:+.5f} best={summary['utility_mean']:+.5f} "
            f"lift={summary['lift_mean']:+.5f} evals={summary['evaluations_mean']:.1f}"
        )
    print(
        f"overall         : {len(records)} deployments, "
        f"uniform={manifest['uniform_mean']:+.5f} best={manifest['utility_mean']:+.5f} "
        f"lift={manifest['utility_mean'] - manifest['uniform_mean']:+.5f}"
    )
    print(
        f"evaluations     : {manifest['evaluations_mean']:.1f} mean "
        f"(budget {args.budget})"
    )
    print(
        f"timing          : {mean_seconds:.2f} s/deployment, {total_seconds:.1f} s this run, "
        f"~{extrapolated / 60.0:.1f} min for the {len(targets)} solved here, "
        f"~{manifest['seconds_extrapolated_full_library'] / 60.0:.1f} min for the "
        f"full {len(selected)}-entry split"
    )
    print(f"reference       : {_portable_path(output)}")
    print(f"manifest        : {_portable_path(manifest_path)}")
    return 0


def _source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        name = str(record["source"])
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())

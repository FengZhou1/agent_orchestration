from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from agent_orch.data import file_sha256
from agent_orch.deployment import DeploymentLibraryBuilder, write_coverage_csv
from agent_orch.schema.loader import ScenarioLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the offline stratified feasible deployment library."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--min-entries", type=int, default=128)
    parser.add_argument("--max-entries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--coverage-csv",
        default=None,
        help="Coverage CSV path (default: alongside the library JSON).",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    library = DeploymentLibraryBuilder(scenario).build(
        min_entries=args.min_entries, max_entries=args.max_entries
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / f"deployment_library_{scenario.id}.json"
    coverage_path = (
        Path(args.coverage_csv).resolve()
        if args.coverage_csv
        else output_dir / f"deployment_library_{scenario.id}.coverage.csv"
    )
    library.save(library_path)
    write_coverage_csv(library, coverage_path)

    strata_counts = library.strata_counts()
    model_sets = library.active_model_sets()
    manifest = {
        "artifact_type": "deployment_library",
        "scenario": _portable_path(scenario_path),
        "scenario_id": scenario.id,
        "scenario_hash": library.scenario_hash,
        "scenario_file_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "n_entries": len(library.entries),
        "min_entries": args.min_entries,
        "max_entries": args.max_entries,
        "seed": args.seed,
        "strata_counts": strata_counts,
        "n_strata": len(strata_counts),
        "n_active_model_sets": len(model_sets),
        "active_model_sets": ["+".join(item) for item in sorted(model_sets)],
        "shortfall_reason": library.metadata.get("shortfall_reason"),
        "library_json": _portable_path(library_path),
        "library_json_sha256": file_sha256(library_path),
        "coverage_csv": _portable_path(coverage_path),
        "generated_by": "scripts/build_deployment_library.py",
        "command": " ".join(["python", _portable_path(Path(__file__)), *sys.argv[1:]]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / f"deployment_library_{scenario.id}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"scenario          : {scenario.id} ({_portable_path(scenario_path)})")
    print(f"library           : {_portable_path(library_path)}")
    print(f"coverage csv      : {_portable_path(coverage_path)}")
    print(f"manifest          : {_portable_path(manifest_path)}")
    print(f"entries           : {len(library.entries)}")
    print(f"strata            : {len(strata_counts)}")
    print(f"distinct model sets: {len(model_sets)}")
    print("per-stratum counts:")
    for name in sorted(strata_counts):
        print(f"  {name:<32} {strata_counts[name]:>4}")
    if manifest["shortfall_reason"]:
        print(f"shortfall         : {manifest['shortfall_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

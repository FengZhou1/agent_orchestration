"""Generate and validate fixed synthetic Agent pools; never execute a policy."""
from __future__ import annotations

import argparse
from dataclasses import fields, asdict
import hashlib
import json
from pathlib import Path

import yaml

from agent_orch.schema.loader import ScenarioLoader
from agent_orch.parameters import DEFAULT_SETTINGS, load_settings, audit_scenario, require_consistent, settings_receipt
from agent_orch.workload.infrastructure import build_infrastructure
from agent_orch.workload.agent_pool import (
    GeneratorConfig, generate_pool, digest, pool_statistics, scenario_mapping,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infrastructure", type=Path,
                        help="Optional registered infrastructure; otherwise build a random edge network")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--legacy-smoke", action="store_true",
                        help="Explicit compatibility diagnostic, not a registered experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/agent_generator.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--applications", type=int)
    parser.add_argument("--tool-types", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--subsets", default="20,50,100,200")
    parser.add_argument("--calibration-passes", type=int)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(GeneratorConfig)}
    if unknown:
        raise ValueError(f"Unknown generator settings: {sorted(unknown)}")
    for name in ("applications", "tool_types", "seed", "calibration_passes"):
        if getattr(args, name) is not None:
            raw[name] = getattr(args, name)
    config = GeneratorConfig(**raw)
    subset_counts = sorted({int(x) for x in args.subsets.split(",") if x})
    if any(n < 1 or n > config.applications for n in subset_counts):
        raise ValueError("All subset sizes must fit in the fixed pool")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; generated data is not overwritten")
    settings = load_settings(args.settings)
    template = (ScenarioLoader.load(args.infrastructure) if args.infrastructure
                else build_infrastructure(settings))
    if not args.legacy_smoke:
        require_consistent(audit_scenario(template, settings, infrastructure_only=True))
    def progress(iteration, statistics):
        p = statistics["input"]
        o = statistics["output"]
        print(f"length pass {iteration}: input p50/p90={p['p50']:.0f}/{p['p90']:.0f}; "
              f"output={o['p50']:.0f}/{o['p90']:.0f}", flush=True)
    pool = generate_pool(template, config, progress=progress)
    validation = pool.validate(template)
    payload = pool.to_dict()
    pool_digest = digest(payload)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pool.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
        allow_nan=False), encoding="utf-8")
    scenarios = {}
    for n in subset_counts:
        scene = pool.scenario(template, n)
        path = args.output / f"applications_{n}.yaml"
        path.write_text(yaml.safe_dump(scenario_mapping(scene), sort_keys=False, allow_unicode=True), encoding="utf-8")
        # Verify the actual on-disk file through the production loader.
        reloaded = ScenarioLoader.load(path)
        if scenario_mapping(reloaded) != scenario_mapping(scene):
            raise AssertionError("Scenario round trip changed application parameters")
        scenarios[str(n)] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "application_ids": list(reloaded.applications), "actual_tool_types": len(reloaded.tools),
            "statistics": pool_statistics(pool.applications[:n], reloaded.tools)}
    manifest = {"pool_digest": pool_digest, "generator_config": asdict(config),
        "infrastructure_source": str(args.infrastructure.resolve()) if args.infrastructure else "registered_random_network",
        "infrastructure_sha256": (hashlib.sha256(args.infrastructure.read_bytes()).hexdigest()
                                  if args.infrastructure else digest(scenario_mapping(template))),
        "parameter_receipt": settings_receipt(settings), "legacy_smoke": args.legacy_smoke,
        "validation": validation, "scenarios": scenarios,
        "generator_source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(),
                *sorted((Path(__file__).resolve().parents[1] / "src/agent_orch/workload").glob("agent_*.py"))]},
        "scope": "synthetic structure/length/loader validation; no PPO results or SLO feasibility filtering"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
        allow_nan=False), encoding="utf-8")
    print(f"Generated {len(pool.applications)} applications, {len(pool.tools)} used tools; "
          f"topologies={validation['unique_topologies']}; pool={pool_digest[:12]}", flush=True)


if __name__ == "__main__":
    main()

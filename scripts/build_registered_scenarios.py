"""Generate registered random edge networks and fixed nested workload scenarios.

No Abilene/GEANT template, learning policy, SLO filtering or teacher is used.
One workload pool is generated/reloaded and explicitly remapped across networks.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import yaml

from agent_orch.parameters import (
    DEFAULT_SETTINGS, audit_scenario, load_settings, require_consistent,
    require_formal_ready, settings_receipt,
)
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload.agent_pool import (
    GeneratorConfig, digest, generate_pool, pool_from_mapping, scenario_mapping,
)
from agent_orch.workload.infrastructure import build_infrastructure, remap_pool_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--servers", help="Comma-separated registered counts; defaults to profile count")
    parser.add_argument("--subsets", help="Comma-separated nested application counts")
    parser.add_argument("--pool", type=Path, help="Reuse a saved pool without graph/token regeneration")
    parser.add_argument("--applications", type=int, help="Explicit pre-experiment pool size")
    parser.add_argument("--tool-types", type=int)
    parser.add_argument("--calibration-passes", type=int, help="Explicit generation diagnostic override")
    parser.add_argument("--formal", action="store_true", help="Require an author-frozen parameter profile")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    source_receipt = settings_receipt(settings)
    if args.formal:
        require_formal_ready(settings)
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; prior pools are not overwritten")
    counts = sorted({int(x) for x in args.servers.split(",")}) if args.servers else [
        settings["infrastructure"]["server_count"]]
    networks = {n: build_infrastructure(settings, n) for n in counts}
    for network in networks.values():
        require_consistent(audit_scenario(network, settings, infrastructure_only=True))
    generator_raw = yaml.safe_load(Path(settings["generator"]).read_text(encoding="utf-8"))
    if set(generator_raw) - {f.name for f in fields(GeneratorConfig)}:
        raise ValueError("Unknown generator field")
    for name in ("applications", "tool_types", "calibration_passes"):
        if getattr(args, name) is not None:
            generator_raw[name] = getattr(args, name)
    config = GeneratorConfig(**generator_raw)
    if args.pool:
        payload = json.loads(args.pool.read_text(encoding="utf-8"))
        if any(getattr(args, name) is not None for name in ("applications", "tool_types", "calibration_passes")):
            raise ValueError("Pool reuse cannot override generation settings")
    else:
        def progress(iteration, stats):
            print(f"generation pass {iteration}: input P50/P90="
                  f"{stats['input']['p50']:.0f}/{stats['input']['p90']:.0f}", flush=True)
        payload = generate_pool(next(iter(networks.values())), config, progress=progress).to_dict()
    pool_size = len(payload["applications"])
    subsets = (sorted({int(x) for x in args.subsets.split(",")}) if args.subsets else
               [n for n in settings["application_counts"] if n <= pool_size])
    if not subsets or any(n <= 0 or n > pool_size for n in subsets):
        raise ValueError("Nested subset sizes must fit the one saved pool")
    # Materialize/validate every scene before writing a result directory.
    scenes = []
    for n, network in networks.items():
        mapped = remap_pool_payload(payload, network)
        pool = pool_from_mapping(mapped, network)
        for a in subsets:
            scene = pool.scenario(network, a)
            # Pool.scenario retains infrastructure identity in metadata. Include N
            # in the scenario ID too, so trajectories cannot cross networks.
            from dataclasses import replace
            scene = replace(scene, id=f"{scene.id}-n{n}")
            report = audit_scenario(scene, settings)
            require_consistent(report)
            scenes.append((n, a, scene, report))
    if settings_receipt(settings) != source_receipt:
        raise ValueError("Parameter sources changed during generation; do not stamp mixed provenance")
    args.output.mkdir(parents=True)
    (args.output / "pool.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                        allow_nan=False), encoding="utf-8")
    manifest = {"scope": "registered synthetic pre-experiment; not convergence or real-server validation",
                "receipt": source_receipt, "pending": settings["pending"],
                "pool_digest": digest(payload), "source_pool": str(args.pool) if args.pool else None,
                "generator_config": payload["config"], "scenarios": {}}
    for n, network in networks.items():
        (args.output / f"infrastructure_n{n}.yaml").write_text(
            yaml.safe_dump(scenario_mapping(network), sort_keys=False), encoding="utf-8")
    for n, a, scene, report in scenes:
        name = f"servers_{n}_applications_{a}.yaml"
        path = args.output / name
        path.write_text(yaml.safe_dump(scenario_mapping(scene), sort_keys=False), encoding="utf-8")
        reloaded = ScenarioLoader.load(path)
        if scenario_mapping(scene) != scenario_mapping(reloaded):
            raise AssertionError("Scenario changed during round trip")
        audit_name = name.replace(".yaml", "_audit.json")
        (args.output / audit_name).write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                       allow_nan=False), encoding="utf-8")
        manifest["scenarios"][name] = {"scenario_id": scene.id, "applications": a,
            "servers": n, "gpu_total": sum(s.gpu_count for s in scene.servers.values()),
            "gpu_servers": sum(s.gpu_count > 0 for s in scene.servers.values()),
            "actual_tool_types": len(scene.tools), "candidates": len(scene.candidates),
            "total_arrival_rps": sum(sum(app.ingress_rates.values()) for app in scene.applications.values()),
            "application_ids": list(scene.applications), "audit": audit_name,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                       allow_nan=False), encoding="utf-8")
    print(f"Generated {len(scenes)} scenes from one pool; no inherited Abilene parameters", flush=True)


if __name__ == "__main__":
    main()

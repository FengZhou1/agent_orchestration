"""Tool-scale scenes from a saved pool; no graph/token resampling or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload.agent_pool import (
    pool_from_mapping, rebind_pool, scenario_mapping, digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--infrastructure", type=Path, required=True)
    parser.add_argument("--tool-counts", required=True, help="e.g. 20,50,100,200")
    parser.add_argument("--shared-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    template = ScenarioLoader.load(args.infrastructure)
    raw = json.loads(args.pool.read_text(encoding="utf-8"))
    pool = pool_from_mapping(raw, template)
    counts = sorted({int(x) for x in args.tool_counts.split(",")})
    variants = [rebind_pool(pool, template, n, args.shared_fraction, uniform_profile=True) for n in counts]
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for n, variant in zip(counts, variants):
        path = args.output / f"tools_{n}.yaml"
        scene = variant.scenario(template)
        path.write_text(yaml.safe_dump(scenario_mapping(scene), sort_keys=False, allow_unicode=True), encoding="utf-8")
        loaded = ScenarioLoader.load(path)
        assert len(loaded.tools) == n
        records[str(n)] = {"file": path.name, "validation": variant.validate(template),
                           "bindings": variant.provenance["bindings"]}
    (args.output / "manifest.json").write_text(json.dumps({"source_pool_digest": digest(raw),
        "scope": "fixed graphs/probabilities/lengths/SLOs/quality/arrivals; rebinding only, uniform tool profiles",
        "variants": records}, indent=2), encoding="utf-8")
    print(f"Exported {len(variants)} tool-count scenarios from the same pool")


if __name__ == "__main__":
    main()

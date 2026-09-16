from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from agent_orch.data import DatasetManifest, file_sha256
from agent_orch.schema.loader import ScenarioLoader


FAMILIES = (
    "interactive_retrieval",
    "transactional_tool",
    "deep_research",
    "coding_agent",
)


def export_catalogs(scenario_path: str | Path, output: str | Path) -> None:
    scenario_path = Path(scenario_path).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    infrastructure = {
        "metadata": scenario.metadata,
        "servers": [vars(server) for server in scenario.servers.values()],
        "links": [vars(link) for link in scenario.links],
    }
    (destination / "infrastructure_catalog.yaml").write_text(
        yaml.safe_dump(infrastructure, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    workflow = {
        "metadata": scenario.metadata,
        "applications": [
            {
                "id": app.id,
                "family": app.family,
                "template_id": app.template_id,
                "length_class": app.length_class,
                "pattern_flows": [
                    {
                        "id": flow.id,
                        "probability": flow.probability,
                        "final_node": flow.final_node,
                        "chains": [list(chain) for chain in flow.chains],
                    }
                    for flow in app.pattern_flows
                ],
            }
            for app in scenario.applications.values()
        ],
    }
    (destination / "workflow_catalog.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    workload_rows = []
    for app in scenario.applications.values():
        input_tokens = sum(
            app.visit_probability(node.id) * node.prompt_tokens[MODEL]
            for node in app.nodes.values()
            if node.type.value == "llm"
        )
        output_tokens = sum(
            app.visit_probability(node.id) * node.output_tokens[MODEL]
            for node in app.nodes.values()
            if node.type.value == "llm"
        )
        workload_rows.append(
            {
                "application": app.id,
                "family": app.family,
                "template_id": app.template_id,
                "length_class": app.length_class,
                "input_tokens": round(input_tokens),
                "output_tokens": round(output_tokens),
                "pattern_flow_count": len(app.pattern_flows),
            }
        )
    pd.DataFrame(workload_rows).to_csv(
        destination / "preconstructed_workload_catalog.csv", index=False
    )
    service_rows = []
    for service in scenario.tools.values():
        for server, rate in service.service_rate.items():
            service_rows.append(
                {
                    "service": service.id,
                    "server": server,
                    "vcpu": service.cpu_cores,
                    "memory_gb": service.memory_gb,
                    "stable_rate_rps": rate,
                    "arrival_scv": service.arrival_scv,
                    "running_cost_per_slot": service.running_cost_per_slot,
                    "source_status": "reference range; replace with pinned low-load measurements",
                }
            )
    pd.DataFrame(service_rows).to_csv(
        destination / "stateless_service_catalog.csv", index=False
    )

    centroids: dict[str, tuple[float, float]] = {}
    for length_class in ("short", "medium", "long"):
        nodes = [
            node
            for app in scenario.applications.values()
            if app.length_class == length_class
            for node in app.nodes.values()
            if node.type.value == "llm"
        ]
        centroids[length_class] = (
            sum(node.prompt_tokens[MODEL] for node in nodes) / len(nodes),
            sum(node.output_tokens[MODEL] for node in nodes) / len(nodes),
        )
    rows = []
    compositions = [
        {family: 0.25 for family in FAMILIES},
        *[
            {
                family: (0.60 if family == dominant else (0.20 if index == 0 else 0.10))
                for index, family in enumerate(
                    [item for item in FAMILIES if item != dominant]
                )
            }
            | {dominant: 0.60}
            for dominant in FAMILIES
        ],
    ]
    # Correct the non-dominant weights to 0.20/0.10/0.10 in stable family order.
    normalized_compositions = []
    for composition in compositions:
        if max(composition.values()) == 0.25:
            normalized_compositions.append(composition)
            continue
        dominant = max(composition, key=composition.get)
        others = [family for family in FAMILIES if family != dominant]
        normalized_compositions.append(
            {dominant: 0.60, others[0]: 0.20, others[1]: 0.10, others[2]: 0.10}
        )
    for config in scenario.llm_configs.values():
        for length_class, (prompt, output_tokens) in centroids.items():
            for load_fraction in (0.20, 0.40, 0.60, 0.80, 0.95, 1.05):
                for composition in normalized_compositions:
                    rows.append(
                        {
                            "model": config.model,
                            "config": config.id,
                            "length_class": length_class,
                            "prompt_tokens": round(prompt),
                            "output_tokens": round(output_tokens),
                            "load_fraction": load_fraction,
                            "long_request_fraction": 1.0 if length_class == "long" else 0.0,
                            **{f"{family}_fraction": composition[family] for family in FAMILIES},
                        }
                    )
    pd.DataFrame(rows).to_csv(destination / "llm_profile_request_grid.csv", index=False)
    DatasetManifest(
        artifact_type="reference-catalog-bundle",
        source_name="compiled benchmark scenario",
        source_url="https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026",
        source_version=str(scenario.metadata.get("source_catalog_sha256", "unknown")),
        license="See each source entry in scenario metadata",
        source_checksum_sha256=file_sha256(scenario_path),
        preprocessing_command=f"python scripts/export_reference_catalogs.py --scenario {scenario_path}",
        random_seed=int(scenario.metadata.get("seed", 2026)),
        parameters={
            "scenario_id": scenario.id,
            "data_sources": scenario.metadata.get("data_sources", {}),
            "profile_load_fractions": [0.20, 0.40, 0.60, 0.80, 0.95, 1.05],
            "workload_catalog": "preconstructed application patterns and quantile templates",
        },
    ).write(destination / "manifest.json")


MODEL = "qwen3-14b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--output", default="data/catalogs/main_abilene")
    args = parser.parse_args()
    export_catalogs(args.scenario, args.output)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

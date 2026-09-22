from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
import yaml

from agent_orch.schema.loader import ScenarioLoader


MODEL_IDS = ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b")
GIB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep a serving-template margin below the 32,768-token engine limit while
# retaining the long-context workload class.
WORKLOAD_TOKEN_BUDGET = 32_000
GPU_MEMORY_GB = {"A10": 24.0, "L20": 48.0, "H20": 96.0}
GPU_COST_PER_HOUR = {"A10": 1.0, "L20": 2.0, "H20": 4.0}
# The analytical model uses the non-sparse BF16 peak rates as its hardware
# layer.  Hence eta_cmp = eta_bw = 1; no fitted value is embedded in a
# scenario file.  Any calibrated equivalent rates belong to validation only.
GPU_PEAK_FLOPS = {"A10": 125e12, "L20": 119.5e12, "H20": 148e12}
GPU_BANDWIDTH = {"A10": 600e9, "L20": 864e9, "H20": 4000e9}

PRECONSTRUCTED_FAMILY_KEYS = {
    "interactive_retrieval": "interactive_rag",
    "transactional_tool": "transactional_tool",
    "deep_research": "deep_research",
    "coding_agent": "coding_agent",
}
PRECONSTRUCTED_WORKLOADS_PATH = (
    REPO_ROOT / "data" / "preconstructed_agent_workloads.yaml"
)
PRECONSTRUCTED_WORKLOADS = yaml.safe_load(
    PRECONSTRUCTED_WORKLOADS_PATH.read_text(encoding="utf-8")
)["applications"]
PRECONSTRUCTED_QUANTILES = (0.50, 0.65, 0.80, 0.90, 0.95)


SOURCE_CATALOG = {
    "arrivals": {
        "name": "stationary Poisson arrivals",
        "url": "docs/experiment_protocol.md",
        "role": "parameterized application request rates and load sweeps",
    },
    "tokens": {
        "name": "preconstructed agent workload catalog",
        "url": "data/preconstructed_agent_workloads.yaml",
        "role": "application-conditioned workflow and LLM token characteristics",
    },
    "workflows": {
        "name": "TraceLab v2 and BFCL V3/V4",
        "url": "https://gorilla.cs.berkeley.edu/leaderboard",
        "role": "agent execution patterns and quality tasks",
    },
    "stateless_services": {
        "name": "Alibaba Microservices v2021 and DeathStarBench",
        "url": "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021",
        "role": "service-graph structure and low-load processing profiles",
    },
    "infrastructure": {
        "name": "Alibaba GPU Trace v2026",
        "url": "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026",
        "role": "GPU-server heterogeneity only",
    },
    "network": {
        "name": "SNDlib 1.0",
        "url": "https://sndlib.put.poznan.pl/networks.overview.action",
        "role": "Abilene and GEANT topology and coordinates",
    },
}


PRECONSTRUCTED_LENGTH_CLASSES = ("short", "short", "medium", "medium", "long")


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


TOPOLOGIES: dict[str, dict[str, Any]] = {
    "abilene": {
        "nodes": {
            "ATLAM5": (-84.3833, 33.75),
            "ATLAng": (-85.50, 34.50),
            "CHINng": (-87.6167, 41.8333),
            "DNVRng": (-105.00, 40.75),
            "HSTNng": (-95.517364, 29.770031),
            "IPLSng": (-86.159535, 39.780622),
            "KSCYng": (-96.596704, 38.961694),
            "LOSAng": (-118.25, 34.05),
            "NYCMng": (-73.9667, 40.7833),
            "SNVAng": (-122.02553, 37.38575),
            "STTLng": (-122.30, 47.60),
            "WASHng": (-77.026842, 38.897303),
        },
        "edges": (
            ("ATLAng", "ATLAM5"), ("HSTNng", "ATLAng"),
            ("IPLSng", "ATLAng"), ("WASHng", "ATLAng"),
            ("IPLSng", "CHINng"), ("NYCMng", "CHINng"),
            ("KSCYng", "DNVRng"), ("SNVAng", "DNVRng"),
            ("STTLng", "DNVRng"), ("KSCYng", "HSTNng"),
            ("LOSAng", "HSTNng"), ("KSCYng", "IPLSng"),
            ("SNVAng", "LOSAng"), ("WASHng", "NYCMng"),
            ("STTLng", "SNVAng"),
        ),
    },
    "geant": {
        "nodes": {
            "at1.at": (16.3729, 48.2091), "be1.be": (4.3518, 50.8469),
            "ch1.ch": (6.1399, 46.2038), "cz1.cz": (14.4423, 50.0785),
            "de1.de": (8.6842, 50.1122), "es1.es": (-3.7033, 40.4167),
            "fr1.fr": (2.351, 48.8566), "gr1.gr": (23.5808, 37.9778),
            "hr1.hr": (15.9644, 45.8071), "hu1.hu": (19.0936, 47.4976),
            "ie1.ie": (-6.2573, 53.3416), "il1.il": (34.8097, 32.0714),
            "it1.it": (9.19, 45.4642), "lu1.lu": (6.1296, 49.6112),
            "nl1.nl": (4.9407, 52.3236), "ny1.ny": (-73.94384, 40.6698),
            "pl1.pl": (16.8874, 52.3963), "pt1.pt": (-9.1363, 38.7073),
            "se1.se": (17.8742, 59.3617), "si1.si": (14.5148, 46.0574),
            "sk1.sk": (17.1297, 48.1531), "uk1.uk": (-0.1264, 51.5086),
        },
        "edges": (
            ("at1.at", "ch1.ch"), ("at1.at", "de1.de"),
            ("at1.at", "hu1.hu"), ("at1.at", "ny1.ny"),
            ("at1.at", "si1.si"), ("be1.be", "fr1.fr"),
            ("be1.be", "lu1.lu"), ("be1.be", "nl1.nl"),
            ("ch1.ch", "fr1.fr"), ("ch1.ch", "it1.it"),
            ("cz1.cz", "de1.de"), ("cz1.cz", "pl1.pl"),
            ("cz1.cz", "sk1.sk"), ("de1.de", "fr1.fr"),
            ("de1.de", "gr1.gr"), ("de1.de", "ie1.ie"),
            ("de1.de", "it1.it"), ("de1.de", "nl1.nl"),
            ("de1.de", "se1.se"), ("es1.es", "fr1.fr"),
            ("es1.es", "it1.it"), ("es1.es", "pt1.pt"),
            ("fr1.fr", "lu1.lu"), ("fr1.fr", "uk1.uk"),
            ("gr1.gr", "it1.it"), ("hr1.hr", "hu1.hu"),
            ("hr1.hr", "si1.si"), ("hu1.hu", "sk1.sk"),
            ("ie1.ie", "uk1.uk"), ("il1.il", "it1.it"),
            ("il1.il", "nl1.nl"), ("nl1.nl", "uk1.uk"),
            ("ny1.ny", "uk1.uk"), ("pl1.pl", "se1.se"),
            ("pt1.pt", "uk1.uk"), ("se1.se", "uk1.uk"),
        ),
    },
}


SERVICE_PROFILE = {
    "web_search": (2, 2.0, 4.0, 0.005, 0.080),
    "information_retrieval": (2, 4.0, 6.0, 0.020, 0.500),
    "code_execution": (4, 8.0, 2.0, 0.050, 0.200),
    "file_processing": (2, 4.0, 5.0, 0.500, 0.400),
    "result_verification": (1, 2.0, 8.0, 0.020, 0.020),
    "external_api": (1, 1.0, 3.0, 0.010, 0.100),
    "knowledge_graph": (2, 4.0, 4.0, 0.030, 0.650),
    "data_transform": (2, 3.0, 5.0, 0.200, 0.150),
}


FAMILY_QUALITY = {
    "interactive_retrieval": (0.64, 0.72, 0.81, 0.89),
    "transactional_tool": (0.58, 0.68, 0.80, 0.91),
    "deep_research": (0.48, 0.62, 0.77, 0.91),
    "coding_agent": (0.40, 0.55, 0.70, 0.86),
}


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(value))


def _link_capacity(distance_km: float) -> float:
    if distance_km < 500.0:
        return 400.0
    if distance_km < 1500.0:
        return 150.0
    return 50.0


def _gpu_assignment(count: int) -> list[tuple[str, int]]:
    # Interleave accelerator classes across the topology.  One 12-node cycle
    # contains four A10, five dual-L20, and three H20 servers, so the main
    # scenario is not dominated by the 4B-only A10 configuration.
    pattern = (
        ("L20", 2), ("A10", 1), ("H20", 1), ("L20", 2),
        ("A10", 1), ("L20", 2), ("H20", 1), ("A10", 1),
        ("L20", 2), ("H20", 1), ("A10", 1), ("L20", 2),
    )
    return [pattern[index % len(pattern)] for index in range(count)]


def _models() -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen3-4b", "parameter_count": 4.0e9, "layers": 36,
            "hidden_size": 2560, "weight_bytes": 2 * 4.0e9,
            "kv_bytes_per_token": 147456.0,
        },
        {
            "id": "qwen3-8b", "parameter_count": 8.2e9, "layers": 36,
            "hidden_size": 4096, "weight_bytes": 2 * 8.2e9,
            "kv_bytes_per_token": 147456.0,
        },
        {
            "id": "qwen3-14b", "parameter_count": 14.8e9, "layers": 40,
            "hidden_size": 5120, "weight_bytes": 2 * 14.8e9,
            "kv_bytes_per_token": 163840.0,
        },
        {
            "id": "qwen3-32b", "parameter_count": 32.8e9, "layers": 64,
            "hidden_size": 5120, "weight_bytes": 2 * 32.8e9,
            "kv_bytes_per_token": 262144.0,
        },
    ]


def _config(
    model: dict[str, Any], gpu_type: str, gpu_count: int = 1
) -> dict[str, Any]:
    model_id = model["id"]
    suffix = f"{gpu_count}x{gpu_type.lower()}" if gpu_count > 1 else gpu_type.lower()
    weight_gib_per_gpu = model["weight_bytes"] / gpu_count / GIB
    reserved = math.ceil(weight_gib_per_gpu + 4.0)
    available_bytes = max(
        GIB,
        (GPU_MEMORY_GB[gpu_type] * 0.9 - weight_gib_per_gpu - 2.0) * GIB,
    )
    kv_capacity = math.floor(available_bytes * gpu_count / model["kv_bytes_per_token"])
    hourly_cost = gpu_count * GPU_COST_PER_HOUR[gpu_type]
    return {
        "id": f"{model_id}-{suffix}",
        "model": model_id,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "gpu_share": 1.0,
        "reserved_memory_gb_per_gpu": float(reserved),
        "effective_flops": GPU_PEAK_FLOPS[gpu_type] * gpu_count,
        "effective_bandwidth_bytes_s": GPU_BANDWIDTH[gpu_type] * gpu_count,
        "kv_token_capacity": float(kv_capacity),
        "running_cost_per_slot": hourly_cost / 3600.0,
        "load_cost": hourly_cost / 60.0,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
        "chunked_prefill": True,
        "prefix_cache": False,
    }


def _llm_configs(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {model["id"]: model for model in models}
    return [
        _config(by_id["qwen3-4b"], "A10"),
        _config(by_id["qwen3-4b"], "L20"),
        _config(by_id["qwen3-8b"], "L20"),
        _config(by_id["qwen3-8b"], "H20"),
        _config(by_id["qwen3-14b"], "L20"),
        _config(by_id["qwen3-14b"], "H20"),
        _config(by_id["qwen3-32b"], "H20"),
        _config(by_id["qwen3-32b"], "L20", gpu_count=2),
    ]


def _node(node_id: str, kind: str, prompt: float = 0, output: float = 0, tool: str | None = None, deadline: float | None = None) -> dict[str, Any]:
    if kind == "tool":
        return {"id": node_id, "type": "tool", "tool": tool}
    item: dict[str, Any] = {
        "id": node_id,
        "type": "llm",
        "prompt_tokens": {model: round(prompt) for model in MODEL_IDS},
        "output_tokens": {model: round(output) for model in MODEL_IDS},
    }
    if deadline is not None:
        item["stage_deadline_s"] = deadline
    return item


def _tool_type(node_id: str) -> str:
    """Map workload-specific service names to reusable stateless services."""
    if node_id in {"hybrid_search", "search_a", "search_b", "targeted_search_a", "targeted_search_b"}:
        return "web_search"
    if node_id in {"dense_retrieval", "db_single", "db_chain", "read_repository", "symbol_search", "read_test_logs"}:
        return "information_retrieval"
    if node_id in {"api_single", "api_chain"}:
        return "external_api"
    if node_id in {"verification", "compensation"}:
        return "result_verification"
    if node_id in {"edit_first", "edit_retry"}:
        return "file_processing"
    if node_id in {"test_first", "test_retry"}:
        return "code_execution"
    if node_id.startswith("search"):
        return "web_search"
    if node_id.startswith("db"):
        return "information_retrieval"
    if node_id.startswith("api"):
        return "external_api"
    if node_id.startswith("test"):
        return "code_execution"
    return "information_retrieval"


def _quantile_value(statistics: dict[str, float], quantile: float) -> int:
    p50 = max(float(statistics["p50"]), 1.0)
    p95 = max(float(statistics["p95"]), p50)
    sigma = math.log(p95 / p50) / NormalDist().inv_cdf(0.95)
    return max(1, round(math.exp(math.log(p50) + sigma * NormalDist().inv_cdf(quantile))))


def _choice_pattern(option: dict[str, Any]) -> list[list[str]]:
    if "chains" in option:
        return [list(chain) for chain in option["chains"]]
    branches = [list(branch) for branch in option.get("parallel_branches", [])]
    join_chain = list(option.get("join_chain", []))
    return [branch + join_chain for branch in branches]


def _append_pattern(prefixes: list[list[str]], suffixes: list[list[str]]) -> list[list[str]]:
    return [prefix + suffix for prefix in prefixes for suffix in suffixes]


def _preconstructed_workflow(
    workload: dict[str, Any], family: str, template_index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quantile = PRECONSTRUCTED_QUANTILES[template_index % len(PRECONSTRUCTED_QUANTILES)]
    nodes: list[dict[str, Any]] = []
    for node_id, raw_node in workload["nodes"].items():
        if raw_node["type"] == "llm":
            output_tokens = min(
                _quantile_value(raw_node["output_tokens"], quantile),
                WORKLOAD_TOKEN_BUDGET - 1,
            )
            prompt_tokens = min(
                _quantile_value(raw_node["input_tokens"], quantile),
                WORKLOAD_TOKEN_BUDGET - output_tokens,
            )
            nodes.append(
                _node(
                    node_id,
                    "llm",
                    prompt_tokens,
                    output_tokens,
                )
            )
        else:
            nodes.append(_node(node_id, "tool", tool=_tool_type(node_id)))

    choice_nodes = workload.get("choices", [])
    root = choice_nodes[0]["llm_node"] if choice_nodes else next(
        node["id"] for node in nodes if node["type"] == "llm"
    )
    prefixes = [[root]]
    mandatory_parallel = workload.get("mandatory_parallel", [])
    if mandatory_parallel:
        parallel = mandatory_parallel[0]
        prefixes = [
            [parallel["llm_node"]] + list(branch) + [parallel["join_node"]]
            for branch in parallel["branches"]
        ]
    if workload.get("mandatory_chain"):
        chain = list(workload["mandatory_chain"])
        prefixes = [prefix + (chain[1:] if prefix[-1] == chain[0] else chain) for prefix in prefixes]

    patterns: list[tuple[float, list[list[str]], tuple[int, ...]]] = [
        (1.0, prefixes, tuple())
    ]
    for choice in choice_nodes:
        expanded: list[tuple[float, list[list[str]], tuple[int, ...]]] = []
        for probability, current, selected in patterns:
            for option_index, option in enumerate(choice["options"]):
                expanded.append(
                    (
                        probability * float(option["probability"]),
                        _append_pattern(current, _choice_pattern(option)),
                        selected + (option_index,),
                    )
                )
        patterns = expanded

    flows = []
    for flow_index, (probability, chains, selected) in enumerate(patterns, start=1):
        flows.append(
            {
                "id": f"{family}-flow-{flow_index:02d}",
                "probability": probability,
                "final_node": chains[0][-1],
                "chains": chains,
            }
        )
    return nodes, flows, dict(workload.get("slo", {}))


def _communication(
    nodes: list[dict[str, Any]],
    flows: list[dict[str, Any]],
    service_parameters: dict[str, dict[str, float]],
) -> dict[str, float]:
    node_map = {node["id"]: node for node in nodes}
    edges = {
        (source, target)
        for flow in flows
        for chain in flow["chains"]
        for source, target in zip(chain[:-1], chain[1:])
    }
    result: dict[str, float] = {}
    for model in MODEL_IDS:
        for source, target in sorted(edges):
            source_node, target_node = node_map[source], node_map[target]
            if target_node["type"] == "tool":
                data_mb = service_parameters[target_node["tool"]]["request_mb"]
            elif source_node["type"] == "tool":
                data_mb = service_parameters[source_node["tool"]]["response_mb"]
            else:
                data_mb = 4.0 * source_node["output_tokens"][model] / 1e6
            result[f"{model}:{source}->{target}"] = round(float(data_mb), 6)
    return result


def _application(
    family: str,
    template_index: int,
    ingress: str,
    rate: float,
    service_parameters: dict[str, dict[str, float]],
    workload: dict[str, Any],
    extended_services: bool = False,
) -> dict[str, Any]:
    nodes, flows, slo = _preconstructed_workflow(workload, family, template_index)
    if extended_services:
        for node in nodes:
            if template_index % 4 == 2 and node.get("tool") == "external_api":
                node["tool"] = "data_transform"
            if template_index % 4 == 3 and node.get("tool") == "information_retrieval":
                node["tool"] = "knowledge_graph"
    llm_nodes = [node for node in nodes if node["type"] == "llm"]
    first = next(node for node in nodes if node["type"] == "llm")
    final_output = {
        model: sum(
            float(flow["probability"])
            * next(node for node in llm_nodes if node["id"] == flow["final_node"])["output_tokens"][model]
            for flow in flows
        )
        for model in MODEL_IDS
    }
    slo = slo or {"type": "cmp", "deadline_s": 40.0}
    quality = dict(zip(MODEL_IDS, FAMILY_QUALITY[family]))
    app_id = f"{family}-t{template_index + 1:02d}"
    return {
        "id": app_id,
        "family": family,
        "template_id": f"{family}:{template_index + 1}",
        "length_class": PRECONSTRUCTED_LENGTH_CLASSES[
            template_index % len(PRECONSTRUCTED_LENGTH_CLASSES)
        ],
        "ingress_rates": {ingress: rate},
        "slo": slo,
        "quality": quality,
        "entry_data_mb": {
            model: round(4.0 * first["prompt_tokens"][model] / 1e6, 6)
            for model in MODEL_IDS
        },
        "exit_data_mb": {
            model: round(4.0 * final_output[model] / 1e6, 6)
            for model in MODEL_IDS
        },
        "edge_data_mb": _communication(nodes, flows, service_parameters),
        "nodes": nodes,
        "pattern_flows": flows,
    }


def _service_parameters(profile: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    parameters = {
        service: {
            "cores": float(cores),
            "memory_gb": float(memory),
            "stable_rate_rps": float(rate_per_core * cores),
            "request_mb": float(request_mb),
            "response_mb": float(response_mb),
        }
        for service, (cores, memory, rate_per_core, request_mb, response_mb)
        in SERVICE_PROFILE.items()
    }
    if profile is None:
        return parameters
    required = {
        "service", "vcpu", "stable_rate_rps",
        "request_mb", "response_mb",
    }
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"Stateless-service profile is missing columns: {sorted(missing)}")
    for service, values in parameters.items():
        subset = profile[profile["service"].astype(str) == service].copy()
        if subset.empty:
            continue
        distance = (subset["vcpu"].astype(float) - values["cores"]).abs()
        subset = subset[distance == distance.min()]
        for column in (
            "stable_rate_rps", "request_mb", "response_mb"
        ):
            measured = subset[column].dropna().astype(float)
            if not measured.empty:
                values[column] = float(measured.median())
    return parameters


def _apply_preconstructed_service_times(
    parameters: dict[str, dict[str, float]],
    workloads: dict[str, dict[str, Any]],
) -> None:
    """Use the preconstructed service timings while retaining shared service pools."""
    observations: dict[str, list[float]] = {}
    returns: dict[str, list[float]] = {}
    for workload in workloads.values():
        for node_id, node in workload["nodes"].items():
            if node["type"] != "stateless_service":
                continue
            service = _tool_type(node_id)
            observations.setdefault(service, []).append(float(node["mean_service_seconds"]))
            returns.setdefault(service, []).append(float(node["return_tokens"]["p50"]))
    for service, values in observations.items():
        if service not in parameters or not values:
            continue
        parameters[service]["stable_rate_rps"] = 1.0 / max(sum(values) / len(values), 1.0e-6)
        parameters[service]["response_mb"] = (
            4.0 * sum(returns[service]) / max(len(returns[service]), 1) / 1.0e6
        )


def build_scenario(
    topology_name: str,
    application_count: int,
    seed: int = 2026,
    infrastructure_servers: list[dict[str, Any]] | None = None,
    service_profile: pd.DataFrame | None = None,
    workloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    topology = TOPOLOGIES[topology_name]
    node_ids = list(topology["nodes"])
    if infrastructure_servers is not None and len(infrastructure_servers) < len(node_ids):
        raise ValueError(
            f"Infrastructure catalog has {len(infrastructure_servers)} rows; "
            f"{len(node_ids)} are required for {topology_name}"
        )
    assignments = _gpu_assignment(len(node_ids))
    servers = []
    for index, (node_id, (default_gpu_type, default_gpu_count)) in enumerate(
        zip(node_ids, assignments)
    ):
        source = infrastructure_servers[index] if infrastructure_servers else None
        gpu_type = str(source["gpu_type"]).upper() if source else default_gpu_type
        gpu_count = int(source["gpu_count"]) if source else default_gpu_count
        base_cpu = {"A10": 32, "L20": 48, "H20": 64}[gpu_type]
        cpu = int(source["cpu_cores"]) if source else base_cpu + 8 * (index % 2)
        memory = float(source.get("memory_gb", 4 * cpu)) if source else float(4 * cpu)
        gpu_memory = (
            float(source.get("gpu_memory_gb", GPU_MEMORY_GB[gpu_type]))
            if source else GPU_MEMORY_GB[gpu_type]
        )
        servers.append(
            {
                "id": node_id,
                "cpu_cores": cpu,
                "memory_gb": memory,
                "gpu_count": gpu_count,
                "gpu_memory_gb": gpu_memory,
                "gpu_type": gpu_type,
            }
        )
    links = []
    for source, target in topology["edges"]:
        distance = _haversine_km(topology["nodes"][source], topology["nodes"][target])
        capacity = _link_capacity(distance)
        propagation = min(20.0, max(1.0, distance / 200.0))
        for u, v in ((source, target), (target, source)):
            links.append(
                {
                    "source": u, "target": v, "capacity_mbps": capacity,
                    "propagation_ms": round(propagation, 3),
                    "cost_per_mb": 1e-5,
                }
            )
    models = _models()
    configs = _llm_configs(models)
    candidates = []
    for server in servers:
        for config in configs:
            if config["gpu_type"] != server["gpu_type"]:
                continue
            if config["gpu_count"] > server["gpu_count"]:
                continue
            if config["reserved_memory_gb_per_gpu"] > server["gpu_memory_gb"]:
                continue
            candidates.append(
                {
                    "id": f"{config['id']}@{server['id']}",
                    "model": config["model"],
                    "config": config["id"],
                    "server": server["id"],
                }
            )
    workload_catalog = workloads or PRECONSTRUCTED_WORKLOADS
    service_parameters = _service_parameters(service_profile)
    if service_profile is None:
        _apply_preconstructed_service_times(service_parameters, workload_catalog)
    tools = []
    median_cpu = float(pd.Series([server["cpu_cores"] for server in servers]).median())
    for service_id, parameters in service_parameters.items():
        cores = int(parameters["cores"])
        memory = float(parameters["memory_gb"])
        base_rate = float(parameters["stable_rate_rps"])
        rates = {
            server["id"]: round(
                base_rate * (server["cpu_cores"] / median_cpu) ** 0.20, 3
            )
            for server in servers
        }
        hourly_cost = cores * 0.05
        tools.append(
            {
                "id": service_id,
                "cpu_cores": cores,
                "memory_gb": memory,
                "service_rate": rates,
                "arrival_scv": 1.0,
                "running_cost_per_slot": hourly_cost / 3600.0,
                "start_cost": hourly_cost / 60.0,
            }
        )
    families = (
        "interactive_retrieval", "transactional_tool", "deep_research", "coding_agent"
    )
    applications = []
    base, remainder = divmod(application_count, len(families))
    family_counts = {
        family: base + (1 if index < remainder else 0)
        for index, family in enumerate(families)
    }
    # The checked-in rates provide a stable reference point for stationary
    # Poisson experiments. Load sweeps multiply these rates uniformly.
    total_family_rate = 0.001
    cursor = 0
    for family in families:
        count = family_counts[family]
        for template_index in range(count):
            applications.append(
                _application(
                    family,
                    template_index,
                    node_ids[cursor % len(node_ids)],
                    total_family_rate / count,
                    service_parameters,
                    workload_catalog[PRECONSTRUCTED_FAMILY_KEYS[family]],
                    extended_services=application_count > 20,
                )
            )
            cursor += 1
    used_services = {
        node["tool"]
        for application in applications
        for node in application["nodes"]
        if node["type"] == "tool"
    }
    tools = [tool for tool in tools if tool["id"] in used_services]
    source_digest = hashlib.sha256(
        yaml.safe_dump(SOURCE_CATALOG, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"agent-{topology_name}-{application_count}",
        "metadata": {
            "role": "main" if topology_name == "abilene" else "scale",
            "generated_by": "scripts/build_benchmark_scenarios.py",
            "generated_on": date.today().isoformat(),
            "seed": seed,
            "source_catalog_sha256": source_digest,
            "data_sources": SOURCE_CATALOG,
            "preconstructed_workload_families": list(PRECONSTRUCTED_FAMILY_KEYS.values()),
            "preconstructed_quantiles": list(PRECONSTRUCTED_QUANTILES),
            "workload_class_count": len(applications),
            "pattern_flow_count": sum(
                len(application["pattern_flows"]) for application in applications
            ),
            "pattern_flow_expansion": (
                "all probabilistic choices enumerated; parallel chains retained within each flow"
            ),
            "request_token_budget": WORKLOAD_TOKEN_BUDGET,
            "arrival_process": {
                "distribution": "stationary intensity (mean-field steady state)",
                "base_total_rate_rps": 4.0 * total_family_rate,
                "load_levels": [0.40, 0.65, 0.85, 1.05],
                "load_basis": "reference stable capacity from calibrate_load_levels.py",
            },
            "units": {
                "arrival_rate": "request/s",
                "service_rate": "request/s",
                "application_data": "MB/request",
                "link_load": "Mbit/s",
                "latency": "s",
            },
            "quality_status": "reference values; replace with pinned benchmark runs",
            "slo_status": "reference thresholds; replace with low-load P95 calibration",
            "cost_normalization": "A10-hour=1, L20-hour=2, H20-hour=4",
            "default_workload": (
                "preconstructed agent workflow characteristics with a stationary arrival intensity"
            ),
            "infrastructure_status": (
                "processed Alibaba GPU catalog" if infrastructure_servers
                else "deterministic reference fixture"
            ),
            "stateless_service_status": (
                "processed service profile" if service_profile is not None
                else "reference 2--8 request/s/core range"
            ),
        },
        "simulation": {
            "slot_seconds": 1.0,
            "prefill_chunk_tokens": 512,
            "overload_delay_s": 600.0,
            "orchestration_period_s": 60.0,
            "max_tool_replicas_per_server": 4,
        },
        "reward": {
            "cost_weight": 0.25, "latency_weight": 0.25,
            "goodput_weight": 0.25, "quality_weight": 0.25,
        },
        "servers": servers,
        "links": links,
        "models": models,
        "llm_configs": configs,
        "candidates": candidates,
        "tools": tools,
        "applications": applications,
    }


def stress_variants(main: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = {}
    network = deepcopy(main)
    network["id"] += "-network-0p5"
    network["metadata"]["role"] = "stress-network"
    for link in network["links"]:
        link["capacity_mbps"] *= 0.5
    variants["stress_network_0p5.yaml"] = network

    service = deepcopy(main)
    service["id"] += "-service-0p5"
    service["metadata"]["role"] = "stress-service"
    for tool in service["tools"]:
        tool["service_rate"] = {
            server: 0.5 * rate for server, rate in tool["service_rate"].items()
        }
    variants["stress_service_0p5.yaml"] = service

    gpu = deepcopy(main)
    gpu["id"] += "-gpu-unavailable"
    gpu["metadata"]["role"] = "stress-gpu"
    gpu["candidates"] = [
        candidate
        for index, candidate in enumerate(gpu["candidates"])
        if index % 4 != 0
    ]
    variants["stress_gpu_unavailable.yaml"] = gpu

    # The protocol lists an arrival-burst stress case alongside the other three.
    # It is expressed on the scenario so the burst travels with the scenario's
    # metadata (the trace generator reads it); the intensity itself lives in the
    # arrival process, not in the resource parameters.
    burst = deepcopy(main)
    burst["id"] += "-arrival-burst"
    burst["metadata"]["role"] = "stress-arrival"
    burst["metadata"]["arrival_burst"] = {
        "pattern": "gaussian_burst",
        "period_slots": 60,
        "sigma_slots": 8.0,
        "phase": 0.25,
        "low_fraction": 0.4,
        "jitter": 0.05,
        "note": (
            "periodic Gaussian bursts on top of the load-level baseline rate; "
            "consumed by ArrivalTrace.gaussian_burst_intensity"
        ),
    }
    variants["stress_arrival_burst.yaml"] = burst
    return variants


def write_scenario(raw: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    ScenarioLoader.load(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="configs/benchmarks")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--infrastructure-catalog")
    parser.add_argument("--service-profile")
    parser.add_argument(
        "--preconstructed-workloads",
        default=str(PRECONSTRUCTED_WORKLOADS_PATH),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    workload_path = Path(args.preconstructed_workloads).resolve()
    workloads = yaml.safe_load(workload_path.read_text(encoding="utf-8"))["applications"]
    infrastructure = None
    if args.infrastructure_catalog:
        infrastructure = yaml.safe_load(
            Path(args.infrastructure_catalog).read_text(encoding="utf-8")
        )["servers"]
    service_profile = (
        pd.read_csv(args.service_profile) if args.service_profile else None
    )
    main_scenario = build_scenario(
        "abilene", 20, args.seed, infrastructure, service_profile, workloads
    )
    scale_scenario = build_scenario(
        "geant", 50, args.seed, infrastructure, service_profile, workloads
    )
    calibration_inputs = {}
    for label, raw_path in (
        ("infrastructure_catalog", args.infrastructure_catalog),
        ("stateless_service_profile", args.service_profile),
    ):
        if raw_path:
            path = Path(raw_path).resolve()
            calibration_inputs[label] = {
                "path": _portable_path(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    for scenario in (main_scenario, scale_scenario):
        scenario["metadata"]["calibration_inputs"] = calibration_inputs
        scenario["metadata"]["preconstructed_workload_path"] = _portable_path(workload_path)
        scenario["metadata"]["preconstructed_workload_sha256"] = hashlib.sha256(
            workload_path.read_bytes()
        ).hexdigest()
    write_scenario(main_scenario, output / "main_abilene.yaml")
    write_scenario(scale_scenario, output / "scale_geant.yaml")
    for filename, scenario in stress_variants(main_scenario).items():
        write_scenario(scenario, output / filename)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

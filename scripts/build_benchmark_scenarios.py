from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agent_orch.schema.loader import ScenarioLoader


MODEL_IDS = ("qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b")
GPU_MEMORY_GB = {"A10": 24.0, "L20": 48.0, "H20": 96.0}
GPU_COST_PER_HOUR = {"A10": 1.0, "L20": 2.0, "H20": 4.0}
GPU_EFFECTIVE_FLOPS = {"A10": 60e12, "L20": 90e12, "H20": 120e12}
GPU_BANDWIDTH = {"A10": 600e9, "L20": 864e9, "H20": 3500e9}

SOURCE_CATALOG = {
    "arrivals": {
        "name": "stationary Poisson arrivals",
        "url": "docs/experiment_protocol.md",
        "role": "parameterized application request rates and load sweeps",
    },
    "tokens": {
        "name": "JITServe Table 2",
        "url": "https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf",
        "role": "application-conditioned input and output token statistics",
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


# JITServe Table 2 reports request-level input and output token statistics for
# chatbot and deep-research workloads under single and compound execution.
# Each row is mapped to the closest application family in the benchmark.
JITSERVE_WORKLOADS: dict[str, dict[str, Any]] = {
    "interactive_retrieval": {
        "workload": "Chatbot",
        "request_type": "Single",
        "input": {"mean": 93, "std": 244, "p50": 27, "p95": 391},
        "output": {"mean": 318, "std": 313, "p50": 225, "p95": 1024},
    },
    "transactional_tool": {
        "workload": "Deep Research",
        "request_type": "Single",
        "input": {"mean": 1911, "std": 2781, "p50": 403, "p95": 7573},
        "output": {"mean": 534, "std": 644, "p50": 410, "p95": 1544},
    },
    "deep_research": {
        "workload": "Deep Research",
        "request_type": "Compound",
        "input": {"mean": 12223, "std": 8407, "p50": 10807, "p95": 29282},
        "output": {"mean": 3541, "std": 2370, "p50": 3148, "p95": 7525},
    },
    "coding_agent": {
        "workload": "Chatbot",
        "request_type": "Compound",
        "input": {"mean": 1300, "std": 912, "p50": 1097, "p95": 2767},
        "output": {"mean": 4458, "std": 1176, "p50": 4417, "p95": 6452},
    },
}

JITSERVE_LENGTH_CLASSES = ("short", "short", "medium", "medium", "long")


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
    "web_search": (2, 2.0, 4.0, 1.5, 0.005, 0.080),
    "information_retrieval": (2, 4.0, 6.0, 1.1, 0.020, 0.500),
    "code_execution": (4, 8.0, 2.0, 2.0, 0.050, 0.200),
    "file_processing": (2, 4.0, 5.0, 1.2, 0.500, 0.400),
    "result_verification": (1, 2.0, 8.0, 0.8, 0.020, 0.020),
    "external_api": (1, 1.0, 3.0, 2.5, 0.010, 0.100),
    "knowledge_graph": (2, 4.0, 4.0, 1.4, 0.030, 0.650),
    "data_transform": (2, 3.0, 5.0, 1.3, 0.200, 0.150),
}


FAMILY_QUALITY = {
    "interactive_retrieval": (0.72, 0.81, 0.89),
    "transactional_tool": (0.68, 0.80, 0.91),
    "deep_research": (0.62, 0.77, 0.91),
    "coding_agent": (0.55, 0.70, 0.86),
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
    # NVIDIA-only proportions normalized from the Alibaba GPU v2026 fleet mix.
    pattern = (
        ("H20", 1), ("L20", 2), ("A10", 1), ("H20", 1),
        ("L20", 2), ("H20", 1), ("A10", 1), ("H20", 1),
        ("L20", 2), ("H20", 1), ("L20", 2), ("A10", 1),
    )
    return [pattern[index % len(pattern)] for index in range(count)]


def _models() -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen2.5-7b", "parameter_count": 7.61e9, "layers": 28,
            "hidden_size": 3584, "weight_bytes": 2 * 7.61e9,
            "kv_bytes_per_token": 114688.0,
        },
        {
            "id": "qwen2.5-14b", "parameter_count": 14.7e9, "layers": 48,
            "hidden_size": 5120, "weight_bytes": 2 * 14.7e9,
            "kv_bytes_per_token": 393216.0,
        },
        {
            "id": "qwen2.5-32b", "parameter_count": 32.5e9, "layers": 64,
            "hidden_size": 5120, "weight_bytes": 2 * 32.5e9,
            "kv_bytes_per_token": 524288.0,
        },
    ]


def _config(
    model: dict[str, Any], gpu_type: str, gpu_count: int = 1
) -> dict[str, Any]:
    model_id = model["id"]
    suffix = f"{gpu_count}x{gpu_type.lower()}" if gpu_count > 1 else gpu_type.lower()
    weight_gb_per_gpu = model["weight_bytes"] / gpu_count / 1e9
    reserved = math.ceil(weight_gb_per_gpu + 4.0)
    available_bytes = max(
        1e9,
        (GPU_MEMORY_GB[gpu_type] * 0.9 - weight_gb_per_gpu - 2.0) * 1e9,
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
        "effective_flops": GPU_EFFECTIVE_FLOPS[gpu_type] * gpu_count,
        "effective_bandwidth_bytes_s": GPU_BANDWIDTH[gpu_type] * gpu_count,
        "effective_concurrency": 16 if model_id == "qwen2.5-32b" else 24,
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
        _config(by_id["qwen2.5-7b"], "A10"),
        _config(by_id["qwen2.5-7b"], "L20"),
        _config(by_id["qwen2.5-7b"], "H20"),
        _config(by_id["qwen2.5-14b"], "L20"),
        _config(by_id["qwen2.5-14b"], "H20"),
        _config(by_id["qwen2.5-32b"], "H20"),
        _config(by_id["qwen2.5-32b"], "L20", gpu_count=2),
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


def _workflow(family: str, scale: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if family == "interactive_retrieval":
        nodes = [
            _node("plan", "llm", 256 * scale, 24 * scale),
            _node("search", "tool", tool="web_search"),
            _node("retrieve", "tool", tool="information_retrieval"),
            _node("final", "llm", 640 * scale, 72 * scale),
        ]
        flows = [
            {"id": "search", "probability": 0.70, "final_node": "final", "chains": [["plan", "search", "final"]]},
            {"id": "search-retrieve", "probability": 0.30, "final_node": "final", "chains": [["plan", "search", "retrieve", "final"]]},
        ]
        slo = {"type": "lat", "ttft_s": 2.0, "tbt_s": 0.15}
    elif family == "transactional_tool":
        nodes = [
            _node("plan", "llm", 384 * scale, 32 * scale, deadline=3.0),
            _node("api", "tool", tool="external_api"),
            _node("verify", "tool", tool="result_verification"),
            _node("file", "tool", tool="file_processing"),
            _node("final", "llm", 768 * scale, 64 * scale),
        ]
        flows = [
            {"id": "direct", "probability": 0.55, "final_node": "final", "chains": [["plan", "api", "final"]]},
            {"id": "verified", "probability": 0.45, "final_node": "final", "chains": [["plan", "api", "file", "verify", "final"]]},
        ]
        slo = {"type": "ddl", "deadline_s": 8.0}
    elif family == "deep_research":
        nodes = [
            _node("plan", "llm", 768 * scale, 64 * scale, deadline=5.0),
            _node("search", "tool", tool="web_search"),
            _node("analyze_web", "llm", 1536 * scale, 128 * scale, deadline=12.0),
            _node("retrieve", "tool", tool="information_retrieval"),
            _node("analyze_docs", "llm", 2048 * scale, 160 * scale, deadline=15.0),
            _node("api", "tool", tool="external_api"),
            _node("file", "tool", tool="file_processing"),
            _node("verify", "tool", tool="result_verification"),
            _node("final", "llm", 3072 * scale, 256 * scale),
        ]
        flows = [
            {
                "id": "parallel-core", "probability": 0.65, "final_node": "final",
                "chains": [
                    ["plan", "search", "analyze_web", "verify", "final"],
                    ["plan", "retrieve", "analyze_docs", "final"],
                ],
            },
            {
                "id": "parallel-expanded", "probability": 0.35, "final_node": "final",
                "chains": [
                    ["plan", "search", "analyze_web", "api", "verify", "final"],
                    ["plan", "retrieve", "analyze_docs", "file", "verify", "final"],
                ],
            },
        ]
        slo = {"type": "cmp", "ttft_s": 4.0, "tbt_s": 0.20, "deadline_s": 25.0}
    else:
        nodes = [
            _node("plan", "llm", 1024 * scale, 96 * scale, deadline=6.0),
            _node("read", "tool", tool="file_processing"),
            _node("design", "llm", 2048 * scale, 160 * scale, deadline=14.0),
            _node("edit", "tool", tool="file_processing"),
            _node("implement", "llm", 4096 * scale, 256 * scale, deadline=24.0),
            _node("unit", "tool", tool="code_execution"),
            _node("integration", "tool", tool="code_execution"),
            _node("inspect", "llm", 3072 * scale, 160 * scale, deadline=32.0),
            _node("verify", "tool", tool="result_verification"),
            _node("final", "llm", 2048 * scale, 192 * scale),
        ]
        flows = [
            {
                "id": "serial-test", "probability": 0.60, "final_node": "final",
                "chains": [["plan", "read", "design", "edit", "implement", "unit", "inspect", "verify", "final"]],
            },
            {
                "id": "parallel-test", "probability": 0.40, "final_node": "final",
                "chains": [
                    ["plan", "read", "design", "edit", "implement", "unit", "inspect", "verify", "final"],
                    ["plan", "read", "design", "edit", "implement", "integration", "inspect", "verify", "final"],
                ],
            },
        ]
        slo = {"type": "cmp", "ttft_s": 5.0, "tbt_s": 0.25, "deadline_s": 40.0}
    return nodes, flows, slo


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


def _jitserve_anchor(statistics: dict[str, int], template_index: int) -> int:
    """Return five ordered token anchors from the reported P50, mean, and P95."""
    p50 = float(statistics["p50"])
    average = float(statistics["mean"])
    p95 = float(statistics["p95"])
    anchors = (
        p50,
        math.sqrt(p50 * average),
        average,
        math.sqrt(average * p95),
        p95,
    )
    return max(1, round(anchors[template_index % len(anchors)]))


def _apply_jitserve_tokens(
    nodes: list[dict[str, Any]],
    flows: list[dict[str, Any]],
    target_input: int,
    target_output: int,
) -> None:
    """Scale LLM-node token demands to a request-level JITServe anchor."""
    visit_probability = {
        node["id"]: sum(
            float(flow["probability"])
            for flow in flows
            if node["id"] in {item for chain in flow["chains"] for item in chain}
        )
        for node in nodes
        if node["type"] == "llm"
    }
    input_total = sum(
        visit_probability[node["id"]] * float(node["prompt_tokens"][MODEL_IDS[0]])
        for node in nodes
        if node["type"] == "llm"
    )
    output_total = sum(
        visit_probability[node["id"]] * float(node["output_tokens"][MODEL_IDS[0]])
        for node in nodes
        if node["type"] == "llm"
    )
    input_scale = target_input / input_total
    output_scale = target_output / output_total
    for node in nodes:
        if node["type"] != "llm":
            continue
        prompt = max(1, round(float(node["prompt_tokens"][MODEL_IDS[0]]) * input_scale))
        output = max(1, round(float(node["output_tokens"][MODEL_IDS[0]]) * output_scale))
        node["prompt_tokens"] = {model: prompt for model in MODEL_IDS}
        node["output_tokens"] = {model: output for model in MODEL_IDS}


def _application(
    family: str,
    template_index: int,
    ingress: str,
    rate: float,
    service_parameters: dict[str, dict[str, float]],
    extended_services: bool = False,
) -> dict[str, Any]:
    workload = JITSERVE_WORKLOADS[family]
    target_input = _jitserve_anchor(workload["input"], template_index)
    target_output = _jitserve_anchor(workload["output"], template_index)
    nodes, flows, slo = _workflow(family, 1.0)
    _apply_jitserve_tokens(nodes, flows, target_input, target_output)
    if extended_services:
        for node in nodes:
            if template_index % 4 == 2 and node.get("tool") == "external_api":
                node["tool"] = "data_transform"
            if template_index % 4 == 3 and node.get("tool") == "information_retrieval":
                node["tool"] = "knowledge_graph"
    llm_nodes = [node for node in nodes if node["type"] == "llm"]
    final_id = flows[0]["final_node"]
    final = next(node for node in llm_nodes if node["id"] == final_id)
    first = llm_nodes[0]
    quality = dict(zip(MODEL_IDS, FAMILY_QUALITY[family]))
    app_id = f"{family}-t{template_index + 1:02d}"
    return {
        "id": app_id,
        "family": family,
        "template_id": f"{family}:{template_index + 1}",
        "length_class": JITSERVE_LENGTH_CLASSES[
            template_index % len(JITSERVE_LENGTH_CLASSES)
        ],
        "ingress_rates": {ingress: rate},
        "slo": slo,
        "quality": quality,
        "entry_data_mb": {
            model: round(4.0 * first["prompt_tokens"][model] / 1e6, 6)
            for model in MODEL_IDS
        },
        "exit_data_mb": {
            model: round(4.0 * final["output_tokens"][model] / 1e6, 6)
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
            "service_scv": float(scv),
            "request_mb": float(request_mb),
            "response_mb": float(response_mb),
        }
        for service, (cores, memory, rate_per_core, scv, request_mb, response_mb)
        in SERVICE_PROFILE.items()
    }
    if profile is None:
        return parameters
    required = {
        "service", "vcpu", "stable_rate_rps", "service_scv",
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
            "stable_rate_rps", "service_scv", "request_mb", "response_mb"
        ):
            measured = subset[column].dropna().astype(float)
            if not measured.empty:
                values[column] = float(measured.median())
    return parameters


def build_scenario(
    topology_name: str,
    application_count: int,
    seed: int = 2026,
    infrastructure_servers: list[dict[str, Any]] | None = None,
    service_profile: pd.DataFrame | None = None,
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
    service_parameters = _service_parameters(service_profile)
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
                "service_scv": float(parameters["service_scv"]),
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
    total_family_rate = 0.01125
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
            "jitserve_workload_profiles": JITSERVE_WORKLOADS,
            "arrival_process": {
                "distribution": "stationary Poisson",
                "base_total_rate_rps": 4.0 * total_family_rate,
                "load_scales": [0.5, 1.0, 2.0, 3.0],
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
                "JITServe Table 2 token characteristics with stationary Poisson arrivals"
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
            "overload_delay_s": 60.0,
            "deployment_period_slots": 60,
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
    args = parser.parse_args()
    output = Path(args.output).resolve()
    infrastructure = None
    if args.infrastructure_catalog:
        infrastructure = yaml.safe_load(
            Path(args.infrastructure_catalog).read_text(encoding="utf-8")
        )["servers"]
    service_profile = (
        pd.read_csv(args.service_profile) if args.service_profile else None
    )
    main_scenario = build_scenario(
        "abilene", 20, args.seed, infrastructure, service_profile
    )
    scale_scenario = build_scenario(
        "geant", 50, args.seed, infrastructure, service_profile
    )
    calibration_inputs = {}
    for label, raw_path in (
        ("infrastructure_catalog", args.infrastructure_catalog),
        ("stateless_service_profile", args.service_profile),
    ):
        if raw_path:
            path = Path(raw_path).resolve()
            calibration_inputs[label] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    for scenario in (main_scenario, scale_scenario):
        scenario["metadata"]["calibration_inputs"] = calibration_inputs
    write_scenario(main_scenario, output / "main_abilene.yaml")
    write_scenario(scale_scenario, output / "scale_geant.yaml")
    for filename, scenario in stress_variants(main_scenario).items():
        write_scenario(scenario, output / filename)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

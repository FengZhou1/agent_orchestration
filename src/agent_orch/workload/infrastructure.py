"""Synthetic infrastructure from the parameter register, independent of SNDlib."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from agent_orch.parameters import settings_receipt
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import (
    CandidateInstance, LinkSpec, LLMConfigSpec, ModelSpec, RewardSpec,
    Scenario, ServerSpec, SimulationSpec,
)
from .agent_pool import keyed_seed


def build_infrastructure(settings, server_count=None, seed=None):
    cfg, engine = settings["infrastructure"], settings["engine"]
    missing = [name for name in ("gpu_count_probabilities", "gpu_type_probabilities",
               "seconds_per_time_unit", "non_kv_memory_gib_per_gpu") if cfg[name] is None]
    if missing:
        raise ValueError("Infrastructure choices still unset: " + ", ".join(missing))
    count = cfg["server_count"] if server_count is None else server_count
    seed = cfg["seed"] if seed is None else seed
    if count not in settings["server_counts"]:
        raise ValueError("Server count must be a registered scenario point")
    if cfg["topology"] != "mst_plus_edges" or not 2 <= cfg["average_degree"] <= count - 1:
        raise ValueError("Invalid synthetic topology or average degree")
    def probabilities(name, length):
        p = np.asarray(cfg[name], dtype=float)
        if len(p) != length or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1):
            raise ValueError(f"{name} must be a normalized nonnegative vector")
        return p
    inventory_p = probabilities("gpu_count_probabilities", 3)
    type_p = probabilities("gpu_type_probabilities", 2)
    if cfg["seconds_per_time_unit"] <= 0 or cfg["non_kv_memory_gib_per_gpu"] < 0:
        raise ValueError("Invalid time conversion or non-KV assumption")
    gpu_types = ("RTX4090", "RTX_A6000")
    gpu_specs = {"RTX4090": (24., 1008e9), "RTX_A6000": (48., 768e9)}
    servers = {}
    coordinates = {}
    for i in range(count):
        # Prefix-stable hardware: extending N does not redraw existing hosts.
        rng = np.random.default_rng(keyed_seed(seed, "server", i))
        n = f"edge-{i:03d}"
        gpus = int(rng.choice(3, p=inventory_p))
        gpu = str(rng.choice(gpu_types, p=type_p))
        servers[n] = ServerSpec(n, int(rng.integers(cfg["cpu_cores"][0], cfg["cpu_cores"][1] + 1)),
                               float(rng.uniform(*cfg["memory_gb"])), gpus,
                               gpu_specs[gpu][0] if gpus else 0., gpu if gpus else "none")
        # Unitless latent geometry: no invented meter-scale PHY claim.
        coordinates[n] = rng.random(2)
    graph = nx.Graph()
    graph.add_nodes_from(servers)
    ids = list(servers)
    for i, u in enumerate(ids):
        for v in ids[i + 1:]:
            graph.add_edge(u, v, weight=float(np.linalg.norm(coordinates[u] - coordinates[v])))
    network = nx.minimum_spanning_tree(graph)
    rng = np.random.default_rng(keyed_seed(seed, "extra-links"))
    missing_edges = sorted(set(graph.edges) - set(network.edges))
    extra = max(0, round(count * cfg["average_degree"] / 2) - network.number_of_edges())
    for index in rng.permutation(len(missing_edges))[:extra]:
        u, v = missing_edges[index]
        network.add_edge(u, v)
    links = []
    for u, v in sorted(network.edges):
        rng = np.random.default_rng(keyed_seed(seed, "link:" + u + ":" + v))
        link = LinkSpec(u, v, float(rng.uniform(*cfg["link_capacity_mbps"])),
                        float(rng.uniform(*cfg["propagation_time_units"])) *
                        cfg["seconds_per_time_unit"] * 1000, cfg["network_cost_per_mb"])
        # NetworkBackend stores directed arcs. One undirected adjacency must
        # supply both directions, with an independent budget per direction.
        links.extend((link, replace(link, source=v, target=u)))
    entries = yaml.safe_load(Path(settings["model_catalog"]).read_text(encoding="utf-8"))["models"]
    models, configs, candidates = {}, {}, {}
    for m in entries:
        if engine["max_model_len"] > m["max_position_embeddings"]:
            raise ValueError("Configured context exceeds pinned model window")
        kv_bytes = 2 * 2 * m["layers"] * m["kv_heads"] * m["head_dim"]
        model = ModelSpec(m["id"], m["parameter_count"], m["layers"], m["hidden_size"], m["weight_bytes"], kv_bytes)
        models[model.id] = model
        for gpu in gpu_types:
            memory, bandwidth = gpu_specs[gpu]
            for cards in (1, 2):
                # Physical memory is explicit GiB here; network MB remains decimal.
                kv_total_bytes = cards * (memory * engine["gpu_memory_utilization"] -
                    cfg["non_kv_memory_gib_per_gpu"]) * 2**30 - model.weight_bytes
                kv_capacity = math.floor(kv_total_bytes / kv_bytes)
                if kv_capacity < engine["max_model_len"]:
                    continue # no candidate unless at least one configured context fits
                compatible = [n for n, s in servers.items() if s.gpu_type == gpu and s.gpu_count >= cards]
                if not compatible:
                    continue
                speedup = 1. if cards == 1 else cfg["tensor_parallel_speedup_2"]
                name = f"{model.id}-{gpu}-tp{cards}"
                running = cfg["gpu_running_cost_per_s"][gpu] * cards
                configs[name] = LLMConfigSpec(name, model.id, cards, 1., memory,
                    cfg["effective_flops"][gpu] * speedup,
                    bandwidth * cfg["bandwidth_efficiency"] * speedup,
                    kv_capacity, running, running * cfg["loading_cost_to_rate_ratio_s"], gpu,
                    **{k: engine[k] for k in ("dtype", "gpu_memory_utilization", "max_model_len",
                       "max_num_batched_tokens", "max_num_seqs", "chunked_prefill", "prefix_cache")})
                for n in compatible:
                    key = f"{name}@{n}"
                    candidates[key] = CandidateInstance(key, model.id, name, n)
    if {c.model for c in candidates.values()} != set(models):
        raise ValueError("Sampled inventory cannot host every registered model tier; change declared "
                         "inventory/seed, not silently resample/filter the topology")
    scene = Scenario(f"synthetic-edge-n{count}-s{seed}", servers, tuple(links), models, configs,
        {}, candidates, {}, metadata={"parameter_source": "parameter_register",
            "parameter_receipt": settings_receipt(settings), "status": settings["status"],
            "infrastructure_settings": {**cfg, "server_count": count, "seed": seed},
            "model_provenance": entries, "engine": engine,
            "link_semantics": "bidirectional adjacency; capacity/load/cost accounted per directed arc",
            "units": {"gpu_memory_gb": "GiB", "memory_gb": "GB", "network": "Mbps/decimal MB",
                      "effective_flops": "FLOP/s, synthetic assumption", "cost": "relative unit/s"}},
        simulation=SimulationSpec(**settings["simulation"]),
        reward=RewardSpec(**{k: settings["objective"][k] for k in
            ("cost_weight", "latency_weight", "goodput_weight", "quality_weight")}))
    ScenarioLoader.validate(scene)
    return scene


def remap_pool_payload(payload, infrastructure):
    """Preserve graph/token/SLO/tool identity and total demand across network sizes."""
    import copy
    result = copy.deepcopy(payload)
    cfg = result["config"]
    servers = sorted(infrastructure.servers)
    for i, record in enumerate(result["applications"]):
        app = record["spec"]
        total = sum(app["ingress_rates"].values())
        rng = np.random.default_rng(keyed_seed(cfg["seed"], "network-ingress:" + app["id"]))
        active = [n for n in servers if rng.random() < cfg["ingress_probability"]]
        if not active:
            active = [str(rng.choice(servers))]
        weights = rng.dirichlet(np.ones(len(active)))
        app["ingress_rates"] = {n: total * float(w) for n, w in zip(active, weights)}
        record["record"]["eligible_ingress"] = servers
        record["record"]["zero_ingress"] = sorted(set(servers) - set(active))
    for tool in result["tools"]:
        rates = set(tool["service_rate"].values())
        if len(rates) != 1:
            raise ValueError("Network remapping requires the registered homogeneous single-core rates")
        tool["service_rate"] = {n: next(iter(rates)) for n in servers}
    result["provenance"]["network_remapping"] = {
        "method": "seeded sparse ingress, preserving per-application total rate",
        "infrastructure": infrastructure.id}
    return result

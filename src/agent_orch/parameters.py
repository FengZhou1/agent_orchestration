"""Executable parameter-register companion and fail-closed experiment checks.

The Markdown register retains authority over provenance and author decisions.
This module checks concrete values; an unchecked research choice is never PASS.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import math
from pathlib import Path
import re

import networkx as nx
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = ROOT / "configs/experiment_settings.yaml"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_settings(path=DEFAULT_SETTINGS):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"register", "status", "protocol", "server_counts", "application_counts",
                "tool_counts", "generator", "model_catalog", "infrastructure", "engine",
                "simulation", "variation", "objective", "ppo", "training", "pending"}
    if set(settings) != expected:
        raise ValueError(f"Settings sections differ: {sorted(set(settings) ^ expected)}")
    if settings["protocol"] != "slot_joint_v7":
        raise ValueError("Only slot_joint_v7 is the current parameter-register protocol")
    for name in ("register", "generator", "model_catalog"):
        settings[name] = str((ROOT / settings[name]).resolve())
        if not Path(settings[name]).is_file():
            raise FileNotFoundError(settings[name])
    settings["_path"] = str(path)
    if settings["status"] not in ("provisional", "frozen"):
        raise ValueError("Parameter status must be provisional or frozen")
    ratio = settings["training"]["phase_ratio"]
    if len(ratio) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratio):
        raise ValueError("Three positive stage ratios are required")
    if settings["training"]["evaluation_protocol"] != "training_trajectory_replay":
        raise ValueError("Only the saved training trajectory may be evaluated")
    if settings["variation"]["gpu_unavailable_probability"] != 0:
        raise ValueError("Main registered inventory is fixed; GPU faults need a separate protocol")
    return settings


def settings_receipt(settings):
    return {"settings_path": settings["_path"],
            "settings_sha256": file_sha256(settings["_path"]),
            "register_sha256": file_sha256(settings["register"]),
            "generator_sha256": file_sha256(settings["generator"]),
            "model_catalog_sha256": file_sha256(settings["model_catalog"])}


def require_formal_ready(settings):
    if settings["status"] != "frozen" or settings["pending"]:
        raise ValueError("Formal training blocked: parameter register contains unconfirmed "
                         "choices; inspect settings.pending and freeze them with the author first")


def make_ppo_config(settings, **overrides):
    from agent_orch.agents.config import PPOConfig
    values = {**settings["ppo"], **overrides}
    known = {f.name for f in fields(PPOConfig)}
    if set(values) - known:
        raise ValueError(f"Unknown PPO fields: {sorted(set(values) - known)}")
    for name in ("constraint_limits", "initial_lagrange_multipliers",
                 "lagrangian_learning_rates", "max_lagrange_multipliers"):
        if name in values:
            values[name] = tuple(values[name])
    if values.get("exploration_mode") != "none":
        raise ValueError("The main protocol does not use RND/ICM")
    return PPOConfig(**values)


def ppo_config_from_manifest(manifest):
    """Snapshot architecture must come from the training receipt, not today's defaults."""
    from agent_orch.agents.config import PPOConfig
    if "ppo_config" in manifest:
        values = dict(manifest["ppo_config"])
        for name in ("constraint_limits", "initial_lagrange_multipliers",
                     "lagrangian_learning_rates", "max_lagrange_multipliers"):
            values[name] = tuple(values[name])
        return PPOConfig(**values)
    # Explicit compatibility for earlier v7 diagnostic runs (never new training).
    return PPOConfig(hidden_layers=2, hidden_size=128, shared_composition_head=bool(
        manifest["training_settings"].get("shared_model_head", False)))


def audit_scenario(scenario, settings, *, infrastructure_only=False):
    """Check enforceable numeric/structural contracts and disclose remaining scope."""
    checks = []
    def check(ids, name, valid, observed):
        checks.append({"parameters": ids.split(), "check": name,
                       "status": "pass" if valid else "fail", "observed": observed})
    def within(value, bounds):
        return math.isfinite(value) and bounds[0] <= value <= bounds[1]
    infra, engine = settings["infrastructure"], settings["engine"]
    check("SC06", "server count is a registered scenario point",
          len(scenario.servers) in settings["server_counts"], len(scenario.servers))
    graph = nx.DiGraph()
    graph.add_nodes_from(scenario.servers)
    graph.add_edges_from((e.source, e.target) for e in scenario.links)
    check("NT01", "strongly connected bidirectional multi-hop network", bool(scenario.servers) and
          nx.is_strongly_connected(graph),
          len(scenario.links))
    check("HW01 HW02", "CPU and tool RAM within registered ranges",
          all(within(s.cpu_cores, infra["cpu_cores"]) and within(s.memory_gb, infra["memory_gb"])
              for s in scenario.servers.values()),
          {n: [s.cpu_cores, s.memory_gb] for n, s in scenario.servers.items()})
    check("HW04 HW05 HW06", "GPU inventory uses registered types and physical memory",
          all(s.gpu_count in (0, 1, 2) and (s.gpu_count == 0 or
              s.gpu_type in ("RTX4090", "RTX_A6000") and s.gpu_memory_gb ==
              {"RTX4090": 24.0, "RTX_A6000": 48.0}[s.gpu_type])
              for s in scenario.servers.values()),
          {n: [s.gpu_type, s.gpu_count, s.gpu_memory_gb] for n, s in scenario.servers.items()})
    check("NT03", "effective link rate range",
          all(within(e.capacity_mbps, infra["link_capacity_mbps"]) for e in scenario.links),
          [e.capacity_mbps for e in scenario.links])
    catalogue = yaml.safe_load(Path(settings["model_catalog"]).read_text(encoding="utf-8"))["models"]
    expected_models = {m["id"]: m for m in catalogue}
    check("LM01 LM02 LM04 LM05", "pinned model structure, weight and KV bytes",
          set(scenario.models) == set(expected_models) and all(
              m.parameter_count == expected_models[k]["parameter_count"] and
              m.weight_bytes == expected_models[k]["weight_bytes"] and
              m.layers == expected_models[k]["layers"] and
              m.hidden_size == expected_models[k]["hidden_size"] and
              m.kv_bytes_per_token == 2 * 2 * m.layers * expected_models[k]["kv_heads"] *
              expected_models[k]["head_dim"] for k, m in scenario.models.items() if k in expected_models),
          list(scenario.models))
    config_fields = ("dtype", "gpu_memory_utilization", "max_model_len",
                     "max_num_batched_tokens", "max_num_seqs", "chunked_prefill", "prefix_cache")
    check("LM03 LM08 LM09 LM10 LM13 LM14 LM15", "engine parameters",
          bool(scenario.llm_configs) and all(
              all(getattr(c, field) == engine[field] for field in config_fields)
              for c in scenario.llm_configs.values()),
          {k: {f: getattr(c, f) for f in config_fields} for k, c in scenario.llm_configs.items()})
    check("LM19", "all model tiers have compatible candidates",
          {c.model for c in scenario.candidates.values()} == set(scenario.models),
          len(scenario.candidates))
    derive_ok = True
    for c in scenario.llm_configs.values():
        if c.gpu_type not in infra["effective_flops"] or c.model not in scenario.models:
            derive_ok = False
            continue
        memory, bandwidth = {"RTX4090": (24., 1008e9), "RTX_A6000": (48., 768e9)}[c.gpu_type]
        m = scenario.models[c.model]
        scale = 1. if c.gpu_count == 1 else infra["tensor_parallel_speedup_2"]
        nonkv = infra["non_kv_memory_gib_per_gpu"]
        expected_kv = math.floor((c.gpu_count * (memory * engine["gpu_memory_utilization"] - nonkv) *
                                 2**30 - m.weight_bytes) / m.kv_bytes_per_token) if nonkv is not None else -1
        running = infra["gpu_running_cost_per_s"][c.gpu_type] * c.gpu_count
        derive_ok &= (c.gpu_count in (1, 2) and c.gpu_share == 1 and
                      c.reserved_memory_gb_per_gpu == memory and c.kv_token_capacity == expected_kv and
                      c.kv_token_capacity >= c.max_model_len and
                      c.effective_flops == infra["effective_flops"][c.gpu_type] * scale and
                      c.effective_bandwidth_bytes_s == bandwidth * infra["bandwidth_efficiency"] * scale and
                      c.running_cost_per_slot == running and
                      c.load_cost == running * infra["loading_cost_to_rate_ratio_s"])
    check("HW08 HW09 HW10 LM16 LM17 LM18 CO01 CO04", "GPU candidates derive from declared assumptions",
          derive_ok, "GiB capacity; pinned BF16 weights; synthetic efficiencies, not measured profiles")
    check("LM11 NT10 NT11 NT12", "simulation uses explicit fixed conventions",
          all(getattr(scenario.simulation, k) == v for k, v in settings["simulation"].items()),
          {k: getattr(scenario.simulation, k) for k in settings["simulation"]})
    if not infrastructure_only:
        check("SC02", "application count is a registered nested-subset point",
              len(scenario.applications) in settings["application_counts"], len(scenario.applications))
        check("SC04", "catalogue contains exactly positive-probability used tools",
              set(scenario.tools) == {a.nodes[n].tool for a in scenario.applications.values()
                                     for f in a.pattern_flows if f.probability > 0
                                     for n in f.nodes if a.nodes[n].type.value == "tool"},
              len(scenario.tools))
        check("SV01 SV02 SV03 SV04 SV06", "tool CPU, RAM, homogeneous rate and variability",
              all(t.cpu_cores == 1 and within(t.memory_gb, (.5, 3)) and t.arrival_scv == 1 and
                  set(t.service_rate) == set(scenario.servers) and
                  len(set(t.service_rate.values())) == 1 and
                  all(within(v, (2, 8)) for v in t.service_rate.values()) for t in scenario.tools.values()),
              len(scenario.tools))
        generator = yaml.safe_load(Path(settings["generator"]).read_text(encoding="utf-8"))
        check("CO02 CO04", "tool prices use registered per-core/per-memory coefficients",
              all(math.isclose(t.running_cost_per_slot, generator["tool_cpu_cost_per_s"] +
                  t.memory_gb * generator["tool_memory_cost_per_gb_s"]) and
                  math.isclose(t.start_cost, t.running_cost_per_slot * generator["tool_start_cost_ratio_s"])
                  for t in scenario.tools.values()), len(scenario.tools))
        check("QU01 QU02", "quality is a declared bounded synthetic proxy",
              all(all(within(v, generator["quality_range"]) for v in a.quality.values())
                  for a in scenario.applications.values()), generator["quality_range"])
        graph_ok = True
        lengths_ok = True
        for a in scenario.applications.values():
            g = nx.DiGraph()
            g.add_nodes_from(a.nodes)
            for f in a.pattern_flows:
                g.add_edges_from(f.edges)
            nl = sum(n.type.value == "llm" for n in a.nodes.values())
            nt = len(a.nodes) - nl
            dag = nx.is_directed_acyclic_graph(g)
            depth = nx.dag_longest_path_length(g) + 1 if dag else math.inf
            graph_ok &= dag and ((nl == 1 and nt == 0 and depth == 1) or
                                 (2 <= nl <= 15 and 1 <= nt <= 30 and 2 <= depth <= 30))
            graph_ok &= set(a.nodes) == {n for f in a.pattern_flows if f.probability > 0 for n in f.nodes}
            for n in a.nodes.values():
                if n.type.value == "llm":
                    lengths_ok &= all(256 <= n.prompt_tokens[m] <= 16384 and
                                      16 <= n.output_tokens[m] <= 2048 and
                                      n.prompt_tokens[m] + n.output_tokens[m] <= 32768
                                      for m in scenario.models)
        check("AG03 AG04 AG06 AG08", "full DAG/node bounds and positive pattern coverage", graph_ok,
              len(scenario.applications))
        check("TK05 TK09 TK14 TK15", "node representative token ranges", lengths_ok,
              "Representative checks do not replace raw per-pattern budget verification")
        check("SC07 SC09 AR05", "sparse ingress and nonnegative conserved application demand",
              all(set(a.ingress_rates) <= set(scenario.servers) and
                  sum(a.ingress_rates.values()) > 0 and
                  all(math.isfinite(r) and r >= 0 for r in a.ingress_rates.values())
                  for a in scenario.applications.values()),
              {a.id: sum(a.ingress_rates.values()) for a in scenario.applications.values()})
    covered = {k for c in checks for k in c["parameters"]}
    rows = []
    for line in Path(settings["register"]).read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*((?:SC|AG|TK|SV|HW|LM|NT|AR|SL|QU|CO|OB|PR|RL)\d\d)\s*\|", line)
        if match:
            columns = [c.strip() for c in line.split("|")[1:-1]]
            key = match[1]
            evidence = [c for c in checks if key in c["parameters"]]
            rows.append({"id": key, "name": columns[1], "register_status": columns[-1],
                         "coverage": ("checked" if key in covered else "manual_or_runtime"),
                         "checks": [c["check"] for c in evidence]})
    return {"scope": "base-scenario numeric and structural audit; not a real-engine/profile certificate",
            "consistent": all(c["status"] == "pass" for c in checks),
            "formal_ready": settings["status"] == "frozen" and not settings["pending"],
            "checks": checks, "register_rows": rows, "pending": settings["pending"],
            "receipt": settings_receipt(settings)}


def require_consistent(report):
    failures = [c["check"] for c in report["checks"] if c["status"] == "fail"]
    if failures:
        raise ValueError("Scenario conflicts with parameter register: " + "; ".join(failures))

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from .models import (
    ApplicationSpec,
    CandidateInstance,
    LinkSpec,
    LLMConfigSpec,
    ModelSpec,
    NodeType,
    PatternFlow,
    RewardSpec,
    Scenario,
    ServerSpec,
    SimulationSpec,
    SLOSpec,
    SLOType,
    ToolSpec,
    WorkflowNode,
)


def _edge_data(raw: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for key, value in raw.items():
        model, edge = key.split(":", 1)
        source, target = edge.split("->", 1)
        result[(model, source, target)] = float(value)
    return result


class ScenarioLoader:
    @staticmethod
    def load(path: str | Path) -> Scenario:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        servers = {item["id"]: ServerSpec(**item) for item in raw["servers"]}
        links = tuple(LinkSpec(**item) for item in raw["links"])
        models = {item["id"]: ModelSpec(**item) for item in raw["models"]}
        configs = {}
        for item in raw["llm_configs"]:
            values = dict(item)
            configs[values["id"]] = LLMConfigSpec(**values)
        tools = {item["id"]: ToolSpec(**item) for item in raw["tools"]}
        candidates = {
            item["id"]: CandidateInstance(**item) for item in raw["candidates"]
        }

        applications: dict[str, ApplicationSpec] = {}
        for item in raw["applications"]:
            nodes = {
                node["id"]: WorkflowNode(
                    id=node["id"],
                    type=NodeType(node["type"]),
                    tool=node.get("tool"),
                    prompt_tokens={k: float(v) for k, v in node.get("prompt_tokens", {}).items()},
                    output_tokens={k: float(v) for k, v in node.get("output_tokens", {}).items()},
                    stage_deadline_s=node.get("stage_deadline_s"),
                )
                for node in item["nodes"]
            }
            flows = tuple(
                PatternFlow(
                    id=flow["id"],
                    probability=float(flow["probability"]),
                    final_node=flow["final_node"],
                    chains=tuple(tuple(chain) for chain in flow["chains"]),
                )
                for flow in item["pattern_flows"]
            )
            slo_raw = item["slo"]
            slo = SLOSpec(
                type=SLOType(slo_raw["type"]),
                ttft_s=slo_raw.get("ttft_s"),
                tbt_s=slo_raw.get("tbt_s"),
                deadline_s=slo_raw.get("deadline_s"),
            )
            applications[item["id"]] = ApplicationSpec(
                id=item["id"],
                ingress_rates={k: float(v) for k, v in item["ingress_rates"].items()},
                nodes=nodes,
                pattern_flows=flows,
                slo=slo,
                quality={k: float(v) for k, v in item["quality"].items()},
                entry_data_mb={k: float(v) for k, v in item["entry_data_mb"].items()},
                exit_data_mb={k: float(v) for k, v in item["exit_data_mb"].items()},
                edge_data_mb=_edge_data(item.get("edge_data_mb", {})),
                family=str(item.get("family", "unspecified")),
                template_id=str(item.get("template_id", "unspecified")),
                length_class=str(item.get("length_class", "unspecified")),
            )

        simulation = SimulationSpec(**raw.get("simulation", {}))
        reward = RewardSpec(**raw.get("reward", {}))
        scenario = Scenario(
            id=raw["id"],
            servers=servers,
            links=links,
            models=models,
            llm_configs=configs,
            tools=tools,
            candidates=candidates,
            applications=applications,
            metadata=dict(raw.get("metadata", {})),
            simulation=simulation,
            reward=reward,
        )
        ScenarioLoader.validate(scenario)
        return scenario

    @staticmethod
    def validate(scenario: Scenario) -> None:
        if scenario.simulation.slot_seconds <= 0.0:
            raise ValueError("slot_seconds must be positive")
        if scenario.simulation.deployment_period_slots <= 0:
            raise ValueError("deployment_period_slots must be positive")
        for link in scenario.links:
            if link.source not in scenario.servers or link.target not in scenario.servers:
                raise ValueError(f"Link {link.source}->{link.target} references an unknown server")
            if link.capacity_mbps <= 0.0 or link.propagation_ms < 0.0:
                raise ValueError(f"Link {link.source}->{link.target} has invalid parameters")
        reward_weights = (
            scenario.reward.cost_weight,
            scenario.reward.latency_weight,
            scenario.reward.goodput_weight,
            scenario.reward.quality_weight,
        )
        if any(weight < 0.0 for weight in reward_weights):
            raise ValueError("Reward weights must be non-negative")
        if abs(sum(reward_weights) - 1.0) > 1e-9:
            raise ValueError("Reward weights must sum to one")
        for candidate in scenario.candidates.values():
            if candidate.model not in scenario.models:
                raise ValueError(f"Unknown model in candidate {candidate.id}")
            if candidate.config not in scenario.llm_configs:
                raise ValueError(f"Unknown config in candidate {candidate.id}")
            if candidate.server not in scenario.servers:
                raise ValueError(f"Unknown server in candidate {candidate.id}")
            if scenario.llm_configs[candidate.config].model != candidate.model:
                raise ValueError(f"Candidate {candidate.id} uses a mismatched config")
            config = scenario.llm_configs[candidate.config]
            server = scenario.servers[candidate.server]
            if config.gpu_count > server.gpu_count:
                raise ValueError(f"Candidate {candidate.id} requests more GPUs than its server")
            if config.reserved_memory_gb_per_gpu > server.gpu_memory_gb:
                raise ValueError(f"Candidate {candidate.id} exceeds per-GPU memory")
            if config.gpu_type != "generic" and server.gpu_type != config.gpu_type:
                raise ValueError(f"Candidate {candidate.id} uses an incompatible GPU type")

        for config in scenario.llm_configs.values():
            if not 0.0 < config.gpu_memory_utilization <= 1.0:
                raise ValueError(f"Invalid GPU memory utilization in {config.id}")
            if config.max_model_len <= 0 or config.max_num_batched_tokens <= 0:
                raise ValueError(f"Invalid vLLM token limits in {config.id}")
            if config.max_num_seqs <= 0:
                raise ValueError(f"Invalid concurrency in {config.id}")

        for tool in scenario.tools.values():
            if tool.arrival_scv < 0.0:
                raise ValueError(f"Invalid variability parameter in {tool.id}")

        for tool in scenario.tools.values():
            missing_servers = set(scenario.servers) - set(tool.service_rate)
            if missing_servers:
                raise ValueError(
                    f"Stateless service {tool.id} lacks service rates for {sorted(missing_servers)}"
                )
            if any(rate <= 0.0 for rate in tool.service_rate.values()):
                raise ValueError(f"Stateless service {tool.id} has a non-positive service rate")

        for app in scenario.applications.values():
            if not app.ingress_rates:
                raise ValueError(f"Application {app.id} has no ingress")
            if set(app.ingress_rates) - set(scenario.servers):
                raise ValueError(f"Application {app.id} references an unknown ingress")
            if any(rate < 0.0 for rate in app.ingress_rates.values()):
                raise ValueError(f"Application {app.id} has a negative arrival rate")
            if set(app.quality) != set(scenario.models):
                raise ValueError(f"Application {app.id} lacks model quality values")
            if set(app.entry_data_mb) != set(scenario.models) or set(app.exit_data_mb) != set(scenario.models):
                raise ValueError(f"Application {app.id} lacks model communication data")
            total_probability = sum(flow.probability for flow in app.pattern_flows)
            if abs(total_probability - 1.0) > 1e-9:
                raise ValueError(f"Pattern-flow probabilities of {app.id} do not sum to one")
            for flow in app.pattern_flows:
                if flow.final_node not in app.nodes:
                    raise ValueError(f"Unknown final node {flow.final_node}")
                if app.nodes[flow.final_node].type is not NodeType.LLM:
                    raise ValueError("Every pattern flow must end at an LLM node")
                for chain in flow.chains:
                    if not chain or chain[-1] != flow.final_node:
                        raise ValueError(f"Every chain of {flow.id} must end at its final LLM")
                    if any(node not in app.nodes for node in chain):
                        raise ValueError(f"Unknown node in a chain of {flow.id}")
                graph = nx.DiGraph()
                graph.add_nodes_from(flow.nodes)
                graph.add_edges_from(flow.edges)
                if not nx.is_directed_acyclic_graph(graph):
                    raise ValueError(f"Pattern flow {flow.id} is not acyclic")
                if any(not nx.has_path(graph, source, flow.final_node) for source in flow.sources):
                    raise ValueError(f"A source of {flow.id} cannot reach its final LLM")
            for node in app.nodes.values():
                if node.type is NodeType.LLM:
                    missing_prompt = set(scenario.models) - set(node.prompt_tokens)
                    missing_output = set(scenario.models) - set(node.output_tokens)
                    if missing_prompt or missing_output:
                        raise ValueError(
                            f"LLM node {app.id}:{node.id} lacks token features for all models"
                        )
                    if any(value <= 0.0 for value in node.prompt_tokens.values()):
                        raise ValueError(f"LLM node {app.id}:{node.id} has invalid input tokens")
                    if any(value <= 0.0 for value in node.output_tokens.values()):
                        raise ValueError(f"LLM node {app.id}:{node.id} has invalid output tokens")
                elif node.tool not in scenario.tools:
                    raise ValueError(f"Node {app.id}:{node.id} references an unknown service")

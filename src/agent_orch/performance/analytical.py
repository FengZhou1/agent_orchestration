from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from agent_orch.schema.models import (
    DeploymentDecision,
    LLMClassPerformance,
    NodeType,
    RoutingDecision,
    Scenario,
)

from .llm import ServiceDemand, service_demand
from .network import Edge, NetworkBackend
from .queueing import llm_waiting_time, tool_response_time

if TYPE_CHECKING:
    from agent_orch.backends import ProfileBackend


LLMClass = tuple[str, str, str]
ToolPool = tuple[str, str]


@dataclass
class AnalyticalResult:
    llm_performance: dict[LLMClass, LLMClassPerformance]
    llm_utilization: dict[str, float]
    llm_kv_stable: dict[str, bool]
    tool_delay: dict[ToolPool, float]
    tool_utilization: dict[ToolPool, float]
    link_loads_mbit: dict[Edge, float]
    node_server_distribution: dict[tuple[str, str, str, str, str], dict[str, float]]
    violations: list[str] = field(default_factory=list)


class AnalyticalBackend:
    def __init__(
        self, scenario: Scenario, profile_backend: "ProfileBackend | None" = None
    ):
        self.scenario = scenario
        self.profile_backend = profile_backend
        self.network = NetworkBackend(scenario.links, scenario.simulation.slot_seconds)

    def evaluate(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> AnalyticalResult:
        llm_arrivals = self.llm_arrivals(routing, arrival_rates)
        llm_perf, llm_util, kv_stable, violations = self._llm_performance(
            deployment, llm_arrivals
        )
        distributions = self.node_distributions(deployment, routing)
        tool_arrivals = self.tool_arrivals(routing, distributions, arrival_rates)
        tool_delay, tool_util, tool_violations = self._tool_performance(
            deployment, tool_arrivals
        )
        violations.extend(tool_violations)
        link_loads = self.link_loads(routing, distributions, arrival_rates)
        for edge, utilization in self.network.utilization(link_loads).items():
            if utilization >= 1.0:
                violations.append(f"link_overload:{edge}")
        return AnalyticalResult(
            llm_perf,
            llm_util,
            kv_stable,
            tool_delay,
            tool_util,
            link_loads,
            distributions,
            violations,
        )

    def llm_arrivals(
        self,
        routing: RoutingDecision,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[LLMClass, float]:
        arrivals: dict[LLMClass, float] = defaultdict(float)
        for app in self.scenario.applications.values():
            for node in app.nodes.values():
                if node.type is not NodeType.LLM:
                    continue
                visit = app.visit_probability(node.id)
                for ingress, base_rate in app.ingress_rates.items():
                    rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                    for candidate_id in self.scenario.candidates:
                        share = routing.llm_share.get(
                            (app.id, ingress, node.id, candidate_id), 0.0
                        )
                        arrivals[(app.id, node.id, candidate_id)] += rate * visit * share
        return dict(arrivals)

    def _llm_performance(
        self,
        deployment: DeploymentDecision,
        arrivals: dict[LLMClass, float],
    ) -> tuple[
        dict[LLMClass, LLMClassPerformance],
        dict[str, float],
        dict[str, bool],
        list[str],
    ]:
        if self.profile_backend is not None:
            return self._profile_llm_performance(deployment, arrivals)
        demands: dict[LLMClass, ServiceDemand] = {}
        rates_by_instance: dict[str, float] = defaultdict(float)
        for key, rate in arrivals.items():
            app_id, node_id, candidate_id = key
            if rate <= 0.0:
                continue
            candidate = self.scenario.candidates[candidate_id]
            if deployment.llm_active.get(candidate_id, 0) != 1:
                continue
            model = self.scenario.models[candidate.model]
            config = self.scenario.llm_configs[candidate.config]
            node = self.scenario.applications[app_id].nodes[node_id]
            demand = service_demand(
                model,
                config,
                node.prompt_tokens[candidate.model],
                node.output_tokens[candidate.model],
                self.scenario.simulation.prefill_chunk_tokens,
            )
            demands[key] = demand
            rates_by_instance[candidate_id] += rate

        perf: dict[LLMClass, LLMClassPerformance] = {}
        utilization: dict[str, float] = {}
        kv_stable: dict[str, bool] = {}
        violations: list[str] = []
        for candidate_id, total_rate in rates_by_instance.items():
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            keys = [key for key in demands if key[2] == candidate_id]
            mean = sum(arrivals[key] * demands[key].service_s for key in keys) / total_rate
            second = sum(
                arrivals[key] * demands[key].service_s**2 for key in keys
            ) / total_rate
            wait, rho, overloaded = llm_waiting_time(
                total_rate,
                mean,
                second,
                config.effective_concurrency,
                self.scenario.simulation.overload_delay_s,
            )
            utilization[candidate_id] = rho
            if overloaded:
                violations.append(f"llm_queue_overload:{candidate_id}")

            weighted_iterations = sum(
                arrivals[key]
                * (
                    math.ceil(
                        self.scenario.applications[key[0]].nodes[key[1]].prompt_tokens[
                            candidate.model
                        ]
                        / self.scenario.simulation.prefill_chunk_tokens
                    )
                    + max(
                        0,
                        round(
                            self.scenario.applications[key[0]].nodes[key[1]].output_tokens[
                                candidate.model
                            ]
                        )
                        - 1,
                    )
                )
                for key in keys
            )
            mean_iteration = (
                sum(arrivals[key] * demands[key].service_s for key in keys)
                / weighted_iterations
                if weighted_iterations > 0
                else 0.0
            )
            max_context = max(
                self.scenario.applications[key[0]].nodes[key[1]].prompt_tokens[
                    candidate.model
                ]
                + self.scenario.applications[key[0]].nodes[key[1]].output_tokens[
                    candidate.model
                ]
                for key in keys
            )
            delta = max_context / config.kv_token_capacity
            kv_demand_rate = sum(
                arrivals[key] * demands[key].kv_work_tokens for key in keys
            )
            kv_supply_rate = (
                (1.0 - delta) * config.kv_token_capacity / mean_iteration
                if mean_iteration > 0 and delta < 1.0
                else 0.0
            )
            stable = delta < 1.0 and kv_demand_rate < kv_supply_rate
            kv_stable[candidate_id] = stable
            if not stable:
                violations.append(f"llm_kv_overload:{candidate_id}")

            for key in keys:
                demand = demands[key]
                app_id, node_id, _ = key
                output = max(
                    1,
                    round(
                        self.scenario.applications[app_id].nodes[node_id].output_tokens[
                            candidate.model
                        ]
                    ),
                )
                tbt = demand.decode_s / (output - 1) if output > 1 else 0.0
                perf[key] = LLMClassPerformance(
                    service_s=demand.service_s,
                    prefill_s=demand.prefill_s,
                    decode_s=demand.decode_s,
                    ttft_s=wait + demand.prefill_s,
                    tbt_s=tbt,
                    response_s=wait + demand.service_s,
                )

        for key, rate in arrivals.items():
            if rate > 0.0 and key not in perf:
                violations.append(f"llm_unserved:{key[2]}")
        return perf, utilization, kv_stable, violations

    def _profile_llm_performance(
        self,
        deployment: DeploymentDecision,
        arrivals: dict[LLMClass, float],
    ) -> tuple[
        dict[LLMClass, LLMClassPerformance],
        dict[str, float],
        dict[str, bool],
        list[str],
    ]:
        assert self.profile_backend is not None
        perf: dict[LLMClass, LLMClassPerformance] = {}
        utilization: dict[str, float] = {}
        kv_stable: dict[str, bool] = {}
        violations: list[str] = []
        by_instance: dict[str, list[LLMClass]] = defaultdict(list)
        for key, rate in arrivals.items():
            if rate > 0.0:
                by_instance[key[2]].append(key)
        for candidate_id, keys in by_instance.items():
            if deployment.llm_active.get(candidate_id, 0) != 1:
                violations.append(f"llm_unserved:{candidate_id}")
                continue
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            total_rate = sum(arrivals[key] for key in keys)
            long_rate = sum(
                arrivals[key]
                for key in keys
                if (
                    self.scenario.applications[key[0]].nodes[key[1]].prompt_tokens[
                        candidate.model
                    ]
                    + self.scenario.applications[key[0]].nodes[key[1]].output_tokens[
                        candidate.model
                    ]
                    >= 1024
                )
            )
            long_fraction = long_rate / total_rate if total_rate > 0.0 else 0.0
            capacities = []
            kv_values = []
            for key in keys:
                app_id, node_id, _ = key
                node = self.scenario.applications[app_id].nodes[node_id]
                estimate = self.profile_backend.estimate(
                    candidate.model,
                    candidate.config,
                    node.prompt_tokens[candidate.model],
                    node.output_tokens[candidate.model],
                    total_rate,
                    long_fraction,
                )
                capacities.append(estimate.stable_capacity_rps)
                kv_values.append(estimate.kv_tokens)
                perf[key] = LLMClassPerformance(
                    service_s=estimate.response_s,
                    prefill_s=estimate.ttft_s,
                    decode_s=max(0.0, estimate.response_s - estimate.ttft_s),
                    ttft_s=estimate.ttft_s,
                    tbt_s=estimate.tbt_s,
                    response_s=estimate.response_s,
                )
            capacity = min(capacities) if capacities else 0.0
            utilization[candidate_id] = total_rate / capacity if capacity > 0.0 else 1.0e6
            stable = total_rate < capacity and max(kv_values, default=0.0) < config.kv_token_capacity
            kv_stable[candidate_id] = stable
            if total_rate >= capacity:
                violations.append(f"llm_queue_overload:{candidate_id}")
            if max(kv_values, default=0.0) >= config.kv_token_capacity:
                violations.append(f"llm_kv_overload:{candidate_id}")
        return perf, utilization, kv_stable, violations

    def node_distributions(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
    ) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
        result: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        for node_id in flow.nodes:
                            node = app.nodes[node_id]
                            key = (app.id, ingress, model, flow.id, node_id)
                            if node.type is NodeType.LLM:
                                dist: dict[str, float] = defaultdict(float)
                                for candidate_id, candidate in self.scenario.candidates.items():
                                    if candidate.model != model:
                                        continue
                                    absolute = routing.llm_share.get(
                                        (app.id, ingress, node_id, candidate_id), 0.0
                                    )
                                    if absolute > 0.0:
                                        dist[candidate.server] += absolute / model_share
                                result[key] = _normalize(dist)

                        unresolved = [
                            edge
                            for edge in flow.edges
                            if app.nodes[edge[1]].type is NodeType.TOOL
                        ]
                        for _ in range(len(flow.nodes)):
                            next_unresolved = []
                            for source, target in unresolved:
                                source_key = (app.id, ingress, model, flow.id, source)
                                if source_key not in result:
                                    next_unresolved.append((source, target))
                                    continue
                                target_key = (app.id, ingress, model, flow.id, target)
                                target_dist: dict[str, float] = defaultdict(float)
                                for u, source_probability in result[source_key].items():
                                    for v in self.scenario.servers:
                                        probability = routing.tool_route.get(
                                            (app.id, source, target, u, v), 0.0
                                        )
                                        target_dist[v] += source_probability * probability
                                if target_key in result:
                                    for server, probability in result[target_key].items():
                                        target_dist[server] += probability
                                result[target_key] = _normalize(target_dist)
                            unresolved = next_unresolved
                            if not unresolved:
                                break
        return result

    def tool_arrivals(
        self,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[ToolPool, float]:
        arrivals: dict[ToolPool, float] = defaultdict(float)
        for app in self.scenario.applications.values():
            for ingress, base_rate in app.ingress_rates.items():
                rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        flow_rate = rate * model_share * flow.probability
                        for node_id in flow.nodes:
                            node = app.nodes[node_id]
                            if node.type is not NodeType.TOOL or node.tool is None:
                                continue
                            key = (app.id, ingress, model, flow.id, node_id)
                            for server, probability in distributions.get(key, {}).items():
                                arrivals[(node.tool, server)] += flow_rate * probability
        return dict(arrivals)

    def _tool_performance(
        self,
        deployment: DeploymentDecision,
        arrivals: dict[ToolPool, float],
    ) -> tuple[dict[ToolPool, float], dict[ToolPool, float], list[str]]:
        delays: dict[ToolPool, float] = {}
        utilizations: dict[ToolPool, float] = {}
        violations: list[str] = []
        for pool, arrival_rate in arrivals.items():
            tool_id, server = pool
            spec = self.scenario.tools[tool_id]
            replicas = deployment.tool_replicas.get(pool, 0)
            wait, process, rho, overloaded = tool_response_time(
                arrival_rate,
                spec.service_rate[server],
                replicas,
                spec.arrival_scv,
                self.scenario.simulation.overload_delay_s,
            )
            delays[pool] = wait + process
            utilizations[pool] = rho
            if overloaded:
                violations.append(f"tool_overload:{tool_id}@{server}")
        return delays, utilizations, violations

    def link_loads(
        self,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[Edge, float]:
        loads: dict[Edge, float] = {}
        for app in self.scenario.applications.values():
            for ingress, base_rate in app.ingress_rates.items():
                arrival_rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        rate = arrival_rate * model_share * flow.probability
                        for source in flow.sources:
                            key = (app.id, ingress, model, flow.id, source)
                            for server, probability in distributions.get(key, {}).items():
                                self.network.add_traffic(
                                    loads,
                                    ingress,
                                    server,
                                    rate * probability,
                                    app.entry_data_mb[model],
                                )
                        final_key = (app.id, ingress, model, flow.id, flow.final_node)
                        for server, probability in distributions.get(final_key, {}).items():
                            self.network.add_traffic(
                                loads,
                                server,
                                ingress,
                                rate * probability,
                                app.exit_data_mb[model],
                            )
                        for source, target in flow.edges:
                            data_mb = app.edge_data_mb.get((model, source, target), 0.0)
                            for u, v, probability in self.edge_pair_distribution(
                                app.id,
                                ingress,
                                model,
                                flow.id,
                                source,
                                target,
                                routing,
                                distributions,
                            ):
                                self.network.add_traffic(
                                    loads, u, v, rate * probability, data_mb
                                )
        return loads

    def edge_pair_distribution(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        source: str,
        target: str,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
    ) -> list[tuple[str, str, float]]:
        app = self.scenario.applications[app_id]
        source_dist = distributions.get((app_id, ingress, model, flow_id, source), {})
        target_dist = distributions.get((app_id, ingress, model, flow_id, target), {})
        pairs: list[tuple[str, str, float]] = []
        if app.nodes[target].type is NodeType.TOOL:
            for u, source_probability in source_dist.items():
                for v in self.scenario.servers:
                    p = routing.tool_route.get((app_id, source, target, u, v), 0.0)
                    if p > 0.0:
                        pairs.append((u, v, source_probability * p))
        else:
            for u, source_probability in source_dist.items():
                for v, target_probability in target_dist.items():
                    pairs.append((u, v, source_probability * target_probability))
        total = sum(item[2] for item in pairs)
        return [(u, v, p / total) for u, v, p in pairs] if total > 0.0 else []


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in values.values() if value > 0.0)
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in values.items() if value > 0.0}

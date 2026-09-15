from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

from agent_orch.schema.models import (
    DeploymentDecision,
    LLMClassPerformance,
    LLMConfigSpec,
    ModelSpec,
    NodeType,
    RoutingDecision,
    Scenario,
)

from .llm import (
    mean_decode_context,
    mean_service_time,
    residency_capacity,
    service_curve,
    steady_active_concurrency,
    throughput_capacity,
)
from .network import Edge, NetworkBackend
from .queueing import tool_response_time


LLMClass = tuple[str, str, str]
ToolPool = tuple[str, str]

@dataclass
class AnalyticalResult:
    llm_performance: dict[LLMClass, LLMClassPerformance]
    llm_utilization: dict[str, float]
    llm_kv_stable: dict[str, bool]
    tool_delay: dict[ToolPool, float]
    tool_utilization: dict[ToolPool, float]
    link_load_mbps: dict[Edge, float]
    node_server_distribution: dict[tuple[str, str, str, str, str], dict[str, float]]
    violations: list[str] = field(default_factory=list)
    llm_instance_performance: dict[str, "LLMInstancePerformance"] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LLMInstancePerformance:
    arrival_rate_rps: float
    active_concurrency: float
    active_kv_tokens: float
    resident_capacity: int
    kv_slack: float
    mean_service_s: float
    throughput_capacity_rps: float
    capacity_concurrency: float
    utilization: float
    stable: bool
    fixed_point_residual: float


def evaluate_llm_instance(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    arrival_rate_rps: float,
    chunk_tokens: int,
) -> tuple[LLMInstancePerformance, list[LLMClassPerformance]]:
    """Steady-state performance of one LLM instance under mixed call classes.

    ``classes`` holds the ``(prompt tokens, output tokens)`` of every call class
    routed to the instance and ``weights`` its share of the call rate.  The
    instance runs the fixed deployment configuration in ``config``; only its
    class composition and total offer rate vary.
    """
    peer_context = mean_decode_context(classes, weights)
    curves = [
        service_curve(
            model, config, prompt, output, chunk_tokens, peer_decode_context=peer_context
        )
        for prompt, output in classes
    ]
    residency = residency_capacity(
        classes,
        weights,
        config.kv_token_capacity,
        config.max_num_seqs,
        chunk_tokens,
    )
    # The steady concurrency follows from Little's law on the same service
    # curve; the capacity is the best sustained throughput of that curve below
    # the KV/sequence residency limit.
    batch, converged, residual = steady_active_concurrency(
        curves, weights, arrival_rate_rps, residency.capacity
    )
    capacity, capacity_concurrency = throughput_capacity(
        curves, weights, residency.capacity
    )
    utilization = arrival_rate_rps / capacity if capacity > 0.0 else math.inf
    stable = (
        converged
        and residency.capacity >= 1
        and batch < residency.capacity
        and residency.kv_slack > 0.0
        and utilization < 1.0
    )
    instance = LLMInstancePerformance(
        arrival_rate_rps=arrival_rate_rps,
        active_concurrency=batch,
        active_kv_tokens=residency.active_kv_tokens,
        resident_capacity=residency.capacity,
        kv_slack=residency.kv_slack,
        mean_service_s=mean_service_time(curves, weights, max(1.0, batch)),
        throughput_capacity_rps=capacity,
        capacity_concurrency=capacity_concurrency,
        utilization=utilization,
        stable=stable,
        fixed_point_residual=residual,
    )
    per_class: list[LLMClassPerformance] = []
    for (_, output), curve in zip(classes, curves):
        prefill = curve.prefill_at(max(1.0, batch))
        decode = curve.decode_at(max(1.0, batch))
        count = max(1, round(output))
        per_class.append(
            LLMClassPerformance(
                service_s=prefill + decode,
                prefill_s=prefill,
                decode_s=decode,
                ttft_s=prefill,
                tbt_s=decode / (count - 1) if count > 1 else 0.0,
                response_s=prefill + decode,
            )
        )
    return instance, per_class


class AnalyticalBackend:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.network = NetworkBackend(
            scenario.links,
            scenario.simulation.slot_seconds,
            scenario.simulation.overload_delay_s,
        )

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
        instance_metrics = getattr(self, "_last_llm_instance_performance", {})
        return AnalyticalResult(
            llm_perf,
            llm_util,
            kv_stable,
            tool_delay,
            tool_util,
            link_loads,
            distributions,
            violations,
            instance_metrics,
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
        classes_by_instance: dict[str, list[LLMClass]] = defaultdict(list)
        for key, rate in arrivals.items():
            if rate <= 0.0:
                continue
            if deployment.llm_active.get(key[2], 0) != 1:
                continue
            classes_by_instance[key[2]].append(key)

        chunk_tokens = self.scenario.simulation.prefill_chunk_tokens
        perf: dict[LLMClass, LLMClassPerformance] = {}
        utilization: dict[str, float] = {}
        kv_stable: dict[str, bool] = {}
        violations: list[str] = []
        instance_metrics: dict[str, LLMInstancePerformance] = {}
        for candidate_id, keys in classes_by_instance.items():
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            model = self.scenario.models[candidate.model]
            rates = [arrivals[key] for key in keys]
            call_classes = [
                (
                    self.scenario.applications[key[0]]
                    .nodes[key[1]]
                    .prompt_tokens[candidate.model],
                    self.scenario.applications[key[0]]
                    .nodes[key[1]]
                    .output_tokens[candidate.model],
                )
                for key in keys
            ]
            instance, class_performance = evaluate_llm_instance(
                model, config, call_classes, rates, sum(rates), chunk_tokens
            )
            instance_metrics[candidate_id] = instance
            utilization[candidate_id] = instance.utilization
            kv_stable[candidate_id] = instance.stable
            if not instance.stable:
                violations.append(f"llm_queue_overload:{candidate_id}")
            if instance.resident_capacity < 1 or instance.kv_slack <= 0.0:
                violations.append(f"llm_kv_overload:{candidate_id}")
            for key, value in zip(keys, class_performance):
                perf[key] = value
        for key, rate in arrivals.items():
            if rate > 0.0 and key not in perf:
                violations.append(f"llm_unserved:{key[2]}")
        self._last_llm_instance_performance = instance_metrics
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

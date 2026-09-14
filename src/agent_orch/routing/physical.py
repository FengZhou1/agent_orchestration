from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from agent_orch.performance.llm import service_demand
from agent_orch.performance.network import NetworkBackend
from agent_orch.performance.queueing import tool_response_time
from agent_orch.schema.models import (
    ApplicationSpec,
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    Scenario,
    SlotMetrics,
    WorkflowNode,
)


@dataclass(frozen=True)
class PhysicalRouterConfig:
    utilization_epsilon: float = 0.05
    utilization_cap: float = 0.95
    llm_inverse_temperature: float = 1.0
    tool_inverse_temperature: float = 1.0


class PhysicalRouter:
    """Capacity-, latency-, and utilization-aware physical probability router."""

    def __init__(
        self,
        scenario: Scenario,
        config: PhysicalRouterConfig = PhysicalRouterConfig(),
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.network = NetworkBackend(scenario.links)

    def route(
        self,
        deployment: DeploymentDecision,
        model_share: Mapping[tuple[str, str, str], float],
        previous_metrics: SlotMetrics | None = None,
    ) -> RoutingDecision:
        llm_share: dict[tuple[str, str, str, str], float] = {}
        service_route: dict[tuple[str, str, str, str, str], float] = {}

        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    for model in self.scenario.models:
                        conditional = self._llm_conditional_probabilities(
                            deployment,
                            app,
                            ingress,
                            node,
                            model,
                            previous_metrics,
                        )
                        self._validate_distribution(
                            conditional,
                            f"LLM route {app.id}/{ingress}/{node.id}/{model}",
                        )
                        share = max(
                            0.0,
                            float(model_share.get((app.id, ingress, model), 0.0)),
                        )
                        for candidate_id, probability in conditional.items():
                            llm_share[(app.id, ingress, node.id, candidate_id)] = (
                                share * probability
                            )

            service_edges = {
                edge
                for flow in app.pattern_flows
                for edge in flow.edges
                if app.nodes[edge[1]].type is NodeType.TOOL
            }
            for source, target in sorted(service_edges):
                service_id = app.nodes[target].tool or ""
                for source_server in self.scenario.servers:
                    conditional = self._service_conditional_probabilities(
                        deployment,
                        service_id,
                        source_server,
                        previous_metrics,
                    )
                    self._validate_distribution(
                        conditional,
                        f"service route {app.id}/{source}/{target}/{source_server}",
                    )
                    for destination, probability in conditional.items():
                        service_route[
                            (app.id, source, target, source_server, destination)
                        ] = probability

        return RoutingDecision(dict(model_share), llm_share, service_route)

    def _llm_conditional_probabilities(
        self,
        deployment: DeploymentDecision,
        app: ApplicationSpec,
        ingress: str,
        node: WorkflowNode,
        model: str,
        previous_metrics: SlotMetrics | None,
    ) -> dict[str, float]:
        costs: dict[str, float] = {}
        for candidate_id, candidate in self.scenario.candidates.items():
            if (
                not deployment.llm_active.get(candidate_id, 0)
                or candidate.model != model
            ):
                continue
            config = self.scenario.llm_configs[candidate.config]
            demand = service_demand(
                self.scenario.models[model],
                config,
                node.prompt_tokens[model],
                node.output_tokens[model],
                self.scenario.simulation.prefill_chunk_tokens,
            )
            utilization = (
                previous_metrics.llm_utilization.get(candidate_id, 0.0)
                if previous_metrics is not None
                else 0.0
            )
            if utilization >= 1.0:
                continue
            residual = self._residual_capacity(utilization)
            network_delay = self._llm_predecessor_network_delay(
                deployment,
                app,
                ingress,
                node.id,
                model,
                candidate.server,
                previous_metrics,
            )
            if network_delay is None:
                continue
            costs[candidate_id] = network_delay + demand.service_s / residual
        return _softmin(costs, self.config.llm_inverse_temperature)

    def _service_conditional_probabilities(
        self,
        deployment: DeploymentDecision,
        service_id: str,
        source_server: str,
        previous_metrics: SlotMetrics | None,
    ) -> dict[str, float]:
        costs: dict[str, float] = {}
        service = self.scenario.tools.get(service_id)
        if service is None:
            return {}
        for destination in self.scenario.servers:
            replicas = deployment.tool_replicas.get((service_id, destination), 0)
            rate = service.service_rate.get(destination, 0.0)
            if replicas <= 0 or rate <= 0.0:
                continue
            utilization = (
                previous_metrics.tool_utilization.get(
                    f"{service_id}@{destination}", 0.0
                )
                if previous_metrics is not None
                else 0.0
            )
            if utilization >= 1.0:
                continue
            network_delay = self._network_delay(
                source_server, destination, previous_metrics
            )
            if network_delay is None:
                continue
            inferred_arrival = max(0.0, utilization) * replicas * rate
            wait, processing, _, overloaded = tool_response_time(
                inferred_arrival,
                rate,
                replicas,
                service.arrival_scv,
                self.scenario.simulation.overload_delay_s,
            )
            if overloaded:
                continue
            costs[destination] = network_delay + wait + processing
        return _softmin(costs, self.config.tool_inverse_temperature)

    def _residual_capacity(self, utilization: float) -> float:
        return max(
            self.config.utilization_epsilon,
            1.0 - min(max(0.0, utilization), self.config.utilization_cap),
        )

    @staticmethod
    def _validate_distribution(
        probabilities: Mapping[str, float], label: str
    ) -> None:
        if probabilities and not math.isclose(
            sum(probabilities.values()), 1.0, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            raise RuntimeError(f"{label} is not normalized")

    def _network_delay(
        self,
        source: str,
        target: str,
        previous_metrics: SlotMetrics | None,
    ) -> float | None:
        try:
            path = self.network.path(source, target)
        except ValueError:
            return None
        loads = {}
        for edge in path:
            key = f"{edge[0]}->{edge[1]}"
            utilization = (
                previous_metrics.link_utilization.get(key, 0.0)
                if previous_metrics is not None
                else 0.0
            )
            if utilization >= 1.0:
                return None
            loads[edge] = utilization * self.network.links[edge].capacity_mbps
        return self.network.path_delay(source, target, loads)

    def _llm_predecessor_network_delay(
        self,
        deployment: DeploymentDecision,
        app: ApplicationSpec,
        ingress: str,
        node_id: str,
        model: str,
        destination: str,
        previous_metrics: SlotMetrics | None,
    ) -> float | None:
        edge_probability: dict[tuple[str, str], float] = {}
        for flow in app.pattern_flows:
            for edge in flow.edges:
                if edge[1] == node_id:
                    edge_probability[edge] = (
                        edge_probability.get(edge, 0.0) + flow.probability
                    )
        if not edge_probability:
            return self._network_delay(ingress, destination, previous_metrics)

        weighted_delays: list[tuple[float, float]] = []
        for (source_id, target_id), probability in edge_probability.items():
            locations = self._node_location_distribution(
                deployment, app, source_id, model
            )
            if not locations:
                return None
            for source_server, location_probability in locations.items():
                delay = self._network_delay(
                    source_server, destination, previous_metrics
                )
                if delay is None:
                    return None
                weighted_delays.append(
                    (probability * location_probability, delay)
                )
        total_weight = sum(weight for weight, _ in weighted_delays)
        if total_weight <= 1.0e-12:
            return None
        return sum(weight * delay for weight, delay in weighted_delays) / total_weight

    def _node_location_distribution(
        self,
        deployment: DeploymentDecision,
        app: ApplicationSpec,
        node_id: str,
        model: str,
    ) -> dict[str, float]:
        node = app.nodes[node_id]
        weights: dict[str, float] = {}
        if node.type is NodeType.LLM:
            for candidate_id, active in deployment.llm_active.items():
                candidate = self.scenario.candidates[candidate_id]
                if active and candidate.model == model:
                    weights[candidate.server] = weights.get(candidate.server, 0.0) + 1.0
        else:
            for server_id in self.scenario.servers:
                replicas = deployment.tool_replicas.get((node.tool or "", server_id), 0)
                if replicas > 0:
                    weights[server_id] = float(replicas)
        return _normalize(weights)

def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(positive.values())
    if total <= 1.0e-12:
        return {}
    return {key: value / total for key, value in positive.items() if value > 0.0}


def _softmin(costs: Mapping[str, float], inverse_temperature: float) -> dict[str, float]:
    if not costs:
        return {}
    minimum = min(costs.values())
    scores = {
        key: math.exp(-inverse_temperature * (value - minimum))
        for key, value in costs.items()
    }
    return _normalize(scores)

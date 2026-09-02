from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agent_orch.performance.llm import service_demand
from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import (
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
                            ingress,
                            node,
                            model,
                            previous_metrics,
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
                    for destination, probability in conditional.items():
                        service_route[
                            (app.id, source, target, source_server, destination)
                        ] = probability

        return RoutingDecision(dict(model_share), llm_share, service_route)

    def _llm_conditional_probabilities(
        self,
        deployment: DeploymentDecision,
        ingress: str,
        node: WorkflowNode,
        model: str,
        previous_metrics: SlotMetrics | None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for candidate_id, candidate in self.scenario.candidates.items():
            if (
                not deployment.llm_active.get(candidate_id, 0)
                or candidate.model != model
            ):
                continue
            propagation = self._propagation_delay(ingress, candidate.server)
            if propagation is None:
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
            residual = self._residual_capacity(utilization)
            scores[candidate_id] = 1.0 / max(
                1.0e-12, propagation + demand.service_s / residual
            )
        return _normalize(scores)

    def _service_conditional_probabilities(
        self,
        deployment: DeploymentDecision,
        service_id: str,
        source_server: str,
        previous_metrics: SlotMetrics | None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        service = self.scenario.tools.get(service_id)
        if service is None:
            return scores
        for destination in self.scenario.servers:
            replicas = deployment.tool_replicas.get((service_id, destination), 0)
            rate = service.service_rate.get(destination, 0.0)
            if replicas <= 0 or rate <= 0.0:
                continue
            propagation = self._propagation_delay(source_server, destination)
            if propagation is None:
                continue
            utilization = (
                previous_metrics.tool_utilization.get(
                    f"{service_id}@{destination}", 0.0
                )
                if previous_metrics is not None
                else 0.0
            )
            residual = self._residual_capacity(utilization)
            processing = 1.0 / (replicas * rate * residual)
            scores[destination] = 1.0 / max(1.0e-12, propagation + processing)
        return _normalize(scores)

    def _residual_capacity(self, utilization: float) -> float:
        return max(
            self.config.utilization_epsilon,
            1.0 - min(max(0.0, utilization), self.config.utilization_cap),
        )

    def _propagation_delay(self, source: str, target: str) -> float | None:
        try:
            path = self.network.path(source, target)
        except ValueError:
            return None
        return sum(self.network.links[edge].propagation_ms for edge in path) / 1000.0


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(positive.values())
    if total <= 1.0e-12:
        return {}
    return {key: value / total for key, value in positive.items() if value > 0.0}

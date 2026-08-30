from __future__ import annotations

from collections import defaultdict

from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import DeploymentDecision, NodeType, RoutingDecision, Scenario


class ActionDecoder:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.capacity = CapacityPlanner(scenario)

    def validate_deployment(self, decision: DeploymentDecision) -> None:
        if not self.capacity.deployment_feasible(decision):
            raise ValueError("Deployment violates a physical resource constraint")

    def validate_routing(
        self, deployment: DeploymentDecision, routing: RoutingDecision, atol: float = 1e-7
    ) -> None:
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                model_sum = sum(
                    routing.model_share.get((app.id, ingress, model), 0.0)
                    for model in self.scenario.models
                )
                if abs(model_sum - 1.0) > atol:
                    raise ValueError(f"Model shares do not sum to one for {app.id}@{ingress}")
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    for node in app.nodes.values():
                        if node.type is not NodeType.LLM:
                            continue
                        instance_sum = sum(
                            routing.llm_share.get(
                                (app.id, ingress, node.id, candidate.id), 0.0
                            )
                            for candidate in self.scenario.candidates.values()
                            if candidate.model == model
                        )
                        if abs(instance_sum - model_share) > atol:
                            raise ValueError(
                                f"LLM shares do not sum to model share for {app.id}:{node.id}:{model}"
                            )
        for app in self.scenario.applications.values():
            tool_edges = {
                edge
                for flow in app.pattern_flows
                for edge in flow.edges
                if app.nodes[edge[1]].type is NodeType.TOOL
            }
            for source, target in tool_edges:
                tool_id = app.nodes[target].tool
                destinations = [
                    server
                    for server in self.scenario.servers
                    if deployment.tool_replicas.get((tool_id or "", server), 0) > 0
                ]
                for u in self.scenario.servers:
                    route_sum = sum(
                        routing.tool_route.get((app.id, source, target, u, v), 0.0)
                        for v in destinations
                    )
                    if destinations and abs(route_sum - 1.0) > atol:
                        raise ValueError(
                            f"Tool route does not sum to one for {app.id}:{source}->{target}@{u}"
                        )

    @staticmethod
    def normalize(values: dict[str, float]) -> dict[str, float]:
        positive = {key: max(0.0, value) for key, value in values.items()}
        total = sum(positive.values())
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in positive.items() if value > 0.0}


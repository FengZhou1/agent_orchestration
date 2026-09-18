from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from agent_orch.action_decoder import ActionDecoder
from agent_orch.performance import AnalyticalBackend
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.schema.models import (
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    Scenario,
    SlotMetrics,
    Transition,
)
from agent_orch.workload import ArrivalTrace


class Simulator:
    def __init__(self, scenario: Scenario, max_mapping_samples: int = 4096):
        self.scenario = scenario
        self.backend = AnalyticalBackend(scenario)
        self.workflow = WorkflowEvaluator(
            scenario, self.backend, max_mapping_samples=max_mapping_samples
        )
        self.decoder = ActionDecoder(scenario)
        self.slot = 0
        self.rng = np.random.default_rng(0)
        self.previous_deployment = self._empty_deployment()
        self.last_metrics: SlotMetrics | None = None
        self.arrival_trace: ArrivalTrace | None = None

    def _empty_deployment(self) -> DeploymentDecision:
        return DeploymentDecision(
            llm_active={candidate_id: 0 for candidate_id in self.scenario.candidates},
            tool_replicas={
                (tool_id, server_id): 0
                for tool_id in self.scenario.tools
                for server_id in self.scenario.servers
            },
        )

    def reset(self, seed: int = 0) -> dict[str, Any]:
        self.slot = 0
        self.rng = np.random.default_rng(seed)
        self.previous_deployment = self._empty_deployment()
        self.last_metrics = None
        return self.observation()

    def set_arrival_trace(self, trace: ArrivalTrace | None) -> None:
        self.arrival_trace = trace

    def current_arrival_rates(self) -> dict[tuple[str, str], float]:
        if self.arrival_trace is None:
            return {
                (app.id, ingress): rate
                for app in self.scenario.applications.values()
                for ingress, rate in app.ingress_rates.items()
            }
        return self.arrival_trace.at(self.slot, self.scenario)

    def evaluate_period(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
    ) -> Transition:
        self.decoder.validate_deployment(deployment)
        self.decoder.validate_routing(deployment, routing, allow_unserved=True)
        arrival_rates = self.current_arrival_rates()
        analytical = self.backend.evaluate(deployment, routing, arrival_rates)
        analytical.violations.extend(
            self._routing_unserved_violations(deployment, routing, arrival_rates)
        )
        workflow = self.workflow.evaluate(
            deployment, routing, analytical, arrival_rates
        )
        cost = self._cost(deployment, analytical.link_load_mbps)
        violations = len(set(analytical.violations))
        attainment = (
            workflow.goodput_rps / workflow.total_arrival_rps
            if workflow.total_arrival_rps > 0.0
            else 0.0
        )
        metrics = SlotMetrics(
            slot=self.slot,
            cost=cost,
            mean_latency_s=workflow.mean_latency_s,
            goodput_rps=workflow.goodput_rps,
            quality=workflow.quality,
            total_arrival_rps=workflow.total_arrival_rps,
            slo_attainment=attainment,
            violations=violations,
            app_latency_s=workflow.app_latency_s,
            llm_utilization=analytical.llm_utilization,
            tool_utilization={f"{h}@{n}": value for (h, n), value in analytical.tool_utilization.items()},
            link_utilization=self.backend.network.utilization(analytical.link_load_mbps),
            diagnostics={
                "violation_labels": sorted(set(analytical.violations)),
                "kv_stable": analytical.llm_kv_stable,
                "flow_metrics": workflow.flow_metrics,
                "active_llm_instances": sorted(
                    candidate_id
                    for candidate_id, active in deployment.llm_active.items()
                    if active
                ),
                "stateless_service_replicas": {
                    f"{tool_id}@{server_id}": int(replicas)
                    for (tool_id, server_id), replicas in deployment.tool_replicas.items()
                    if replicas > 0
                },
                "model_composition": {
                    f"{app_id}@{ingress}:{model_id}": float(share)
                    for (app_id, ingress, model_id), share in routing.model_share.items()
                },
            },
        )
        reward_components = {
            "cost_raw": cost,
            "latency_raw": workflow.mean_latency_s,
            "goodput_raw": workflow.goodput_rps,
            "quality_raw": workflow.quality,
            "constraint_count": float(violations),
        }
        self.previous_deployment = deployment.copy()
        self.last_metrics = metrics
        self.slot += 1
        return Transition(
            observation=self.observation(deployment, analytical),
            reward_components=reward_components,
            metrics=metrics,
        )

    # Compatibility alias for non-RL baselines.  The environment invokes
    # evaluate_period once after composition and never advances a physical
    # slot during deployment substeps.
    def step(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
    ) -> Transition:
        return self.evaluate_period(deployment, routing)

    def _routing_unserved_violations(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        arrival_rates: dict[tuple[str, str], float],
    ) -> list[str]:
        violations: list[str] = []
        for app in self.scenario.applications.values():
            service_edges = {
                edge
                for flow in app.pattern_flows
                for edge in flow.edges
                if app.nodes[edge[1]].type is NodeType.TOOL
            }
            for ingress in app.ingress_rates:
                if arrival_rates.get((app.id, ingress), 0.0) <= 0.0:
                    continue
                model_sum = sum(
                    routing.model_share.get((app.id, ingress, model), 0.0)
                    for model in self.scenario.models
                )
                if model_sum < 1.0 - 1.0e-7:
                    violations.append(f"llm_unserved:{app.id}@{ingress}")
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    for model in self.scenario.models:
                        requested = routing.model_share.get(
                            (app.id, ingress, model), 0.0
                        )
                        if requested <= 1.0e-12:
                            continue
                        routed = sum(
                            routing.llm_share.get(
                                (app.id, ingress, node.id, candidate.id), 0.0
                            )
                            for candidate in self.scenario.candidates.values()
                            if candidate.model == model
                        )
                        if routed < requested - 1.0e-7:
                            violations.append(
                                f"llm_unserved:{app.id}:{node.id}:{model}"
                            )
            for source, target in service_edges:
                service_id = app.nodes[target].tool or ""
                replicas = sum(
                    deployment.tool_replicas.get((service_id, server), 0)
                    for server in self.scenario.servers
                )
                has_route = any(
                    key[0] == app.id and key[1] == source and key[2] == target
                    for key in routing.tool_route
                )
                if replicas <= 0 or not has_route:
                    violations.append(f"service_unserved:{app.id}:{source}->{target}")
        return violations

    def observation(
        self,
        deployment: DeploymentDecision | None = None,
        analytical: Any | None = None,
    ) -> dict[str, Any]:
        deployment = deployment or self.previous_deployment
        return {
            "slot": self.slot,
            "deployment": {
                "llm_active": dict(deployment.llm_active),
                "tool_replicas": {
                    f"{tool}@{server}": value
                    for (tool, server), value in deployment.tool_replicas.items()
                },
            },
            "workload": {
                app.id: {
                    ingress: self.current_arrival_rates().get((app.id, ingress), base_rate)
                    for ingress, base_rate in app.ingress_rates.items()
                }
                for app in self.scenario.applications.values()
            },
            "runtime": {
                "llm_utilization": dict(analytical.llm_utilization) if analytical else {},
                "tool_utilization": {
                    f"{h}@{n}": value
                    for (h, n), value in analytical.tool_utilization.items()
                }
                if analytical
                else {},
                "link_utilization": self.backend.network.utilization(
                    analytical.link_load_mbps
                )
                if analytical
                else {},
            },
        }

    def _cost(
        self,
        deployment: DeploymentDecision,
        link_load_mbps: dict[tuple[str, str], float],
    ) -> float:
        cost = 0.0
        period_seconds = self.scenario.simulation.orchestration_period_s
        for candidate_id, active in deployment.llm_active.items():
            if not active:
                continue
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            cost += config.running_cost_per_slot * period_seconds
            if not self.previous_deployment.llm_active.get(candidate_id, 0):
                cost += config.load_cost
        for pool, replicas in deployment.tool_replicas.items():
            tool_id, _ = pool
            tool = self.scenario.tools[tool_id]
            cost += replicas * tool.running_cost_per_slot * period_seconds
            started = max(0, replicas - self.previous_deployment.tool_replicas.get(pool, 0))
            cost += started * tool.start_cost
        cost += self.backend.network.traffic_cost(link_load_mbps) * period_seconds
        return cost


def metrics_to_dict(metrics: SlotMetrics) -> dict[str, Any]:
    return asdict(metrics)

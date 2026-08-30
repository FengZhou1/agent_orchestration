from __future__ import annotations

from collections import defaultdict

import numpy as np

from agent_orch.capacity import CapacityPlanner
from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import (
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    Scenario,
    SlotMetrics,
)


class BasePolicy:
    def __init__(self, scenario: Scenario, seed: int = 0):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.planner = CapacityPlanner(scenario)

    def deployment(self) -> DeploymentDecision:
        return self.planner.initial_deployment()

    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        raise NotImplementedError

    def decide(self) -> tuple[DeploymentDecision, RoutingDecision]:
        deployment = self.deployment()
        return deployment, self.routing(deployment)

    def _active_by_model(self, deployment: DeploymentDecision) -> dict[str, list[str]]:
        active: dict[str, list[str]] = defaultdict(list)
        for candidate_id, enabled in deployment.llm_active.items():
            if enabled:
                active[self.scenario.candidates[candidate_id].model].append(candidate_id)
        return active

    def _tool_routes(
        self,
        deployment: DeploymentDecision,
        randomize: bool = False,
        metrics: SlotMetrics | None = None,
    ) -> dict[tuple[str, str, str, str, str], float]:
        routes: dict[tuple[str, str, str, str, str], float] = {}
        network = NetworkBackend(self.scenario.links)
        for app in self.scenario.applications.values():
            edges = {
                edge
                for flow in app.pattern_flows
                for edge in flow.edges
                if app.nodes[edge[1]].type is NodeType.TOOL
            }
            for source, target in edges:
                tool_id = app.nodes[target].tool or ""
                destinations = [
                    server
                    for server in self.scenario.servers
                    if deployment.tool_replicas.get((tool_id, server), 0) > 0
                ]
                for u in self.scenario.servers:
                    if not destinations:
                        continue
                    if randomize:
                        weights = self.rng.dirichlet(np.ones(len(destinations)))
                    else:
                        scores = []
                        for v in destinations:
                            propagation = sum(
                                network.links[edge].propagation_ms
                                for edge in network.path(u, v)
                            ) / 1000.0
                            processing = 1.0 / self.scenario.tools[tool_id].service_rate[v]
                            utilization = (
                                metrics.tool_utilization.get(f"{tool_id}@{v}", 0.0)
                                if metrics
                                else 0.0
                            )
                            congestion = processing / max(
                                1e-3, 1.0 - min(utilization, 0.999)
                            )
                            scores.append(1.0 / max(1e-9, propagation + congestion))
                        weights = np.asarray(scores) / np.sum(scores)
                    for v, weight in zip(destinations, weights):
                        routes[(app.id, source, target, u, v)] = float(weight)
        return routes


class StaticPolicy(BasePolicy):
    def deployment(self) -> DeploymentDecision:
        deployment = self.planner.initial_deployment()
        selected = min(
            self.scenario.candidates.values(),
            key=lambda candidate: self.scenario.llm_configs[
                candidate.config
            ].running_cost_per_slot,
        )
        for candidate_id in deployment.llm_active:
            deployment.llm_active[candidate_id] = int(candidate_id == selected.id)
        return deployment

    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        active = self._active_by_model(deployment)
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        available_models = sorted(model for model, ids in active.items() if ids)
        selected_model = available_models[0]
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share[(app.id, ingress, model)] = float(model == selected_model)
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    instances = active[selected_model]
                    for candidate_id in instances:
                        llm_share[(app.id, ingress, node.id, candidate_id)] = 1.0 / len(instances)
        return RoutingDecision(model_share, llm_share, self._tool_routes(deployment))


class GreedyPolicy(BasePolicy):
    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        active = self._active_by_model(deployment)
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        for app in self.scenario.applications.values():
            feasible_models = [model for model, ids in active.items() if ids]
            selected = max(
                feasible_models,
                key=lambda model: app.quality[model]
                - 0.02
                * min(
                    self.scenario.llm_configs[self.scenario.candidates[cid].config].running_cost_per_slot
                    for cid in active[model]
                ),
            )
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share[(app.id, ingress, model)] = float(model == selected)
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    scores = []
                    for candidate_id in active[selected]:
                        candidate = self.scenario.candidates[candidate_id]
                        config = self.scenario.llm_configs[candidate.config]
                        scores.append(config.effective_flops / config.running_cost_per_slot)
                    weights = np.asarray(scores) / np.sum(scores)
                    for candidate_id, weight in zip(active[selected], weights):
                        llm_share[(app.id, ingress, node.id, candidate_id)] = float(weight)
        return RoutingDecision(model_share, llm_share, self._tool_routes(deployment))


class RandomPolicy(BasePolicy):
    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        active = self._active_by_model(deployment)
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        feasible_models = [model for model, ids in active.items() if ids]
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                weights = self.rng.dirichlet(np.ones(len(feasible_models)))
                for model in self.scenario.models:
                    model_share[(app.id, ingress, model)] = 0.0
                for model, weight in zip(feasible_models, weights):
                    model_share[(app.id, ingress, model)] = float(weight)
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    for model in feasible_models:
                        instance_weights = self.rng.dirichlet(np.ones(len(active[model])))
                        for candidate_id, conditional in zip(active[model], instance_weights):
                            llm_share[(app.id, ingress, node.id, candidate_id)] = float(
                                model_share[(app.id, ingress, model)] * conditional
                            )
        return RoutingDecision(
            model_share,
            llm_share,
            self._tool_routes(deployment, randomize=True),
        )


class EqualSplitPolicy(BasePolicy):
    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        active = self._active_by_model(deployment)
        available_models = [model for model, instances in active.items() if instances]
        model_probability = 1.0 / len(available_models)
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share[(app.id, ingress, model)] = (
                        model_probability if model in available_models else 0.0
                    )
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    for model in available_models:
                        for candidate_id in active[model]:
                            llm_share[(app.id, ingress, node.id, candidate_id)] = (
                                model_probability / len(active[model])
                            )
        return RoutingDecision(model_share, llm_share, self._tool_routes(deployment))


class LeastLoadPolicy(BasePolicy):
    def routing(
        self, deployment: DeploymentDecision, metrics: SlotMetrics | None = None
    ) -> RoutingDecision:
        active = self._active_by_model(deployment)
        available_models = [model for model, instances in active.items() if instances]
        utilization = metrics.llm_utilization if metrics else {}
        model_scores = {
            model: sum(
                1.0 / max(1e-3, 1.0 + utilization.get(candidate_id, 0.0))
                for candidate_id in active[model]
            )
            for model in available_models
        }
        model_total = sum(model_scores.values())
        model_weights = {
            model: score / model_total for model, score in model_scores.items()
        }
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share[(app.id, ingress, model)] = model_weights.get(model, 0.0)
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    for model in available_models:
                        instance_scores = {
                            candidate_id: 1.0
                            / max(1e-3, 1.0 + utilization.get(candidate_id, 0.0))
                            for candidate_id in active[model]
                        }
                        total = sum(instance_scores.values())
                        for candidate_id, score in instance_scores.items():
                            llm_share[(app.id, ingress, node.id, candidate_id)] = (
                                model_weights[model] * score / total
                            )
        return RoutingDecision(
            model_share,
            llm_share,
            self._tool_routes(deployment, metrics=metrics),
        )


def make_policy(name: str, scenario: Scenario, seed: int = 0) -> BasePolicy:
    policies = {
        "static": StaticPolicy,
        "equal": EqualSplitPolicy,
        "least_load": LeastLoadPolicy,
        "greedy": GreedyPolicy,
        "random": RandomPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown policy {name}; choose from {sorted(policies)}")
    return policies[name](scenario, seed)

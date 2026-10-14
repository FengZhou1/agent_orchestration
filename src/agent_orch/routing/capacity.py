"""Manuscript capacity-proportional physical routing (algorithm 2).

    phi[a,g,i,l] = x[a,g,m] * mu_ref[l] / sum(mu_ref[reachable same-model l])
    p[a,i,j,u,v] = z[h,v] * mu[h,v] / sum(z[h,w] * mu[h,w])

Historical utilization and predicted class latency do not enter these rules.
Empty support stays empty; Simulator reports unserved requests as violations.
"""
from __future__ import annotations

import math
import time
from typing import Mapping

from agent_orch.performance.analytical import reference_llm_capacity
from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import DeploymentDecision, NodeType, RoutingDecision, Scenario, SlotMetrics


class CapacityProportionalRouter:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.network = NetworkBackend(scenario.links)
        self.reference_capacities: dict[str, float] = {}
        self._reachability: dict[tuple[str, str], bool] = {}
        self.last_duration_s = 0.0

    def _reachable(self, source: str, target: str) -> bool:
        key = (source, target)
        if key not in self._reachability:
            try:
                self.network.path(source, target)
            except ValueError:
                self._reachability[key] = False
            else:
                self._reachability[key] = True
        return self._reachability[key]

    def route(
        self,
        deployment: DeploymentDecision,
        model_share: Mapping[tuple[str, str, str], float],
        previous_metrics: SlotMetrics | None = None,
        arrival_rates: Mapping[tuple[str, str], float] | None = None,
    ) -> RoutingDecision:
        started = time.perf_counter()
        scenario = self.scenario
        active: dict[str, list[str]] = {}
        for cid, candidate in scenario.candidates.items():
            if deployment.llm_active.get(cid, 0):
                active.setdefault(candidate.model, []).append(cid)

        capacities = {}
        for model, candidates in active.items():
            classes, weights = [], []
            for app in scenario.applications.values():
                rate = sum(
                    max(0.0, float((arrival_rates or {}).get((app.id, g), base)))
                    * max(0.0, float(model_share.get((app.id, g, model), 0.0)))
                    for g, base in app.ingress_rates.items()
                )
                for node in app.nodes.values():
                    if node.type is NodeType.LLM:
                        classes.append((node.prompt_tokens[model], node.output_tokens[model]))
                        weights.append(app.visit_probability(node.id) * rate)
            if not classes:
                continue
            # Paper's explicit fallback: equal inference-flow weights when
            # the entire model has no requested traffic in this slot.
            if sum(weights) <= 0.0:
                weights = [1.0] * len(classes)
            else:
                # Zero-rate flows are absent from the requested mixture; they
                # must not constrain KV slack through their maximum length.
                positive = [(item, weight) for item, weight in zip(classes, weights) if weight > 0.0]
                classes, weights = map(list, zip(*positive))
            for cid in candidates:
                config = scenario.llm_configs[scenario.candidates[cid].config]
                capacity, _ = reference_llm_capacity(
                    scenario.models[model], config, classes, weights,
                    scenario.simulation.prefill_chunk_tokens,
                )
                capacities[cid] = capacity
        self.reference_capacities = capacities

        llm_share = {}
        # One conditional distribution per (ingress, model), shared by all
        # applications/stages having the same reachable candidate set.
        conditional = {}
        for app in scenario.applications.values():
            for ingress in app.ingress_rates:
                for model, candidates in active.items():
                    share = max(0.0, float(model_share.get((app.id, ingress, model), 0.0)))
                    if share <= 0.0:
                        continue
                    key = (ingress, model)
                    if key not in conditional:
                        conditional[key] = _capacity_distribution({
                            cid: capacities.get(cid, 0.0) for cid in candidates
                            if self._reachable(ingress, scenario.candidates[cid].server)
                        })
                    for node in app.nodes.values():
                        if node.type is NodeType.LLM:
                            for cid, probability in conditional[key].items():
                                llm_share[(app.id, ingress, node.id, cid)] = share * probability

        service_route, service_conditional = {}, {}
        for app in scenario.applications.values():
            edges = sorted({edge for flow in app.pattern_flows for edge in flow.edges
                            if app.nodes[edge[1]].type is NodeType.TOOL})
            for source, target in edges:
                service_id = app.nodes[target].tool or ""
                service = scenario.tools[service_id]
                for u in scenario.servers:
                    key = (service_id, u)
                    if key not in service_conditional:
                        service_conditional[key] = _capacity_distribution({
                            v: deployment.tool_replicas.get((service_id, v), 0)
                            * service.service_rate.get(v, 0.0)
                            for v in scenario.servers if self._reachable(u, v)
                        })
                    for v, probability in service_conditional[key].items():
                        service_route[(app.id, source, target, u, v)] = probability
        result = RoutingDecision(dict(model_share), llm_share, service_route)
        self.last_duration_s = time.perf_counter() - started
        return result


def _capacity_distribution(values: Mapping[str, float]) -> dict[str, float]:
    positive = {key: float(value) for key, value in values.items()
                if math.isfinite(value) and value > 0.0}
    total = sum(positive.values())
    return {key: value / total for key, value in positive.items()} if total > 0.0 else {}

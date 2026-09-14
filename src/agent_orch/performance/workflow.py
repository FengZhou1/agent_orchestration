from __future__ import annotations

from dataclasses import dataclass, field

from agent_orch.schema.models import (
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    SLOType,
    Scenario,
)

from .analytical import AnalyticalBackend, AnalyticalResult


@dataclass
class WorkflowResult:
    app_latency_s: dict[str, float]
    mean_latency_s: float
    goodput_rps: float
    quality: float
    total_arrival_rps: float
    flow_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


class WorkflowEvaluator:
    def __init__(self, scenario: Scenario, backend: AnalyticalBackend):
        self.scenario = scenario
        self.backend = backend

    def evaluate(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        analytical: AnalyticalResult,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> WorkflowResult:
        app_latency: dict[str, float] = {}
        flow_metrics: dict[str, dict[str, float]] = {}
        goodput = 0.0
        total_arrival = 0.0
        latency_numerator_system = 0.0
        quality_numerator = 0.0

        for app in self.scenario.applications.values():
            app_rate = sum(
                (arrival_rates or {}).get((app.id, ingress), base_rate)
                for ingress, base_rate in app.ingress_rates.items()
            )
            total_arrival += app_rate
            latency_numerator = 0.0
            for ingress, base_rate in app.ingress_rates.items():
                ingress_rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    quality_numerator += ingress_rate * model_share * app.quality[model]
                    for flow in app.pattern_flows:
                        probability = ingress_rate * model_share * flow.probability
                        e2e, ttft, tbt, stage_ok, stage_times = self._flow_latency(
                            app.id,
                            ingress,
                            model,
                            flow.id,
                            routing,
                            analytical,
                        )
                        latency_numerator += probability * e2e
                        satisfied = self.slo_satisfied(
                            app.id, e2e, ttft, tbt, stage_ok
                        )
                        if satisfied:
                            goodput += probability
                        flow_metrics[f"{app.id}:{ingress}:{model}:{flow.id}"] = {
                            "e2e_s": e2e,
                            "ttft_s": ttft,
                            "tbt_s": tbt,
                            "slo": float(satisfied),
                            "weight_rps": probability,
                            **{
                                f"stage:{node_id}_s": value
                                for node_id, value in stage_times.items()
                            },
                        }
            app_latency[app.id] = latency_numerator / app_rate if app_rate > 0.0 else 0.0
            latency_numerator_system += latency_numerator

        mean_latency = (
            latency_numerator_system / total_arrival if total_arrival > 0.0 else 0.0
        )
        quality = quality_numerator / total_arrival if total_arrival > 0.0 else 0.0
        return WorkflowResult(
            app_latency,
            mean_latency,
            goodput,
            quality,
            total_arrival,
            flow_metrics,
        )

    def _flow_latency(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> tuple[float, float, float, bool, dict[str, float]]:
        app = self.scenario.applications[app_id]
        flow = next(flow for flow in app.pattern_flows if flow.id == flow_id)
        prefix_delays: list[float] = []
        for chain in flow.chains:
            delay = self._entry_delay(app_id, ingress, model, flow_id, chain[0], result)
            for source, target in zip(chain[:-1], chain[1:]):
                delay += self._node_response(
                    app_id, ingress, model, flow_id, source, routing, result
                )
                delay += self._edge_delay(
                    app_id,
                    ingress,
                    model,
                    flow_id,
                    source,
                    target,
                    routing,
                    result,
                )
            prefix_delays.append(delay)

        critical_prefix = max(prefix_delays, default=self.scenario.simulation.overload_delay_s)
        final_response, final_ttft, final_tbt = self._final_llm_performance(
            app_id, ingress, model, flow.final_node, routing, result
        )
        exit_delay = self._exit_delay(
            app_id, ingress, model, flow_id, flow.final_node, result
        )
        first_token_return = self._first_token_return_delay(
            app_id, ingress, model, flow_id, flow.final_node, result
        )
        e2e = critical_prefix + final_response + exit_delay
        ttft = critical_prefix + final_ttft + first_token_return
        stage_times = self._stage_completion_times(
            app_id, ingress, model, flow_id, routing, result
        )
        stage_ok = all(
            value <= float(app.nodes[node_id].stage_deadline_s)
            for node_id, value in stage_times.items()
            if app.nodes[node_id].stage_deadline_s is not None
        )
        return e2e, ttft, final_tbt, stage_ok, stage_times

    def _node_response(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        node_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> float:
        app = self.scenario.applications[app_id]
        node = app.nodes[node_id]
        if node.type is NodeType.LLM:
            model_share = routing.model_share.get((app_id, ingress, model), 0.0)
            if model_share <= 0.0:
                return self.scenario.simulation.overload_delay_s
            total = 0.0
            covered = 0.0
            for candidate_id, candidate in self.scenario.candidates.items():
                if candidate.model != model:
                    continue
                probability = routing.llm_share.get(
                    (app_id, ingress, node_id, candidate_id), 0.0
                ) / model_share
                if probability <= 0.0:
                    continue
                perf = result.llm_performance.get((app_id, node_id, candidate_id))
                if perf is None:
                    total += probability * self.scenario.simulation.overload_delay_s
                else:
                    total += probability * perf.response_s
                covered += probability
            if covered < 1.0 - 1e-8:
                total += (1.0 - covered) * self.scenario.simulation.overload_delay_s
            return total

        distribution = result.node_server_distribution.get(
            (app_id, ingress, model, flow_id, node_id), {}
        )
        total = 0.0
        for server, probability in distribution.items():
            total += probability * result.tool_delay.get(
                (node.tool or "", server), self.scenario.simulation.overload_delay_s
            )
        if sum(distribution.values()) < 1.0 - 1e-8:
            total += (1.0 - sum(distribution.values())) * self.scenario.simulation.overload_delay_s
        return total

    def _edge_delay(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        source: str,
        target: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> float:
        pairs = self.backend.edge_pair_distribution(
            app_id,
            ingress,
            model,
            flow_id,
            source,
            target,
            routing,
            result.node_server_distribution,
        )
        if not pairs:
            return self.scenario.simulation.overload_delay_s
        return sum(
            probability
            * self.backend.network.path_delay(u, v, result.link_load_mbps)
            for u, v, probability in pairs
        )

    def _entry_delay(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        node_id: str,
        result: AnalyticalResult,
    ) -> float:
        distribution = result.node_server_distribution.get(
            (app_id, ingress, model, flow_id, node_id), {}
        )
        return sum(
            probability
            * self.backend.network.path_delay(
                ingress, server, result.link_load_mbps
            )
            for server, probability in distribution.items()
        )

    def _exit_delay(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        final_node: str,
        result: AnalyticalResult,
    ) -> float:
        distribution = result.node_server_distribution.get(
            (app_id, ingress, model, flow_id, final_node), {}
        )
        return sum(
            probability
            * self.backend.network.path_delay(
                server, ingress, result.link_load_mbps
            )
            for server, probability in distribution.items()
        )

    def _first_token_return_delay(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        final_node: str,
        result: AnalyticalResult,
    ) -> float:
        app = self.scenario.applications[app_id]
        output_tokens = max(1.0, app.nodes[final_node].output_tokens[model])
        token_data = app.exit_data_mb[model] / output_tokens
        distribution = result.node_server_distribution.get(
            (app_id, ingress, model, flow_id, final_node), {}
        )
        return sum(
            probability
            * self.backend.network.first_token_return_delay(server, ingress, token_data)
            for server, probability in distribution.items()
        )

    def _final_llm_performance(
        self,
        app_id: str,
        ingress: str,
        model: str,
        node_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> tuple[float, float, float]:
        model_share = routing.model_share.get((app_id, ingress, model), 0.0)
        response = ttft = tbt = covered = 0.0
        if model_share <= 0.0:
            delay = self.scenario.simulation.overload_delay_s
            return delay, delay, delay
        for candidate_id, candidate in self.scenario.candidates.items():
            if candidate.model != model:
                continue
            probability = routing.llm_share.get(
                (app_id, ingress, node_id, candidate_id), 0.0
            ) / model_share
            if probability <= 0.0:
                continue
            perf = result.llm_performance.get((app_id, node_id, candidate_id))
            if perf is None:
                response += probability * self.scenario.simulation.overload_delay_s
                ttft += probability * self.scenario.simulation.overload_delay_s
                tbt += probability * self.scenario.simulation.overload_delay_s
            else:
                response += probability * perf.response_s
                ttft += probability * perf.ttft_s
                tbt += probability * perf.tbt_s
            covered += probability
        if covered < 1.0 - 1e-8:
            missing = 1.0 - covered
            response += missing * self.scenario.simulation.overload_delay_s
            ttft += missing * self.scenario.simulation.overload_delay_s
            tbt += missing * self.scenario.simulation.overload_delay_s
        return response, ttft, tbt

    def _stage_completion_times(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> dict[str, float]:
        app = self.scenario.applications[app_id]
        flow = next(flow for flow in app.pattern_flows if flow.id == flow_id)
        completion: dict[str, float] = {}
        for node in app.nodes.values():
            if (
                node.type is not NodeType.LLM
                or node.id == flow.final_node
                or node.id not in flow.nodes
            ):
                continue
            prefixes = []
            for chain in flow.chains:
                if node.id not in chain:
                    continue
                index = chain.index(node.id)
                prefix = chain[: index + 1]
                delay = self._entry_delay(
                    app_id, ingress, model, flow_id, prefix[0], result
                )
                for source, target in zip(prefix[:-1], prefix[1:]):
                    delay += self._node_response(
                        app_id, ingress, model, flow_id, source, routing, result
                    )
                    delay += self._edge_delay(
                        app_id,
                        ingress,
                        model,
                        flow_id,
                        source,
                        target,
                        routing,
                        result,
                    )
                delay += self._node_response(
                    app_id, ingress, model, flow_id, node.id, routing, result
                )
                prefixes.append(delay)
            if prefixes:
                completion[node.id] = max(prefixes)
        return completion

    def slo_satisfied(
        self,
        app_id: str,
        e2e_s: float,
        ttft_s: float,
        tbt_s: float,
        stage_ok: bool,
    ) -> bool:
        slo = self.scenario.applications[app_id].slo
        if slo.type is SLOType.LATENCY:
            return ttft_s <= float(slo.ttft_s) and tbt_s <= float(slo.tbt_s)
        if slo.type is SLOType.DEADLINE:
            return e2e_s <= float(slo.deadline_s)
        return (
            e2e_s <= float(slo.deadline_s)
            and ttft_s <= float(slo.ttft_s)
            and tbt_s <= float(slo.tbt_s)
            and stage_ok
        )

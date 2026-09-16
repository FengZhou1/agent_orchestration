from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools

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
    _MAX_MAPPING_SAMPLES = 4096

    def __init__(self, scenario: Scenario, backend: AnalyticalBackend):
        self.scenario = scenario
        self.backend = backend
        self._mapping_cache: dict[tuple[str, str, str, str], list[tuple[dict[str, tuple[str, str | None]], float]]] = {}

    def evaluate(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        analytical: AnalyticalResult,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> WorkflowResult:
        self._mapping_cache = {}
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
                        e2e, ttft, tbt, slo_probability, stage_times = self._flow_summary(
                            app.id,
                            ingress,
                            model,
                            flow.id,
                            routing,
                            analytical,
                        )
                        latency_numerator += probability * e2e
                        goodput += probability * slo_probability
                        flow_metrics[f"{app.id}:{ingress}:{model}:{flow.id}"] = {
                            "e2e_s": e2e,
                            "ttft_s": ttft,
                            "tbt_s": tbt,
                            "slo": float(slo_probability),
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
        e2e, ttft, tbt, slo_probability, stage_times = self._flow_summary(
            app_id, ingress, model, flow_id, routing, result
        )
        return e2e, ttft, tbt, slo_probability >= 1.0 - 1.0e-8, stage_times

    def _flow_summary(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> tuple[float, float, float, float, dict[str, float]]:
        """Evaluate the critical path before averaging over physical mappings."""
        app = self.scenario.applications[app_id]
        flow = next(flow for flow in app.pattern_flows if flow.id == flow_id)
        outcomes = self._physical_mapping_outcomes(
            app_id, ingress, model, flow_id, routing, result
        )
        if not outcomes:
            delay = self.scenario.simulation.overload_delay_s
            return delay, delay, delay, 0.0, {}

        e2e_sum = ttft_sum = tbt_sum = 0.0
        slo_sum = 0.0
        stage_sums: dict[str, float] = {}
        for mapping, probability in outcomes:
            e2e, ttft, tbt, stage_times = self._mapping_latency(
                app_id, ingress, model, flow, mapping, result
            )
            stage_ok = all(
                value <= float(app.nodes[node_id].stage_deadline_s)
                for node_id, value in stage_times.items()
                if app.nodes[node_id].stage_deadline_s is not None
            )
            e2e_sum += probability * e2e
            ttft_sum += probability * ttft
            tbt_sum += probability * tbt
            if self.slo_satisfied(app_id, e2e, ttft, tbt, stage_ok):
                slo_sum += probability
            for node_id, value in stage_times.items():
                stage_sums[node_id] = stage_sums.get(node_id, 0.0) + probability * value
        return e2e_sum, ttft_sum, tbt_sum, slo_sum, stage_sums

    def _physical_mapping_outcomes(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        routing: RoutingDecision,
        result: AnalyticalResult,
    ) -> list[tuple[dict[str, tuple[str, str | None]], float]]:
        """Construct the joint physical mapping distribution for one pattern flow.

        Each LLM node carries its candidate instance and server; each stateless
        node carries its selected server.  Exact enumeration is used for small
        supports.  Large products are evaluated by a deterministic weighted
        sample so that the critical-path maximum remains inside the expectation.
        """
        app = self.scenario.applications[app_id]
        flow = next(flow for flow in app.pattern_flows if flow.id == flow_id)
        cache_key = (app_id, ingress, model, flow_id)
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]
        choices: list[list[tuple[str, tuple[str, str | None], float]]] = []
        model_share = routing.model_share.get((app_id, ingress, model), 0.0)
        for node_id in sorted(flow.nodes):
            node = app.nodes[node_id]
            node_choices: list[tuple[str, tuple[str, str | None], float]] = []
            if node.type is NodeType.LLM and model_share > 0.0:
                for candidate_id, candidate in self.scenario.candidates.items():
                    if candidate.model != model:
                        continue
                    probability = routing.llm_share.get(
                        (app_id, ingress, node_id, candidate_id), 0.0
                    ) / model_share
                    if probability > 0.0:
                        node_choices.append(
                            (node_id, (candidate.server, candidate_id), probability)
                        )
            else:
                distribution = result.node_server_distribution.get(
                    (app_id, ingress, model, flow_id, node_id), {}
                )
                for server, probability in distribution.items():
                    if probability > 0.0:
                        node_choices.append((node_id, (server, None), probability))
            if not node_choices:
                node_choices = [(node_id, ("", None), 1.0)]
            total = sum(item[2] for item in node_choices)
            choices.append([
                (node_id, location, probability / total)
                for node_id, location, probability in node_choices
            ])

        cardinality = 1
        for node_choices in choices:
            cardinality *= len(node_choices)
        outcomes: list[tuple[dict[str, tuple[str, str | None]], float]] = []
        if cardinality <= self._MAX_MAPPING_SAMPLES:
            for combination in itertools.product(*choices):
                probability = 1.0
                mapping: dict[str, tuple[str, str | None]] = {}
                for node_id, location, weight in combination:
                    mapping[node_id] = location
                    probability *= weight
                outcomes.append((mapping, probability))
            self._mapping_cache[cache_key] = outcomes
            return outcomes

        seed_material = f"{app_id}|{ingress}|{model}|{flow_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        import numpy as np
        rng = np.random.default_rng(seed)
        for _ in range(self._MAX_MAPPING_SAMPLES):
            mapping = {}
            for node_choices in choices:
                probabilities = np.asarray([item[2] for item in node_choices], dtype=float)
                probabilities /= probabilities.sum()
                index = int(rng.choice(len(node_choices), p=probabilities))
                node_id, location, _ = node_choices[index]
                mapping[node_id] = location
            outcomes.append((mapping, 1.0 / self._MAX_MAPPING_SAMPLES))
        self._mapping_cache[cache_key] = outcomes
        return outcomes

    def _mapping_latency(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow,
        mapping: dict[str, tuple[str, str | None]],
        result: AnalyticalResult,
    ) -> tuple[float, float, float, dict[str, float]]:
        app = self.scenario.applications[app_id]
        overload = self.scenario.simulation.overload_delay_s

        def location(node_id: str) -> str:
            return mapping.get(node_id, ("", None))[0]

        def node_response(node_id: str) -> float:
            node = app.nodes[node_id]
            server, candidate_id = mapping.get(node_id, ("", None))
            if not server:
                return overload
            if node.type is NodeType.LLM:
                perf = result.llm_performance.get((app_id, node_id, candidate_id or ""))
            else:
                perf = None
            if node.type is NodeType.LLM:
                return perf.response_s if perf is not None else overload
            return result.tool_delay.get((node.tool or "", server), overload)

        prefix_delays: list[float] = []
        for chain in flow.chains:
            first = location(chain[0])
            delay = overload if not first else self.backend.network.path_delay(
                ingress, first, result.link_load_mbps
            )
            for source, target in zip(chain[:-1], chain[1:]):
                delay += node_response(source)
                source_server, target_server = location(source), location(target)
                if not source_server or not target_server:
                    delay += overload
                else:
                    delay += self.backend.network.path_delay(
                        source_server, target_server, result.link_load_mbps
                    )
            prefix_delays.append(delay)
        critical_prefix = max(prefix_delays, default=overload)
        final_server = location(flow.final_node)
        final_candidate = mapping.get(flow.final_node, ("", None))[1]
        final_perf = result.llm_performance.get(
            (app_id, flow.final_node, final_candidate or "")
        )
        if final_perf is None or not final_server:
            final_response = final_ttft = final_tbt = overload
        else:
            final_response, final_ttft, final_tbt = (
                final_perf.response_s,
                final_perf.ttft_s,
                final_perf.tbt_s,
            )
        exit_delay = overload if not final_server else self.backend.network.path_delay(
            final_server, ingress, result.link_load_mbps
        )
        output_tokens = max(1.0, app.nodes[flow.final_node].output_tokens[model])
        token_data = app.exit_data_mb[model] / output_tokens
        first_token_return = overload if not final_server else self.backend.network.first_token_return_delay(
            final_server, ingress, token_data
        )
        stage_times: dict[str, float] = {}
        for node in app.nodes.values():
            if node.type is not NodeType.LLM or node.id == flow.final_node or node.id not in flow.nodes:
                continue
            prefixes: list[float] = []
            for chain in flow.chains:
                if node.id not in chain:
                    continue
                index = chain.index(node.id)
                prefix = chain[: index + 1]
                first_server = location(prefix[0])
                delay = overload if not first_server else self.backend.network.path_delay(
                    ingress, first_server, result.link_load_mbps
                )
                for source, target in zip(prefix[:-1], prefix[1:]):
                    delay += node_response(source)
                    source_server, target_server = location(source), location(target)
                    delay += overload if not source_server or not target_server else self.backend.network.path_delay(
                        source_server, target_server, result.link_load_mbps
                    )
                delay += node_response(node.id)
                prefixes.append(delay)
            if prefixes:
                stage_times[node.id] = max(prefixes)
        return (
            critical_prefix + final_response + exit_delay,
            critical_prefix + final_ttft + first_token_return,
            final_tbt,
            stage_times,
        )

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

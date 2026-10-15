from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import itertools
import math
import networkx as nx
import numpy as np

from agent_orch.schema.models import (
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    SLOType,
    Scenario,
)

from .analytical import AnalyticalBackend, AnalyticalResult
from .workflow_batch import IndexedWorkflowKernel
from .workflow_runtime import WorkflowRuntime


@lru_cache(maxsize=4096)
def _workflow_topology(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]):
    """Cache graph facts only, preserving NetworkX's exact traversal orders."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    order = tuple(nx.lexicographical_topological_sort(graph))
    parents = tuple((node, tuple(graph.predecessors(node))) for node in order)
    roots = tuple(node for node in graph if graph.in_degree(node) == 0)
    return parents, roots


@dataclass
class WorkflowResult:
    app_latency_s: dict[str, float]
    mean_latency_s: float | None
    goodput_rps: float
    quality: float
    total_arrival_rps: float
    flow_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    app_quality: dict[str, float] = field(default_factory=dict)
    completed_rps: float = 0.0
    app_completed_rps: dict[str, float] = field(default_factory=dict)


class WorkflowEvaluator:
    _MAX_MAPPING_SAMPLES = 4096
    DEFAULT_BATCH_EVALUATION = True

    def __init__(
        self,
        scenario: Scenario,
        backend: AnalyticalBackend,
        max_mapping_samples: int = _MAX_MAPPING_SAMPLES,
        *,
        batch_evaluation: bool | None = None,
        runtime: WorkflowRuntime | None = None,
    ):
        self.scenario = scenario
        self.backend = backend
        self.max_mapping_samples = max(1, int(max_mapping_samples))
        self.batch_evaluation = self.DEFAULT_BATCH_EVALUATION if batch_evaluation is None else batch_evaluation
        self._batch_kernel = None
        self._mapping_cache: dict[tuple[str, str, str, str], list[tuple[dict[str, tuple[str, str | None]], float]]] = {}
        self._path_delay_cache: dict[tuple[str, str], float] = {}
        self._first_token_delay_cache: dict[tuple[str, str, float], float] = {}
        self._completion_probability = {}
        self._path_feasible_cache: dict[tuple[str, str], bool] = {}
        self.runtime = runtime or WorkflowRuntime()
        self._array_summaries = None
        self._choice_cache = {}

    def evaluate(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        analytical: AnalyticalResult,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> WorkflowResult:
        self._batch_kernel = None
        self._mapping_cache = {}
        self._path_delay_cache = {}
        self._first_token_delay_cache = {}
        self._completion_probability = {}
        # Feasibility depends on this evaluation's link loads, not just topology.
        self._path_feasible_cache = {}
        self._choice_cache = {}
        self._array_summaries = None
        summaries = None
        if self.runtime.options.engine == 'array':
            if not self.batch_evaluation:
                raise ValueError('Scalar workflow oracle requires the reference engine')
            summaries = self.runtime.applications(self.scenario.applications.values(),
                lambda app: self._prepare_application(app, routing, analytical))
        app_latency: dict[str, float] = {}
        app_quality: dict[str, float] = {}
        flow_metrics: dict[str, dict[str, float]] = {}
        goodput = 0.0
        total_arrival = 0.0
        latency_numerator_system = 0.0
        quality_numerator = 0.0
        app_completed = {}

        for app in self.scenario.applications.values():
            if summaries is not None:
                self._array_summaries = next(summaries)
            app_rate = sum(
                (arrival_rates or {}).get((app.id, ingress), base_rate)
                for ingress, base_rate in app.ingress_rates.items()
            )
            total_arrival += app_rate
            latency_numerator = 0.0
            app_quality_numerator = 0.0
            completed_rate = 0.0
            for ingress, base_rate in app.ingress_rates.items():
                ingress_rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    quality_numerator += ingress_rate * model_share * app.quality[model]
                    app_quality_numerator += (
                        ingress_rate * model_share * app.quality[model]
                    )
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
                        completion = self._completion_probability[(app.id, ingress, model, flow.id)]
                        completed_rate += probability * completion
                        if completion > 0:
                            latency_numerator += probability * completion * e2e
                        goodput += probability * slo_probability
                        flow_metrics[f"{app.id}:{ingress}:{model}:{flow.id}"] = {
                            "e2e_s": e2e,
                            "ttft_s": ttft,
                            "tbt_s": tbt,
                            "slo": float(slo_probability),
                            "weight_rps": probability,
                            "completion_probability": completion,
                            "completed_rps": probability * completion,
                            **{
                                f"stage:{node_id}_s": value
                                for node_id, value in stage_times.items()
                            },
                        }
            app_completed[app.id] = completed_rate
            if completed_rate > 0:
                app_latency[app.id] = latency_numerator / completed_rate
            app_quality[app.id] = (
                app_quality_numerator / app_rate if app_rate > 0.0 else 0.0
            )
            latency_numerator_system += latency_numerator

        if summaries is not None:
            # Consume the final yield so normal generator disposal cannot look
            # like an aborted evaluation and close the run-owned process pool.
            if next(summaries, None) is not None:
                raise RuntimeError('Unexpected extra application result')
        mean_latency = (sum(app_latency.values()) / len(app_latency)
                        if app_latency else None)
        quality = quality_numerator / total_arrival if total_arrival > 0.0 else 0.0
        workflow_result = WorkflowResult(
            app_latency,
            mean_latency,
            goodput,
            quality,
            total_arrival,
            flow_metrics,
            app_quality,
            sum(app_completed.values()),
            app_completed,
        )
        self._batch_kernel = None
        self._array_summaries = None
        self._choice_cache = {}
        return workflow_result

    def _prepare_application(self, app, routing, result):
        if self._batch_kernel is None:
            self._batch_kernel = IndexedWorkflowKernel(self, result)
        kernel = self._batch_kernel
        requests = []
        # The cache is evaluation-local and bounded to the current application.
        self._choice_cache = {}
        llm_choices = {}
        for ingress in app.ingress_rates:
            for model in self.scenario.models:
                share = routing.model_share.get((app.id, ingress, model), 0.0)
                if share <= 0:
                    continue
                for flow in app.pattern_flows:
                    nodes = tuple(sorted(flow.nodes))
                    choices = []
                    for name in nodes:
                        node = app.nodes[name]
                        key = (ingress, model, name)
                        if node.type is NodeType.LLM and key in llm_choices:
                            choices.append(llm_choices[key])
                            continue
                        raw = []
                        if node.type is NodeType.LLM:
                            for cid, candidate in self.scenario.candidates.items():
                                if candidate.model == model:
                                    p = routing.llm_share.get((app.id, ingress, name, cid), 0.0) / share
                                    if p > 0.0:
                                        raw.append((kernel.server_index.get(candidate.server, 0), kernel.candidate_index[cid], p))
                        else:
                            distribution = result.node_server_distribution.get((app.id, ingress, model, flow.id, name), {})
                            raw = [(kernel.server_index.get(s, 0), 0, p) for s, p in distribution.items() if p > 0.0]
                        if not raw:
                            raw = [(0, 0, 1.0)]
                        total = sum(item[2] for item in raw)
                        normalized = tuple((s, c, p / total) for s, c, p in raw)
                        if normalized not in self._choice_cache:
                            self._choice_cache[normalized] = (
                                np.asarray([v[0] for v in normalized], dtype=np.int32),
                                np.asarray([v[1] for v in normalized], dtype=np.int32),
                                np.asarray([v[2] for v in normalized], dtype=np.float64))
                        choice = self._choice_cache[normalized]
                        if node.type is NodeType.LLM:
                            llm_choices[key] = choice
                        choices.append(choice)
                    requests.append(((app.id, ingress, model, flow.id), nodes, choices, flow))
        context = kernel.compact(app, [(r[0][1], r[0][2], r[3]) for r in requests])
        return app, context, requests, self.max_mapping_samples

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

    def _flow_summary(self, app_id, ingress, model, flow_id, routing, result):
        if self._array_summaries is not None:
            key = (app_id, ingress, model, flow_id)
            value, completed = self._array_summaries[key]
            self._completion_probability[key] = completed
            return value
        if not self.batch_evaluation:
            return self._flow_summary_scalar(app_id, ingress, model, flow_id, routing, result)
        app = self.scenario.applications[app_id]
        flow = next(flow for flow in app.pattern_flows if flow.id == flow_id)
        outcomes = self._physical_mapping_outcomes(app_id, ingress, model, flow_id, routing, result)
        if self._batch_kernel is None or self._batch_kernel.result is not result:
            self._batch_kernel = IndexedWorkflowKernel(self, result)
        batch = self._batch_kernel.evaluate(app_id, ingress, model, flow, outcomes)
        e2e_sum = ttft_sum = tbt_sum = slo_sum = completed = 0.0
        stage_sums = {}
        # Keep scalar float64 additions in exactly the original sample order.
        for row, (_, probability) in enumerate(outcomes):
            if not batch.valid[row]:
                continue
            completed += probability
            e2e_sum += probability * float(batch.e2e[row])
            ttft_sum += probability * float(batch.ttft[row])
            tbt_sum += probability * float(batch.tbt[row])
            if batch.slo[row]:
                slo_sum += probability
            for name, column in batch.stages:
                stage_sums[name] = stage_sums.get(name, 0.0) + probability * float(batch.completions[row, column])
        self._completion_probability[(app_id, ingress, model, flow_id)] = completed
        if not completed:
            return None, None, None, 0.0, {}
        return (e2e_sum / completed, ttft_sum / completed, tbt_sum / completed,
                slo_sum, {n: value / completed for n, value in stage_sums.items()})

    def _flow_summary_scalar(
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
        e2e_sum = ttft_sum = tbt_sum = 0.0
        slo_sum = 0.0
        completed = 0.0
        stage_sums: dict[str, float] = {}
        for mapping, probability in outcomes:
            if not self._mapping_complete(app_id, ingress, flow, mapping, result):
                continue
            e2e, ttft, tbt, stage_times = self._mapping_latency(
                app_id, ingress, model, flow, mapping, result
            )
            if not all(math.isfinite(v) for v in (e2e, ttft, tbt, *stage_times.values())):
                continue
            completed += probability
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
        self._completion_probability[(app_id, ingress, model, flow_id)] = completed
        if not completed:
            return None, None, None, 0.0, {}
        return (e2e_sum / completed, ttft_sum / completed, tbt_sum / completed,
                slo_sum, {n: value / completed for n, value in stage_sums.items()})

    def _mapping_complete(self, app_id, ingress, flow, mapping, result):
        """Completion feasibility is separate from satisfying a latency SLO."""
        app = self.scenario.applications[app_id]
        for node_id in flow.nodes:
            server, candidate = mapping.get(node_id, ("", None))
            if not server:
                return False
            node = app.nodes[node_id]
            if node.type is NodeType.LLM:
                perf = result.llm_performance.get((app_id, node_id, candidate or ""))
                instance = result.llm_instance_performance.get(candidate)
                if (perf is None or not math.isfinite(perf.response_s)
                        or result.llm_utilization.get(candidate, 0) >= 1
                        or result.llm_kv_stable.get(candidate) is False
                        or (instance is not None and not instance.stable)):
                    return False
            else:
                pool = (node.tool or "", server)
                if (pool not in result.tool_delay or not math.isfinite(result.tool_delay[pool])
                        or result.tool_utilization.get(pool, 0) >= 1):
                    return False
        _, roots = _workflow_topology(tuple(flow.nodes), tuple(flow.edges))
        transfers = [(ingress, mapping[n][0]) for n in roots]
        transfers += [(mapping[u][0], mapping[v][0]) for u, v in flow.edges]
        transfers.append((mapping[flow.final_node][0], ingress))
        return all(self._path_feasible(u, v, result) for u, v in transfers)

    def _path_feasible(self, source, target, result):
        key = (source, target)
        if key not in self._path_feasible_cache:
            try:
                self._path_feasible_cache[key] = all(
                    result.link_load_mbps.get(edge, 0) < self.backend.network.links[edge].capacity_mbps
                    for edge in self.backend.network.path(source, target))
            except ValueError:
                self._path_feasible_cache[key] = False
        return self._path_feasible_cache[key]

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
        if cardinality <= self.max_mapping_samples:
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
        prepared = []
        for node_choices in choices:
            probabilities = np.asarray([item[2] for item in node_choices], dtype=float)
            probabilities /= probabilities.sum()
            prepared.append((node_choices, probabilities))
        for _ in range(self.max_mapping_samples):
            mapping = {}
            for node_choices, probabilities in prepared:
                index = int(rng.choice(len(node_choices), p=probabilities))
                node_id, location, _ = node_choices[index]
                mapping[node_id] = location
            outcomes.append((mapping, 1.0 / self.max_mapping_samples))
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

        ordered_parents, _ = _workflow_topology(tuple(flow.nodes), tuple(flow.edges))
        starts = {}
        completions = {}
        for node_id, parents in ordered_parents:
            destination = location(node_id)
            if not parents:
                starts[node_id] = overload if not destination else self._path_delay(ingress, destination, result)
            else:
                starts[node_id] = max(completions[parent] + (
                    self._path_delay(location(parent), destination, result)
                    if location(parent) and destination else overload
                ) for parent in parents)
            completions[node_id] = starts[node_id] + node_response(node_id)
        critical_prefix = starts[flow.final_node]
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
        exit_delay = overload if not final_server else self._path_delay(
            final_server, ingress, result
        )
        output_tokens = max(1.0, app.nodes[flow.final_node].output_tokens[model])
        token_data = app.exit_payload(flow.id, model) / output_tokens
        first_token_return = overload if not final_server else self._first_token_delay(
            final_server, ingress, token_data
        )
        stage_times: dict[str, float] = {}
        for node in app.nodes.values():
            if node.type is not NodeType.LLM or node.id == flow.final_node or node.id not in flow.nodes:
                continue
            stage_times[node.id] = completions[node.id]
        return (
            critical_prefix + final_response + exit_delay,
            critical_prefix + final_ttft + first_token_return,
            final_tbt,
            stage_times,
        )

    def _path_delay(
        self,
        source: str,
        target: str,
        result: AnalyticalResult,
    ) -> float:
        key = (source, target)
        if key not in self._path_delay_cache:
            self._path_delay_cache[key] = self.backend.network.path_delay(
                source, target, result.link_load_mbps
            )
        return self._path_delay_cache[key]

    def _first_token_delay(
        self,
        source: str,
        target: str,
        token_data_mb: float,
    ) -> float:
        key = (source, target, float(token_data_mb))
        if key not in self._first_token_delay_cache:
            self._first_token_delay_cache[key] = (
                self.backend.network.first_token_return_delay(
                    source, target, token_data_mb
                )
            )
        return self._first_token_delay_cache[key]

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
            * self._path_delay(u, v, result)
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
            * self._path_delay(ingress, server, result)
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
            * self._path_delay(server, ingress, result)
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
        token_data = app.exit_payload(flow_id, model) / output_tokens
        distribution = result.node_server_distribution.get(
            (app_id, ingress, model, flow_id, final_node), {}
        )
        return sum(
            probability
            * self._first_token_delay(server, ingress, token_data)
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

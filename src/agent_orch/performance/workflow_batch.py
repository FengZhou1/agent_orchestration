"""Indexed, bounded-memory evaluation of the existing physical mapping samples.

No sampling or probability aggregation lives here. Float64 operations follow
the scalar recurrence; the caller retains its original sequential weighted sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from weakref import proxy

import networkx as nx
import numpy as np

from agent_orch.schema.models import NodeType, SLOType
from .workflow_mapping import MappingBatch


@dataclass(frozen=True)
class IndexedPattern:
    nodes: tuple[str, ...]
    ordered_parents: tuple[tuple[int, tuple[int, ...]], ...]
    roots: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    final: int


@lru_cache(maxsize=4096)
def indexed_pattern(nodes, edges, final):
    """Compile actual graph values, never a graph ID alone."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    index = {node: i for i, node in enumerate(nodes)}
    return IndexedPattern(nodes,
        tuple((index[node], tuple(index[p] for p in graph.predecessors(node)))
              for node in nx.lexicographical_topological_sort(graph)),
        tuple(index[node] for node in graph if graph.in_degree(node) == 0),
        tuple((index[u], index[v]) for u, v in edges), index[final])


@dataclass
class BatchValues:
    valid: np.ndarray
    slo: np.ndarray
    e2e: np.ndarray
    ttft: np.ndarray
    tbt: np.ndarray
    completions: np.ndarray
    stages: tuple[tuple[str, int], ...]


class IndexedWorkflowKernel:
    """Numeric lookup tables live for one analytical evaluation only.

    Only the current application's LLM tables are retained. Each mapping batch
    is at most the unchanged sampler's bound, not all applications or a rollout.
    Index zero denotes a missing/unknown location or instance.
    """

    def __init__(self, evaluator, result):
        self.evaluator = proxy(evaluator)
        self.scenario = evaluator.scenario
        self.result = result
        self.overload = self.scenario.simulation.overload_delay_s
        self.servers = ('', *self.scenario.servers)
        self.server_index = {name: i for i, name in enumerate(self.servers)}
        self.candidates = ('', *self.scenario.candidates)
        self.candidate_index = {name: i for i, name in enumerate(self.candidates)}
        self.tool_index = {name: i for i, name in enumerate(self.scenario.tools)}
        self.tool_response = np.full((len(self.tool_index), len(self.servers)), self.overload, dtype=np.float64)
        self.tool_valid = np.zeros(self.tool_response.shape, dtype=bool)
        for tool, row in self.tool_index.items():
            for column, server in enumerate(self.servers[1:], 1):
                pool = (tool, server)
                if pool in result.tool_delay:
                    value = result.tool_delay[pool]
                    self.tool_response[row, column] = value
                    self.tool_valid[row, column] = math.isfinite(value) and not (result.tool_utilization.get(pool, 0) >= 1)
        self.network_delay = None
        self.network_valid = None
        self._app_id = None
        self._llm_table = None

    def _network(self):
        if self.network_delay is not None:
            return
        count = len(self.servers)
        self.network_delay = np.full((count, count), self.overload, dtype=np.float64)
        self.network_valid = np.zeros((count, count), dtype=bool)
        for i, source in enumerate(self.servers[1:], 1):
            for j, target in enumerate(self.servers[1:], 1):
                valid = self.evaluator._path_feasible(source, target, self.result)
                self.network_valid[i, j] = valid
                if valid:
                    self.network_delay[i, j] = self.evaluator._path_delay(source, target, self.result)

    def _llm(self, app):
        if self._app_id != app.id:
            self._app_id = app.id
            self._llm_table = None
        if self._llm_table is not None:
            return self._llm_table
        nodes = {n.id: i for i, n in enumerate(n for n in app.nodes.values() if n.type is NodeType.LLM)}
        shape = (len(nodes), len(self.candidates))
        response = np.full(shape, self.overload, dtype=np.float64)
        ttft, tbt = response.copy(), response.copy()
        valid = np.zeros(shape, dtype=bool)
        for node, row in nodes.items():
            for column, candidate in enumerate(self.candidates[1:], 1):
                perf = self.result.llm_performance.get((app.id, node, candidate))
                if perf is None:
                    continue
                response[row, column], ttft[row, column], tbt[row, column] = perf.response_s, perf.ttft_s, perf.tbt_s
                instance = self.result.llm_instance_performance.get(candidate)
                valid[row, column] = (math.isfinite(perf.response_s)
                    and not (self.result.llm_utilization.get(candidate, 0) >= 1)
                    and self.result.llm_kv_stable.get(candidate) is not False
                    and (instance is None or instance.stable))
        tables = nodes, response, ttft, tbt, valid
        # Candidate IDs already identify their models; do not duplicate this
        # all-candidate table when the same application's model group changes.
        self._llm_table = tables
        return tables

    def evaluate(self, app_id, ingress, model, flow, outcomes):
        app = self.scenario.applications[app_id]
        plan = indexed_pattern(tuple(flow.nodes), tuple(flow.edges), flow.final_node)
        if isinstance(outcomes, MappingBatch):
            permutation = [outcomes.nodes.index(node) for node in plan.nodes]
            servers = outcomes.servers[:, permutation]
            candidates = outcomes.candidates[:, permutation]
            return self.evaluate_arrays(app, ingress, model, flow, servers, candidates)
        count, width = len(outcomes), len(plan.nodes)
        servers = np.zeros((count, width), dtype=np.int32)
        candidates = np.zeros((count, width), dtype=np.int32)
        for row, (mapping, _) in enumerate(outcomes):
            for column, node in enumerate(plan.nodes):
                server, candidate = mapping.get(node, ('', None))
                servers[row, column] = self.server_index.get(server, 0)
                candidates[row, column] = self.candidate_index.get(candidate, 0)

        return self.evaluate_arrays(app, ingress, model, flow, servers, candidates)

    def evaluate_arrays(self, app, ingress, model, flow, servers, candidates):
        plan = indexed_pattern(tuple(flow.nodes), tuple(flow.edges), flow.final_node)
        count, width = servers.shape

        node_index, llm_response, llm_ttft, llm_tbt, llm_valid = self._llm(app)
        response = np.full((count, width), self.overload, dtype=np.float64)
        valid = np.all(servers != 0, axis=1)
        for column, name in enumerate(plan.nodes):
            node = app.nodes[name]
            if node.type is NodeType.LLM:
                response[:, column] = llm_response[node_index[name], candidates[:, column]]
                valid &= llm_valid[node_index[name], candidates[:, column]]
            else:
                tool = self.tool_index.get(node.tool or '')
                if tool is None:
                    valid[:] = False
                else:
                    response[:, column] = self.tool_response[tool, servers[:, column]]
                    valid &= self.tool_valid[tool, servers[:, column]]

        stages = tuple((node.id, plan.nodes.index(node.id)) for node in app.nodes.values()
                       if node.type is NodeType.LLM and node.id != flow.final_node and node.id in plan.nodes)
        completions = np.zeros_like(response)
        e2e = np.zeros(count, dtype=np.float64)
        ttft, tbt = e2e.copy(), e2e.copy()
        slo = np.zeros(count, dtype=bool)
        if not valid.any():
            return BatchValues(valid, slo, e2e, ttft, tbt, completions, stages)

        self._network()
        ingress_index = self.server_index.get(ingress, 0)
        for root in plan.roots:
            valid &= self.network_valid[ingress_index, servers[:, root]]
        for source, target in plan.edges:
            valid &= self.network_valid[servers[:, source], servers[:, target]]
        final_servers = servers[:, plan.final]
        valid &= self.network_valid[final_servers, ingress_index]
        if not valid.any():
            return BatchValues(valid, slo, e2e, ttft, tbt, completions, stages)

        final_start = None
        with np.errstate(invalid='ignore', over='ignore'):
            for column, parents in plan.ordered_parents:
                if not parents:
                    start = self.network_delay[ingress_index, servers[:, column]]
                else:
                    parent = parents[0]
                    start = completions[:, parent] + self.network_delay[servers[:, parent], servers[:, column]]
                    for parent in parents[1:]:
                        alternative = completions[:, parent] + self.network_delay[servers[:, parent], servers[:, column]]
                        # Python max retains its first operand on ties/NaN comparisons.
                        start = np.where(alternative > start, alternative, start)
                completions[:, column] = start + response[:, column]
                if column == plan.final:
                    final_start = start
            final_row = node_index[flow.final_node]
            final_candidates = candidates[:, plan.final]
            exit_delay = self.network_delay[final_servers, ingress_index]
            e2e = final_start + llm_response[final_row, final_candidates] + exit_delay
            output_tokens = max(1.0, app.nodes[flow.final_node].output_tokens[model])
            token_data = app.exit_payload(flow.id, model) / output_tokens
            first_return = np.full(len(self.servers), self.overload, dtype=np.float64)
            for server in np.unique(final_servers[valid]):
                first_return[server] = self.evaluator._first_token_delay(self.servers[server], ingress, token_data)
            ttft = final_start + llm_ttft[final_row, final_candidates] + first_return[final_servers]
            tbt = llm_tbt[final_row, final_candidates].copy()

        valid &= np.isfinite(e2e) & np.isfinite(ttft) & np.isfinite(tbt)
        stage_ok = np.ones(count, dtype=bool)
        for name, column in stages:
            valid &= np.isfinite(completions[:, column])
            deadline = app.nodes[name].stage_deadline_s
            if deadline is not None:
                stage_ok &= completions[:, column] <= float(deadline)
        contract = app.slo
        if contract.type is SLOType.LATENCY:
            slo = (ttft <= float(contract.ttft_s)) & (tbt <= float(contract.tbt_s))
        elif contract.type is SLOType.DEADLINE:
            slo = e2e <= float(contract.deadline_s)
        else:
            slo = ((e2e <= float(contract.deadline_s)) & (ttft <= float(contract.ttft_s))
                   & (tbt <= float(contract.tbt_s)) & stage_ok)
        return BatchValues(valid, slo & valid, e2e, ttft, tbt, completions, stages)

    def compact(self, app, requests):
        """Application-only numeric context, with shared arrays pickled once/task.

        Dynamic tables are rebuilt on EVERY evaluation after the full global
        loads have been computed. No backend, environment or policy is sent.
        """
        self._network()
        first_returns = {}
        for ingress, model, flow in requests:
            output = max(1.0, app.nodes[flow.final_node].output_tokens[model])
            token_data = app.exit_payload(flow.id, model) / output
            for server in self.servers[1:]:
                key = (server, ingress, token_data)
                if key not in first_returns and self.network_valid[self.server_index[server], self.server_index[ingress]]:
                    first_returns[key] = self.evaluator._first_token_delay(*key)
        return CompactKernel(self.overload, self.servers, self.server_index,
            self.tool_index, self.tool_response, self.tool_valid,
            self.network_delay, self.network_valid, self._llm(app), first_returns)


@dataclass
class CompactKernel:
    overload: float
    servers: tuple
    server_index: dict
    tool_index: dict
    tool_response: np.ndarray
    tool_valid: np.ndarray
    network_delay: np.ndarray
    network_valid: np.ndarray
    llm_tables: tuple
    first_returns: dict

    # Reuse exactly the reference array recurrence, not a second approximation.
    evaluate_arrays = IndexedWorkflowKernel.evaluate_arrays

    @property
    def evaluator(self):
        return self

    def _llm(self, app):
        return self.llm_tables

    def _network(self):
        pass

    def _first_token_delay(self, source, target, data):
        return self.first_returns[(source, target, data)]

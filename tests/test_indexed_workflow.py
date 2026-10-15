from dataclasses import replace
import math

import numpy as np
import pytest

from agent_orch.performance.analytical import AnalyticalBackend, AnalyticalResult
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.performance.workflow_batch import IndexedWorkflowKernel
from agent_orch.performance.workflow_mapping import MappingBatch
from agent_orch.schema.models import (
    LLMClassPerformance, NodeType, PatternFlow, RoutingDecision, SLOSpec, SLOType, WorkflowNode,
)


@pytest.fixture
def parallel_case(scenario):
    original = next(iter(scenario.applications.values()))
    model = max(scenario.models, key=lambda m: sum(c.model == m for c in scenario.candidates.values()))
    candidates = [c for c in scenario.candidates.values() if c.model == model]
    assert len(candidates) >= 2
    tool = next(iter(scenario.tools))
    nodes = {name: WorkflowNode(name, NodeType.LLM,
        prompt_tokens={m: 256. for m in scenario.models}, output_tokens={m: 8. for m in scenario.models},
        stage_deadline_s=10.) for name in ('root', 'b', 'join')}
    nodes['a'] = WorkflowNode('a', NodeType.TOOL, tool=tool)
    flow = PatternFlow('fork', 1., 'join', (), ('root', 'a', 'b', 'join'),
        (('root', 'a'), ('root', 'b'), ('a', 'join'), ('b', 'join')))
    ingress = next(iter(scenario.servers))
    app = replace(original, nodes=nodes, pattern_flows=(flow,), ingress_rates={ingress: .1},
        slo=SLOSpec(SLOType.COMPOUND, ttft_s=10., tbt_s=1., deadline_s=10.))
    scene = replace(scenario, applications={app.id: app})
    performance = {}
    for i, node in enumerate(('root', 'b', 'join')):
        for j, candidate in enumerate(candidates):
            response = .2 + i * .13 + j * .071
            performance[(app.id, node, candidate.id)] = LLMClassPerformance(response, .03, response-.03,
                .03+j*.01, .02+j*.003, response)
    servers = list(scene.servers)
    result = AnalyticalResult(performance, {c.id: .1 for c in candidates}, {c.id: True for c in candidates},
        {(tool, n): .07+i*.03 for i, n in enumerate(servers)},
        {(tool, n): .1 for n in servers}, {}, {})
    rng = np.random.default_rng(182)
    outcomes = []
    for _ in range(17):
        mapping = {}
        for name in flow.nodes:
            if nodes[name].type is NodeType.LLM:
                candidate = candidates[int(rng.integers(len(candidates)))]
                mapping[name] = candidate.server, candidate.id
            else:
                mapping[name] = str(rng.choice(servers)), None
        outcomes.append((mapping, float(rng.random())))
    total = sum(p for _, p in outcomes)
    outcomes = [(mapping, p / total) for mapping, p in outcomes]
    routing = RoutingDecision({(app.id, ingress, model): 1.}, {}, {})
    return scene, app, model, ingress, flow, result, routing, outcomes


def compare(scene, app, model, ingress, flow, result, routing, outcomes):
    scalar = WorkflowEvaluator(scene, AnalyticalBackend(scene), batch_evaluation=False)
    batch = WorkflowEvaluator(scene, AnalyticalBackend(scene), batch_evaluation=True)
    scalar._physical_mapping_outcomes = lambda *args: outcomes
    batch._physical_mapping_outcomes = lambda *args: outcomes
    # Full workflow objects include completion-conditioned delay and goodput.
    expected = scalar.evaluate(None, routing, result)
    actual = batch.evaluate(None, routing, result)
    assert actual == expected
    assert batch._completion_probability == scalar._completion_probability
    kernel = IndexedWorkflowKernel(batch, result)
    values = kernel.evaluate(app.id, ingress, model, flow, outcomes)
    servers = np.asarray([[kernel.server_index.get(mapping.get(n, ('', None))[0], 0)
                           for n in flow.nodes] for mapping, _ in outcomes], dtype=np.int32).reshape(len(outcomes), len(flow.nodes))
    candidates = np.asarray([[kernel.candidate_index.get(mapping.get(n, ('', None))[1], 0)
                              for n in flow.nodes] for mapping, _ in outcomes], dtype=np.int32).reshape(len(outcomes), len(flow.nodes))
    direct = kernel.evaluate(app.id, ingress, model, flow,
        MappingBatch(tuple(flow.nodes), servers, candidates, np.asarray([p for _, p in outcomes])))
    compact = kernel.compact(app, [(ingress, model, flow)]).evaluate_arrays(app, ingress, model, flow, servers, candidates)
    for alternative in (direct, compact):
        for name in ('valid', 'slo', 'e2e', 'ttft', 'tbt', 'completions'):
            np.testing.assert_array_equal(getattr(values, name), getattr(alternative, name))
    for row, (mapping, _) in enumerate(outcomes):
        if not scalar._mapping_complete(app.id, ingress, flow, mapping, result):
            assert not values.valid[row]
            continue
        e2e, ttft, tbt, stages = scalar._mapping_latency(app.id, ingress, model, flow, mapping, result)
        finite = all(math.isfinite(v) for v in (e2e, ttft, tbt, *stages.values()))
        assert values.valid[row] == finite
        if finite:
            # Bit-level comparisons also catch signed-zero/arithmetic order drift.
            for a, b in zip((values.e2e[row], values.ttft[row], values.tbt[row]), (e2e, ttft, tbt)):
                assert np.float64(a).tobytes() == np.float64(b).tobytes()
            for name, column in values.stages:
                assert np.float64(values.completions[row, column]).tobytes() == np.float64(stages[name]).tobytes()
    return actual


@pytest.mark.parametrize('kind', list(SLOType))
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_batch_matches_scalar_at_exact_and_adjacent_slo_boundaries(parallel_case, kind, direction):
    scene, app, model, ingress, flow, result, routing, outcomes = parallel_case
    scalar = WorkflowEvaluator(scene, AnalyticalBackend(scene), batch_evaluation=False)
    e2e, ttft, tbt, _ = scalar._mapping_latency(app.id, ingress, model, flow, outcomes[0][0], result)
    def threshold(value):
        return float(np.nextafter(value, np.inf if direction > 0 else -np.inf)) if direction else value
    app = replace(app, slo=SLOSpec(kind, threshold(ttft), threshold(tbt), threshold(e2e)))
    scene = replace(scene, applications={app.id: app})
    compare(scene, app, model, ingress, flow, result, routing, outcomes)


@pytest.mark.parametrize('failure', ['missing', 'unknown_server', 'tool', 'llm', 'kv', 'network', 'nonfinite', 'intermediate_ttft',
                                    'llm_nan_util', 'tool_nan_util', 'disconnected'])
def test_completion_and_failure_boundaries(parallel_case, failure):
    scene, app, model, ingress, flow, result, routing, outcomes = parallel_case
    if failure == 'missing':
        bad = dict(outcomes[0][0])
        bad.pop('join')
        outcomes = [(bad, .2), *[(m, p*.8) for m, p in outcomes]]
    elif failure == 'unknown_server':
        bad = dict(outcomes[0][0])
        bad['root'] = ('not-a-server', bad['root'][1])
        outcomes = [(bad, .2), *[(m, p*.8) for m, p in outcomes]]
    elif failure in ('tool', 'tool_nan_util'):
        result = replace(result, tool_utilization={next(iter(result.tool_delay)): math.nan if failure=='tool_nan_util' else 1.})
    elif failure in ('llm', 'kv'):
        candidate = next(iter(result.llm_utilization))
        result = replace(result, **({'llm_utilization': {candidate: 1.}} if failure == 'llm' else
                                   {'llm_kv_stable': {candidate: False}}))
    elif failure == 'network':
        result = replace(result, link_load_mbps={(l.source, l.target): l.capacity_mbps for l in scene.links})
    elif failure == 'disconnected':
        scene = replace(scene, links=())
    elif failure == 'llm_nan_util':
        result = replace(result, llm_utilization={next(iter(result.llm_utilization)): math.nan})
    else:
        performance = {k: replace(v) for k, v in result.llm_performance.items()}
        for key, value in performance.items():
            if key[1] == ('join' if failure == 'nonfinite' else 'root'):
                value.ttft_s = math.inf
        result = replace(result, llm_performance=performance)
    compare(scene, app, model, ingress, flow, result, routing, outcomes)


def test_pure_inference_single_output_and_empty_mappings(parallel_case):
    scene, app, model, ingress, _, result, routing, outcomes = parallel_case
    final = replace(app.nodes['join'], output_tokens={m: 1. for m in scene.models})
    flow = PatternFlow('pure', 1., 'join', (('join',),))
    app = replace(app, nodes={'join': final}, pattern_flows=(flow,))
    scene = replace(scene, applications={app.id: app})
    result = replace(result, llm_performance={k: replace(v, tbt_s=0.) for k,v in result.llm_performance.items() if k[1]=='join'})
    outcomes = [({'join': mapping['join']}, p) for mapping,p in outcomes]
    compare(scene, app, model, ingress, flow, result, routing, outcomes)
    empty = compare(scene, app, model, ingress, flow, result, routing, [])
    assert empty.mean_latency_s is None and empty.goodput_rps == 0.


def test_tables_are_rebuilt_when_evaluation_loads_change(parallel_case):
    scene, app, model, ingress, flow, result, routing, outcomes = parallel_case
    evaluator = WorkflowEvaluator(scene, AnalyticalBackend(scene))
    evaluator._physical_mapping_outcomes = lambda *args: outcomes
    before = evaluator.evaluate(None, routing, result)
    assert evaluator._batch_kernel is None
    bad = replace(result, llm_utilization={c: 1. for c in result.llm_utilization})
    assert evaluator.evaluate(None, routing, bad).completed_rps == 0.
    assert evaluator.evaluate(None, routing, result) == before


def test_multiple_entry_nodes_preserve_join_wait(parallel_case):
    scene, app, model, ingress, flow, result, routing, outcomes = parallel_case
    flow = replace(flow, graph_edges=tuple(e for e in flow.edges if e != ('root', 'b')))
    app = replace(app, pattern_flows=(flow,))
    scene = replace(scene, applications={app.id: app})
    compare(scene, app, model, ingress, flow, result, routing, outcomes)

from __future__ import annotations

import copy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import NodeType, PatternFlow
from agent_orch.workload.agent_graph import GraphExpander, compose, validate_graph
from agent_orch.workload.agent_lengths import generate_lengths, merge_messages
from agent_orch.workload.agent_pool import (
    GeneratorConfig, generate_pool, digest, pool_from_mapping, scenario_mapping,
    rebind_pool, same_application,
)


@pytest.fixture(scope="module")
def small_pool():
    template = ScenarioLoader.load("configs/toy.yaml")
    config = GeneratorConfig(applications=20, tool_types=30, calibration_passes=0)
    return template, generate_pool(template, config)


@pytest.mark.parametrize("seed", list(range(20)))
def test_expansion_conserves_size_and_positive_patterns(seed):
    graph = GraphExpander(np.random.default_rng(seed)).expand(8, 15)
    stats = validate_graph(graph)
    assert stats["llm_nodes"] == 8
    assert stats["tool_nodes"] == 15
    assert stats["patterns"] <= 32
    assert sum(p.probability for p in graph.patterns) == pytest.approx(1)


def test_and_is_not_normalized_and_or_keeps_exact_alternatives():
    expander = GraphExpander(np.random.default_rng(2))
    a, b = expander.atom("tool"), expander.atom("tool")
    both = compose([a, b], "parallel")
    assert len(both.variants) == 1
    assert len(both.variants[0].nodes) == 2
    assert both.variants[0].probability == 1
    either = compose([a, b], "exclusive", [.25, .75])
    assert [len(v.nodes) for v in either.variants] == [1, 1]
    assert [v.probability for v in either.variants] == [.25, .75]
    choices = compose([either, either], "serial")
    assert [v.probability for v in choices.variants] == [.0625, .1875, .1875, .5625]


def test_single_llm_exception_and_unsupported_tool_case():
    g = GraphExpander(np.random.default_rng(0)).expand(1, 0)
    assert validate_graph(g)["depth"] == 1
    with pytest.raises(ValueError):
        GraphExpander(np.random.default_rng(0)).expand(1, 1)


def test_parallel_cap_does_not_limit_full_width():
    widths = [validate_graph(GraphExpander(np.random.default_rng(s)).expand(5, 8))["width"]
              for s in range(20)]
    assert max(widths) > 4


def test_joint_lengths_reuse_prefix_messages_and_preserve_all_patterns():
    graph = GraphExpander(np.random.default_rng(1)).expand(5, 8)
    lengths = generate_lengths(graph, 71)
    assert len(lengths["patterns"]) == len(graph.patterns)
    roots = [record["calls"]["l0"]["input_tokens"] for record in lengths["patterns"].values()]
    assert len(set(roots)) == 1
    assert merge_messages([{"prefix": 100, "left": 30}, {"prefix": 100, "right": 40}]) == {
        "prefix": 100, "left": 30, "right": 40}
    for record in lengths["patterns"].values():
        assert record["total_tokens"] <= 128000
        for node, call in record["calls"].items():
            assert sum(call["input_messages"].values()) == call["input_tokens"]
            assert call["input_tokens"] + call["output_tokens"] <= 32768
            assert call["new_input_tokens"] >= 0


def test_generation_is_deterministic_and_roundtrips_without_resampling(small_pool):
    template, pool = small_pool
    again = generate_pool(template, pool.config)
    assert digest(pool.to_dict()) == digest(again.to_dict())
    loaded = pool_from_mapping(pool.to_dict(), template)
    assert digest(loaded.to_dict()) == digest(pool.to_dict())
    raw = scenario_mapping(pool.scenario(template, 10))
    assert scenario_mapping(ScenarioLoader.from_mapping(raw)) == raw


def test_nested_subsets_and_no_unused_tools(small_pool):
    template, pool = small_pool
    first, second = pool.scenario(template, 5), pool.scenario(template, 10)
    assert list(first.applications) == list(second.applications)[:5]
    for scene in (first, second):
        used = {n.tool for a in scene.applications.values() for n in a.nodes.values() if n.type is NodeType.TOOL}
        assert used == set(scene.tools)
    stats = pool.validate(template)
    assert stats["application_equal_pattern_weighted"]["execution_llms"]["p50"] == 2


def test_application_identity_ignores_identifier_not_payload(small_pool):
    _, pool = small_pool
    original = pool.applications[0]
    renamed = copy.deepcopy(original)
    renamed.spec = replace(renamed.spec, id="a-different-name")
    assert same_application(original, renamed)
    changed = copy.deepcopy(renamed)
    edges = dict(changed.spec.edge_data_mb)
    if edges:
        key = next(iter(edges))
        edges[key] += 1
        changed.spec = replace(changed.spec, edge_data_mb=edges)
    else:
        changed.spec = replace(changed.spec, entry_data_mb={k: v + 1 for k, v in changed.spec.entry_data_mb.items()})
    assert not same_application(original, changed)


def test_rebinding_only_changes_service_binding_and_profiles(small_pool):
    template, pool = small_pool
    variant = rebind_pool(pool, template, 12)
    assert len(variant.tools) == 12
    for a, b in zip(pool.applications, variant.applications):
        assert a.graph == b.graph
        assert a.lengths == b.lengths
        assert a.spec.slo == b.spec.slo
        assert a.spec.quality == b.spec.quality
        assert a.spec.ingress_rates == b.spec.ingress_rates
        for n in a.spec.nodes:
            assert a.spec.nodes[n].prompt_tokens == b.spec.nodes[n].prompt_tokens
            assert a.spec.nodes[n].output_tokens == b.spec.nodes[n].output_tokens
    assert len({t.memory_gb for t in variant.tools.values()}) == 1
    assert len({tuple(t.service_rate.values()) for t in variant.tools.values()}) == 1


def test_generator_never_filters_on_greedy_slo_success(small_pool, monkeypatch):
    template, _ = small_pool
    from agent_orch.capacity import CapacityPlanner
    def forbidden(*args, **kwargs):
        raise AssertionError("Generator must not ask a planner or policy")
    monkeypatch.setattr(CapacityPlanner, "initial_deployment", forbidden)
    pool = generate_pool(template, GeneratorConfig(applications=6, tool_types=8, calibration_passes=0))
    assert len(pool.applications) == 6


def test_compact_graph_matches_path_enumeration_in_workflow(small_pool):
    template, pool = small_pool
    compact = pool.scenario(template, 2)
    apps = {}
    for app in compact.applications.values():
        flows = []
        for flow in app.pattern_flows:
            graph = nx.DiGraph()
            graph.add_nodes_from(flow.nodes)
            graph.add_edges_from(flow.edges)
            paths = []
            for source in flow.sources:
                if source == flow.final_node:
                    paths.append((source,))
                else:
                    paths.extend(tuple(p) for p in nx.all_simple_paths(graph, source, flow.final_node))
            flows.append(PatternFlow(flow.id, flow.probability, flow.final_node, tuple(paths)))
        apps[app.id] = replace(app, pattern_flows=tuple(flows))
    legacy = replace(compact, applications=apps)
    from agent_orch.capacity import CapacityPlanner
    from agent_orch.routing import CapacityProportionalRouter
    from agent_orch.simulator import Simulator
    placement = CapacityPlanner(compact).initial_deployment()
    shares = {(a.id, g, m): 1 / len(compact.models) for a in compact.applications.values()
              for g in a.ingress_rates for m in compact.models}
    routing = CapacityProportionalRouter(compact).route(placement, shares)
    a = Simulator(compact, max_mapping_samples=8).step(placement, routing).metrics
    b = Simulator(legacy, max_mapping_samples=8).step(placement, routing).metrics
    assert a.mean_latency_s == pytest.approx(b.mean_latency_s)
    assert a.goodput_rps == pytest.approx(b.goodput_rps)


def test_invalid_tampered_budget_is_rejected(small_pool):
    template, pool = small_pool
    raw = pool.to_dict()
    raw["applications"][0]["lengths"]["patterns"]["p0"]["total_tokens"] = 128001
    with pytest.raises(ValueError, match="budget"):
        pool_from_mapping(raw, template)


def test_generated_scenario_uses_fixed_slot_workload_and_production_env(small_pool):
    template, pool = small_pool
    scene = pool.scenario(template, 2)
    from agent_orch.workload import SlotTrajectory
    from agent_orch.envs import SlotSequentialJointEnv
    trace = SlotTrajectory.sample(scene, 2, 71)
    assert trace.slots[0].prompt_tokens == trace.slots[1].prompt_tokens
    assert trace.slots[0].flow_probabilities == trace.slots[1].flow_probabilities
    env = SlotSequentialJointEnv(scene, trace, mapping_samples=2)
    env.training_phase = "composition"
    obs, _ = env.reset(seed=0)
    count = 0
    while True:
        obs, _, done, _, info = env.step({"model": np.ones(env.layout.model_action_size)})
        count += 1
        if info.get("period_complete"):
            assert info["physical_slot"] == 0
            break
    assert count == len(env.layout.model_groups)


def test_control_semantics_tampering_is_rejected(small_pool):
    template, pool = small_pool
    raw = pool.to_dict()
    record = next(r for r in raw["applications"] if r["control"]["op"] == "response_or_extend")
    record["control"]["direct_probability"] *= .5
    with pytest.raises(ValueError, match="control semantics"):
        pool_from_mapping(raw, template)


def test_output_payload_is_pattern_specific(small_pool):
    _, pool = small_pool
    for app in pool.applications:
        for flow in app.spec.pattern_flows:
            for model in app.spec.quality:
                assert app.spec.exit_payload(flow.id, model) == pytest.approx(
                    app.lengths["outputs"][flow.final_node] * 4 / 1e6)

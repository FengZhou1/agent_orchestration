from dataclasses import replace
import json
import math

import numpy as np
import pytest

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlanner
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec, ReferenceScales
from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import RoutingDecision, LinkSpec, SLOSpec, SLOType
from agent_orch.simulator import Simulator
from agent_orch.simulator.core import metrics_to_dict
from agent_orch.workload import SlotTrajectory, SlotVariationSpec


def test_network_fixed_window_not_physical_slot():
    links = (LinkSpec('a', 'b', 100, 1),)
    load = {('a', 'b'): 20}
    assert NetworkBackend(links, 1).path_delay('a', 'b', load) == pytest.approx(.201)
    assert NetworkBackend(links, 600).path_delay('a', 'b', load) == pytest.approx(.201)
    assert NetworkBackend(links, 600, accounting_window_s=2).path_delay('a', 'b', load) == pytest.approx(.401)


def test_cost_and_latency_invariant_to_slot_duration(scenario):
    deployment, routing = GreedyPolicy(scenario).decide()
    short = Simulator(scenario).step(deployment, routing).metrics
    long_scene = replace(scenario, simulation=replace(scenario.simulation, slot_seconds=600))
    long = Simulator(long_scene).step(deployment, routing).metrics
    assert short.cost == pytest.approx(long.cost)
    assert short.mean_latency_s == pytest.approx(long.mean_latency_s)
    ref_a = ReferenceScales.from_scenario(scenario)
    ref_b = ReferenceScales.from_scenario(long_scene)
    assert ref_a.cost_max == pytest.approx(ref_b.cost_max)


def test_all_unserved_latency_is_null_not_zero(scenario):
    metrics = Simulator(scenario).step(CapacityPlanner(scenario).empty_deployment(),
                                      RoutingDecision({}, {}, {})).metrics
    assert metrics.total_arrival_rps > 0
    assert metrics.completed_rps == 0
    assert metrics.failed_rps == pytest.approx(metrics.total_arrival_rps)
    assert metrics.mean_latency_s is None
    assert metrics.app_latency_s == {}
    assert metrics.goodput_rps == 0
    assert json.loads(json.dumps(metrics_to_dict(metrics), allow_nan=False))['mean_latency_s'] is None
    value = ObjectiveEvaluator(scenario, ObjectiveSpec.slo_constrained()).evaluate(
        metrics, {(a.id, g): r for a in scenario.applications.values() for g, r in a.ingress_rates.items()})
    assert math.isfinite(value.utility)
    assert value.components['latency_normalized'] == 1


def test_partial_unserved_excluded_from_latency_denominator(scenario):
    dep, route = GreedyPolicy(scenario).decide()
    target = next(iter(scenario.applications))
    route.llm_share = {k: v for k, v in route.llm_share.items() if k[0] != target}
    metrics = Simulator(scenario).step(dep, route).metrics
    assert metrics.app_completed_rps[target] == 0
    assert target not in metrics.app_latency_s
    assert metrics.failed_rps > 0
    assert metrics.mean_latency_s == pytest.approx(np.mean(list(metrics.app_latency_s.values())))


def test_overload_cannot_become_goodput_by_relaxing_slo(scenario):
    apps = {a.id: replace(a, slo=SLOSpec(SLOType.DEADLINE, deadline_s=1e12))
            for a in scenario.applications.values()}
    scene = replace(scenario, applications=apps)
    dep, route = GreedyPolicy(scene).decide()
    sim = Simulator(scene)
    result = sim.backend.evaluate(dep, route)
    result.llm_utilization = {c: 2 for c in result.llm_utilization}
    out = sim.workflow.evaluate(dep, route, result)
    assert out.completed_rps == 0
    assert out.goodput_rps == 0
    assert out.mean_latency_s is None


def test_completed_slo_miss_still_enters_latency(scenario):
    scene = replace(scenario, applications={a.id: replace(a, slo=SLOSpec(SLOType.DEADLINE, deadline_s=1e-12))
                                           for a in scenario.applications.values()})
    dep, route = GreedyPolicy(scene).decide()
    metrics = Simulator(scene).step(dep, route).metrics
    assert metrics.completed_rps > 0
    assert metrics.goodput_rps == 0
    assert metrics.mean_latency_s > 0


def test_cost_above_reference_not_clipped(scenario):
    evaluator = ObjectiveEvaluator(scenario)
    value = evaluator.evaluate_arrays(cost=2 * evaluator.references.cost_max,
        mean_latency=0, attainment=1, quality=1,
        app_latency={a.id: 0 for a in scenario.applications.values()},
        arrival_rates={(a.id, g): r for a in scenario.applications.values() for g, r in a.ingress_rates.items()})
    assert value.components['cost_normalized'] == pytest.approx(2)


def test_deployment_accepts_more_than_four_replicas_with_resource_mask(scenario):
    trajectory = SlotTrajectory.sample(scenario, 1, 31, SlotVariationSpec(
        arrival_range=(1, 1), bandwidth_range=(1, 1), cpu_range=(1, 1), memory_range=(1, 1)))
    env = SlotSequentialJointEnv(scenario, trajectory, mapping_samples=8)
    obs, _ = env.reset(seed=0)
    for _ in scenario.candidates:
        obs, _, _, _, _ = env.step({'deploy': 0})
    kind, tool, server = env._current_demand
    assert kind == 'tool'
    assert scenario.tool_replica_limit(tool, server) > 4
    assert obs['deploy_mask'][5]
    env.step({'deploy': 5})
    assert env.current_deployment.tool_replicas[(tool, server)] == 5
    assert env.planner.deployment_feasible(env.current_deployment)


def test_null_latency_has_finite_observations_rewards_and_differences(scenario):
    env = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 17), mapping_samples=8)
    for episode in range(2):
        observation, _ = env.reset(seed=episode)
        done = False
        while not done:
            assert all(np.isfinite(v).all() for v in observation.values())
            action = {'deploy': 0, 'model': np.zeros(env.layout.model_action_size)}
            observation, reward, done, _, info = env.step(action)
            assert math.isfinite(reward)
        assert info['metrics'].mean_latency_s is None
        assert info['metrics'].completed_rps == 0
        assert info['reward_components']['latency_raw'] is None
        assert info['reward_components']['delta_latency_raw'] is None


def test_reference_places_tools_across_resource_feasible_servers(scenario):
    servers = {n: replace(s, cpu_cores=2) for n, s in scenario.servers.items()}
    template = next(iter(scenario.tools.values()))
    ids = list(scenario.tools) + [f'extra-{i}' for i in range(3)]
    tools = {h: replace(template, id=h, cpu_cores=1, memory_gb=.1,
                        service_rate={n: 1 for n in servers}) for h in ids}
    scene = replace(scenario, servers=servers, tools=tools)
    dep = CapacityPlanner(scene).initial_deployment()
    assert CapacityPlanner(scene).deployment_feasible(dep)
    assert sum(dep.tool_replicas.values()) == len(ids)
    assert len({n for (h,n), z in dep.tool_replicas.items() if z}) > 1


def test_zero_demand_is_not_a_service_failure(scenario):
    scene = replace(scenario, applications={a.id: replace(a, ingress_rates={g: 0 for g in a.ingress_rates})
                                           for a in scenario.applications.values()})
    metrics = Simulator(scene).step(CapacityPlanner(scene).empty_deployment(),
                                      RoutingDecision({}, {}, {})).metrics
    zero = {(a.id,g): 0 for a in scenario.applications.values() for g in a.ingress_rates}
    value = ObjectiveEvaluator(scenario, ObjectiveSpec.slo_constrained()).evaluate(metrics, zero)
    assert value.components['latency_normalized'] == 0
    assert value.constraints[-1] == 0
    assert metrics.completed_rps == metrics.total_arrival_rps == 0
    assert metrics.mean_latency_s is None


def test_partial_mapping_latency_is_conditioned_on_completion(scenario, monkeypatch):
    dep, route = GreedyPolicy(scenario).decide()
    sim = Simulator(scenario)
    result = sim.backend.evaluate(dep, route)
    app = next(iter(scenario.applications.values()))
    ingress = next(iter(app.ingress_rates))
    model = next(m for m in scenario.models if route.model_share.get((app.id, ingress, m), 0) > 0)
    flow = app.pattern_flows[0]
    mappings = sim.workflow._physical_mapping_outcomes(app.id, ingress, model, flow.id, route, result)
    good = next(m for m, _ in mappings if sim.workflow._mapping_complete(app.id, ingress, flow, m, result))
    bad = dict(good)
    bad[flow.final_node] = ('', None)
    expected = sim.workflow._mapping_latency(app.id, ingress, model, flow, good, result)[0]
    monkeypatch.setattr(sim.workflow, '_physical_mapping_outcomes', lambda *args: [(good, .25), (bad, .75)])
    summary = sim.workflow._flow_summary(app.id, ingress, model, flow.id, route, result)
    assert summary[0] == pytest.approx(expected)
    assert sim.workflow._completion_probability[(app.id, ingress, model, flow.id)] == .25
    assert summary[3] <= .25

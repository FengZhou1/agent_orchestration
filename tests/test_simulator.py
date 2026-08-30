import math

import pytest

from agent_orch.baselines import GreedyPolicy, RandomPolicy
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.simulator import Simulator


def _all_finite(value):
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def test_simulator_is_seed_deterministic(scenario):
    policy_a = RandomPolicy(scenario, seed=11)
    dep_a, route_a = policy_a.decide()
    sim_a = Simulator(scenario)
    sim_a.reset(11)
    metrics_a = sim_a.step(dep_a, route_a).metrics

    policy_b = RandomPolicy(scenario, seed=11)
    dep_b, route_b = policy_b.decide()
    sim_b = Simulator(scenario)
    sim_b.reset(11)
    metrics_b = sim_b.step(dep_b, route_b).metrics
    assert metrics_a.cost == pytest.approx(metrics_b.cost)
    assert metrics_a.mean_latency_s == pytest.approx(metrics_b.mean_latency_s)
    assert metrics_a.goodput_rps == pytest.approx(metrics_b.goodput_rps)
    assert metrics_a.quality == pytest.approx(metrics_b.quality)


def test_parallel_flow_uses_maximum_branch_not_sum(scenario):
    policy = GreedyPolicy(scenario, seed=7)
    deployment, routing = policy.decide()
    simulator = Simulator(scenario)
    analytical = simulator.backend.evaluate(deployment, routing)
    evaluator: WorkflowEvaluator = simulator.workflow
    app_id, ingress, model, flow_id = "research-agent", "n2", "large", "parallel-research"
    app = scenario.applications[app_id]
    flow = app.pattern_flows[0]
    prefixes = []
    for chain in flow.chains:
        delay = evaluator._entry_delay(app_id, ingress, model, flow_id, chain[0], analytical)
        for source, target in zip(chain[:-1], chain[1:]):
            delay += evaluator._node_response(
                app_id, ingress, model, flow_id, source, routing, analytical
            )
            delay += evaluator._edge_delay(
                app_id, ingress, model, flow_id, source, target, routing, analytical
            )
        prefixes.append(delay)
    e2e, _, _, _ = evaluator._flow_latency(
        app_id, ingress, model, flow_id, routing, analytical
    )
    final_response = evaluator._final_llm_performance(
        app_id, ingress, model, flow.final_node, routing, analytical
    )[0]
    exit_delay = evaluator._exit_delay(
        app_id, ingress, model, flow_id, flow.final_node, analytical
    )
    assert e2e == pytest.approx(max(prefixes) + final_response + exit_delay)
    assert e2e < sum(prefixes) + final_response + exit_delay


def test_analytical_backend_never_returns_nonfinite_metrics(scenario):
    policy = GreedyPolicy(scenario, seed=7)
    deployment, routing = policy.decide()
    simulator = Simulator(scenario)
    metrics = simulator.step(deployment, routing).metrics
    assert _all_finite(metrics.__dict__)


def test_system_latency_is_request_rate_weighted(scenario):
    policy = GreedyPolicy(scenario, seed=3)
    deployment, routing = policy.decide()
    metrics = Simulator(scenario).step(deployment, routing).metrics
    weighted = sum(
        sum(app.ingress_rates.values()) * metrics.app_latency_s[app.id]
        for app in scenario.applications.values()
    ) / sum(
        sum(app.ingress_rates.values()) for app in scenario.applications.values()
    )
    assert metrics.mean_latency_s == pytest.approx(weighted)

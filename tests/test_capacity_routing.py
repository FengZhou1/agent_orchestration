from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

from agent_orch.capacity import CapacityPlanner
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.performance import analytical
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.models import NodeType, SlotMetrics
from agent_orch.simulator import Simulator
from agent_orch.workload import SlotTrajectory


def shares_of(scenario, fraction=.75):
    return {(app.id, ingress, model): fraction if model == "small" else 1-fraction
            for app in scenario.applications.values() for ingress in app.ingress_rates
            for model in scenario.models}


def test_capacity_formula_uses_global_flow_mixture_and_preserves_model_share(scenario, monkeypatch):
    config = replace(scenario.llm_configs["small-edge"], id="small-fast")
    candidates = dict(scenario.candidates)
    candidates["small-n2"] = replace(candidates["small-n2"], config=config.id)
    scenario = replace(scenario, candidates=candidates,
                       llm_configs={**scenario.llm_configs, config.id: config})
    deployment = CapacityPlanner(scenario).initial_deployment()
    deployment.llm_active["small-n2"] = 1
    captures = []

    def capacity(model, config, classes, weights, chunk):
        captures.append((model.id, config.id, classes, weights))
        return (8. if config.id == "small-fast" else 2.), 1.

    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", capacity)
    shares = shares_of(scenario)
    rates = {(app.id, g): 2. * rate for app in scenario.applications.values()
             for g, rate in app.ingress_rates.items()}
    router = CapacityProportionalRouter(scenario)
    decision = router.route(deployment, shares, arrival_rates=rates)
    for app in scenario.applications.values():
        for g in app.ingress_rates:
            for node in app.nodes.values():
                if node.type is NodeType.LLM:
                    assert decision.llm_share[(app.id, g, node.id, "small-n0")] == pytest.approx(.75 * .2)
                    assert decision.llm_share[(app.id, g, node.id, "small-n2")] == pytest.approx(.75 * .8)
    expected = [app.visit_probability(node.id) * sum(
        rates[(app.id, g)] * shares[(app.id, g, "small")] for g in app.ingress_rates
    ) for app in scenario.applications.values() for node in app.nodes.values()
        if node.type is NodeType.LLM]
    assert captures[0][3] == pytest.approx(expected)
    assert decision.model_share == shares
    arrivals = analytical.AnalyticalBackend(scenario).llm_arrivals(decision, rates)
    loads = {cid: sum(rate for key, rate in arrivals.items() if key[2] == cid)
             for cid in ("small-n0", "small-n2")}
    assert loads["small-n0"] / 2. == pytest.approx(loads["small-n2"] / 8.)
    # This rule is independent of historical overload scores.
    history = SlotMetrics(0, 0, 0, 0, 0, 0, 0, 0,
                          llm_utilization={cid: 2. for cid in deployment.llm_active},
                          tool_utilization={"search@n0": 2.})
    assert router.route(deployment, shares, history, rates) == decision


def test_tool_routes_use_replicas_times_heterogeneous_rate(scenario):
    deployment = CapacityPlanner(scenario).initial_deployment()
    for key in deployment.tool_replicas:
        deployment.tool_replicas[key] = 0
    deployment.tool_replicas[("search", "n0")] = 1
    deployment.tool_replicas[("search", "n3")] = 4
    result = CapacityProportionalRouter(scenario).route(deployment, shares_of(scenario))
    for key, value in result.tool_route.items():
        if scenario.applications[key[0]].nodes[key[2]].tool == "search":
            assert value == pytest.approx(30 / 190 if key[4] == "n0" else 160 / 190)


def test_reachability_and_zero_capacity_leave_unserved_support(scenario, monkeypatch):
    scenario = replace(scenario, links=())
    deployment = CapacityPlanner(scenario).empty_deployment()
    deployment.llm_active["small-n2"] = 1
    deployment.tool_replicas[("search", "n2")] = 1
    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", lambda *args: (1., 1.))
    decision = CapacityProportionalRouter(scenario).route(deployment, shares_of(scenario, 1))
    assert not any(key[1] == "n0" for key in decision.llm_share)
    assert all(key[3] == "small-n2" for key in decision.llm_share)
    assert all(key[3] == key[4] == "n2" for key in decision.tool_route)
    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", lambda *args: (0., 1.))
    assert not CapacityProportionalRouter(scenario).route(deployment, shares_of(scenario, 1)).llm_share


def test_unserved_actual_tool_source_is_reported_even_if_idle_source_has_route(scenario, monkeypatch):
    scenario = replace(scenario, links=())
    deployment = CapacityPlanner(scenario).empty_deployment()
    deployment.llm_active["small-n0"] = 1
    deployment.tool_replicas[("search", "n2")] = 1
    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", lambda *args: (1., 1.))
    decision = CapacityProportionalRouter(scenario).route(deployment, shares_of(scenario, 1))
    metrics = Simulator(scenario, max_mapping_samples=8).step(deployment, decision).metrics
    assert "service_unserved:chat-search:llm0->search0@n0" in metrics.diagnostics["violation_labels"]


def test_zero_model_demand_uses_equal_flow_fallback(scenario, monkeypatch):
    observed = []
    def capacity(model, config, classes, weights, chunk):
        observed.append((model.id, weights))
        return 1., 1.
    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", capacity)
    deployment = CapacityPlanner(scenario).initial_deployment()
    CapacityProportionalRouter(scenario).route(deployment, shares_of(scenario, 1))
    assert next(w for m, w in observed if m == "large") == [1.] * 4


def test_zero_rate_flow_does_not_constrain_positive_model_mixture(scenario, monkeypatch):
    observed = []
    def capacity(model, config, classes, weights, chunk):
        if model.id == "small":
            observed.append((classes, weights))
        return 1., 1.
    monkeypatch.setattr("agent_orch.routing.capacity.reference_llm_capacity", capacity)
    shares = shares_of(scenario, 0)
    shares[("chat-search", "n0", "small")] = 1.
    shares[("chat-search", "n0", "large")] = 0.
    CapacityProportionalRouter(scenario).route(CapacityPlanner(scenario).initial_deployment(), shares)
    assert observed[0][0] == [(256, 24), (512, 48)]
    assert all(weight > 0 for weight in observed[0][1])


def test_capacity_only_matches_full_model_and_cache_keys_are_exact(scenario, monkeypatch):
    model, config = scenario.models["small"], scenario.llm_configs["small-edge"]
    classes, weights = [(256., 24.), (512., 48.)], [2., 1.]
    analytical._reference_llm_capacity_cached.cache_clear()
    calls = []
    original = analytical.two_mode_capacity
    def counted(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(analytical, "two_mode_capacity", counted)
    capacity, concurrency = analytical.reference_llm_capacity(model, config, classes, weights, 512)
    instance, _ = analytical.evaluate_llm_instance(model, config, classes, weights, .001, 512, "macro")
    assert instance.throughput_capacity_rps == capacity
    assert instance.capacity_concurrency == concurrency
    assert len(calls) == 1
    analytical.reference_llm_capacity(model, config, classes, [1., 2.], 512)
    analytical.reference_llm_capacity(model, config, [(257., 24.), (512., 48.)], weights, 512)
    analytical.reference_llm_capacity(model, replace(config, effective_flops=config.effective_flops * .5), classes, weights, 512)
    analytical.reference_llm_capacity(model, config, classes, weights, 256)
    assert len(calls) == 5
    # No operating point is needed for the routing capacity calculation.
    monkeypatch.setattr(analytical, "two_mode_operating_point", lambda *a, **kw: pytest.fail("full evaluation"))
    analytical.reference_llm_capacity(model, config, [(258., 24.)], [1.], 512)


def complete_slot(env, observation):
    steps = 0
    while True:
        observation, reward, done, _, info = env.step({
            "deploy": 0, "model": np.ones(env.layout.model_action_size, dtype=np.float32)
        })
        steps += 1
        if info.get("period_complete"):
            return observation, info, steps


def test_frozen_deployment_shortcut_matches_replay_and_reuses_slot_router(scenario):
    trace = SlotTrajectory.sample(scenario, 2, 37)
    fast = SlotSequentialJointEnv(scenario, trace, mapping_samples=8)
    replay = SlotSequentialJointEnv(scenario, trace, mapping_samples=8, replay_frozen_deployment_steps=True)
    fast.training_phase = replay.training_phase = "composition"
    obs, _ = fast.reset(seed=1)
    slow, _ = replay.reset(seed=1)
    router = fast.physical_router
    assert obs["action_type"] == fast.COMPOSITION
    assert slow["action_type"] == replay.LLM_DEPLOYMENT
    replay_steps = 0
    while slow["action_type"] != replay.COMPOSITION:
        slow, _, _, _, _ = replay.step({"deploy": 0})
        replay_steps += 1
    for key in ("features", "routing_features", "deployment_features", "model_mask"):
        np.testing.assert_array_equal(obs[key], slow[key])
    for _ in range(2):
        obs, a, steps = complete_slot(fast, obs)
        slow, b, slow_steps = complete_slot(replay, slow)
        assert steps == len(fast.layout.model_groups)
        assert slow_steps + replay_steps == steps + len(scenario.candidates) + len(scenario.tools) * len(scenario.servers)
        replay_steps = 0
        assert a["deployment_steps"] == 0
        assert a["utility"] == pytest.approx(b["utility"], abs=1e-12)
        assert a["metrics"] == b["metrics"]
    fast.reset(seed=999)
    assert fast.physical_router is router
    legacy_state = {**fast.reward_state(), "routing_method": "removed_method"}
    with pytest.raises(ValueError, match="routing protocol"):
        fast.restore_reward_state(legacy_state)


def test_only_current_capacity_protocol_is_supported(scenario):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_slot_joint import environment_options, PROTOCOL, EVALUATION_PROTOCOL
    manifest = {"protocol": PROTOCOL, "physical_routing": "capacity_proportional",
                "evaluation_protocol": EVALUATION_PROTOCOL}
    assert environment_options(manifest)["routing_method"] == "capacity_proportional"
    with pytest.raises(ValueError):
        environment_options({"protocol": "slot_joint_v4"})
    with pytest.raises(ValueError):
        environment_options({**manifest, "physical_routing": "removed_method"})
    with pytest.raises(ValueError, match="Only capacity_proportional"):
        SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 0),
                               routing_method="removed_method")
    from run_slot_joint import _evaluate_baselines
    with pytest.raises(ValueError, match="Only capacity_proportional"):
        _evaluate_baselines(scenario, SlotTrajectory.sample(scenario, 1, 0), 1, None, None,
                            routing_method="removed_method")

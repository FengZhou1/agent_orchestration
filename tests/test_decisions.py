import pytest

from agent_orch.action_decoder import ActionDecoder
from agent_orch.baselines import GreedyPolicy, make_policy


def test_greedy_decision_satisfies_probability_constraints(scenario):
    policy = GreedyPolicy(scenario, seed=7)
    deployment, routing = policy.decide()
    decoder = ActionDecoder(scenario)
    decoder.validate_deployment(deployment)
    decoder.validate_routing(deployment, routing)


def test_all_baseline_decisions_satisfy_probability_constraints(scenario):
    decoder = ActionDecoder(scenario)
    for name in ("static", "equal", "least_load", "random", "greedy"):
        deployment, routing = make_policy(name, scenario, seed=11).decide()
        decoder.validate_deployment(deployment)
        decoder.validate_routing(deployment, routing)


def test_static_homogeneous_uses_one_llm_candidate(scenario):
    deployment = make_policy("static", scenario).deployment()
    assert sum(deployment.llm_active.values()) == 1


def test_tool_resource_violation_is_rejected(scenario):
    policy = GreedyPolicy(scenario, seed=7)
    deployment = policy.deployment()
    deployment.tool_replicas[("search", "n0")] = 100
    with pytest.raises(ValueError):
        ActionDecoder(scenario).validate_deployment(deployment)

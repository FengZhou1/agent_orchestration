import math

from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import NodeType


def test_ahcp_aggregates_agent_visits_into_tool_and_model_load(scenario):
    planner = CapacityPlanner(scenario)
    arrivals = {
        (app.id, ingress): rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    plan = planner.plan(arrivals, period_index=0)

    expected_tool = {tool_id: 0.0 for tool_id in scenario.tools}
    expected_model_total = 0.0
    for app in scenario.applications.values():
        app_rate = sum(arrivals[(app.id, ingress)] for ingress in app.ingress_rates)
        for node in app.nodes.values():
            visits = app.visit_probability(node.id)
            if node.type is NodeType.TOOL:
                expected_tool[node.tool] += app_rate * visits
            else:
                expected_model_total += app_rate * visits

    for tool_id, expected in expected_tool.items():
        assert math.isclose(plan.tool_arrival[tool_id], expected)
        assert plan.tool_required[tool_id] >= 1
    assert math.isclose(sum(plan.model_arrival.values()), expected_model_total)
    assert all(value > 0.0 for value in plan.candidate_capacity.values())
    assert all(value >= 0.0 for value in plan.model_required_capacity.values())


def test_ahcp_planning_shares_are_normalized_and_follow_history(scenario):
    planner = CapacityPlanner(scenario)
    arrivals = {
        (app.id, ingress): rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    preferred = next(iter(scenario.models))
    history = {
        (app.id, ingress, model): float(model == preferred)
        for app in scenario.applications.values()
        for ingress in app.ingress_rates
        for model in scenario.models
    }
    plan = planner.plan(arrivals, history, period_index=3)
    for app in scenario.applications.values():
        for ingress in app.ingress_rates:
            shares = [
                plan.planning_model_share[(app.id, ingress, model)]
                for model in scenario.models
            ]
            assert math.isclose(sum(shares), 1.0)
            assert plan.planning_model_share[(app.id, ingress, preferred)] > max(
                plan.planning_model_share[(app.id, ingress, model)]
                for model in scenario.models
                if model != preferred
            )

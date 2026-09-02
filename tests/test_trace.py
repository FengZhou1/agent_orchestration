import pytest

from agent_orch.baselines import GreedyPolicy
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def test_stationary_poisson_intensity_is_preserved_in_every_slot(scenario):
    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, slots=5, rate_scale=2.0
    )
    expected_rate = sum(
        rate
        for app in scenario.applications.values()
        for rate in app.ingress_rates.values()
    ) * 2.0
    assert all(
        sum(slot.values()) == pytest.approx(expected_rate)
        for slot in trace.rates.values()
    )

    first = ArrivalTrace.stationary_poisson(scenario, slots=5, seed=13)
    second = ArrivalTrace.stationary_poisson(scenario, slots=5, seed=99)
    assert first.rates == second.rates

    policy = GreedyPolicy(scenario, seed=13)
    deployment, routing = policy.decide()
    simulator = Simulator(scenario)
    simulator.set_arrival_trace(trace)
    simulator.reset(13)
    expected = sum(trace.at(0, scenario).values())
    metrics = simulator.step(deployment, routing).metrics
    assert metrics.total_arrival_rps == expected

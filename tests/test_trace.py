from agent_orch.baselines import GreedyPolicy
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def test_trace_changes_slot_arrival_and_is_seed_deterministic(scenario):
    first = ArrivalTrace.synthetic_bursty(scenario, slots=5, seed=13)
    second = ArrivalTrace.synthetic_bursty(scenario, slots=5, seed=13)
    assert first.rates == second.rates

    policy = GreedyPolicy(scenario, seed=13)
    deployment, routing = policy.decide()
    simulator = Simulator(scenario)
    simulator.set_arrival_trace(first)
    simulator.reset(13)
    expected = sum(first.at(0, scenario).values())
    metrics = simulator.step(deployment, routing).metrics
    assert metrics.total_arrival_rps == expected


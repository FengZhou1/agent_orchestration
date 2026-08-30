import pytest

from agent_orch.performance.queueing import llm_waiting_time


def test_single_server_matches_pollaczek_khinchine():
    arrival = 0.4
    mean = 1.0
    second = 2.0
    wait, rho, overloaded = llm_waiting_time(arrival, mean, second, 1, 60.0)
    expected = arrival * second / (2.0 * (1.0 - arrival * mean))
    assert not overloaded
    assert rho == pytest.approx(0.4)
    assert wait == pytest.approx(expected)


def test_waiting_time_is_monotone_in_load():
    low = llm_waiting_time(0.2, 0.5, 0.3, 2, 60.0)[0]
    high = llm_waiting_time(2.0, 0.5, 0.3, 2, 60.0)[0]
    assert high > low


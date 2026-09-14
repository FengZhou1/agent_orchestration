import pytest

from agent_orch.performance.queueing import llm_waiting_time, tool_response_time


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


def test_tool_waiting_time_matches_gi_m_c_single_server():
    wait, process, rho, overloaded = tool_response_time(1.0, 4.0, 1, 1.0, 60.0)
    assert not overloaded
    assert rho == pytest.approx(0.25)
    assert process == pytest.approx(0.25)
    # (C_A^2 + 1) / 2 * P_W / (c * mu - lambda), with P_W = rho for c = 1
    assert wait == pytest.approx(0.25 / 3.0)


def test_tool_waiting_time_grows_with_arrival_variability():
    low = tool_response_time(1.0, 4.0, 1, 1.0, 60.0)[0]
    high = tool_response_time(1.0, 4.0, 1, 2.0, 60.0)[0]
    assert high > low
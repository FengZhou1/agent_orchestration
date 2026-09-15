import pytest

from agent_orch.performance.queueing import tool_response_time


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
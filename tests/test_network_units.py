import pytest

from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import LinkSpec


def test_offered_load_is_mbps_and_independent_of_slot_duration():
    links = (LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0),)
    one_second = NetworkBackend(links, slot_seconds=1.0)
    ten_seconds = NetworkBackend(links, slot_seconds=10.0)
    load_one = {}
    load_ten = {}
    one_second.add_traffic(load_one, "a", "b", request_rate=2.0, data_mb_per_request=1.0)
    ten_seconds.add_traffic(load_ten, "a", "b", request_rate=2.0, data_mb_per_request=1.0)
    assert load_one == load_ten == {("a", "b"): 16.0}
    assert one_second.utilization(load_one)["a->b"] == pytest.approx(0.16)


def test_hop_delay_uses_aggregate_offered_volume_over_link_rate():
    links = (LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0),)
    network = NetworkBackend(links, slot_seconds=1.0)
    loads = {("a", "b"): 20.0}
    # 1 ms propagation plus 20 Mbit / 100 Mbps = 0.2 s transmission time
    assert network.path_delay("a", "b", loads) == pytest.approx(0.201)


def test_local_transfer_has_zero_network_delay():
    links = (LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0),)
    network = NetworkBackend(links)
    assert network.path_delay("a", "a", {}) == 0.0


def test_saturated_link_returns_overload_delay():
    links = (LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0),)
    network = NetworkBackend(links, overload_delay_s=60.0)
    assert network.path_delay("a", "b", {("a", "b"): 100.0}) == pytest.approx(60.0)


def test_network_traffic_cost_converts_mbps_to_mb_per_slot():
    links = (
        LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0, cost_per_mb=0.5),
    )
    network = NetworkBackend(links, slot_seconds=10.0)
    assert network.traffic_cost({("a", "b"): 16.0}) == pytest.approx(10.0)
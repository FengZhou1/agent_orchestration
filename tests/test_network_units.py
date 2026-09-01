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


def test_request_delay_uses_its_payload_not_whole_slot_traffic():
    links = (LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0),)
    network = NetworkBackend(links)
    loads = {("a", "b"): 20.0}
    assert network.path_delay("a", "b", 1.0, loads) == pytest.approx(0.081)


def test_network_traffic_cost_converts_mbps_to_mb_per_slot():
    links = (
        LinkSpec("a", "b", capacity_mbps=100.0, propagation_ms=1.0, cost_per_mb=0.5),
    )
    network = NetworkBackend(links, slot_seconds=10.0)
    assert network.traffic_cost({("a", "b"): 16.0}) == pytest.approx(10.0)

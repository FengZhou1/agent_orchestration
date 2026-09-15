from __future__ import annotations

def erlang_c(servers: int, utilization: float) -> float:
    """Return the Erlang-C waiting probability for total utilization in [0, 1)."""
    if servers <= 0:
        return 1.0
    if utilization <= 0.0:
        return 0.0
    if utilization >= 1.0:
        return 1.0
    offered = servers * utilization
    terms = [1.0]
    for k in range(1, servers):
        terms.append(terms[-1] * offered / k)
    tail = terms[-1] * offered / servers / (1.0 - utilization)
    return tail / (sum(terms) + tail)


def tool_response_time(
    arrival_rate: float,
    service_rate: float,
    replicas: int,
    arrival_scv: float,
    overload_delay: float,
) -> tuple[float, float, float, bool]:
    if arrival_rate <= 0.0:
        processing = 1.0 / service_rate
        return 0.0, processing, 0.0, False
    if replicas <= 0 or service_rate <= 0.0:
        return overload_delay, 0.0, 1.0e6, True
    utilization = arrival_rate / (replicas * service_rate)
    processing = 1.0 / service_rate
    if utilization >= 1.0:
        return overload_delay, processing, utilization, True
    wait_probability = erlang_c(replicas, utilization)
    wait = (arrival_scv + 1.0) / 2.0
    wait *= wait_probability / (replicas * service_rate - arrival_rate)
    return max(0.0, wait), processing, utilization, False

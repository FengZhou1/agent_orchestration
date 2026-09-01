from __future__ import annotations

from dataclasses import dataclass

from agent_orch.performance.analytical import AnalyticalBackend, AnalyticalResult
from agent_orch.schema.models import DeploymentDecision, RoutingDecision, Scenario


@dataclass(frozen=True)
class CapacityEstimate:
    stable_capacity_rps: float
    arrival_scale: float
    limiting_utilization: float
    limiting_resource: str


def _scaled_arrivals(scenario: Scenario, factor: float) -> dict[tuple[str, str], float]:
    return {
        (app.id, ingress): factor * rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }


def _stability_margin(
    backend: AnalyticalBackend, result: AnalyticalResult
) -> tuple[bool, float, str]:
    resources: dict[str, float] = {
        **{f"llm:{key}": value for key, value in result.llm_utilization.items()},
        **{f"service:{tool}@{server}": value for (tool, server), value in result.tool_utilization.items()},
        **{
            f"link:{edge}": value
            for edge, value in backend.network.utilization(result.link_load_mbps).items()
        },
    }
    for candidate, stable in result.llm_kv_stable.items():
        if not stable:
            resources[f"kv:{candidate}"] = max(resources.get(f"llm:{candidate}", 0.0), 1.0)
    resource, utilization = max(resources.items(), key=lambda item: item[1], default=("none", 0.0))
    invalid = any(
        label.startswith(("llm_unserved", "llm_queue_overload", "llm_kv_overload", "tool_overload", "link_overload"))
        for label in result.violations
    )
    stable = not invalid and utilization < 1.0 and all(result.llm_kv_stable.values())
    return stable, float(utilization), resource


def estimate_reference_capacity(
    scenario: Scenario,
    deployment: DeploymentDecision,
    routing: RoutingDecision,
    backend: AnalyticalBackend,
    tolerance: float = 1e-3,
    maximum_scale: float = 1024.0,
) -> CapacityEstimate:
    """Find the largest common arrival-rate scale satisfying all capacity limits."""
    base_rate = sum(
        rate
        for app in scenario.applications.values()
        for rate in app.ingress_rates.values()
    )
    if base_rate <= 0.0:
        raise ValueError("Reference scenario must have positive arrival rates")

    low, high = 0.0, 1.0
    while high < maximum_scale:
        result = backend.evaluate(deployment, routing, _scaled_arrivals(scenario, high))
        stable, _, _ = _stability_margin(backend, result)
        if not stable:
            break
        low, high = high, min(maximum_scale, 2.0 * high)
    if high == maximum_scale:
        result = backend.evaluate(deployment, routing, _scaled_arrivals(scenario, high))
        if _stability_margin(backend, result)[0]:
            raise ValueError("No stability boundary found below maximum_scale")

    while high - low > tolerance * max(1.0, high):
        middle = 0.5 * (low + high)
        result = backend.evaluate(deployment, routing, _scaled_arrivals(scenario, middle))
        if _stability_margin(backend, result)[0]:
            low = middle
        else:
            high = middle

    result = backend.evaluate(deployment, routing, _scaled_arrivals(scenario, low))
    _, utilization, resource = _stability_margin(backend, result)
    upper_result = backend.evaluate(
        deployment, routing, _scaled_arrivals(scenario, high)
    )
    for prefix, label in (
        ("llm_kv_overload:", "kv:"),
        ("llm_queue_overload:", "llm:"),
        ("tool_overload:", "service:"),
        ("link_overload:", "link:"),
    ):
        violation = next(
            (item for item in upper_result.violations if item.startswith(prefix)), None
        )
        if violation is not None:
            resource = label + violation.removeprefix(prefix)
            utilization = max(1.0, utilization)
            break
    return CapacityEstimate(base_rate * low, low, utilization, resource)

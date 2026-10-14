"""Fixed scales from feasible placements, not extrema of impossible deployments.

Historical cost_min/cost_max field names encode 0 and a positive reference
cost rate. Cost ratios may exceed one; they are never clipped to that scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from agent_orch.schema.models import Scenario

from .spec import ObjectiveSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_orch.deployment import DeploymentLibrary


def app_latency_reference(scenario: Scenario, app_id: str) -> float:
    """The per-application latency scale that SLO thresholds are expressed in."""

    app = scenario.applications[app_id]
    if app.slo.deadline_s is not None:
        return max(float(app.slo.deadline_s), 1.0e-9)
    ttft = app.slo.ttft_s or scenario.simulation.overload_delay_s
    tbt = app.slo.tbt_s or 0.0
    output_tokens = sum(
        flow.probability
        * float(np.mean(list(app.nodes[flow.final_node].output_tokens.values())))
        for flow in app.pattern_flows
    )
    return max(ttft + max(0.0, output_tokens - 1.0) * tbt, 1.0e-9)


def arrival_weighted_latency_reference(scenario: Scenario) -> float:
    """Arrival-weighted mean of the per-application latency scales."""

    weighted = 0.0
    total = 0.0
    for app in scenario.applications.values():
        for rate in app.ingress_rates.values():
            weight = max(0.0, float(rate))
            weighted += weight * app_latency_reference(scenario, app.id)
            total += weight
    if total > 0.0:
        return max(weighted / total, 1.0e-12)
    return max(
        float(
            np.mean(
                [app_latency_reference(scenario, app_id) for app_id in scenario.applications]
            )
        ),
        1.0e-12,
    )


def _theoretical_cost_bounds(scenario: Scenario) -> tuple[float, float]:
    from agent_orch.capacity import CapacityPlanner
    from agent_orch.cost import operating_cost_rate
    placement = CapacityPlanner(scenario).initial_deployment()
    return 0.0, max(operating_cost_rate(scenario, placement), 1e-9)


def _library_cost_bounds(
    scenario: Scenario, library: "DeploymentLibrary"
) -> tuple[float, float]:
    """Fixed scale from feasible placements, never an upper bound or clip."""
    from agent_orch.cost import operating_cost_rate
    steady = [operating_cost_rate(scenario, entry.to_deployment()) for entry in library.entries]
    if not steady:
        return _theoretical_cost_bounds(scenario)
    return 0.0, max(float(np.median(steady)), 1e-9)


@dataclass(frozen=True)
class ReferenceScales:
    """Normalisation scales for cost and latency."""

    cost_min: float
    cost_max: float
    cost_source: str
    latency_reference: float
    app_latency: dict[str, float]

    def app_latency_reference(self, app_id: str) -> float:
        return self.app_latency.get(app_id, self.latency_reference)

    @property
    def cost_span(self) -> float:
        return max(self.cost_max - self.cost_min, 1.0e-12)

    @staticmethod
    def from_scenario(
        scenario: Scenario,
        spec: ObjectiveSpec | None = None,
        library: "DeploymentLibrary | None" = None,
    ) -> "ReferenceScales":
        spec = spec or ObjectiveSpec.legacy()
        if spec.cost_bounds == "library" and library is not None and len(library) > 0:
            cost_min, cost_max = _library_cost_bounds(scenario, library)
            source = "fixed-feasible-library-median-rate"
        else:
            cost_min, cost_max = _theoretical_cost_bounds(scenario)
            source = "fixed-greedy-operating-rate"
        return ReferenceScales(
            cost_min=cost_min,
            cost_max=cost_max,
            cost_source=source,
            latency_reference=arrival_weighted_latency_reference(scenario),
            app_latency={
                app_id: app_latency_reference(scenario, app_id)
                for app_id in scenario.applications
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cost_min": self.cost_min,
            "cost_max": self.cost_max,
            "cost_source": self.cost_source,
            "latency_reference": self.latency_reference,
            "app_latency": dict(self.app_latency),
        }

    @staticmethod
    def from_mapping(payload: Mapping[str, object]) -> "ReferenceScales":
        return ReferenceScales(
            cost_min=float(payload["cost_min"]),  # type: ignore[arg-type]
            cost_max=float(payload["cost_max"]),  # type: ignore[arg-type]
            cost_source=str(payload.get("cost_source", "unknown")),
            latency_reference=float(payload["latency_reference"]),  # type: ignore[arg-type]
            app_latency={
                str(key): float(value)
                for key, value in dict(payload.get("app_latency", {})).items()  # type: ignore[arg-type]
            },
        )

"""Metric scales used to normalise the objective.

The cost range is taken from the feasible deployment library whenever one is
available.  Deriving it from the catalogue of *candidates* instead makes the
range far wider than any deployment the system can actually reach, which
collapses the normalised cost onto a near-constant and removes cost from the
objective entirely.
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
    period_seconds = scenario.simulation.orchestration_period_s
    minimum_llm = min(
        scenario.llm_configs[candidate.config].running_cost_per_slot * period_seconds
        for candidate in scenario.candidates.values()
    )
    minimum_services = sum(
        tool.running_cost_per_slot * period_seconds for tool in scenario.tools.values()
    )
    maximum = sum(
        scenario.llm_configs[candidate.config].running_cost_per_slot * period_seconds
        + scenario.llm_configs[candidate.config].load_cost
        for candidate in scenario.candidates.values()
    )
    maximum += sum(
        scenario.simulation.max_tool_replicas_per_server
        * (tool.running_cost_per_slot * period_seconds + tool.start_cost)
        for tool in scenario.tools.values()
        for _ in scenario.servers
    )
    minimum = float(minimum_llm + minimum_services)
    return minimum, float(max(maximum, minimum + 1.0))


def _library_cost_bounds(
    scenario: Scenario, library: "DeploymentLibrary"
) -> tuple[float, float]:
    """Steady-cost range over feasible deployments, widened by switch-on cost.

    ``DeploymentEntry.cost_per_period`` counts running cost only, while the
    simulator charges ``load_cost`` when a candidate switches on and
    ``start_cost`` per newly started replica.  The upper bound adds the largest
    switch-on surcharge reachable inside the library so a legitimately
    expensive deployment is not pinned at the top of the normalised range.
    """

    steady = [float(entry.cost_per_period) for entry in library.entries]
    if not steady:
        return _theoretical_cost_bounds(scenario)
    surcharge = 0.0
    for entry in library.entries:
        llm_switch = sum(
            scenario.llm_configs[scenario.candidates[cid].config].load_cost
            for cid, active in entry.llm_active.items()
            if active
        )
        tool_switch = sum(
            scenario.tools[tool_id].start_cost * int(replicas)
            for (tool_id, _server), replicas in entry.tool_replicas.items()
        )
        surcharge = max(surcharge, float(llm_switch + tool_switch))
    minimum = float(min(steady))
    maximum = float(max(steady)) + surcharge
    return minimum, float(max(maximum, minimum + 1.0e-9))


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
            source = "library+switch-on"
        else:
            cost_min, cost_max = _theoretical_cost_bounds(scenario)
            source = "theoretical"
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

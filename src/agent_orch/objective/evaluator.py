"""Per-period objective and constraint evaluation.

The evaluator is the single place where a ``SlotMetrics`` record turns into a
scalar utility and a constraint vector, so the objective can be audited,
versioned and swapped without touching the environment or the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from agent_orch.schema.models import Scenario, SlotMetrics

from .references import ReferenceScales
from .spec import ObjectiveSpec

ArrivalRates = Mapping[tuple[str, str], float]


@dataclass(frozen=True)
class ObjectiveValue:
    """One period's objective value, constraint vector and audit diagnostics."""

    utility: float
    components: dict[str, float]
    constraints: tuple[float, ...]
    diagnostics: dict[str, float] = field(default_factory=dict)


class ObjectiveEvaluator:
    """Turn metrics into a utility and a constraint vector for one scenario."""

    def __init__(
        self,
        scenario: Scenario,
        spec: ObjectiveSpec | None = None,
        references: ReferenceScales | None = None,
    ) -> None:
        self.scenario = scenario
        self.spec = spec or ObjectiveSpec.legacy()
        self.references = references or ReferenceScales.from_scenario(scenario, self.spec)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return self.spec.constraint_names

    @property
    def constraint_count(self) -> int:
        return len(self.constraint_names)

    @property
    def cost_min(self) -> float:
        return self.references.cost_min

    @property
    def cost_max(self) -> float:
        return self.references.cost_max

    @property
    def latency_reference(self) -> float:
        return self.references.latency_reference

    def app_latency_reference(self, app_id: str) -> float:
        return self.references.app_latency_reference(app_id)

    def evaluate(self, metrics: SlotMetrics, arrival_rates: ArrivalRates) -> ObjectiveValue:
        return self.evaluate_arrays(
            cost=metrics.cost,
            mean_latency=metrics.mean_latency_s,
            attainment=metrics.slo_attainment,
            quality=metrics.quality,
            app_latency=metrics.app_latency_s,
            arrival_rates=arrival_rates,
            llm_utilization=metrics.llm_utilization,
            tool_utilization=metrics.tool_utilization,
            violation_labels=tuple(metrics.diagnostics.get("violation_labels", ())),
        )

    def evaluate_arrays(
        self,
        *,
        cost: float,
        mean_latency: float,
        attainment: float,
        quality: float,
        app_latency: Mapping[str, float],
        arrival_rates: ArrivalRates,
        llm_utilization: Mapping[str, float] | None = None,
        tool_utilization: Mapping[str, float] | None = None,
        violation_labels: Sequence[str] = (),
    ) -> ObjectiveValue:
        spec = self.spec
        references = self.references

        cost_normalized = float(
            np.clip((cost - references.cost_min) / references.cost_span, 0.0, 1.0)
        )
        latency_normalized, saturation_fraction = self._latency_term(
            mean_latency=mean_latency,
            app_latency=app_latency,
            arrival_rates=arrival_rates,
        )

        components = {
            "cost_normalized": cost_normalized,
            "latency_normalized": float(latency_normalized),
            "goodput_normalized": float(np.clip(attainment, 0.0, 1.0)),
            "quality_normalized": float(np.clip(quality, 0.0, 1.0)),
        }
        utility = (
            spec.goodput_weight * components["goodput_normalized"]
            + spec.quality_weight * components["quality_normalized"]
            - spec.cost_weight * components["cost_normalized"]
            - spec.latency_weight * components["latency_normalized"]
        )
        components["utility"] = float(utility)

        constraints = self._constraint_vector(
            attainment=attainment,
            llm_utilization=llm_utilization or {},
            tool_utilization=tool_utilization or {},
            violation_labels=violation_labels,
        )
        diagnostics = {
            "cost": float(cost),
            "mean_latency_s": float(mean_latency),
            "slo_attainment": float(attainment),
            "quality": float(quality),
            "latency_saturation_fraction": float(saturation_fraction),
            "cost_clip_high": float(cost >= references.cost_max),
            "cost_clip_low": float(cost <= references.cost_min),
        }
        for index, name in enumerate(self.constraint_names):
            diagnostics[f"constraint_{name}"] = float(constraints[index])
        return ObjectiveValue(
            utility=float(utility),
            components=components,
            constraints=tuple(float(value) for value in constraints),
            diagnostics=diagnostics,
        )

    def _latency_term(
        self,
        *,
        mean_latency: float,
        app_latency: Mapping[str, float],
        arrival_rates: ArrivalRates,
    ) -> tuple[float, float]:
        references = self.references
        total_rate = sum(max(0.0, float(rate)) for rate in arrival_rates.values())
        if total_rate > 0.0 and app_latency:
            numerator = 0.0
            saturated_weight = 0.0
            for app in self.scenario.applications.values():
                weight = sum(
                    max(0.0, float(arrival_rates.get((app.id, ingress), 0.0)))
                    for ingress in app.ingress_rates
                )
                if weight <= 0.0:
                    continue
                ratio = app_latency.get(app.id, mean_latency) / references.app_latency_reference(
                    app.id
                )
                numerator += weight * self._saturate(ratio)
                if ratio >= 1.0:
                    saturated_weight += weight
            return numerator / total_rate, saturated_weight / total_rate
        mean_reference = float(
            np.mean(list(references.app_latency.values()))
            if references.app_latency
            else references.latency_reference
        )
        ratio = mean_latency / max(mean_reference, 1.0e-12)
        return self._saturate(ratio), float(ratio >= 1.0)

    def _saturate(self, ratio: float) -> float:
        ratio = max(0.0, float(ratio))
        if self.spec.latency_map == "smooth":
            return ratio / (1.0 + ratio)
        return min(1.0, ratio)

    def _constraint_vector(
        self,
        *,
        attainment: float,
        llm_utilization: Mapping[str, float],
        tool_utilization: Mapping[str, float],
        violation_labels: Sequence[str],
    ) -> list[float]:
        spec = self.spec
        llm = max(
            [max(0.0, float(value) - spec.llm_utilization_target) for value in llm_utilization.values()]
            or [0.0]
        )
        service = max(
            [
                max(0.0, float(value) - spec.service_utilization_target)
                for value in tool_utilization.values()
            ]
            or [0.0]
        )
        labels = tuple(violation_labels)
        llm += float(any(label.startswith("llm_") for label in labels))
        service += float(
            any(label.startswith(("service_", "tool_", "link_")) for label in labels)
        )
        values = [float(llm), float(service)]
        if spec.attainment_target is not None:
            values.append(max(0.0, spec.attainment_target - float(attainment)))
        return values

    def zero_constraints(self) -> np.ndarray:
        return np.zeros(self.constraint_count, dtype=np.float32)

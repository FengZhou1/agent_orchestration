from __future__ import annotations

from dataclasses import dataclass

from agent_orch.schema.models import Scenario


@dataclass(frozen=True)
class ArrivalTrace:
    """Per-slot arrival intensities used by the analytical queueing model."""

    rates: dict[int, dict[tuple[str, str], float]]

    def at(self, slot: int, scenario: Scenario) -> dict[tuple[str, str], float]:
        defaults = {
            (app.id, ingress): rate
            for app in scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        }
        if slot not in self.rates:
            return defaults
        dense = {key: 0.0 for key in defaults}
        dense.update(self.rates[slot])
        return dense

    @staticmethod
    def stationary_poisson_intensity(
        scenario: Scenario,
        slots: int,
        rate_scale: float = 1.0,
    ) -> "ArrivalTrace":
        """Return the stationary Poisson intensity for every analytical slot."""
        if slots <= 0:
            raise ValueError("slots must be positive")
        if rate_scale < 0.0:
            raise ValueError("rate_scale must be non-negative")
        intensities = {
            (app.id, ingress): rate_scale * float(rate)
            for app in scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        }
        return ArrivalTrace({slot: dict(intensities) for slot in range(slots)})

    @staticmethod
    def stationary_poisson(
        scenario: Scenario,
        slots: int,
        seed: int = 0,
        rate_scale: float = 1.0,
    ) -> "ArrivalTrace":
        """Backward-compatible alias for the analytical Poisson intensity."""
        del seed
        return ArrivalTrace.stationary_poisson_intensity(
            scenario,
            slots,
            rate_scale=rate_scale,
        )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent_orch.schema.models import Scenario


@dataclass(frozen=True)
class ArrivalTrace:
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
    def from_csv(path: str | Path) -> "ArrivalTrace":
        frame = pd.read_csv(path)
        required = {"slot", "application", "ingress", "rate_rps"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Arrival trace is missing columns: {sorted(missing)}")
        rates: dict[int, dict[tuple[str, str], float]] = {}
        for row in frame.itertuples(index=False):
            rate = max(0.0, float(row.rate_rps))
            rates.setdefault(int(row.slot), {})[(str(row.application), str(row.ingress))] = rate
        if rates:
            for slot in range(min(rates), max(rates) + 1):
                rates.setdefault(slot, {})
        return ArrivalTrace(rates)

    def to_frame(self, scenario: Scenario | None = None) -> pd.DataFrame:
        if scenario is None:
            records = [
                {
                    "slot": slot,
                    "application": application,
                    "ingress": ingress,
                    "rate_rps": rate,
                }
                for slot, slot_rates in sorted(self.rates.items())
                for (application, ingress), rate in sorted(slot_rates.items())
            ]
        else:
            keys = [
                (app.id, ingress)
                for app in scenario.applications.values()
                for ingress in app.ingress_rates
            ]
            records = [
                {
                    "slot": slot,
                    "application": application,
                    "ingress": ingress,
                    "rate_rps": slot_rates.get((application, ingress), 0.0),
                }
                for slot, slot_rates in sorted(self.rates.items())
                for application, ingress in keys
            ]
        return pd.DataFrame(
            records,
            columns=("slot", "application", "ingress", "rate_rps"),
        )

    def scaled(self, factor: float) -> "ArrivalTrace":
        if factor < 0.0:
            raise ValueError("Arrival-rate scale must be non-negative")
        return ArrivalTrace(
            {
                slot: {key: factor * value for key, value in slot_rates.items()}
                for slot, slot_rates in self.rates.items()
            }
        )

    def window(self, start: int, stop: int) -> "ArrivalTrace":
        if start < 0 or stop < start:
            raise ValueError("Invalid trace window")
        return ArrivalTrace(
            {
                slot - start: dict(self.rates.get(slot, {}))
                for slot in range(start, stop)
            }
        )

    @staticmethod
    def nhpp_control(
        scenario: Scenario,
        source: "ArrivalTrace",
        slots: int,
        seed: int = 0,
        intensity_window_slots: int = 60,
    ) -> "ArrivalTrace":
        """Sample per-slot counts from empirical piecewise-constant intensities."""
        if intensity_window_slots <= 0:
            raise ValueError("intensity_window_slots must be positive")
        rng = np.random.default_rng(seed)
        keys = [
            (app.id, ingress)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
        ]
        rates: dict[int, dict[tuple[str, str], float]] = {}
        seconds = scenario.simulation.slot_seconds
        for slot in range(slots):
            left = (slot // intensity_window_slots) * intensity_window_slots
            right = min(slots, left + intensity_window_slots)
            rates[slot] = {}
            for key in keys:
                samples = [
                    source.at(index, scenario).get(key, 0.0)
                    for index in range(left, right)
                ]
                intensity_rps = float(np.mean(samples)) if samples else 0.0
                count = rng.poisson(intensity_rps * seconds)
                rates[slot][key] = float(count) / seconds
        return ArrivalTrace(rates)

    @staticmethod
    def homogeneous_poisson(
        scenario: Scenario,
        source: "ArrivalTrace",
        slots: int,
        seed: int = 0,
    ) -> "ArrivalTrace":
        """Sample stationary Poisson arrivals with source-trace mean rates."""
        rng = np.random.default_rng(seed)
        seconds = scenario.simulation.slot_seconds
        keys = [
            (app.id, ingress)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
        ]
        means = {
            key: float(
                np.mean(
                    [source.at(slot, scenario).get(key, 0.0) for slot in range(slots)]
                )
            )
            for key in keys
        }
        return ArrivalTrace(
            {
                slot: {
                    key: float(rng.poisson(rate * seconds)) / seconds
                    for key, rate in means.items()
                }
                for slot in range(slots)
            }
        )

    @staticmethod
    def synthetic_bursty(
        scenario: Scenario,
        slots: int,
        seed: int = 0,
        burst_probability: float = 0.08,
        burst_multiplier: float = 2.5,
    ) -> "ArrivalTrace":
        rng = np.random.default_rng(seed)
        current_multiplier = 1.0
        remaining_burst = 0
        rates: dict[int, dict[tuple[str, str], float]] = {}
        for slot in range(slots):
            if remaining_burst <= 0 and rng.random() < burst_probability:
                remaining_burst = int(rng.integers(3, 10))
                current_multiplier = burst_multiplier
            elif remaining_burst <= 0:
                current_multiplier = 1.0
            rates[slot] = {}
            for app in scenario.applications.values():
                for ingress, base_rate in app.ingress_rates.items():
                    noise = float(rng.lognormal(mean=-0.5 * 0.12**2, sigma=0.12))
                    rates[slot][(app.id, ingress)] = base_rate * current_multiplier * noise
            remaining_burst -= 1
        return ArrivalTrace(rates)

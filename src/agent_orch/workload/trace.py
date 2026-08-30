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
        base = {
            (app.id, ingress): rate
            for app in scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        }
        base.update(self.rates.get(slot, {}))
        return base

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
        return ArrivalTrace(rates)

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


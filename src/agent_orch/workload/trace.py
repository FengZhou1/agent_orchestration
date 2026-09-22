from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

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
    def bursty_intensity(
        scenario: Scenario,
        slots: int,
        low_scale: float,
        high_scale: float,
        period: int,
        duty: float = 0.5,
    ) -> "ArrivalTrace":
        """Two-level (diurnal-style) arrival intensity.

        A stationary intensity leaves the deployment decision with a single
        constant optimum, so a deployment policy has nothing to react to.  This
        alternates between a low and a high level with a fixed period, which
        makes the best deployment time-varying and gives a reactive policy
        something to win: the first ``duty`` fraction of every period is high.
        """

        if slots <= 0:
            raise ValueError("slots must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if not 0.0 <= duty <= 1.0:
            raise ValueError("duty must lie in [0, 1]")
        if low_scale < 0.0 or high_scale < 0.0:
            raise ValueError("rate scales must be non-negative")
        base = {
            (app.id, ingress): float(rate)
            for app in scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        }
        high_slots = int(round(period * duty))
        rates: dict[int, dict[tuple[str, str], float]] = {}
        for slot in range(slots):
            scale = high_scale if (slot % period) < high_slots else low_scale
            rates[slot] = {key: scale * value for key, value in base.items()}
        return ArrivalTrace(rates)

    @staticmethod
    def gaussian_burst_intensity(
        scenario: Scenario,
        slots: int,
        base_scale: float,
        burst_scale: float,
        period: int,
        sigma: float,
        phase: float = 0.0,
        seed: int | None = None,
        jitter: float = 0.0,
    ) -> "ArrivalTrace":
        """Baseline intensity with periodic Gaussian bursts on top.

        The deployment decision only becomes time-varying when the intensity
        varies: with a constant intensity there is one fixed optimum and nothing
        to react to.  Each cycle of ``period`` slots carries one burst centred at
        ``phase * period`` with standard deviation ``sigma``, so intensity rises
        from ``base_scale`` to roughly ``base_scale + burst_scale`` and back.

        ``seed`` and ``jitter`` add a multiplicative per-slot perturbation, which
        makes an independent realisation of the same process -- exactly what a
        validation trace has to be if it is to measure adaptation rather than
        repeat a fixed operating point.
        """

        if slots <= 0:
            raise ValueError("slots must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if base_scale < 0.0 or burst_scale < 0.0:
            raise ValueError("rate scales must be non-negative")
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must lie in [0, 1)")
        centre = float(phase) * float(period)
        rng = np.random.default_rng(0 if seed is None else int(seed))
        base = {
            (app.id, ingress): float(rate)
            for app in scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        }
        rates: dict[int, dict[tuple[str, str], float]] = {}
        for slot in range(slots):
            offset = slot % period
            distance = min(
                abs(offset - centre), abs(offset - centre + period), abs(offset - centre - period)
            )
            burst = burst_scale * math.exp(-(distance ** 2) / (2.0 * sigma ** 2))
            scale = base_scale + burst
            if jitter > 0.0:
                scale *= float(1.0 + jitter * (2.0 * rng.random() - 1.0))
            rates[slot] = {key: max(0.0, scale * value) for key, value in base.items()}
        return ArrivalTrace(rates)

    @staticmethod
    def total_intensity(trace: "ArrivalTrace", slot: int, scenario: Scenario) -> float:
        """Total arrival intensity of one slot, for state construction."""

        return float(sum(trace.at(slot, scenario).values()))

    @staticmethod
    def stationary_poisson(
        scenario: Scenario,
        slots: int,
        seed: int = 0,
        rate_scale: float = 1.0,
    ) -> "ArrivalTrace":
        """Compatibility alias for :meth:`stationary_poisson_intensity`.

        The name is historical.  The analytical model consumes an *intensity*
        (requests per second) per slot, not a sampled arrival sequence, so
        ``seed`` is accepted and ignored: two calls with different seeds return
        identical traces.  Use ``stationary_poisson_intensity`` in new code.
        """

        del seed
        return ArrivalTrace.stationary_poisson_intensity(
            scenario,
            slots,
            rate_scale=rate_scale,
        )

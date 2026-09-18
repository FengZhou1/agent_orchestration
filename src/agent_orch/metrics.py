from __future__ import annotations

from collections.abc import Iterable, Mapping


def summarize_slot_metrics(
    records: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate physical-slot metrics using optional slot multiplicities.

    ``slot_weight`` defaults to one.  A weight larger than one represents repeated
    stationary periods and lets analytical experiments collapse identical periods
    without changing their horizon-level statistics.
    """
    rows = list(records)
    if not rows:
        raise ValueError("At least one physical-slot metric record is required")

    weights = [float(row.get("slot_weight", 1.0)) for row in rows]
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("slot_weight must be positive")
    slots = sum(weights)
    total_arrival = sum(
        weight * float(row["total_arrival_rps"])
        for row, weight in zip(rows, weights)
    )
    total_goodput = sum(
        weight * float(row["goodput_rps"])
        for row, weight in zip(rows, weights)
    )

    if total_arrival > 0.0:
        mean_latency = sum(
            weight
            * float(row["total_arrival_rps"])
            * float(row["mean_latency_s"])
            for row, weight in zip(rows, weights)
        ) / total_arrival
        mean_quality = sum(
            weight * float(row["total_arrival_rps"]) * float(row["quality"])
            for row, weight in zip(rows, weights)
        ) / total_arrival
        slo_attainment = total_goodput / total_arrival
    else:
        mean_latency = 0.0
        mean_quality = 0.0
        slo_attainment = 0.0

    violation_slots = sum(
        weight * int(float(row["violations"]) > 0.0)
        for row, weight in zip(rows, weights)
    )
    return {
        "mean_cost": sum(
            weight * float(row["cost"]) for row, weight in zip(rows, weights)
        ) / slots,
        "mean_latency_s": mean_latency,
        "mean_goodput_rps": total_goodput / slots,
        "mean_quality": mean_quality,
        "mean_slo_attainment": slo_attainment,
        "mean_violations": sum(
            weight * float(row["violations"])
            for row, weight in zip(rows, weights)
        ) / slots,
        "violation_slot_fraction": violation_slots / slots,
    }

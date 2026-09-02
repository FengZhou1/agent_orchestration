from __future__ import annotations

from collections.abc import Iterable, Mapping


def summarize_slot_metrics(
    records: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate equal-duration physical-slot metrics using request-rate weights."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one physical-slot metric record is required")

    slots = len(rows)
    total_arrival = sum(float(row["total_arrival_rps"]) for row in rows)
    total_goodput = sum(float(row["goodput_rps"]) for row in rows)

    if total_arrival > 0.0:
        mean_latency = sum(
            float(row["total_arrival_rps"]) * float(row["mean_latency_s"])
            for row in rows
        ) / total_arrival
        mean_quality = sum(
            float(row["total_arrival_rps"]) * float(row["quality"])
            for row in rows
        ) / total_arrival
        slo_attainment = total_goodput / total_arrival
    else:
        mean_latency = 0.0
        mean_quality = 0.0
        slo_attainment = 0.0

    violation_slots = sum(int(float(row["violations"]) > 0.0) for row in rows)
    return {
        "mean_cost": sum(float(row["cost"]) for row in rows) / slots,
        "mean_latency_s": mean_latency,
        "mean_goodput_rps": total_goodput / slots,
        "mean_quality": mean_quality,
        "mean_slo_attainment": slo_attainment,
        "mean_violations": sum(float(row["violations"]) for row in rows) / slots,
        "violation_slot_fraction": violation_slots / slots,
    }

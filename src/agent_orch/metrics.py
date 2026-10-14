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
        # Pool completed requests within each application, then give each
        # application one vote. Unfinished requests never enter this mean.
        app_totals, app_mass = {}, {}
        for row, weight in zip(rows, weights):
            for app, latency in row.get("app_latency_s", {}).items():
                mass = weight * row.get("app_completed_rps", {}).get(app, 0.0)
                app_totals[app] = app_totals.get(app, 0.0) + mass * latency
                app_mass[app] = app_mass.get(app, 0.0) + mass
        means = [app_totals[a] / mass for a, mass in app_mass.items() if mass > 0]
        if means:
            mean_latency = sum(means) / len(means)
        else:
            valid = [(row, weight * row.get("completed_rps", row["total_arrival_rps"]))
                     for row, weight in zip(rows, weights)
                     if row.get("mean_latency_s") is not None]
            mass = sum(w for _, w in valid)
            mean_latency = (sum(w * row["mean_latency_s"] for row, w in valid) / mass
                            if mass > 0 else None)
        mean_quality = sum(
            weight * float(row["total_arrival_rps"]) * float(row["quality"])
            for row, weight in zip(rows, weights)
        ) / total_arrival
        slo_attainment = total_goodput / total_arrival
    else:
        mean_latency = None
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
        "completion_fraction": sum(w * r.get("completed_rps", r["total_arrival_rps"])
                                   for r, w in zip(rows, weights)) / total_arrival
                               if total_arrival > 0 else None,
        "mean_goodput_rps": total_goodput / slots,
        "mean_quality": mean_quality,
        "mean_slo_attainment": slo_attainment,
        "mean_violations": sum(
            weight * float(row["violations"])
            for row, weight in zip(rows, weights)
        ) / slots,
        "violation_slot_fraction": violation_slots / slots,
    }

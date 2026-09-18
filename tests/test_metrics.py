import pytest

from agent_orch.metrics import summarize_slot_metrics


def test_slot_metrics_use_request_rate_weighting():
    records = [
        {
            "cost": 3.0,
            "mean_latency_s": 10.0,
            "goodput_rps": 1.0,
            "quality": 0.8,
            "total_arrival_rps": 2.0,
            "violations": 2,
        },
        {
            "cost": 1.0,
            "mean_latency_s": 0.0,
            "goodput_rps": 0.0,
            "quality": 0.0,
            "total_arrival_rps": 0.0,
            "violations": 0,
        },
        {
            "cost": 2.0,
            "mean_latency_s": 4.0,
            "goodput_rps": 1.0,
            "quality": 0.5,
            "total_arrival_rps": 1.0,
            "violations": 1,
        },
    ]

    summary = summarize_slot_metrics(records)

    assert summary["mean_cost"] == pytest.approx(2.0)
    assert summary["mean_latency_s"] == pytest.approx(8.0)
    assert summary["mean_goodput_rps"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_quality"] == pytest.approx(0.7)
    assert summary["mean_slo_attainment"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_violations"] == pytest.approx(1.0)
    assert summary["violation_slot_fraction"] == pytest.approx(2.0 / 3.0)


def test_slot_metrics_handle_zero_arrival_horizon():
    summary = summarize_slot_metrics(
        [
            {
                "cost": 1.0,
                "mean_latency_s": 0.0,
                "goodput_rps": 0.0,
                "quality": 0.0,
                "total_arrival_rps": 0.0,
                "violations": 0,
            }
        ]
    )

    assert summary["mean_latency_s"] == 0.0
    assert summary["mean_quality"] == 0.0
    assert summary["mean_slo_attainment"] == 0.0
    assert summary["violation_slot_fraction"] == 0.0


def test_slot_metrics_support_collapsed_stationary_periods():
    compressed = [
        {
            "cost": 5.0,
            "mean_latency_s": 10.0,
            "goodput_rps": 1.0,
            "quality": 0.5,
            "total_arrival_rps": 2.0,
            "violations": 1,
            "slot_weight": 1.0,
        },
        {
            "cost": 1.0,
            "mean_latency_s": 4.0,
            "goodput_rps": 2.0,
            "quality": 0.8,
            "total_arrival_rps": 2.0,
            "violations": 0,
            "slot_weight": 3.0,
        },
    ]
    expanded = [
        {key: value for key, value in compressed[0].items() if key != "slot_weight"},
        *[
            {key: value for key, value in compressed[1].items() if key != "slot_weight"}
            for _ in range(3)
        ],
    ]

    assert summarize_slot_metrics(compressed) == pytest.approx(
        summarize_slot_metrics(expanded)
    )

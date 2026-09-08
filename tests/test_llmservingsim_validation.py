from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent_orch.validation.llmservingsim import (
    AnalyticalParameters,
    WorkloadClass,
    allen_cunneen_wait,
    calibrate_effective_concurrency,
    fit_effective_rates,
    generate_poisson_trace,
    read_simulator_output,
    roofline_service,
    saturated_throughput,
)


def test_poisson_trace_is_reproducible_and_has_expected_rate(tmp_path) -> None:
    workload = WorkloadClass("x", 128, 64)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    generate_poisson_trace(first, first_manifest, {"x": workload}, {"x": 1.0}, 50_000, 4.0, 7)
    generate_poisson_trace(second, second_manifest, {"x": workload}, {"x": 1.0}, 50_000, 4.0, 7)
    assert first.read_bytes() == second.read_bytes()
    arrivals = pd.read_csv(first_manifest)["arrival_time_ns"].to_numpy(dtype=float) / 1e9
    observed_rate = 1.0 / np.mean(np.diff(arrivals))
    assert abs(observed_rate - 4.0) / 4.0 < 0.05


def test_simulator_metric_mapping(tmp_path) -> None:
    path = tmp_path / "sim.csv"
    pd.DataFrame(
        [
            {
                "instance id": 0,
                "request id": 3,
                "model": "m",
                "input": 10,
                "output": 5,
                "arrival": 100_000_000,
                "end_time": 2_100_000_000,
                "latency": 2_000_000_000,
                "queuing_delay": 300_000_000,
                "TTFT": 800_000_000,
                "TPOT": 300_000_000,
                "ITL": "[]",
            }
        ]
    ).to_csv(path, index=False)
    row = read_simulator_output(path).iloc[0]
    assert row.waiting_s == 0.3
    assert row.prefill_s == 0.5
    assert row.decode_s == 1.2
    assert row.service_s == 1.7
    assert abs(row.service_s - row.prefill_s - row.decode_s) < 1e-12


def test_effective_rate_fit_recovers_synthetic_service() -> None:
    true = AnalyticalParameters(effective_flops=67e12, effective_bandwidth_bytes_s=745e9)
    workloads = [
        WorkloadClass("a", 64, 64),
        WorkloadClass("b", 512, 128),
        WorkloadClass("c", 2048, 256),
        WorkloadClass("d", 4096, 512),
    ]
    records = []
    for workload in workloads:
        service = roofline_service(workload, true, concurrency=1)
        records.append(
            {
                "prompt_tokens": workload.prompt_tokens,
                "output_tokens": workload.output_tokens,
                "prefill_s": service.prefill_s,
                "decode_s": service.decode_s,
            }
        )
    fitted, info = fit_effective_rates(pd.DataFrame(records))
    assert info["success"]
    for workload in workloads:
        expected = roofline_service(workload, true, concurrency=1)
        actual = roofline_service(workload, fitted, concurrency=1)
        assert abs(actual.prefill_s - expected.prefill_s) / expected.prefill_s < 1e-4
        assert abs(actual.decode_s - expected.decode_s) / expected.decode_s < 1e-4


def test_allen_cunneen_reduces_to_pollaczek_khinchine_for_one_server() -> None:
    arrival = 0.3
    mean = 2.0
    second = 5.0
    waiting, utilization, overloaded = allen_cunneen_wait(arrival, mean, second, 1)
    expected = arrival * second / (2.0 * (1.0 - arrival * mean))
    assert not overloaded
    assert utilization == arrival * mean
    assert abs(waiting - expected) < 1e-12


def test_capacity_and_effective_concurrency_calibration() -> None:
    completion = pd.DataFrame({"end_s": np.arange(1.0, 101.0)})
    assert abs(saturated_throughput(completion) - 1.0) < 0.02
    workload = WorkloadClass("x", 128, 64)
    params = AnalyticalParameters()
    target_b = 4
    service = roofline_service(workload, params, target_b)
    target_capacity = target_b / service.service_s
    fitted_b, candidates = calibrate_effective_concurrency(
        target_capacity, {"x": 1.0}, {"x": workload}, params, maximum=16
    )
    assert fitted_b == target_b
    assert len(candidates) == 16


def test_generated_jsonl_uses_llmservingsim_schema(tmp_path) -> None:
    output = tmp_path / "trace.jsonl"
    manifest = tmp_path / "manifest.csv"
    generate_poisson_trace(
        output,
        manifest,
        {"x": WorkloadClass("x", 93, 318)},
        {"x": 1.0},
        2,
        None,
        0,
        simultaneous=True,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"input_toks": 93, "output_toks": 318, "arrival_time_ns": 0},
        {"input_toks": 93, "output_toks": 318, "arrival_time_ns": 0},
    ]

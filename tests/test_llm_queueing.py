from __future__ import annotations

import math

from agent_orch.performance.llm_queueing import (
    IterationCalibration,
    admission_queue_wait,
    aggregate_iteration_work,
    aggregate_packed_iteration_work,
    class_service_times,
)
from agent_orch.schema.models import ModelSpec


def _model() -> ModelSpec:
    return ModelSpec(
        id="toy",
        parameter_count=1.0e9,
        layers=4,
        hidden_size=1024,
        weight_bytes=2.0e9,
        kv_bytes_per_token=4096.0,
    )


def test_mixed_iteration_counts_weights_once() -> None:
    model = _model()
    one_flops, one_bytes = aggregate_iteration_work(model, 128, 64, 1)
    two_flops, two_bytes = aggregate_iteration_work(model, 128, 64, 2)
    assert two_flops > one_flops
    # The second decode stream adds KV traffic but not a second model-weight read.
    assert two_bytes - one_bytes < model.weight_bytes


def test_packed_prefill_does_not_create_cross_request_attention() -> None:
    model = _model()
    packed_flops, packed_bytes = aggregate_packed_iteration_work(
        model, [(512, 0), (512, 0)], 0, 0
    )
    monolithic_flops, monolithic_bytes = aggregate_iteration_work(model, 1024, 0, 0)
    assert packed_flops < monolithic_flops
    # Independent chunks still share the weight read of one GPU iteration.
    assert packed_bytes < 2.0 * model.weight_bytes
    assert monolithic_bytes == packed_bytes


def test_mg1_wait_is_zero_at_zero_load_and_infinite_at_overload() -> None:
    services = [(1.0, 2.0, 3.0)]
    assert admission_queue_wait(0.0, [(10, 10)], [1.0], services, 1.0, 1.0)[0] == 0.0
    waiting, utilization, overloaded = admission_queue_wait(
        1.1, [(10, 10)], [1.0], services, 1.0, 1.0
    )
    assert math.isinf(waiting)
    assert utilization > 1.0
    assert overloaded


def test_service_time_increases_with_resident_decode_population() -> None:
    calibration = IterationCalibration(1.0e12, 1.0e12, 0.0)
    first = class_service_times(_model(), calibration, 128, 32, 512, 1.0)[2]
    many = class_service_times(_model(), calibration, 128, 32, 512, 8.0)[2]
    assert many > first

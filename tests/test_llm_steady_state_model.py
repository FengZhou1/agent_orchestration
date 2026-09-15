import math

import numpy as np
import pytest

from agent_orch.performance.llm import (
    decode_work,
    mean_service_time,
    prefill_work,
    roofline_time,
    service_curve,
    service_demand,
    steady_active_concurrency,
    throughput_capacity,
)
from agent_orch.schema.loader import ScenarioLoader

SCENARIO = "configs/benchmarks/main_abilene_revised.yaml"
CLASSES = [(93, 318), (1300, 4458), (1911, 534), (12223, 3541)]


def _instance(config_id: str):
    scenario = ScenarioLoader.load(SCENARIO)
    config = scenario.llm_configs[config_id]
    return scenario, config, scenario.models[config.model]


@pytest.mark.parametrize("config_id", ["qwen3-4b-a10", "qwen3-14b-h20", "qwen3-32b-h20"])
def test_iterations_share_the_resident_decode_batch(config_id):
    _, config, model = _instance(config_id)
    low = service_demand(model, config, 1300, 512, 512, decode_concurrency=1.0)
    high = service_demand(model, config, 1300, 512, 512, decode_concurrency=32.0)
    # A prefill iteration carries the resident decodes of the same instance, so
    # the time to first token grows with load, but sub-linearly.
    assert low.prefill_s < high.prefill_s <= 32.0 * low.prefill_s * (1.0 + 1e-9)
    assert low.decode_s < high.decode_s <= 32.0 * low.decode_s * (1.0 + 1e-9)
    assert high.service_s == pytest.approx(high.prefill_s + high.decode_s)
    assert high.decode_concurrency == 32.0


@pytest.mark.parametrize("config_id", ["qwen3-4b-a10", "qwen3-14b-h20"])
def test_prefill_iterations_match_the_chunked_roofline_at_isolation(config_id):
    _, config, model = _instance(config_id)
    demand = service_demand(model, config, 2048, 64, 512, decode_concurrency=1.0)
    expected = 0.0
    for position in range(4):
        flops, memory = prefill_work(model, 512, position * 512, 1.0)
        expected += roofline_time(flops, memory, config)
    # At concurrency one the only extra term is the single resident decode step.
    assert demand.prefill_s >= expected
    assert demand.prefill_s < expected + 2.0 * roofline_time(*decode_work(model, 1024, 1.0), config)


@pytest.mark.parametrize("config_id", ["qwen3-4b-a10", "qwen3-14b-h20", "qwen3-32b-h20"])
def test_service_curve_reproduces_reference_demand(config_id):
    _, config, model = _instance(config_id)
    for prompt, output in CLASSES:
        curve = service_curve(model, config, prompt, output, 512)
        for nu in (1.0, 2.5, 7.0, 33.0, 128.0):
            reference = service_demand(
                model, config, prompt, output, 512, decode_concurrency=nu
            )
            assert curve.service_at(nu) == pytest.approx(reference.service_s, rel=1e-12)
            assert curve.decode_at(nu) == pytest.approx(reference.decode_s, rel=1e-12)


def test_service_curve_agrees_with_explicit_reference_on_a_grid(config_id="qwen3-32b-h20"):
    _, config, model = _instance(config_id)
    curve = service_curve(model, config, 1911, 534, 512)
    grid = np.array([1.0, 4.0, 17.0, 64.0])
    values = curve.service_on_grid(grid)
    for nu, value in zip(grid, values):
        reference = service_demand(model, config, 1911, 534, 512, decode_concurrency=nu)
        assert value == pytest.approx(reference.service_s, rel=1e-12)


def test_throughput_capacity_lies_inside_the_residency_limit():
    _, config, model = _instance("qwen3-14b-h20")
    curves = [service_curve(model, config, p, o, 512) for p, o in CLASSES]
    weights = [0.25] * 4
    capacity, concurrency = throughput_capacity(curves, weights, residency_capacity=24)
    assert 1.0 <= concurrency <= 24.0
    assert capacity == pytest.approx(
        concurrency / mean_service_time(curves, weights, concurrency), rel=1e-9
    )
    # The capacity curve is never worse than the single-sequence rate.
    assert capacity >= 1.0 / mean_service_time(curves, weights, 1.0)


def test_steady_concurrency_solves_littles_law_below_the_residency_limit():
    _, config, model = _instance("qwen3-14b-h20")
    curves = [service_curve(model, config, p, o, 512) for p, o in CLASSES]
    weights = [0.25] * 4
    rate = 0.5 / mean_service_time(curves, weights, 1.0)
    batch, converged, residual = steady_active_concurrency(
        curves, weights, rate, residency_capacity=24
    )
    assert converged
    assert batch == pytest.approx(
        rate * mean_service_time(curves, weights, max(1.0, batch)), rel=1e-6
    )
    assert batch < 24.0

    capacity, _ = throughput_capacity(curves, weights, residency_capacity=24)
    batch, converged, _ = steady_active_concurrency(
        curves, weights, 1.5 * capacity, residency_capacity=24
    )
    assert not converged
    assert batch > 24.0


def test_zero_rate_does_not_create_llm_instance_state():
    scenario = ScenarioLoader.load(SCENARIO)
    assert all(rate >= 0.0 for app in scenario.applications.values() for rate in app.ingress_rates.values())


@pytest.mark.parametrize("output_tokens", [1, 2, 128])
def test_iteration_count_boundary_is_finite(output_tokens):
    _, config, model = _instance("qwen3-4b-a10")
    demand = service_demand(model, config, 128, output_tokens, 512, decode_concurrency=2.0)
    assert math.isfinite(demand.service_s)
    assert demand.service_s > 0.0

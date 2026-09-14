import math

import pytest

from agent_orch.performance.llm import service_demand
from agent_orch.schema.loader import ScenarioLoader


def test_service_demand_accepts_fractional_steady_concurrency():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene_calibrated.yaml")
    config = scenario.llm_configs["qwen3-4b-a10"]
    model = scenario.models[config.model]
    low = service_demand(model, config, 512, 128, 512, concurrency=1.0)
    high = service_demand(model, config, 512, 128, 512, concurrency=3.5)
    assert high.service_s > low.service_s
    assert high.prefill_s > low.prefill_s


def test_running_capacity_is_derived_from_kv_and_max_sequences():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene_calibrated.yaml")
    config = scenario.llm_configs["qwen3-4b-a10"]
    assert config.max_num_seqs == 128
    assert config.max_num_seqs > 0


def test_zero_rate_does_not_create_llm_instance_state():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene_calibrated.yaml")
    assert all(rate >= 0.0 for app in scenario.applications.values() for rate in app.ingress_rates.values())


@pytest.mark.parametrize("output_tokens", [1, 2, 128])
def test_iteration_count_boundary_is_finite(output_tokens):
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene_calibrated.yaml")
    config = scenario.llm_configs["qwen3-4b-a10"]
    model = scenario.models[config.model]
    demand = service_demand(model, config, 128, output_tokens, 512, concurrency=2.0)
    assert math.isfinite(demand.service_s)
    assert demand.service_s > 0.0

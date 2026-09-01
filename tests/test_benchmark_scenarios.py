from pathlib import Path
from runpy import run_path

import pytest

from agent_orch.schema.loader import ScenarioLoader
from agent_orch.baselines import GreedyPolicy
from agent_orch.performance import AnalyticalBackend


BUILDER = run_path("scripts/build_benchmark_scenarios.py", run_name="test_module")


def test_checked_in_benchmark_scenarios_match_protocol():
    root = Path("configs/benchmarks")
    main = ScenarioLoader.load(root / "main_abilene.yaml")
    scale = ScenarioLoader.load(root / "scale_geant.yaml")
    assert (len(main.servers), len(main.links) // 2) == (12, 15)
    assert (len(scale.servers), len(scale.links) // 2) == (22, 36)
    assert len(main.applications) == 20
    assert len(scale.applications) == 50
    assert len(main.tools) == 6
    assert len(scale.tools) == 8
    assert main.simulation.slot_seconds == 1.0
    assert main.simulation.deployment_period_slots == 60
    assert main.metadata["units"]["link_load"] == "Mbit/s"


def test_main_scenario_has_balanced_families_and_length_classes():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    family_counts = {}
    for app in scenario.applications.values():
        family_counts[app.family] = family_counts.get(app.family, 0) + 1
    assert set(family_counts.values()) == {5}
    assert {app.length_class for app in scenario.applications.values()} == {
        "short",
        "medium",
        "long",
    }
    family_rates = {
        family: sum(
            sum(app.ingress_rates.values())
            for app in scenario.applications.values()
            if app.family == family
        )
        for family in family_counts
    }
    assert all(rate == pytest.approx(0.015) for rate in family_rates.values())


def test_main_default_workload_is_stable_for_reference_deployment():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    policy = GreedyPolicy(scenario, 2026)
    deployment = policy.deployment()
    result = AnalyticalBackend(scenario).evaluate(
        deployment, policy.routing(deployment)
    )
    assert all(value < 1.0 for value in result.llm_utilization.values())
    assert all(value < 1.0 for value in result.tool_utilization.values())
    assert all(result.llm_kv_stable.values())
    assert not any("overload" in violation for violation in result.violations)

    stressed_rates = {
        (app.id, ingress): 3.0 * rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    stressed = AnalyticalBackend(scenario).evaluate(
        deployment, policy.routing(deployment), stressed_rates
    )
    assert any("overload" in violation for violation in stressed.violations)
    shared = set(result.llm_performance) & set(stressed.llm_performance)
    assert max(stressed.llm_performance[key].response_s for key in shared) > (
        max(result.llm_performance[key].response_s for key in shared) + 50.0
    )


def test_vllm_configuration_is_fixed_and_gpu_compatible():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    for candidate in scenario.candidates.values():
        config = scenario.llm_configs[candidate.config]
        server = scenario.servers[candidate.server]
        assert config.gpu_type == server.gpu_type
        assert config.gpu_count <= server.gpu_count
        assert config.dtype == "bfloat16"
        assert config.gpu_memory_utilization == pytest.approx(0.9)
        assert config.max_model_len == 32768
        assert config.max_num_batched_tokens == 8192
        assert config.max_num_seqs == 128
        assert config.chunked_prefill
        assert not config.prefix_cache


def test_builder_is_deterministic():
    first = BUILDER["build_scenario"]("abilene", 20, 2026)
    second = BUILDER["build_scenario"]("abilene", 20, 2026)
    first["metadata"].pop("generated_on")
    second["metadata"].pop("generated_on")
    assert first == second

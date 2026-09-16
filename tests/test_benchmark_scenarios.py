import hashlib
import json
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
    assert main.simulation.orchestration_period_s == 60.0
    assert main.simulation.overload_delay_s == 600.0
    assert main.metadata["units"]["link_load"] == "Mbit/s"
    assert main.metadata["workload_class_count"] == 20
    assert main.metadata["pattern_flow_count"] == 65
    assert main.metadata["request_token_budget"] == 32000
    assert not Path(main.metadata["preconstructed_workload_path"]).is_absolute()


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
    assert all(rate == pytest.approx(0.001) for rate in family_rates.values())


def test_preconstructed_workload_quantiles_are_applied_to_each_family():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    source_keys = {
        "interactive_retrieval": "interactive_rag",
        "transactional_tool": "transactional_tool",
        "deep_research": "deep_research",
        "coding_agent": "coding_agent",
    }
    source = BUILDER["PRECONSTRUCTED_WORKLOADS"]
    for family, source_key in source_keys.items():
        application = next(
            app
            for app in scenario.applications.values()
            if app.family == family and app.template_id.endswith(":3")
        )
        workload = source[source_key]
        for node_id, node in workload["nodes"].items():
            if node["type"] != "llm":
                continue
            actual = application.nodes[node_id]
            assert node["input_tokens"]["p50"] <= actual.prompt_tokens["qwen3-14b"] <= node["input_tokens"]["p95"]
            assert node["output_tokens"]["p50"] <= actual.output_tokens["qwen3-14b"] <= node["output_tokens"]["p95"]


def test_preconstructed_choices_are_fully_expanded():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    expected_flow_counts = {
        "interactive_retrieval": 3,
        "transactional_tool": 6,
        "deep_research": 2,
        "coding_agent": 2,
    }
    for family, expected_count in expected_flow_counts.items():
        applications = [app for app in scenario.applications.values() if app.family == family]
        assert applications
        assert {len(app.pattern_flows) for app in applications} == {expected_count}
        assert all(
            sum(flow.probability for flow in app.pattern_flows) == pytest.approx(1.0)
            for app in applications
        )


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
        (app.id, ingress): 4.0 * rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    stressed = AnalyticalBackend(scenario).evaluate(
        deployment, policy.routing(deployment), stressed_rates
    )
    # KV enters as a utilization rather than a hard constraint, so the load
    # response is a monotone rise in bottleneck pressure and latency.
    peak = max(result.llm_utilization.values())
    stressed_peak = max(stressed.llm_utilization.values())
    assert stressed_peak > peak
    shared = set(result.llm_performance) & set(stressed.llm_performance)
    assert max(stressed.llm_performance[key].response_s for key in shared) > max(
        result.llm_performance[key].response_s for key in shared
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
        assert config.kv_token_capacity >= config.max_model_len


def test_application_tokens_keep_engine_margin():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    budget = scenario.metadata["request_token_budget"]
    for application in scenario.applications.values():
        for node in application.nodes.values():
            for model in node.prompt_tokens:
                assert node.prompt_tokens[model] + node.output_tokens[model] <= budget


def test_load_levels_are_calibrated_from_checked_in_main_scenario():
    scenario_path = Path("configs/benchmarks/main_abilene.yaml")
    payload = json.loads(Path("data/processed/load_levels.json").read_text(encoding="utf-8"))
    assert payload["scenario"] == "configs/benchmarks/main_abilene.yaml"
    assert payload["scenario_hash"] == hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    assert [level["target_load"] for level in payload["levels"]] == [0.4, 0.65, 0.85, 1.05]


def test_main_gpu_mix_and_model_deployment_coverage():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    gpu_counts = {gpu_type: 0 for gpu_type in ("A10", "L20", "H20")}
    for server in scenario.servers.values():
        gpu_counts[server.gpu_type] += 1
    assert gpu_counts == {"A10": 4, "L20": 5, "H20": 3}
    assert set(scenario.models) == {
        "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"
    }
    deployed_models = {candidate.model for candidate in scenario.candidates.values()}
    assert deployed_models == set(scenario.models)


def test_candidate_configs_use_intended_gpu_model_pairs():
    scenario = ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")
    pairs = {
        (config.model, config.gpu_type, config.gpu_count)
        for config in scenario.llm_configs.values()
    }
    assert pairs == {
        ("qwen3-4b", "A10", 1),
        ("qwen3-4b", "L20", 1),
        ("qwen3-8b", "L20", 1),
        ("qwen3-8b", "H20", 1),
        ("qwen3-14b", "L20", 1),
        ("qwen3-14b", "H20", 1),
        ("qwen3-32b", "H20", 1),
        ("qwen3-32b", "L20", 2),
    }


def test_builder_is_deterministic():
    first = BUILDER["build_scenario"]("abilene", 20, 2026)
    second = BUILDER["build_scenario"]("abilene", 20, 2026)
    first["metadata"].pop("generated_on")
    second["metadata"].pop("generated_on")
    assert first == second

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import networkx as nx
import pytest
import torch

from agent_orch.agents import StructuredActorCritic
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.envs.layout import StructuredActionLayout
from agent_orch.parameters import (
    audit_scenario, load_settings, make_ppo_config, ppo_config_from_manifest,
    require_consistent, require_formal_ready,
)
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload.agent_pool import (
    GeneratorConfig, digest, generate_pool, pool_from_mapping, scenario_mapping,
)
from agent_orch.workload.infrastructure import build_infrastructure, remap_pool_payload
from agent_orch.workload.slot_trajectory import SlotTrajectory


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def fixed_pool(settings):
    infrastructure = build_infrastructure(settings)
    pool = generate_pool(infrastructure, GeneratorConfig(applications=20, tool_types=20,
                                                         calibration_passes=0))
    return infrastructure, pool


@pytest.mark.parametrize("count", [10, 20, 30, 40, 50])
def test_registered_network_is_connected_and_has_no_legacy_profiles(settings, count):
    scene = build_infrastructure(settings, count)
    assert digest(scenario_mapping(scene)) == digest(scenario_mapping(build_infrastructure(settings, count)))
    assert len(scene.servers) == count
    graph = nx.Graph()
    graph.add_nodes_from(scene.servers)
    graph.add_edges_from((e.source, e.target) for e in scene.links)
    assert nx.is_connected(graph)
    assert graph.number_of_edges() == count * 2
    assert len(scene.links) == count * 4
    from agent_orch.performance.network import NetworkBackend
    network = NetworkBackend(scene.links)
    for u in scene.servers:
        for v in scene.servers:
            network.path(u, v)
    assert {c.model for c in scene.candidates.values()} == set(scene.models)
    assert {s.gpu_type for s in scene.servers.values()} <= {"none", "RTX4090", "RTX_A6000"}
    require_consistent(audit_scenario(scene, settings, infrastructure_only=True))
    assert all(c.max_num_batched_tokens == 2048 and c.max_num_seqs == 256 and
               c.gpu_memory_utilization == .92 and not c.prefix_cache for c in scene.llm_configs.values())


def test_prefix_stable_hardware_and_distinct_trajectory_identity(settings):
    small, big = build_infrastructure(settings, 10), build_infrastructure(settings, 50)
    assert all(big.servers[n] == server for n, server in small.servers.items())
    assert small.id != big.id


def test_inventory_not_silently_resampled_and_unset_assumptions_block(settings):
    changed = copy.deepcopy(settings)
    changed["infrastructure"]["gpu_count_probabilities"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="cannot host every"):
        build_infrastructure(changed)
    changed["infrastructure"]["seconds_per_time_unit"] = None
    with pytest.raises(ValueError, match="unset"):
        build_infrastructure(changed)


def test_model_weights_are_not_rounded_and_kv_is_derived(settings):
    scene = build_infrastructure(settings)
    assert scene.models["qwen3-4b"].weight_bytes == 8044936192
    assert scene.models["qwen3-32b"].weight_bytes == 65524246528
    assert scene.models["qwen3-4b"].kv_bytes_per_token == 147456
    assert scene.models["qwen3-32b"].kv_bytes_per_token == 262144
    for c in scene.llm_configs.values():
        assert c.kv_token_capacity >= c.max_model_len


def test_pool_remapping_preserves_workload_and_total_demand(settings, fixed_pool):
    old, pool = fixed_pool
    new = build_infrastructure(settings, 50)
    mapped = pool_from_mapping(remap_pool_payload(pool.to_dict(), new), new)
    require_consistent(audit_scenario(mapped.scenario(new), settings))
    for a, b in zip(pool.applications, mapped.applications):
        assert a.graph == b.graph and a.lengths == b.lengths
        assert a.spec.nodes == b.spec.nodes
        assert a.spec.quality == b.spec.quality and a.spec.slo == b.spec.slo
        assert a.spec.edge_data_mb == b.spec.edge_data_mb
        assert sum(a.spec.ingress_rates.values()) == pytest.approx(sum(b.spec.ingress_rates.values()))
        assert set(b.spec.ingress_rates) <= set(new.servers)
        assert set(b.record["eligible_ingress"]) == set(new.servers)
    for key, tool in mapped.tools.items():
        assert tool.memory_gb == pool.tools[key].memory_gb
        assert set(tool.service_rate.values()) == set(pool.tools[key].service_rate.values())
        assert set(tool.service_rate) == set(new.servers)


def test_audit_inventory_covers_every_register_row_without_claiming_all_verified(settings, fixed_pool):
    base, pool = fixed_pool
    report = audit_scenario(pool.scenario(base), settings)
    require_consistent(report)
    ids = [row["id"] for row in report["register_rows"]]
    assert len(ids) >= 150 and len(ids) == len(set(ids))
    assert {"RL30", "HW07", "SC06", "NT11", "TK09"} <= set(ids)
    assert any(r["coverage"] == "manual_or_runtime" for r in report["register_rows"])
    assert not report["formal_ready"]
    with pytest.raises(ValueError, match="Formal training blocked"):
        require_formal_ready(settings)


def test_old_abilene_is_rejected_as_a_main_experiment(settings):
    report = audit_scenario(ScenarioLoader.load("configs/benchmarks/main_abilene.yaml"), settings)
    assert not report["consistent"]
    with pytest.raises(ValueError, match="conflicts"):
        require_consistent(report)


def test_registered_actor_depth_and_snapshot_restore(settings):
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 101), mapping_samples=2)
    config = make_ppo_config(settings)
    policy = StructuredActorCritic(env, config)
    assert config.learning_rate == config.composition_learning_rate == 1e-4
    assert config.minibatch_size == 32 and config.update_epochs == 3
    assert config.shared_composition_head and config.exploration_mode == "none"
    for encoder in (policy.deployment_encoder, policy.composition_encoder):
        linear = [m for m in encoder if isinstance(m, torch.nn.Linear)]
        assert len(linear) == 4 and all(m.out_features == 256 for m in linear)
    restored = ppo_config_from_manifest({"ppo_config": asdict(config)})
    replica = StructuredActorCritic(env, restored)
    replica.load_state_dict(policy.state_dict())


def test_public_ppo_defaults_match_registered_learning_candidates(settings):
    from agent_orch.agents import PPOConfig
    default = PPOConfig()
    registered = make_ppo_config(settings)
    for name in ("hidden_layers", "hidden_size", "learning_rate", "composition_learning_rate",
                 "update_epochs", "minibatch_size", "gamma", "gae_lambda", "clip_ratio",
                 "entropy_coefficient", "composition_entropy_coefficient", "shared_composition_head",
                 "exploration_mode", "shaping_coefficient"):
        assert getattr(default, name) == getattr(registered, name)


def test_zero_rate_keys_do_not_create_policy_actions():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    apps = dict(scenario.applications)
    app = next(iter(apps.values()))
    rates = dict(app.ingress_rates)
    spare = next(n for n in scenario.servers if n not in rates)
    rates[spare] = 0.
    apps[app.id] = replace(app, ingress_rates=rates)
    changed = replace(scenario, applications=apps)
    original = StructuredActionLayout.build(scenario)
    actual = StructuredActionLayout.build(changed)
    assert actual.model_groups == original.model_groups
    env = SlotSequentialJointEnv(changed, SlotTrajectory.sample(changed, 2, 101), mapping_samples=2)
    assert env._working_share.shape[0] == len(actual.model_groups)


def test_fixed_trajectory_matches_base_not_just_itself():
    scene = ScenarioLoader.load("configs/toy.yaml")
    trace = SlotTrajectory.sample(scene, 2, 101)
    raw = trace.to_dict()
    a = next(iter(raw["slots"][0]["prompt_tokens"]))
    n = next(iter(raw["slots"][0]["prompt_tokens"][a]))
    m = next(iter(raw["slots"][0]["prompt_tokens"][a][n]))
    raw = copy.deepcopy(raw)
    for slot in raw["slots"]:
        slot["prompt_tokens"][a][n][m] += 1
    corrupt = SlotTrajectory.from_dict(raw)
    with pytest.raises(ValueError, match="differ from the base"):
        corrupt.validate_against(scene)


def test_isolated_server_is_not_hidden_by_edge_only_connectivity(settings):
    scene = build_infrastructure(settings)
    isolated = next(iter(scene.servers))
    changed = replace(scene, links=tuple(e for e in scene.links if isolated not in (e.source, e.target)))
    report = audit_scenario(changed, settings, infrastructure_only=True)
    assert next(c for c in report["checks"] if c["parameters"] == ["NT01"])["status"] == "fail"


def test_undirected_connectivity_does_not_hide_missing_reverse_arcs(settings):
    scene = build_infrastructure(settings)
    changed = replace(scene, links=tuple(e for e in scene.links if e.source < e.target))
    report = audit_scenario(changed, settings, infrastructure_only=True)
    assert next(c for c in report["checks"] if c["parameters"] == ["NT01"])["status"] == "fail"


def test_v7_checkpoint_digest_can_be_verified_in_its_original_nested_location():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_slot_joint import checkpoint_trajectory_digest
    assert checkpoint_trajectory_digest({"trajectory_digest": "first"}) == "first"
    assert checkpoint_trajectory_digest({"environment_reward_state": {"trajectory_digest": "nested"}}) == "nested"
    assert checkpoint_trajectory_digest({}) is None


def test_retired_console_training_requires_explicit_legacy_mode(monkeypatch):
    import sys
    from agent_orch import cli
    monkeypatch.setattr(sys, "argv", ["agent-orch-sim", "train", "--scenario", "configs/toy.yaml"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2

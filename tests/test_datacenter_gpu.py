from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_orch.capacity import CapacityPlanner
from agent_orch.cost import switching_charge, operating_cost_rate
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.hardware import vllm_online_batch_defaults
from agent_orch.parameters import load_settings, audit_scenario, require_consistent
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory
from agent_orch.workload.agent_pool import GeneratorConfig, generate_pool, scenario_mapping
from agent_orch.workload.infrastructure import build_infrastructure, remap_pool_payload


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def infrastructure(settings):
    return build_infrastructure(settings)


@pytest.mark.parametrize("gpu,memory,tokens,sequences", [
    ("A10", 24, 2048, 256), ("L20", 48, 2048, 256),
    ("H20", 96, 8192, 1024), ("A100", 80, 2048, 256),
    ("B200", 192, 16384, 1024),
])
def test_pinned_online_defaults_follow_per_device_memory(settings, gpu, memory, tokens, sequences):
    assert vllm_online_batch_defaults(memory, gpu, settings["engine"]) == {
        "max_num_batched_tokens": tokens, "max_num_seqs": sequences}


def test_tensor_parallel_does_not_multiply_single_gpu_memory_for_defaults(infrastructure):
    assert any(c.gpu_type == "L20" and c.gpu_count == 2 for c in infrastructure.llm_configs.values())
    for c in infrastructure.llm_configs.values():
        if c.gpu_type == "L20":
            assert c.max_num_batched_tokens == 2048 and c.max_num_seqs == 256
        if c.gpu_type == "H20":
            assert c.max_num_batched_tokens == 8192 and c.max_num_seqs == 1024


def _two_replicas(scene):
    groups = {}
    for c in scene.candidates.values():
        groups.setdefault((c.config, c.server), []).append(c)
    return next(sorted(g, key=lambda c: c.replica_index) for g in groups.values()
                if len(g) == 2 and scene.llm_configs[g[0].config].gpu_count == 1)


def test_same_configuration_has_multiple_independent_binary_candidates(infrastructure, settings):
    a, b = _two_replicas(infrastructure)
    assert a.model == b.model and a.config == b.config and a.server == b.server
    assert a.id != b.id and (a.replica_index, b.replica_index) == (0, 1)
    planner = CapacityPlanner(infrastructure)
    deployment = planner.empty_deployment()
    assert planner.feasible_activation(deployment, a.id)
    deployment.llm_active[a.id] = 1
    assert planner.feasible_activation(deployment, b.id)
    deployment.llm_active[b.id] = 1
    assert planner.deployment_feasible(deployment)
    third = next(c for c in infrastructure.candidates.values()
                 if c.server == a.server and c.id not in (a.id, b.id))
    assert not planner.feasible_activation(deployment, third.id)
    assert operating_cost_rate(infrastructure, deployment) == pytest.approx(
        2 * infrastructure.llm_configs[a.config].running_cost_per_slot)
    loaded = ScenarioLoader.from_mapping(scenario_mapping(infrastructure))
    assert loaded.candidates == infrastructure.candidates
    require_consistent(audit_scenario(loaded, settings, infrastructure_only=True))


def test_missing_or_duplicate_replica_is_audit_failure(infrastructure, settings):
    a, b = _two_replicas(infrastructure)
    changed = dict(infrastructure.candidates)
    changed.pop(b.id)
    report = audit_scenario(replace(infrastructure, candidates=changed), settings, infrastructure_only=True)
    assert not report["consistent"]
    changed[b.id] = replace(b, replica_index=a.replica_index)
    assert not audit_scenario(replace(infrastructure, candidates=changed), settings,
                              infrastructure_only=True)["consistent"]


def test_relabeling_identical_replica_is_not_model_loading(infrastructure):
    a, b = _two_replicas(infrastructure)
    planner = CapacityPlanner(infrastructure)
    before, after = planner.empty_deployment(), planner.empty_deployment()
    before.llm_active[a.id] = 1
    after.llm_active[b.id] = 1
    assert switching_charge(infrastructure, after, before) == 0
    after.llm_active[a.id] = 1
    assert switching_charge(infrastructure, after, before) == pytest.approx(
        infrastructure.llm_configs[a.config].load_cost)


def test_replica_resource_mask_and_capacity_router_use_both_instances(infrastructure):
    pool = generate_pool(infrastructure, GeneratorConfig(applications=2, tool_types=2,
                                                         calibration_passes=0))
    scene = pool.scenario(infrastructure)
    a, b = _two_replicas(scene)
    planner = CapacityPlanner(scene)
    dep = planner.empty_deployment()
    dep.llm_active[a.id] = dep.llm_active[b.id] = 1
    shares = {(app.id, g, m): float(m == a.model) for app in scene.applications.values()
              for g in app.ingress_rates for m in scene.models}
    routed = CapacityProportionalRouter(scene).route(dep, shares)
    app = next(iter(scene.applications.values()))
    ingress = next(iter(app.ingress_rates))
    node = next(n.id for n in app.nodes.values() if n.type.value == "llm")
    assert routed.llm_share[(app.id, ingress, node, a.id)] == pytest.approx(.5)
    assert routed.llm_share[(app.id, ingress, node, b.id)] == pytest.approx(.5)
    trajectory = SlotTrajectory.sample(scene, 1, 101)
    env = SlotSequentialJointEnv(scene, trajectory, mapping_samples=2)
    env.current_deployment = dep
    third = next(c for c in scene.candidates.values()
                 if c.server == a.server and c.id not in (a.id, b.id))
    env._current_demand = ("llm", third.id, "")
    assert env._deploy_mask().tolist()[:2] == [1, 0]


def test_changes_do_not_edit_load_or_slo_generation(settings):
    import yaml
    cfg = yaml.safe_load(Path(settings["generator"]).read_text(encoding="utf-8"))
    assert cfg["application_rate_rps"] == [.005, .02]
    assert cfg["ttft_s_range"] == [2., 20.]
    assert cfg["tbt_s_range"] == [.02, .2]
    assert cfg["deadline_s_range"] == [20., 180.]
    assert settings["variation"]["arrival_range"] == [.6, 1.4]


def test_remapping_network_preserves_original_application_rates_and_contracts(infrastructure, settings):
    pool = generate_pool(infrastructure, GeneratorConfig(applications=3, tool_types=3,
                                                         calibration_passes=0))
    payload = pool.to_dict()
    network = build_infrastructure(settings, 50)
    mapped = remap_pool_payload(payload, network)
    for a, b in zip(payload["applications"], mapped["applications"]):
        assert sum(a["spec"]["ingress_rates"].values()) == pytest.approx(
            sum(b["spec"]["ingress_rates"].values()))
        assert a["spec"]["slo"] == b["spec"]["slo"]
        assert a["spec"]["nodes"] == b["spec"]["nodes"]
        assert a["lengths"] == b["lengths"]

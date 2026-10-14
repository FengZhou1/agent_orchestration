from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_orch.parameters import load_settings
from agent_orch.workload.agent_pool import GeneratorConfig, generate_pool, scenario_mapping, rebind_pool
from agent_orch.workload.infrastructure import build_infrastructure
from agent_orch.workload.slo_calibration import (
    VERSION, CalibrationSpec, apply_contracts, calibrate_application, matches_contracts,
    reference_environment, sign_contracts, validate_contracts, weighted_quantile, workload_signature,
)
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import SLOSpec, SLOType


@pytest.fixture(scope="module")
def source():
    infra = build_infrastructure(load_settings())
    pool = generate_pool(infra, GeneratorConfig(applications=3, tool_types=3, calibration_passes=0))
    return pool.scenario(infra)


def test_quantile_uses_probability_not_number_of_catalogued_patterns():
    assert weighted_quantile([1., 100.], [.99, .01], .95) == 1.
    assert weighted_quantile([1., 100.], [.9, .1], .95) == 100.
    assert weighted_quantile([999., 1.], [0., 1.], .95) == 1.
    for values, weights in (([float("nan")], [1.]), ([1.], [0.]), ([1.], [-1.]), ([], [])):
        with pytest.raises(ValueError):
            weighted_quantile(values, weights, .95)


def test_reference_is_fixed_and_does_not_modify_source_rates(source):
    original = scenario_mapping(source)
    app = next(iter(source.applications.values()))
    spec = CalibrationSpec()
    ref, deployment, _ = reference_environment(source, app, spec)
    assert len(ref.candidates) == 1
    candidate = next(iter(ref.candidates.values()))
    assert candidate.model == "qwen3-8b"
    assert ref.llm_configs[candidate.config].gpu_type == "L20"
    assert ref.llm_configs[candidate.config].gpu_count == 1
    assert sum(ref.applications[app.id].ingress_rates.values()) == 1e-6
    assert sum(deployment.tool_replicas.values()) == len(ref.tools)
    assert scenario_mapping(source) == original


@pytest.mark.parametrize("kind", [SLOType.LATENCY, SLOType.DEADLINE, SLOType.COMPOUND])
def test_calibration_and_application_only_change_contract_fields(source, kind):
    app = next(iter(source.applications.values()))
    app = replace(app, slo=SLOSpec(kind, ttft_s=2., tbt_s=.1, deadline_s=30.))
    scene = replace(source, applications={app.id: app})
    before = scenario_mapping(scene)
    record, samples = calibrate_application(scene, app, CalibrationSpec())
    assert len(samples) == len(app.pattern_flows)
    assert record["maximum_reference_utilization"] < .01
    assert record["variants"]["main"]["slo"]["type"] == kind.value
    payload = sign_contracts({"version": VERSION, "applications": {app.id: record}})
    main = apply_contracts(scene, payload, "main")
    tight = apply_contracts(scene, payload, "tight")
    assert matches_contracts(main, payload) and matches_contracts(tight, payload)
    assert main.id != tight.id
    updated = main.applications[app.id]
    assert updated.ingress_rates == app.ingress_rates
    assert updated.quality == app.quality and updated.pattern_flows == app.pattern_flows
    assert updated.edge_data_mb == app.edge_data_mb
    assert workload_signature(updated, main.tools) == workload_signature(app, scene.tools)
    for key, node in updated.nodes.items():
        assert replace(node, stage_deadline_s=app.nodes[key].stage_deadline_s) == app.nodes[key]
    if kind is SLOType.COMPOUND:
        assert all(v <= updated.slo.deadline_s for v in record["variants"]["main"]["stage_deadlines_s"].values())
    assert scenario_mapping(scene) == before
    ScenarioLoader.from_mapping(scenario_mapping(main))


def test_frozen_contracts_ignore_arrival_rate_and_network_scale_but_reject_token_changes(source):
    app = next(iter(source.applications.values()))
    record, _ = calibrate_application(source, app, CalibrationSpec())
    payload = sign_contracts({"version": VERSION, "applications": {app.id: record}})
    scene = replace(source, applications={app.id: replace(app, ingress_rates={
        k: v * 9 for k, v in app.ingress_rates.items()})})
    changed = apply_contracts(scene, payload)
    assert changed.applications[app.id].slo == apply_contracts(
        replace(source, applications={app.id: app}), payload).applications[app.id].slo
    nodes = dict(app.nodes)
    key = next(k for k, n in nodes.items() if n.type.value == "llm")
    prompts = dict(nodes[key].prompt_tokens)
    prompts["qwen3-8b"] += 1
    nodes[key] = replace(nodes[key], prompt_tokens=prompts)
    with pytest.raises(ValueError, match="do not match"):
        apply_contracts(replace(source, applications={app.id: replace(app, nodes=nodes)}), payload)
    corrupted = copy.deepcopy(payload)
    corrupted["applications"][app.id]["reference_quantile_s"]["ttft_s"] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_contracts(corrupted)


def test_declared_main_and_tight_multipliers(source):
    app = next(iter(source.applications.values()))
    app = replace(app, slo=SLOSpec(SLOType.COMPOUND, 2., .1, 30.))
    record, _ = calibrate_application(source, app, CalibrationSpec())
    q = record["reference_quantile_s"]
    main, tight = record["variants"]["main"]["slo"], record["variants"]["tight"]["slo"]
    assert main["ttft_s"] == pytest.approx(q["ttft_s"] * 1.5)
    assert main["tbt_s"] == pytest.approx(q["tbt_s"] * 1.25)
    assert main["deadline_s"] == pytest.approx(q["e2e_s"] * 1.5)
    assert tight["tbt_s"] == q["tbt_s"] # avoids spurious violation at 1.25*0.8
    assert tight["ttft_s"] == pytest.approx(q["ttft_s"] * 1.2)


def test_binding_and_service_capacity_changes_reuse_task_contract(source):
    spec = CalibrationSpec()
    records = {key: calibrate_application(source, app, spec)[0] for key, app in source.applications.items()}
    payload = sign_contracts({"version": VERSION, "applications": records})
    changed_tools = {key: replace(tool, service_rate={n: v * 2 for n, v in tool.service_rate.items()})
                     for key, tool in source.tools.items()}
    altered = replace(source, tools=changed_tools)
    assert {k: a.slo for k, a in apply_contracts(source, payload).applications.items()} == {
        k: a.slo for k, a in apply_contracts(altered, payload).applications.items()}
    main = apply_contracts(source, payload)
    assert apply_contracts(main, payload).id == main.id


def test_reference_timing_does_not_depend_on_old_placeholder_thresholds(source):
    app = next(iter(source.applications.values()))
    loose = replace(app, slo=SLOSpec(app.slo.type, 1e9, 1e9, 1e9))
    unset = replace(app, slo=SLOSpec(app.slo.type))
    a = calibrate_application(source, loose, CalibrationSpec())[0]
    b = calibrate_application(source, unset, CalibrationSpec())[0]
    assert a == b


def test_single_output_token_keeps_zero_tbt(source):
    app = next(iter(source.applications.values()))
    app = replace(app, slo=SLOSpec(SLOType.COMPOUND, 2., .1, 30.), nodes={key:
        replace(node, output_tokens={m: 1. for m in node.output_tokens}) if node.type.value == "llm" else node
        for key, node in app.nodes.items()})
    record, _ = calibrate_application(source, app, CalibrationSpec())
    assert record["reference_quantile_s"]["tbt_s"] == 0.
    assert record["variants"]["main"]["slo"]["tbt_s"] == 0.


def test_configured_contracts_are_complete_and_declared_parameters_match():
    from agent_orch.parameters import configured_contracts
    payload = configured_contracts(load_settings())
    assert len(payload["applications"]) == 200
    assert payload["spec"] == yaml.safe_load(Path("configs/slo_calibration.yaml").read_text(encoding="utf-8"))
    assert min(r["reference_joint_attainment"]["main"] for r in payload["applications"].values()) > .95


def test_tool_rebinding_reuses_the_same_logical_task_slos():
    infra = build_infrastructure(load_settings())
    pool = generate_pool(infra, GeneratorConfig(applications=3, tool_types=3, calibration_passes=0))
    source = pool.scenario(infra)
    records = {key: calibrate_application(source, app, CalibrationSpec())[0] for key, app in source.applications.items()}
    payload = sign_contracts({"version": VERSION, "applications": records})
    changed = rebind_pool(pool, infra, 2).scenario(infra)
    old_slos = {key: app.slo for key, app in apply_contracts(source, payload).applications.items()}
    new_slos = {key: app.slo for key, app in apply_contracts(changed, payload).applications.items()}
    assert old_slos == new_slos

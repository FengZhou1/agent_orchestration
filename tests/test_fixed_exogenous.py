"""The deployment-only control must freeze inputs, not homogenize applications."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from agent_orch.capacity import CapacityPlanner
from agent_orch.deployment.warmup import (
    FixedWarmupDeployments, _capacities, capacity_probes, generate_warmup_deployments,
    reuse_layouts_fixed_exogenous,
)
from agent_orch.workload import SlotTrajectory, SlotVariationSpec


@pytest.fixture(scope="module")
def source(scenario):
    trace = SlotTrajectory.sample(scenario, 4, 101)
    return trace, generate_warmup_deployments(scenario, trace, 20261014)


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def test_all_conditions_equal_base_and_application_heterogeneity_preserved(scenario):
    trace = SlotTrajectory.fixed(scenario, 32, 101)
    assert trace.variation.is_fixed_base
    for t in range(32):
        assert trace.scenario_at(t, scenario) == scenario
        assert trace.slots[t].arrivals == {a.id: a.ingress_rates for a in scenario.applications.values()}
    loaded = SlotTrajectory.from_dict(json.loads(json.dumps(trace.to_dict())))
    loaded.validate_against(scenario)
    assert loaded.digest() == trace.digest()
    assert SlotTrajectory.sample(scenario, 32, 101, SlotVariationSpec.fixed_base()).digest() == trace.digest()
    with pytest.raises(ValueError, match="positive"):
        SlotTrajectory.fixed(scenario, 0, 101)


def test_fixed_snapshots_do_not_alias_base_or_other_slots(scenario):
    trace = SlotTrajectory.fixed(scenario, 2, 101)
    a = next(iter(scenario.applications.values()))
    g = next(k for k, v in a.ingress_rates.items() if v > 0)
    before = a.ingress_rates[g]
    trace.slots[0].arrivals[a.id][g] *= 2
    assert trace.slots[1].arrivals[a.id][g] == before
    assert a.ingress_rates[g] == before
    with pytest.raises(ValueError, match="Fixed-exogenous"):
        trace.validate_against(scenario)


@pytest.mark.parametrize("component", ["resources", "bandwidth"])
def test_fixed_metadata_cannot_disguise_variation(scenario, component):
    trace = SlotTrajectory.fixed(scenario, 2, 101)
    snapshot = copy.deepcopy(trace.slots[1])
    if component == "resources":
        n = next(iter(snapshot.server_resources))
        snapshot.server_resources[n]["memory_gb"] *= .9
    else:
        snapshot = replace(snapshot, link_capacity_mbps=tuple(v*.9 for v in snapshot.link_capacity_mbps))
    trace = replace(trace, slots=(trace.slots[0], snapshot))
    with pytest.raises(ValueError, match="Fixed-exogenous"):
        trace.validate_against(scenario)


def test_layout_rebinding_preserves_every_action_and_refreshes_diagnostics(scenario, source):
    old_trace, old_bank = source
    original = copy.deepcopy(old_bank.to_dict())
    trace = SlotTrajectory.fixed(scenario, len(old_trace), 101)
    bank = reuse_layouts_fixed_exogenous(scenario, trace, old_bank, old_trace)
    assert old_bank.to_dict() == original
    assert bank.trajectory_digest != old_bank.trajectory_digest
    assert bank.digest() != old_bank.digest()
    for t, entry in enumerate(bank.entries):
        assert entry["deployment"] == old_bank.entries[t]["deployment"]
        assert entry["signature"] == old_bank.entries[t]["signature"]
        assert entry["source_context"]["bank_digest"] == old_bank.digest()
        assert entry["attempts"] == 0 and entry["repairs_added"] == []
        assert CapacityPlanner(scenario).deployment_feasible(bank.deployment_at(t))
        capacities, calls = _capacities(scenario)
        assert entry["probes"] == capacity_probes(scenario, bank.deployment_at(t), capacities, calls)
    again = reuse_layouts_fixed_exogenous(scenario, trace, old_bank, old_trace)
    assert bank.digest() == again.digest()
    with pytest.raises(ValueError, match="fixed-base"):
        reuse_layouts_fixed_exogenous(scenario, old_trace, old_bank, old_trace)
    with pytest.raises(ValueError, match="slot count"):
        reuse_layouts_fixed_exogenous(scenario, SlotTrajectory.fixed(scenario, 1, 101), old_bank, old_trace)


def test_build_verify_train_and_resume_fixed_profile(tmp_path, monkeypatch, scenario, source):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_warmup_deployments as build
    import verify_warmup_deployments as verify
    import run_slot_joint as train

    old_trace, old_bank = source
    source_path = tmp_path / "original"
    source_path.mkdir()
    (source_path / "train_trajectory.json").write_text(json.dumps(old_trace.to_dict()), encoding="utf-8")
    old_bank.save(source_path / "warmup_deployments.json")
    original_hashes = hashes(source_path)
    bundle = tmp_path / "fixed"
    monkeypatch.setattr(sys, "argv", ["build", "--scenario", "configs/toy.yaml", "--output", str(bundle),
                                      "--slots", "4", "--fixed-exogenous", "--reuse-deployments", str(source_path), "--smoke"])
    build.main()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["exogenous_mode"] == "fixed_base"
    assert manifest["source_layouts"]["all_layouts_preserved"]
    assert hashes(source_path) == original_hashes

    def forbidden(*args, **kwargs):
        raise AssertionError("Must replay rebinding, not generate new layouts")
    monkeypatch.setattr(verify, "generate_warmup_deployments", forbidden)
    result = bundle / "verification.json"
    monkeypatch.setattr(sys, "argv", ["verify", "--scenario", "configs/toy.yaml", "--bundle", str(bundle),
                                      "--output", str(result), "--regenerate"])
    verify.main()
    assert json.loads(result.read_text(encoding="utf-8"))["deterministic_regeneration"] is True

    run = tmp_path / "run"
    arguments = ["train", "--scenario", "configs/toy.yaml", "--output", str(run), "--warmup-bundle", str(bundle),
                 "--slots", "4", "--updates", "3", "--route-updates", "3", "--deploy-updates", "0",
                 "--fixed-exogenous", "--mapping-samples", "7", "--hidden-size", "16", "--hidden-layers", "1",
                 "--update-epochs", "1", "--minibatch-size", "8", "--smoke"]
    real_train = train.train_ppo
    def interrupt(env, **kwargs):
        save = kwargs["on_checkpoint"]
        def stop(state):
            save(state)
            raise KeyboardInterrupt("test stop")
        return real_train(env, **{**kwargs, "on_checkpoint": stop})
    monkeypatch.setattr(train, "train_ppo", interrupt)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(KeyboardInterrupt): train.main()
    before = hashes(run)
    monkeypatch.setattr(sys, "argv", [a for a in arguments if a != "--fixed-exogenous"] + ["--resume"])
    with pytest.raises(ValueError, match="Variation"):
        train.main()
    assert hashes(run) == before
    monkeypatch.setattr(train, "train_ppo", real_train)
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    train.main()
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert m["training_settings"]["exogenous_mode"] == "fixed_base"
    assert len(json.loads((run / "history.json").read_text(encoding="utf-8"))) == 3
    assert hashes(source_path) == original_hashes
    assert (run / "train_trajectory.json").read_bytes() == (bundle / "train_trajectory.json").read_bytes()


def test_fixed_flag_cannot_be_combined_with_arbitrary_override_or_formal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_slot_joint
    for extra in (["--variation", "other.json"], ["--formal"]):
        monkeypatch.setattr(sys, "argv", ["train", "--scenario", "configs/toy.yaml", "--output", str(tmp_path),
                                          "--updates", "1", "--fixed-exogenous", *extra])
        with pytest.raises(ValueError, match="incompatible"):
            run_slot_joint.main()

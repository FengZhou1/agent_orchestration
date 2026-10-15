"""Budget continuation must retain the trajectory and the complete PPO state."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from agent_orch.agents import train_ppo
from agent_orch.agents.budget import canonical_schedule, validate_resume_schedule
from agent_orch.agents.config import PPOConfig
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.workload import SlotTrajectory


@pytest.mark.parametrize("phase", ["composition", "deployment", "joint"])
def test_explicit_extension_only_changes_final_length(phase):
    old = (("composition", 2), ("deployment", 3), (phase, 4))
    new = (*old[:-1], (phase, 8))
    assert validate_resume_schedule(old, new, allow_extension=True)
    assert canonical_schedule([list(item) for item in old]) == old
    assert not validate_resume_schedule(old, old)
    with pytest.raises(ValueError, match="extend-budget"):
        validate_resume_schedule(old, new)


@pytest.mark.parametrize("old,new", [
    ((("composition", 100),), (("composition", 99),)),
    ((("composition", 100),), (("composition", 100), ("joint", 900))),
    ((("composition", 100),), (("joint", 1000),)),
    ((("composition", 2), ("joint", 2)), (("composition", 3), ("joint", 8))),
    ((("composition", 2), ("deployment", 2)), (("deployment", 2), ("composition", 8))),
    (None, (("composition", 1000),)),
    ((("composition", 100),), None),
])
def test_extension_rejects_stage_changes_and_shortening(old, new):
    with pytest.raises(ValueError, match="final existing phase"):
        validate_resume_schedule(old, new, allow_extension=True)


def _assert_same_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(expected, actual):
            _assert_same_state(a, b)
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


def test_two_plus_two_matches_uninterrupted_training(scenario):
    trajectory = SlotTrajectory.sample(scenario, slots=2, seed=71)
    config = PPOConfig(hidden_size=16, hidden_layers=1, update_epochs=1,
                       minibatch_size=8, exploration_mode="none")

    def run(total, *, resume=None, extend=False):
        states = []
        env = SlotSequentialJointEnv(scenario, trajectory, mapping_samples=7)
        train_ppo(env, updates=total, rollout_steps=1, rollout_periods=2,
                  seed=93, config=config, phase_schedule=(("composition", total),),
                  resume_state=resume, allow_budget_extension=extend,
                  on_checkpoint=lambda state: states.append(copy.deepcopy(state)))
        return states[-1]

    whole = run(4)
    first = run(2)
    with pytest.raises(ValueError, match="extend-budget"):
        run(4, resume=first)
    resumed = run(4, resume=first, extend=True)
    assert first["next_update"] == 2
    assert first["phase_schedule"] == (("composition", 2),)
    assert resumed["next_update"] == 4
    assert resumed["phase_schedule"] == (("composition", 4),)
    for key in whole.keys() - {"history"}:
        _assert_same_state(whole[key], resumed[key])
    assert len(resumed["history"]) == 4
    for a, b in zip(whole["history"], resumed["history"]):
        for key in a:
            if key.startswith("timing_") or key in ("collection_time_s", "optimization_time_s"):
                continue
            _assert_same_state(a[key], b[key])


def _hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_cli_budget_preflight_backup_interruption_and_resume(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_slot_joint

    real_train = run_slot_joint.train_ppo
    arguments = ["run_slot_joint.py", "--scenario", "configs/toy.yaml", "--output", str(tmp_path),
                 "--deploy-updates", "0", "--slots", "2", "--mapping-samples", "7",
                 "--hidden-size", "16", "--hidden-layers", "1", "--update-epochs", "1",
                 "--minibatch-size", "8", "--workflow-engine", "array", "--smoke"]

    def invoke(total, *extra):
        monkeypatch.setattr(sys, "argv", [*arguments, "--updates", str(total),
                                         "--route-updates", str(total), *extra])
        run_slot_joint.main()

    invoke(2)
    original_hashes = _hashes(tmp_path)
    original_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    invoke(4, "--resume", "--extend-budget", "--check-resume")
    assert _hashes(tmp_path) == original_hashes
    with pytest.raises(ValueError, match="extend-budget"):
        invoke(4, "--resume")
    with pytest.raises(ValueError, match="Training settings"):
        invoke(4, "--resume", "--extend-budget", "--update-epochs", "2")
    assert _hashes(tmp_path) == original_hashes

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("intentional interruption before first extended checkpoint")

    monkeypatch.setattr(run_slot_joint, "train_ppo", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(4, "--resume", "--extend-budget")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    event, = manifest["training_budget_extensions"]
    assert event["from"] == [["composition", 2]]
    assert event["to"] == [["composition", 4]]
    assert event["checkpoint_next_update"] == 2
    backup = tmp_path / event["backup"]
    for name in ("checkpoint.pt", "manifest.json", "history.json"):
        assert hashlib.sha256((backup / name).read_bytes()).hexdigest() == original_hashes[name]
    assert manifest["training_settings"]["phase_schedule"] == [["composition", 4]]

    # Repeating the explicit command after a pre-checkpoint interruption is safe.
    monkeypatch.setattr(run_slot_joint, "train_ppo", real_train)
    invoke(4, "--resume", "--extend-budget")
    final_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["training_budget_extensions"] == [event]
    for key in original_manifest.keys() - {"training_settings"}:
        assert final_manifest[key] == original_manifest[key]
    settings = dict(final_manifest["training_settings"])
    settings["phase_schedule"] = [["composition", 2]]
    assert settings == original_manifest["training_settings"]
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["next_update"] == 4
    assert checkpoint["training_budget_extensions"] == [event]
    assert len(checkpoint["history"]) == 4
    assert all(row["training_phase"] == "composition" for row in checkpoint["history"])
    for name in ("train_trajectory.json", "warmup_deployments.json"):
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == original_hashes[name]
    invoke(4, "--resume", "--check-resume")  # No flag needed after a new checkpoint is saved.


@pytest.mark.parametrize("flag", ["--extend-budget", "--check-resume"])
def test_cli_budget_flags_require_resume(tmp_path, monkeypatch, flag):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_slot_joint
    monkeypatch.setattr(sys, "argv", ["run_slot_joint.py", "--scenario", "configs/toy.yaml",
                                      "--output", str(tmp_path), "--updates", "4", flag])
    with pytest.raises(ValueError, match="require --resume"):
        run_slot_joint.main()
    assert not list(tmp_path.iterdir())

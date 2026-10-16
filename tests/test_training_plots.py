"""Do not invent missing performance curves or interpret placeholder KL as data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("plot_slot_joint_run", Path(__file__).parents[1] / "scripts/plot_slot_joint_run.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def test_moving_mean_preserves_missing_observations():
    values = np.array([1., np.nan, 3., 5.])
    np.testing.assert_equal(plots.moving_mean(values, 2), [1., np.nan, 3., 4.])
    np.testing.assert_equal(values, [1., np.nan, 3., 5.])
    with pytest.raises(ValueError):
        plots.moving_mean(values, 0)


def test_inventory_distinguishes_constant_missing_and_placeholder():
    history = [{"update": 0, "mean_constraint_llm": 0., "last_approx_kl": 0., "training_phase": "composition"},
               {"update": 1, "mean_constraint_llm": 0., "last_approx_kl": 0., "timing_new_s": 3.}]
    inventory = plots.metric_inventory(history, {"ppo_config": {"target_kl": None}})
    assert inventory["last_approx_kl"]["status"] == "placeholder_not_measured"
    assert inventory["mean_constraint_llm"]["status"] == "recorded"
    assert inventory["mean_constraint_llm"]["max"] == 0
    assert inventory["timing_new_s"]["missing_count"] == 1
    assert "quality" not in inventory
    np.testing.assert_equal(plots.series(history, "quality"), [np.nan, np.nan])
    assert plots.metric_inventory(history, {"ppo_config": {"target_kl": .1}})["last_approx_kl"]["status"] == "recorded"


@pytest.fixture
def saved_run(tmp_path):
    manifest = {"protocol": "slot_joint_v7", "trajectory_digest": "trace", "warmup_deployment_digest": "bank",
                "evaluation_protocol": "training_trajectory_replay", "slots": 2,
                "training_settings": {"phase_schedule": [["composition", 4]]}}
    history = [{"update": i, "episode_length": 2, "mean_utility": -.1} for i in range(4)]
    evaluation = {**{k: manifest[k] for k in ("protocol", "trajectory_digest", "warmup_deployment_digest", "evaluation_protocol")},
                  "slots": [{"slot": 0}, {"slot": 1}]}
    for name, value in [("manifest.json", manifest), ("history.json", history), ("evaluation.json", evaluation)]:
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    return tmp_path


def test_loading_does_not_change_input_files(saved_run):
    before = {p.name: p.read_bytes() for p in saved_run.iterdir()}
    manifest, history, evaluation = plots.load_run(saved_run)
    assert len(history) == 4 and len(evaluation["slots"]) == manifest["slots"]
    assert {p.name: p.read_bytes() for p in saved_run.iterdir()} == before


@pytest.mark.parametrize("change", ["digest", "duplicate_update", "incomplete", "slot_order", "partial_episode"])
def test_reject_incomparable_or_incomplete_data(saved_run, change):
    name = "evaluation.json" if change in ("digest", "slot_order") else "history.json"
    data = json.loads((saved_run / name).read_text(encoding="utf-8"))
    if change == "digest": data["trajectory_digest"] = "other"
    elif change == "slot_order": data["slots"].reverse()
    elif change == "duplicate_update": data[1]["update"] = 0
    elif change == "incomplete": data.pop()
    else: data[1]["episode_length"] = 1
    (saved_run / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        plots.load_run(saved_run)


def test_existing_figure_directory_is_not_overwritten(saved_run, tmp_path):
    with pytest.raises(FileExistsError):
        plots.plot_run(saved_run, tmp_path)

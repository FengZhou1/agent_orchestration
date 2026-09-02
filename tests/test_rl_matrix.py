from runpy import run_path

import pytest


def test_rl_matrix_auto_has_only_meaningful_combinations():
    namespace = run_path("scripts/run_rl_matrix.py", run_name="test_module")
    combinations = namespace["_combinations"](
        ["joint", "deploy", "route"], ["auto"]
    )
    assert combinations == [
        ("joint", "rnd"),
        ("joint", "no-rnd"),
        ("joint", "unconstrained-rnd"),
        ("deploy", "no-rnd"),
        ("route", "no-rnd"),
    ]


def test_rl_matrix_defaults_to_one_seed_and_one_algorithm():
    namespace = run_path("scripts/run_rl_matrix.py", run_name="test_module")
    assert namespace["DEFAULT_SEEDS"] == "0"
    assert namespace["DEFAULT_MODES"] == "joint"
    assert namespace["DEFAULT_VARIANTS"] == "rnd"


def test_rl_matrix_rejects_exploration_variants_for_single_stage_ablations():
    namespace = run_path("scripts/run_rl_matrix.py", run_name="test_module")
    with pytest.raises(ValueError):
        namespace["_combinations"](["deploy"], ["rnd"])

from runpy import run_path


def test_rl_matrix_auto_has_only_meaningful_combinations():
    namespace = run_path("scripts/run_rl_matrix.py", run_name="test_module")
    combinations = namespace["_combinations"](
        ["joint", "deploy", "route"], ["auto"]
    )
    assert combinations == [
        ("joint", "vanilla"),
        ("joint", "potential"),
        ("joint", "icm"),
        ("deploy", "vanilla"),
        ("route", "vanilla"),
    ]

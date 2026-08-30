from runpy import run_path


NAMESPACE = run_path("scripts/generate_composition_sweep.py", run_name="test_module")


def test_composition_sweep_preserves_combined_rate():
    scenario = {
        "id": "unit",
        "applications": [
            {"id": "short", "ingress_rates": {"n0": 2.0, "n1": 1.0}},
            {"id": "long", "ingress_rates": {"n2": 1.0}},
        ],
    }
    result = NAMESPACE["set_composition"](
        scenario, "short", "long", long_fraction=0.75, total_rate_rps=8.0
    )
    applications = {app["id"]: app for app in result["applications"]}
    short_rate = sum(applications["short"]["ingress_rates"].values())
    long_rate = sum(applications["long"]["ingress_rates"].values())
    assert short_rate == 2.0
    assert long_rate == 6.0
    assert short_rate + long_rate == 8.0

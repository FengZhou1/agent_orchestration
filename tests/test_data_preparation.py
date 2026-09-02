from runpy import run_path

import pandas as pd
import pytest

SERVICES = run_path("scripts/prepare_service_profiles.py", run_name="test_module")
LLM_PROFILES = run_path("scripts/prepare_llm_profiles.py", run_name="test_module")


def test_service_profile_uses_low_load_samples_and_stability_rule():
    frame = pd.DataFrame(
        {
            "service": ["search"] * 8,
            "server": ["n0"] * 8,
            "vcpu": [2] * 8,
            "arrival_rate_rps": [1, 1, 1, 1, 4, 4, 8, 8],
            "latency_ms": [10, 11, 9, 10, 15, 16, 40, 45],
            "error": [0, 0, 0, 0, 0, 0, 1, 1],
            "request_bytes": [1000] * 8,
            "response_bytes": [2000] * 8,
        }
    )
    summary = SERVICES["summarize_measurements"](frame)
    row = summary.iloc[0]
    assert row["stable_rate_rps"] == 4.0
    assert row["mean_service_s"] == pytest.approx(0.01)
    assert row["request_mb"] == pytest.approx(0.001)


def test_llm_profile_validation_and_holdout_report():
    frame = pd.DataFrame(
        {
            "model": ["m"] * 6,
            "config": ["c"] * 6,
            "prompt_tokens": [128] * 6,
            "output_tokens": [32] * 6,
            "arrival_rate_rps": [1, 2, 3, 4, 5, 6],
            "long_request_fraction": [0.0] * 6,
            "ttft_s": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
            "tbt_s": [0.01] * 6,
            "response_s": [0.4, 0.42, 0.44, 0.46, 0.48, 0.50],
            "stable_capacity_rps": [8.0] * 6,
            "kv_tokens": [160.0] * 6,
        }
    )
    validated = LLM_PROFILES["validate_profile"](frame)
    report = LLM_PROFILES["interpolation_holdout"](validated, 0.2, 2026)
    assert set(report) == {
        "ttft_s", "tbt_s", "response_s", "stable_capacity_rps", "kv_tokens"
    }

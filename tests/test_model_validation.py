import pandas as pd
from runpy import run_path


NAMESPACE = run_path("scripts/validate_llm_model.py", run_name="test_module")


def test_model_validation_produces_finite_error_table(scenario):
    frame = pd.DataFrame(
        [
            {
                "model": "small",
                "config": "small-edge",
                "prompt_tokens": 256,
                "output_tokens": 32,
                "arrival_rate_rps": 1.0,
                "ttft_s": 0.2,
                "tbt_s": 0.03,
                "response_s": 1.0,
                "stable_capacity_rps": 10.0,
            }
        ]
    )
    predictions = NAMESPACE["predict_profile_rows"](frame, scenario)
    summary = NAMESPACE["error_summary"](predictions)
    assert len(summary) == 4
    assert summary[["rmse", "median_ape_pct", "p95_ape_pct"]].notna().all().all()

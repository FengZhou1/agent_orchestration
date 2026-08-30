import numpy as np
import pandas as pd
from runpy import run_path


NAMESPACE = run_path("scripts/summarize_results.py", run_name="test_module")


def test_bootstrap_and_paired_statistics_are_reproducible():
    frame = pd.DataFrame(
        {
            "policy": ["base"] * 4 + ["new"] * 4,
            "seed": [0, 1, 2, 3] * 2,
            "score": [1.0, 1.1, 0.9, 1.0, 1.2, 1.3, 1.1, 1.2],
        }
    )
    summary = NAMESPACE["bootstrap_summary"](
        frame, ["policy"], ["score"], samples=200, seed=7
    )
    comparisons = NAMESPACE["paired_comparisons"](
        frame, ["policy"], ["score"], "seed", ("base",)
    )
    assert len(summary) == 2
    np.testing.assert_allclose(
        comparisons.iloc[0]["mean_paired_difference"], 0.2
    )

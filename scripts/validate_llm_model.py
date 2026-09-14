from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_orch.performance.llm import service_demand
from agent_orch.performance.queueing import llm_waiting_time
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import Scenario


OBSERVED_COLUMNS = ("ttft_s", "tbt_s", "response_s", "stable_capacity_rps")


def predict_profile_rows(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    required = {
        "model",
        "config",
        "prompt_tokens",
        "output_tokens",
        "arrival_rate_rps",
        *OBSERVED_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Profile table is missing columns: {sorted(missing)}")
    records = []
    for row in frame.itertuples(index=False):
        model = scenario.models[row.model]
        config = scenario.llm_configs[row.config]
        demand = service_demand(
            model,
            config,
            row.prompt_tokens,
            row.output_tokens,
            scenario.simulation.prefill_chunk_tokens,
        )
        wait, utilization, overloaded = llm_waiting_time(
            row.arrival_rate_rps,
            demand.service_s,
            demand.service_s**2,
            config.max_num_seqs,
            scenario.simulation.overload_delay_s,
        )
        output_tokens = max(1, round(row.output_tokens))
        predicted = {
            "ttft_s": wait + demand.prefill_s,
            "tbt_s": demand.decode_s / (output_tokens - 1) if output_tokens > 1 else 0.0,
            "response_s": wait + demand.service_s,
            "stable_capacity_rps": config.max_num_seqs / demand.service_s,
        }
        record = row._asdict()
        record["predicted_utilization"] = utilization
        record["predicted_overload"] = overloaded
        for metric, value in predicted.items():
            record[f"predicted_{metric}"] = value
            denominator = max(abs(float(record[metric])), 1e-9)
            record[f"ape_{metric}"] = abs(value - float(record[metric])) / denominator
        records.append(record)
    return pd.DataFrame(records)


def error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in OBSERVED_COLUMNS:
        error = predictions[f"predicted_{metric}"] - predictions[metric]
        absolute_percentage = predictions[f"ape_{metric}"]
        rows.append(
            {
                "metric": metric,
                "n": len(predictions),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_ape_pct": 100.0 * float(np.median(absolute_percentage)),
                "p95_ape_pct": 100.0 * float(np.quantile(absolute_percentage, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", default="results/llm_model_validation")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    frame = pd.read_csv(args.profile)
    predictions = predict_profile_rows(frame, scenario)
    summary = error_summary(predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "point_predictions.parquet", index=False)
    summary.to_csv(output / "error_summary.csv", index=False)
    validity = {
        row.metric: {
            "median_within_10pct": bool(row.median_ape_pct <= 10.0),
            "p95_within_20pct": bool(row.p95_ape_pct <= 20.0),
        }
        for row in summary.itertuples(index=False)
    }
    (output / "validity.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

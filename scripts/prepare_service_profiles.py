from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from agent_orch.data import DatasetManifest, file_sha256


REQUIRED = {
    "service",
    "server",
    "vcpu",
    "arrival_rate_rps",
    "latency_ms",
    "error",
}


def summarize_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Service measurements are missing columns: {sorted(missing)}")
    rows = []
    group_columns = ["service", "server", "vcpu"]
    for keys, group in frame.groupby(group_columns, sort=True):
        rates = sorted(float(value) for value in group["arrival_rate_rps"].unique())
        low_rate = rates[0]
        low = group[group["arrival_rate_rps"] == low_rate]
        low_latency_s = low["latency_ms"].astype(float).to_numpy() / 1000.0
        low_mean = float(np.mean(low_latency_s))
        low_variance = float(np.var(low_latency_s, ddof=1)) if len(low_latency_s) > 1 else 0.0
        low_scv = low_variance / max(low_mean**2, 1e-12)
        low_p95_ms = float(np.quantile(low["latency_ms"].astype(float), 0.95))
        stable_rates = []
        for rate in rates:
            point = group[group["arrival_rate_rps"] == rate]
            error_rate = float(point["error"].astype(float).mean())
            p95_ms = float(np.quantile(point["latency_ms"].astype(float), 0.95))
            if error_rate < 0.01 and p95_ms <= 2.0 * low_p95_ms:
                stable_rates.append(rate)
        request_bytes = (
            float(group["request_bytes"].astype(float).mean())
            if "request_bytes" in group
            else np.nan
        )
        response_bytes = (
            float(group["response_bytes"].astype(float).mean())
            if "response_bytes" in group
            else np.nan
        )
        rows.append(
            {
                "service": keys[0],
                "server": keys[1],
                "vcpu": int(keys[2]),
                "low_load_rate_rps": low_rate,
                "mean_service_s": low_mean,
                "service_scv": low_scv,
                "low_load_p95_s": low_p95_ms / 1000.0,
                "stable_rate_rps": max(stable_rates, default=0.0),
                "request_mb": request_bytes / 1e6,
                "response_mb": response_bytes / 1e6,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-name", default="DeathStarBench/vLLM-adjacent service replay")
    parser.add_argument("--source-url", default="https://github.com/delimitrou/DeathStarBench")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--license", default="See source and measurement harness")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/processed/stateless_service_profile.csv")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_measurements(pd.read_csv(source))
    summary.to_csv(output, index=False)
    manifest = DatasetManifest(
        artifact_type="stateless-service-profile",
        source_name=args.source_name,
        source_url=args.source_url,
        source_version=args.source_version,
        license=args.license,
        source_checksum_sha256=file_sha256(source),
        preprocessing_command=" ".join(sys.argv),
        random_seed=args.seed,
        parameters={
            "stability_rule": "error_rate < 0.01 and p95 <= 2 * low_load_p95",
            "processing_time_source": "low-load internal service measurements",
            "end_to_end_rt_used_as_service_time": False,
        },
    )
    manifest.write(output.with_suffix(".manifest.json"))
    print(json.dumps({"profile": str(output), "rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from agent_orch.data import DatasetManifest, file_sha256


BASE_COLUMNS = (
    "model", "config", "prompt_tokens", "output_tokens", "arrival_rate_rps",
    "long_request_fraction",
)
COMPOSITION_COLUMNS = (
    "interactive_retrieval_fraction",
    "transactional_tool_fraction",
    "deep_research_fraction",
    "coding_agent_fraction",
)
OUTPUT_COLUMNS = (
    "ttft_s", "tbt_s", "response_s", "stable_capacity_rps", "kv_tokens",
)


def validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    present = set(COMPOSITION_COLUMNS) & set(frame.columns)
    if present and present != set(COMPOSITION_COLUMNS):
        raise ValueError("Workload-composition columns must be provided as one complete group")
    required = {*BASE_COLUMNS, *OUTPUT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LLM profile is missing columns: {sorted(missing)}")
    result = frame.copy()
    numeric = [*BASE_COLUMNS[2:], *OUTPUT_COLUMNS, *COMPOSITION_COLUMNS]
    for column in numeric:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise")
    positive = (
        "prompt_tokens", "output_tokens", "arrival_rate_rps",
        "stable_capacity_rps",
    )
    if any((result[column] <= 0.0).any() for column in positive):
        raise ValueError("Token counts, arrival rate, and stable capacity must be positive")
    if any((result[column] < 0.0).any() for column in OUTPUT_COLUMNS):
        raise ValueError("Measured profile outputs must be non-negative")
    if present:
        fractions = result[list(COMPOSITION_COLUMNS)]
        if ((fractions < 0.0) | (fractions > 1.0)).any().any():
            raise ValueError("Workload-composition fractions must lie in [0, 1]")
        if not np.allclose(fractions.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Workload-composition fractions must sum to one")
    return result.sort_values(list(BASE_COLUMNS[:2]) + list(BASE_COLUMNS[2:])).reset_index(drop=True)


def interpolation_holdout(
    frame: pd.DataFrame, holdout_fraction: float, seed: int
) -> dict[str, dict[str, float]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    has_composition = set(COMPOSITION_COLUMNS) <= set(frame.columns)
    input_columns = (
        [*BASE_COLUMNS[2:], *COMPOSITION_COLUMNS] if has_composition
        else list(BASE_COLUMNS[2:])
    )
    train_parts, test_parts = [], []
    for _, group in frame.groupby(["model", "config"], sort=True):
        if len(group) < 5:
            train_parts.append(group)
            continue
        count = max(1, int(round(holdout_fraction * len(group))))
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        test_parts.append(group.loc[chosen])
        train_parts.append(group.drop(chosen))
    if not test_parts:
        return {}
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    errors = {column: [] for column in OUTPUT_COLUMNS}
    for row in test.itertuples(index=False):
        candidates = train[
            (train["model"] == row.model) & (train["config"] == row.config)
        ]
        if candidates.empty:
            continue
        point = np.asarray(
            [float(getattr(row, column)) for column in input_columns], dtype=float
        )
        points = candidates[input_columns].to_numpy(dtype=float)
        nearest = int(np.argmin(np.linalg.norm(points - point, axis=1)))
        for column in OUTPUT_COLUMNS:
            observed = float(getattr(row, column))
            predicted = float(candidates.iloc[nearest][column])
            errors[column].append(abs(predicted - observed) / max(abs(observed), 1e-9))
    return {
        column: {
            "n": len(values),
            "median_ape_pct": 100.0 * float(np.median(values)),
            "p95_ape_pct": 100.0 * float(np.quantile(values, 0.95)),
        }
        for column, values in errors.items()
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-name", default="LLMServingSim 2.0")
    parser.add_argument(
        "--source-url", default="https://github.com/casys-kaist/LLMServingSim"
    )
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--license", default="See source and replay harness")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/processed/llm_profile.csv")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = validate_profile(pd.read_csv(source))
    validation = interpolation_holdout(frame, args.holdout_fraction, args.seed)
    frame.to_csv(output, index=False)
    (output.with_suffix(".validation.json")).write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    DatasetManifest(
        artifact_type="llm-performance-profile",
        source_name=args.source_name,
        source_url=args.source_url,
        source_version=args.source_version,
        license=args.license,
        source_checksum_sha256=file_sha256(source),
        preprocessing_command=" ".join(sys.argv),
        random_seed=args.seed,
        parameters={
            "holdout_fraction": args.holdout_fraction,
            "composition_aware": set(COMPOSITION_COLUMNS) <= set(frame.columns),
            "validation": validation,
        },
    ).write(output.with_suffix(".manifest.json"))
    print(json.dumps({"profile": str(output), "rows": len(frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for group, subset in frame.groupby(group_key, sort=True):
        group_values = (group,) if len(group_columns) == 1 else tuple(group)
        identity = dict(zip(group_columns, group_values))
        for metric in metrics:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "n": values.size,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    seed_column: str,
    baseline: tuple[str, ...],
) -> pd.DataFrame:
    baseline_mask = np.logical_and.reduce(
        [frame[column].astype(str) == value for column, value in zip(group_columns, baseline)]
    )
    baseline_frame = frame.loc[baseline_mask]
    rows = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for group, subset in frame.groupby(group_key, sort=True):
        group_values = (str(group),) if len(group_columns) == 1 else tuple(map(str, group))
        if group_values == baseline:
            continue
        identity = dict(zip(group_columns, group_values))
        for metric in metrics:
            paired = baseline_frame[[seed_column, metric]].merge(
                subset[[seed_column, metric]], on=seed_column, suffixes=("_baseline", "_method")
            )
            if paired.empty:
                continue
            difference = (
                paired[f"{metric}_method"] - paired[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            elif difference.size < 2:
                statistic, p_value = np.nan, np.nan
            else:
                result = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            standard_deviation = difference.std(ddof=1) if difference.size > 1 else np.nan
            effect = (
                float(difference.mean() / standard_deviation)
                if np.isfinite(standard_deviation) and standard_deviation > 0.0
                else np.nan
            )
            rows.append(
                {
                    **identity,
                    "baseline": ",".join(baseline),
                    "metric": metric,
                    "paired_n": difference.size,
                    "mean_paired_difference": float(difference.mean()),
                    "cohen_dz": effect,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = np.nan
        for metric, indices in result.groupby("metric").groups.items():
            valid = [index for index in indices if np.isfinite(result.loc[index, "p_value"])]
            adjusted = _holm_adjust(result.loc[valid, "p_value"].to_numpy(dtype=float))
            result.loc[valid, "p_value_holm"] = adjusted
    return result


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted_sorted = np.empty_like(p_values)
    running = 0.0
    count = p_values.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p_values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-columns", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--seed-column", default="seed")
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", default="results/statistics")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    group_columns = [value.strip() for value in args.group_columns.split(",")]
    metrics = [value.strip() for value in args.metrics.split(",")]
    required = {*group_columns, *metrics, args.seed_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = bootstrap_summary(
        frame, group_columns, metrics, args.bootstrap_samples, args.bootstrap_seed
    )
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    summary.to_csv(output / "bootstrap_summary.csv", index=False)

    if args.baseline:
        baseline = tuple(value.strip() for value in args.baseline.split(","))
        if len(baseline) != len(group_columns):
            raise ValueError("Baseline must provide one value per group column")
        comparisons = paired_comparisons(
            frame, group_columns, metrics, args.seed_column, baseline
        )
        comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
        comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

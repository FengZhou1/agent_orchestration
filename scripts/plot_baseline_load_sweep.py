from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


POLICY_LABELS = {
    "static": "Static",
    "equal": "Equal-Split",
    "least_load": "Least-Load",
    "random": "Random",
    "greedy": "Greedy",
}
POLICY_ORDER = ["Static", "Equal-Split", "Least-Load", "Random", "Greedy"]
METRICS = {
    "mean_cost": "Average Cost",
    "mean_latency_s": "Mean Response Time (s)",
    "mean_goodput_rps": "Goodput (req/s)",
    "mean_quality": "Quality Score",
    "mean_slo_attainment": "SLO Attainment (%)",
    "mean_violations": "Violated Constraints per Slot",
    "violation_slot_fraction": "Slots with Any Violation (%)",
}
PLOT_SCALE = {
    "mean_slo_attainment": 100.0,
    "violation_slot_fraction": 100.0,
}


def load_summaries(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("baseline_load_*.summary.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No baseline summaries found in {input_dir}; run the baseline matrix first"
        )
    frames = []
    for path in paths:
        frame = pd.read_parquet(path).copy()
        frame["method"] = frame["policy"].map(POLICY_LABELS)
        if frame["method"].isna().any():
            raise ValueError(f"Unknown policy in {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale, method), group in frame.groupby(["arrival_scale", "method"]):
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            mean = float(values.mean())
            if len(values) > 1:
                half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
            else:
                half_width = 0.0
            rows.append(
                {
                    "arrival_scale": float(scale),
                    "method": method,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": mean,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    palette = dict(zip(POLICY_ORDER, sns.color_palette("colorblind", len(POLICY_ORDER))))
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.2))
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        subset = summary[summary["metric"] == metric]
        for method in POLICY_ORDER:
            values = subset[subset["method"] == method].sort_values("arrival_scale")
            if values.empty:
                continue
            x = values["arrival_scale"].to_numpy(dtype=float)
            scale = PLOT_SCALE.get(metric, 1.0)
            y = scale * values["mean"].to_numpy(dtype=float)
            low = scale * values["ci95_low"].to_numpy(dtype=float)
            high = scale * values["ci95_high"].to_numpy(dtype=float)
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.0,
                      color=palette[method], label=method)
            axis.fill_between(x, low, high, color=palette[method], alpha=0.12,
                              linewidth=0)
        axis.set_xlabel("Arrival-rate multiplier")
        axis.set_ylabel(label)
        axis.set_xticks(sorted(summary["arrival_scale"].unique()))
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(METRICS):]:
        axis.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.subplots_adjust(top=0.91, hspace=0.52, wspace=0.38)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fig_baseline_load_sweep.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_baseline_load_sweep.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results")
    parser.add_argument("--output", default="results/figures_baseline")
    args = parser.parse_args()
    frame = load_summaries(Path(args.input).resolve())
    summary = summarize(frame)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "baseline_load_sweep_seed_summary.parquet", index=False)
    summary.to_csv(output / "baseline_load_sweep_summary.csv", index=False)
    plot(summary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

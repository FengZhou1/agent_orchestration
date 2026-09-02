from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from summarize_results import bootstrap_summary, paired_comparisons


METHOD_ORDER = [
    "Static",
    "Equal-Split",
    "Least-Load",
    "Random",
    "Greedy",
    "PPO-Route",
    "PPO-Deploy",
    "DTS-PPO",
    "DTS-PPO-RND (Unconstrained)",
    "DTS-PPO-RND",
]

BASELINE_LABELS = {
    "static": "Static",
    "equal": "Equal-Split",
    "least_load": "Least-Load",
    "random": "Random",
    "greedy": "Greedy",
}

RL_LABELS = {
    ("route", "no-rnd"): "PPO-Route",
    ("deploy", "no-rnd"): "PPO-Deploy",
    ("joint", "no-rnd"): "DTS-PPO",
    ("joint", "unconstrained-rnd"): "DTS-PPO-RND (Unconstrained)",
    ("joint", "rnd"): "DTS-PPO-RND",
    ("joint", "potential"): "Joint+Potential",
    ("joint", "icm"): "Joint+ICM",
}

METRICS = {
    "mean_cost": "Average Cost",
    "mean_latency_s": "Mean Response Time (s)",
    "mean_quality": "Quality Score",
}

BASELINE_METRICS = {
    "mean_cost": "Average Cost",
    "mean_latency_s": "Mean Response Time (s)",
    "mean_goodput_rps": "Goodput (req/s)",
    "mean_quality": "Quality Score",
    "mean_slo_attainment": "SLO Attainment (%)",
    "mean_violations": "Violated Constraints per Slot",
    "violation_slot_fraction": "Slots with Any Violation (%)",
}
BASELINE_PLOT_SCALE = {
    "mean_slo_attainment": 100.0,
    "violation_slot_fraction": 100.0,
}


def _style() -> None:
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
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_manifest = _load_json(results / "baseline_fixed.manifest.json")
    rl_manifest = _load_json(results / "rl_matrix" / "manifest.json")
    if baseline_manifest["scenario_hash"] != rl_manifest["scenario_hash"]:
        raise ValueError("Baseline and RL scenario hashes differ")
    if baseline_manifest["slots"] != rl_manifest["eval_slots"]:
        raise ValueError("Baseline and RL evaluation horizons differ")
    if baseline_manifest["seeds"] != rl_manifest["seeds"]:
        raise ValueError("Baseline and RL seed sets differ")

    baseline = pd.read_parquet(results / "baseline_fixed.summary.parquet").copy()
    baseline["method"] = baseline["policy"].map(BASELINE_LABELS)
    baseline["family"] = "Baseline"

    rl = pd.read_parquet(results / "rl_matrix" / "run_summary.parquet").copy()
    rl["method"] = [
        RL_LABELS[(mode, variant)]
        for mode, variant in zip(rl["mode"], rl["variant"])
    ]
    rl["family"] = "RL"

    common = [
        "scenario",
        "seed",
        "method",
        "family",
        "mean_cost",
        "mean_latency_s",
        "mean_goodput_rps",
        "mean_quality",
        "mean_slo_attainment",
        "mean_violations",
        "violation_slot_fraction",
    ]
    unified = pd.concat([baseline[common], rl[common]], ignore_index=True)
    unified["method"] = pd.Categorical(
        unified["method"], categories=METHOD_ORDER, ordered=True
    )
    return unified, rl, baseline


def _bootstrap_table(unified: pd.DataFrame) -> pd.DataFrame:
    return bootstrap_summary(
        unified,
        ["method"],
        list(METRICS),
        samples=10_000,
        seed=2026,
    )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=400)
    plt.close(fig)


def load_baseline_summary(path: Path) -> pd.DataFrame:
    baseline = pd.read_parquet(path).copy()
    baseline["method"] = baseline["policy"].map(BASELINE_LABELS)
    if baseline["method"].isna().any():
        unknown = sorted(baseline.loc[baseline["method"].isna(), "policy"].unique())
        raise ValueError(f"Unknown baseline policies: {unknown}")
    return baseline


def plot_baseline_metrics(
    baseline: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    order = [label for label in METHOD_ORDER[:5] if label in set(baseline["method"])]
    colors = dict(zip(order, sns.color_palette("colorblind", len(order))))
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.2))
    for axis, (metric, label) in zip(axes.flat, BASELINE_METRICS.items()):
        table = summary[summary.metric == metric].set_index("method").loc[order]
        scale = BASELINE_PLOT_SCALE.get(metric, 1.0)
        means = scale * table["mean"].to_numpy(dtype=float)
        lower = scale * table["ci95_low"].to_numpy(dtype=float)
        upper = scale * table["ci95_high"].to_numpy(dtype=float)
        x = np.arange(len(order))
        axis.bar(
            x,
            means,
            color=[colors[method] for method in order],
            edgecolor="black",
            linewidth=0.5,
            yerr=np.vstack([means - lower, upper - means]),
            capsize=2,
        )
        for index, method in enumerate(order):
            raw = scale * baseline.loc[
                baseline.method == method, metric
            ].to_numpy(dtype=float)
            axis.scatter(
                np.full(len(raw), index), raw, color="black", s=7, alpha=0.45, zorder=3
            )
        axis.set_ylabel(label)
        axis.set_xticks(x, order, rotation=25, ha="right")
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(BASELINE_METRICS):]:
        axis.set_visible(False)
    fig.subplots_adjust(hspace=0.52, wspace=0.38)
    _save(fig, output, "fig_baseline_metrics")


def plot_baseline_tradeoff(baseline: pd.DataFrame, output: Path) -> None:
    aggregate = (
        baseline.groupby("method", as_index=False)
        .agg(
            cost=("mean_cost", "mean"),
            quality=("mean_quality", "mean"),
            latency=("mean_latency_s", "mean"),
        )
    )
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    norm = plt.Normalize(aggregate.latency.min(), aggregate.latency.max())
    cmap = plt.get_cmap("viridis_r")
    for row in aggregate.itertuples(index=False):
        axis.scatter(
            row.cost,
            row.quality,
            s=42,
            color=cmap(norm(row.latency)),
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axis.annotate(
            row.method,
            (row.cost, row.quality),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.02
    )
    colorbar.set_label("Mean Response Time (s)")
    axis.set_xlabel("Average Cost")
    axis.set_ylabel("Quality Score")
    axis.grid(alpha=0.25)
    _save(fig, output, "fig_baseline_tradeoff")


def write_baseline_tables(
    baseline: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(output / "baseline_seed_summary.parquet", index=False)
    summary.to_csv(output / "baseline_bootstrap_summary.csv", index=False)
    wide = summary.pivot(
        index="method", columns="metric", values=["mean", "ci95_low", "ci95_high"]
    ).reindex(METHOD_ORDER[:5])
    wide.to_csv(output / "baseline_bootstrap_summary_wide.csv")


def plot_main_performance(
    unified: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    palette = dict(zip(METHOD_ORDER, sns.color_palette("colorblind", len(METHOD_ORDER))))
    y_positions = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 4.1), sharey=True)
    rng = np.random.default_rng(2026)
    for axis, (metric, label) in zip(axes, METRICS.items()):
        table = summary[summary.metric == metric].set_index("method")
        for y, method in enumerate(METHOD_ORDER):
            row = table.loc[method]
            color = palette[method]
            raw = unified.loc[unified.method == method, metric].to_numpy(dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=len(raw))
            axis.scatter(raw, y + jitter, s=10, color=color, alpha=0.45, linewidth=0)
            axis.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]],
                fmt="o",
                markersize=4,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
            )
        axis.set_xlabel(label)
        axis.set_yticks(y_positions, METHOD_ORDER)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Method")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, output, "fig1_main_performance")


def plot_pareto(unified: pd.DataFrame, output: Path) -> None:
    aggregate = (
        unified.groupby(["method", "family"], observed=True)
        .agg(
            cost=("mean_cost", "mean"),
            quality=("mean_quality", "mean"),
            latency=("mean_latency_s", "mean"),
        )
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    norm = plt.Normalize(aggregate.latency.min(), aggregate.latency.max())
    cmap = plt.get_cmap("viridis_r")
    markers = {"Baseline": "o", "RL": "s"}
    offsets = {
        "Static": (4, 4),
        "Equal-Split": (4, -10),
        "Least-Load": (4, 4),
        "Random": (4, 10),
        "Greedy": (4, 4),
        "PPO-Route": (4, -10),
        "PPO-Deploy": (4, 4),
        "DTS-PPO": (4, -10),
        "DTS-PPO-RND (Unconstrained)": (-55, -10),
        "DTS-PPO-RND": (4, 4),
        "Joint+Potential": (-55, -10),
        "Joint+ICM": (4, 4),
    }
    for row in aggregate.itertuples(index=False):
        axis.scatter(
            row.cost,
            row.quality,
            s=38,
            marker=markers[row.family],
            color=cmap(norm(row.latency)),
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axis.annotate(
            row.method,
            (row.cost, row.quality),
            xytext=offsets[row.method],
            textcoords="offset points",
            fontsize=6.3,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.02
    )
    colorbar.set_label("Mean Response Time (s)")
    axis.set_xlabel("Average Cost")
    axis.set_ylabel("Quality Score")
    axis.grid(alpha=0.25)
    legend = [
        Line2D([0], [0], marker=marker, linestyle="", color="black", label=family)
        for family, marker in markers.items()
    ]
    axis.legend(handles=legend, loc="lower right", frameon=True)
    _save(fig, output, "fig2_cost_quality_pareto")


def plot_rl_ablation(rl: pd.DataFrame, output: Path) -> None:
    order = [
        "PPO-Route",
        "PPO-Deploy",
        "DTS-PPO",
        "DTS-PPO-RND (Unconstrained)",
        "DTS-PPO-RND",
    ]
    colors = dict(zip(order, sns.color_palette("colorblind", len(order))))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    for axis, (metric, label) in zip(axes, METRICS.items()):
        values = [rl.loc[rl.method == method, metric].to_numpy(dtype=float) for method in order]
        means = np.asarray([value.mean() for value in values])
        intervals = np.asarray([_bootstrap_interval(value) for value in values])
        axis.bar(
            np.arange(len(order)),
            means,
            color=[colors[method] for method in order],
            edgecolor="black",
            linewidth=0.5,
            yerr=np.vstack([means - intervals[:, 0], intervals[:, 1] - means]),
            capsize=2,
        )
        for index, value in enumerate(values):
            axis.scatter(
                np.full(len(value), index), value, color="black", s=7, alpha=0.45, zorder=3
            )
        axis.set_ylabel(label)
        axis.set_xticks(
            np.arange(len(order)),
            ["Route", "Deploy", "DTS", "Unconst.", "DTS+RND"],
            rotation=25,
        )
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, output, "fig3_rl_ablation")


def _bootstrap_interval(values: np.ndarray, samples: int = 5_000) -> tuple[float, float]:
    rng = np.random.default_rng(2026)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_training_history(results: Path) -> pd.DataFrame:
    records = []
    pattern = re.compile(
        r"-(joint|deploy|route)-(rnd|no-rnd|unconstrained-rnd|potential|icm)-s(\d+)-"
    )
    for path in (results / "rl_matrix").glob("*/training_history.json"):
        match = pattern.search(path.parent.name)
        if not match:
            continue
        mode, variant, seed = match.group(1), match.group(2), int(match.group(3))
        method = RL_LABELS[(mode, variant)]
        for record in _load_json(path):
            records.append({"method": method, "seed": seed, **record})
    frame = pd.DataFrame(records)
    frame["reward_smooth"] = frame.groupby(["method", "seed"])["mean_reward"].transform(
        lambda series: series.rolling(5, min_periods=1, center=True).mean()
    )
    return frame


def plot_convergence(history: pd.DataFrame, output: Path) -> None:
    panels = [
        (
            "Joint PPO variants",
            ["DTS-PPO", "DTS-PPO-RND (Unconstrained)", "DTS-PPO-RND"],
        ),
        ("Deployment policy", ["PPO-Deploy"]),
        ("Routing policy", ["PPO-Route"]),
    ]
    palette = dict(
        zip(
            [
                "DTS-PPO",
                "DTS-PPO-RND (Unconstrained)",
                "DTS-PPO-RND",
                "PPO-Deploy",
                "PPO-Route",
            ],
            sns.color_palette("colorblind", 5),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for axis, (title, methods) in zip(axes, panels):
        for method in methods:
            subset = history[history.method == method]
            pivot = subset.pivot(index="update", columns="seed", values="reward_smooth")
            center = pivot.mean(axis=1).to_numpy(dtype=float)
            low, high = _row_bootstrap_band(pivot.to_numpy(dtype=float))
            updates = pivot.index.to_numpy(dtype=float)
            axis.plot(updates, center, label=method, color=palette[method], linewidth=1.2)
            axis.fill_between(updates, low, high, color=palette[method], alpha=0.18, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Training Update")
        axis.set_ylabel("Mean Reward")
        axis.grid(alpha=0.25)
        if len(methods) > 1:
            axis.legend(frameon=False)
    fig.subplots_adjust(wspace=0.34)
    _save(fig, output, "fig4_training_convergence")


def _row_bootstrap_band(values: np.ndarray, samples: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    sampled = rng.integers(0, values.shape[1], size=(samples, values.shape[1]))
    draws = values[:, sampled].mean(axis=2)
    return np.quantile(draws, 0.025, axis=1), np.quantile(draws, 0.975, axis=1)


def plot_overhead(rl: pd.DataFrame, output: Path) -> None:
    order = [
        "PPO-Route",
        "PPO-Deploy",
        "DTS-PPO",
        "DTS-PPO-RND (Unconstrained)",
        "DTS-PPO-RND",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.35))
    specifications = [
        ("train_wall_time_s", "Training Time (min)", 1.0 / 60.0),
        ("mean_decision_time_ms", "Decision Time (ms)", 1.0),
    ]
    for axis, (metric, label, scale) in zip(axes, specifications):
        values = [rl.loc[rl.method == method, metric].to_numpy(dtype=float) * scale for method in order]
        means = [value.mean() for value in values]
        intervals = np.asarray([_bootstrap_interval(value) for value in values])
        axis.bar(
            np.arange(len(order)),
            means,
            color=sns.color_palette("colorblind", len(order)),
            edgecolor="black",
            linewidth=0.5,
            yerr=np.vstack([np.asarray(means) - intervals[:, 0], intervals[:, 1] - np.asarray(means)]),
            capsize=2,
        )
        axis.set_ylabel(label)
        axis.set_xticks(np.arange(len(order)), ["Route", "Deploy", "Joint", "+Pot.", "+ICM"], rotation=30)
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, output, "fig5_rl_overhead")


def plot_bursty_trace(results: Path, output: Path) -> None:
    path = results / "baseline_matrix.parquet"
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    frame["method"] = frame.policy.map(BASELINE_LABELS)
    by_seed = frame.drop_duplicates(["seed", "slot"])[
        ["seed", "slot", "total_arrival_rps"]
    ]
    arrival = by_seed.groupby("slot").total_arrival_rps.mean().rolling(9, center=True, min_periods=1).mean()
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.2), sharex=True)
    axes[0].plot(arrival.index, arrival.values, color="black", linewidth=1.0)
    axes[0].set_ylabel("Arrival Rate\n(req/s)")
    for method in METHOD_ORDER[:5]:
        subset = frame[frame.method == method]
        for metric, axis in (("mean_latency_s", axes[1]), ("quality", axes[2])):
            curve = subset.groupby("slot")[metric].mean().rolling(9, center=True, min_periods=1).mean()
            axis.plot(curve.index, curve.values, label=method, linewidth=0.9)
    axes[1].set_ylabel("Response Time (s)")
    axes[2].set_ylabel("Quality")
    axes[2].set_xlabel("Time Slot")
    axes[1].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.subplots_adjust(hspace=0.23)
    _save(fig, output, "fig6_bursty_workload")


def write_tables(unified: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unified.assign(method=unified.method.astype(str)).to_parquet(
        output / "unified_seed_summary.parquet", index=False
    )
    summary.to_csv(output / "method_bootstrap_summary.csv", index=False)
    wide = (
        summary.pivot(index="method", columns="metric", values=["mean", "ci95_low", "ci95_high"])
        .reindex(METHOD_ORDER)
    )
    wide.to_csv(output / "method_bootstrap_summary_wide.csv")
    comparisons = paired_comparisons(
        unified.assign(method=unified.method.astype(str)),
        ["method"],
        list(METRICS),
        "seed",
        ("DTS-PPO-RND",),
    )
    comparisons.to_csv(output / "paired_comparisons_vs_joint_ppo.csv", index=False)
    latex = unified.groupby("method", observed=True)[list(METRICS)].agg(["mean", "std"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        (output / "method_summary.tex").write_text(
            latex.to_latex(float_format=lambda value: f"{value:.4f}"), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--baseline-summary",
        help="Baseline summary Parquet; when set, only baseline figures are generated",
    )
    parser.add_argument("--output", default="results/figures")
    parser.add_argument("--analysis-output", default="results/analysis")
    args = parser.parse_args()
    _style()
    results = Path(args.results).resolve()
    figure_output = Path(args.output).resolve()
    analysis_output = Path(args.analysis_output).resolve()
    figure_output.mkdir(parents=True, exist_ok=True)
    if args.baseline_summary:
        baseline = load_baseline_summary(Path(args.baseline_summary).resolve())
        summary = bootstrap_summary(
            baseline,
            ["method"],
            list(BASELINE_METRICS),
            samples=10_000,
            seed=2026,
        )
        write_baseline_tables(baseline, summary, analysis_output)
        plot_baseline_metrics(baseline, summary, figure_output)
        plot_baseline_tradeoff(baseline, figure_output)
        print(figure_output)
        return 0
    unified, rl, _ = load_results(results)
    summary = _bootstrap_table(unified)
    write_tables(unified, summary, analysis_output)
    plot_main_performance(unified, summary, figure_output)
    plot_pareto(unified, figure_output)
    plot_rl_ablation(rl, figure_output)
    plot_convergence(load_training_history(results), figure_output)
    plot_overhead(rl, figure_output)
    plot_bursty_trace(results, figure_output)
    print(figure_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

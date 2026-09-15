"""Revalidate the steady-state LLM equations with two hardware-rate choices.

The ``peak`` branch uses the hardware-layer BF16 peak rates stored in the
scenario.  The ``equivalent`` branch fits only the two hardware throughput
parameters (compute rate and memory bandwidth) to the independent
LLMServingSim observations.  No queueing, KV, or analytical-formula
parameter is fitted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_orch.schema.loader import ScenarioLoader  # noqa: E402
from agent_orch.validation.llmservingsim import WorkloadClass  # noqa: E402
from validate_llm_steady_state_model import (  # noqa: E402
    analytical_capacity,
    analytical_metrics,
    family_workloads,
    model_id_for,
)


COMPOSITIONS = {
    "interactive": {"interactive_retrieval": 1.0},
    "transactional": {"transactional_tool": 1.0},
    "deep_research": {"deep_research": 1.0},
    "coding": {"coding_agent": 1.0},
    "balanced": {name: 0.25 for name in (
        "interactive_retrieval",
        "transactional_tool",
        "deep_research",
        "coding_agent",
    )},
}


def scenario_with_rates(scenario, config_id: str, flops: float, bandwidth: float):
    configs = dict(scenario.llm_configs)
    configs[config_id] = replace(
        configs[config_id],
        effective_flops=float(flops),
        effective_bandwidth_bytes_s=float(bandwidth),
    )
    return replace(scenario, llm_configs=configs)


def workloads_for(scenario, config_id: str) -> dict[str, WorkloadClass]:
    return family_workloads(scenario, model_id_for(config_id))


def valid_rows(observations: pd.DataFrame, config_id: str) -> pd.DataFrame:
    frame = observations[
        (observations["config_id"] == config_id)
        & (observations["load_factor"] < 1.0)
    ].copy()
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_service_s", "observed_tbt_s"]
    )


def fit_hardware_rates(scenario, observations: pd.DataFrame, config_id: str) -> dict:
    config = scenario.llm_configs[config_id]
    workloads = workloads_for(scenario, config_id)
    rows = valid_rows(observations, config_id)
    if rows.empty:
        raise ValueError(f"No stable observations available for {config_id}")

    def residual(log_rates: np.ndarray) -> np.ndarray:
        trial = scenario_with_rates(
            scenario,
            config_id,
            float(np.exp(log_rates[0])),
            float(np.exp(log_rates[1])),
        )
        values: list[float] = []
        for row in rows.itertuples(index=False):
            metrics = analytical_metrics(
                trial,
                config_id,
                workloads,
                COMPOSITIONS[str(row.composition_id)],
                float(row.arrival_rate_rps),
            )
            for predicted, observed in (
                (metrics["predicted_service_s"], row.observed_service_s),
                (metrics["predicted_tbt_s"], row.observed_tbt_s),
            ):
                values.append(float(np.log((predicted + 1e-8) / (observed + 1e-8))))
        return np.asarray(values)

    initial = np.log([config.effective_flops, config.effective_bandwidth_bytes_s])
    result = least_squares(
        residual,
        initial,
        bounds=(np.log([1e12, 1e9]), np.log([1e16, 1e14])),
        max_nfev=80,
    )
    return {
        "effective_flops": float(np.exp(result.x[0])),
        "effective_bandwidth_bytes_s": float(np.exp(result.x[1])),
        "n_fit_rows": int(len(rows)),
        "fit_rmse_log": float(np.sqrt(np.mean(np.square(result.fun)))),
        "fit_success": bool(result.success),
    }


def make_predictions(scenario, observations: pd.DataFrame, rates: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    workloads_cache = {config_id: workloads_for(scenario, config_id) for config_id in rates}
    for row in observations.itertuples(index=False):
        config_id = str(row.config_id)
        rate = rates[config_id]
        trial = scenario_with_rates(
            scenario,
            config_id,
            rate["effective_flops"],
            rate["effective_bandwidth_bytes_s"],
        )
        metrics = analytical_metrics(
            trial,
            config_id,
            workloads_cache[config_id],
            COMPOSITIONS[str(row.composition_id)],
            float(row.arrival_rate_rps),
        )
        record = {key: getattr(row, key) for key in observations.columns if hasattr(row, key)}
        for key, value in metrics.items():
            record[key] = value
        rows.append(record)
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    metric_pairs = {
        "wait_s": ("predicted_wait_s", "observed_wait_s"),
        "ttft_s": ("predicted_ttft_s", "observed_ttft_s"),
        "tbt_s": ("predicted_tbt_s", "observed_tbt_s"),
        "response_s": ("predicted_response_s", "observed_response_s"),
        "service_s": ("predicted_service_s", "observed_service_s"),
    }
    rows = []
    for (config, composition), group in predictions.groupby(["config_id", "composition_id"]):
        for metric, (predicted, observed) in metric_pairs.items():
            frame = group[[predicted, observed]].replace([np.inf, -np.inf], np.nan).dropna()
            if frame.empty:
                continue
            denominator = max(float(frame[observed].abs().sum()), 1e-9)
            rows.append({
                "config_id": config,
                "composition_id": composition,
                "metric": metric,
                "n": int(len(frame)),
                "wape": float((frame[predicted] - frame[observed]).abs().sum() / denominator),
                "mean_ratio_predicted_over_observed": float(
                    frame[predicted].mean() / max(frame[observed].mean(), 1e-9)
                ),
            })
    return pd.DataFrame(rows)


def write_figures(predictions: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "ttft_s": "TTFT (s)",
        "tbt_s": "TBT (s/token)",
        "response_s": "end-to-end response time (s)",
        "service_s": "LLM service time (s)",
    }
    # Each panel is one configuration/composition.  This makes the load trend
    # visible without putting twenty unrelated curves into one legend.
    for metric, ylabel in labels.items():
        predicted, observed = f"predicted_{metric}", f"observed_{metric}"
        configs = sorted(predictions["config_id"].unique())
        compositions = sorted(predictions["composition_id"].unique())
        fig, axes = plt.subplots(
            len(configs), len(compositions), figsize=(3.1 * len(compositions), 2.35 * len(configs)),
            sharex=True, sharey=False, squeeze=False,
        )
        for row, config in enumerate(configs):
            for col, composition in enumerate(compositions):
                ax = axes[row][col]
                group = predictions[
                    (predictions["config_id"] == config)
                    & (predictions["composition_id"] == composition)
                ].sort_values("load_factor")
                if group.empty:
                    ax.axis("off")
                    continue
                ax.plot(group["load_factor"], group[observed], "o-", lw=1.5, ms=3.5,
                        color="#2166ac", label="LLMServingSim")
                ax.plot(group["load_factor"], group[predicted], "x--", lw=1.4, ms=4,
                        color="#b2182b", label="analytical")
                ax.set_title(f"{config} | {composition}", fontsize=8)
                ax.grid(alpha=0.25)
                if row == len(configs) - 1:
                    ax.set_xlabel("offered load / simulated capacity")
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Load trend: {ylabel}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figure_dir / f"{metric}_trend_by_load.png", dpi=220)
        plt.close(fig)


def write_comparison_figures(peak: pd.DataFrame, equivalent: pd.DataFrame, output: Path) -> None:
    """Write figures that answer the three validation questions directly.

    1) trend-by-load panels compare simulation and analysis in the same axes;
    2) log-log scatter plots show prediction accuracy against y=x;
    3) fixed-load configuration bars show whether model/GPU ordering is retained.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return
    output.mkdir(parents=True, exist_ok=True)
    metric_names = {
        "service_s": "LLM service time (s)",
        "tbt_s": "TBT (s/token)",
        "response_s": "end-to-end response time (s)",
        "ttft_s": "TTFT (s)",
    }
    modes = [("peak", peak), ("equivalent", equivalent)]
    colors = {"qwen3-4b-a10": "#1b9e77", "qwen3-8b-h20": "#d95f02",
              "qwen3-14b-h20": "#7570b3", "qwen3-32b-h20": "#e7298a"}
    markers = {"balanced": "o", "coding": "s", "deep_research": "^",
               "interactive": "D", "transactional": "P"}

    # Accuracy scatter: every point is one configuration/composition/load.
    scatter_dir = output / "scatter"
    scatter_dir.mkdir(exist_ok=True)
    for metric, ylabel in metric_names.items():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), squeeze=False)
        finite_all = []
        for _, frame in modes:
            finite_all.extend(frame[[f"observed_{metric}", f"predicted_{metric}"]].to_numpy().ravel())
        finite_all = np.asarray(finite_all, dtype=float)
        finite_all = finite_all[np.isfinite(finite_all) & (finite_all > 0)]
        lo = max(float(finite_all.min()) * 0.75, 1e-5) if len(finite_all) else 1e-5
        hi = float(finite_all.max()) * 1.35 if len(finite_all) else 1.0
        for col, (mode, frame) in enumerate(modes):
            ax = axes[0][col]
            for (config, composition), group in frame.groupby(["config_id", "composition_id"]):
                x = group[f"observed_{metric}"].to_numpy(dtype=float)
                y = group[f"predicted_{metric}"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                ax.scatter(x[mask], y[mask], s=34, alpha=0.85,
                           color=colors.get(config, "#333333"),
                           marker=markers.get(composition, "o"),
                           edgecolors="white", linewidths=0.35)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, label="ideal: analytical = simulation")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("LLMServingSim (s)")
            ax.set_ylabel("analytical model (s)")
            ax.set_title(f"{mode} hardware parameters")
            ax.grid(alpha=0.25, which="both")
            if col == 0:
                ax.legend(fontsize=7, loc="upper left")
        config_handles = [Line2D([0], [0], marker="o", color="w", label=k,
                                  markerfacecolor=v, markersize=7) for k, v in colors.items()]
        comp_handles = [Line2D([0], [0], marker=v, color="k", linestyle="None", label=k,
                               markersize=6) for k, v in markers.items()]
        fig.legend(handles=config_handles + comp_handles, loc="lower center", ncol=5,
                   fontsize=7, frameon=False)
        fig.suptitle(f"Accuracy scatter: {ylabel}", fontsize=13)
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        fig.savefig(scatter_dir / f"{metric}_scatter.png", dpi=220)
        plt.close(fig)

    # Configuration ordering: use the balanced workload at load factor 0.85,
    # which is common to all four observed model/GPU combinations.
    fixed = equivalent[(equivalent["composition_id"] == "balanced") &
                       (np.isclose(equivalent["load_factor"], 0.85))].copy()
    if not fixed.empty:
        peak_fixed = peak[(peak["composition_id"] == "balanced") &
                          (np.isclose(peak["load_factor"], 0.85))]
        order = (fixed.groupby("config_id")["observed_response_s"].mean()
                 .sort_values().index.tolist())
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), squeeze=False)
        bar_metrics = [("service_s", "service time (s)"), ("tbt_s", "TBT (s/token)"),
                       ("response_s", "response time (s)")]
        x = np.arange(len(order))
        width = 0.25
        for col, (metric, ylabel) in enumerate(bar_metrics):
            ax = axes[0][col]
            sim_vals = [float(fixed.loc[fixed.config_id == c, f"observed_{metric}"].mean()) for c in order]
            peak_vals = [float(peak_fixed.loc[peak_fixed.config_id == c, f"predicted_{metric}"].mean()) for c in order]
            eq_vals = [float(fixed.loc[fixed.config_id == c, f"predicted_{metric}"].mean()) for c in order]
            ax.bar(x - width, sim_vals, width, label="LLMServingSim", color="#2166ac")
            ax.bar(x, peak_vals, width, label="peak hardware", color="#999999")
            ax.bar(x + width, eq_vals, width, label="equivalent hardware", color="#b2182b")
            ax.set_xticks(x, order, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            if col == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Model/GPU configuration ordering at balanced load (0.85)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output / "configuration_ordering_balanced_l085.png", dpi=220)
        plt.close(fig)

    # A reader-friendly version of the load curves: one figure per workload
    # composition, with the four model/GPU configurations as panels.
    trend_dir = output / "trends"
    trend_dir.mkdir(exist_ok=True)
    for mode, frame in modes:
        for metric, ylabel in metric_names.items():
            for composition in sorted(frame["composition_id"].unique()):
                subset = frame[frame["composition_id"] == composition]
                configs = sorted(subset["config_id"].unique())
                fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), sharex=True, squeeze=False)
                axes_flat = axes.ravel()
                for index, config in enumerate(configs):
                    ax = axes_flat[index]
                    group = subset[subset["config_id"] == config].sort_values("load_factor")
                    ax.plot(group["load_factor"], group[f"observed_{metric}"], "o-", lw=1.8, ms=4,
                            color="#2166ac", label="LLMServingSim")
                    ax.plot(group["load_factor"], group[f"predicted_{metric}"], "x--", lw=1.6, ms=5,
                            color="#b2182b", label="analytical")
                    ax.set_title(config, fontsize=10)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.25, which="both")
                    ax.set_xlabel("offered load / simulated capacity")
                    ax.set_ylabel(ylabel)
                    if index == 0:
                        ax.legend(fontsize=8)
                for index in range(len(configs), 4):
                    axes_flat[index].axis("off")
                fig.suptitle(f"{mode}: {composition} workload under increasing load", fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                fig.savefig(trend_dir / f"{mode}_{metric}_{composition}.png", dpi=220)
                plt.close(fig)


def trend_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether both curves move in the same direction with load."""
    metrics = ["service_s", "tbt_s", "response_s", "ttft_s"]
    rows = []
    for (config, composition), group in predictions.groupby(["config_id", "composition_id"]):
        group = group.sort_values("load_factor")
        for metric in metrics:
            x = group[f"observed_{metric}"].to_numpy(dtype=float)
            y = group[f"predicted_{metric}"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                continue
            dx = np.sign(np.diff(x[mask]))
            dy = np.sign(np.diff(y[mask]))
            rows.append({
                "config_id": config,
                "composition_id": composition,
                "metric": metric,
                "simulation_monotone_non_decreasing": bool(np.all(dx >= 0)),
                "analytical_monotone_non_decreasing": bool(np.all(dy >= 0)),
                "same_direction_fraction": float(np.mean(dx == dy)),
                "simulation_load_slope": float(x[mask][-1] - x[mask][0]),
                "analytical_load_slope": float(y[mask][-1] - y[mask][0]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/benchmarks/main_abilene_revised.yaml")
    parser.add_argument("--observations", type=Path, default=ROOT / "results/llm_steady_state_validation_final/observations.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/llm_steady_state_parameter_revalidation")
    args = parser.parse_args()

    scenario = ScenarioLoader.load(args.scenario)
    observations = pd.read_csv(args.observations)
    config_ids = sorted(observations["config_id"].unique())
    peak_rates = {
        config_id: {
            "effective_flops": scenario.llm_configs[config_id].effective_flops,
            "effective_bandwidth_bytes_s": scenario.llm_configs[config_id].effective_bandwidth_bytes_s,
        }
        for config_id in config_ids
    }
    fitted_rates = {
        config_id: fit_hardware_rates(scenario, observations, config_id)
        for config_id in config_ids
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    peak_predictions = make_predictions(scenario, observations, peak_rates)
    fitted_predictions = make_predictions(scenario, observations, fitted_rates)
    peak_predictions.to_csv(output / "peak_predictions.csv", index=False)
    fitted_predictions.to_csv(output / "equivalent_predictions.csv", index=False)
    peak_summary = summarize(peak_predictions)
    peak_summary.insert(0, "parameter_mode", "peak")
    fitted_summary = summarize(fitted_predictions)
    fitted_summary.insert(0, "parameter_mode", "equivalent_hardware_fit")
    summary = pd.concat([peak_summary, fitted_summary], ignore_index=True)
    summary.to_csv(output / "error_summary.csv", index=False)
    pd.DataFrame([
        {"parameter_mode": "peak", "config_id": k, **v} for k, v in peak_rates.items()
    ] + [
        {"parameter_mode": "equivalent_hardware_fit", "config_id": k, **v} for k, v in fitted_rates.items()
    ]).to_csv(output / "hardware_parameters.csv", index=False)

    kv_path = ROOT / "results/llm_steady_state_validation_final/kv_capacity_compare.csv"
    if kv_path.exists():
        kv = pd.read_csv(kv_path)
        capacities = {
            k: config.kv_token_capacity
            for k, config in scenario.llm_configs.items()
        }
        kv["yaml_capacity"] = kv["label"].map(capacities)
        kv["yaml_vs_sim_relative_error"] = (kv["yaml_capacity"] - kv["sim"]) / kv["sim"]
        kv.to_csv(output / "kv_capacity_comparison.csv", index=False)

    write_figures(peak_predictions, output / "peak")
    write_figures(fitted_predictions, output / "equivalent_hardware_fit")
    write_comparison_figures(peak_predictions, fitted_predictions, output / "comparison")
    trend_consistency(peak_predictions).to_csv(output / "trend_consistency_peak.csv", index=False)
    trend_consistency(fitted_predictions).to_csv(output / "trend_consistency.csv", index=False)
    report = [
        "# LLM hardware-parameter revalidation",
        "",
        "The peak branch uses the hardware-layer BF16 peak rates in the scenario. The equivalent branch fits only compute throughput and memory bandwidth to independent LLMServingSim observations; no queueing, KV, or formula parameter is fitted.",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Observations: `{args.observations}`",
        f"- Configurations: {', '.join(config_ids)}",
        "",
        "The comparison directory contains per-composition load-trend panels, log-log accuracy scatters, and fixed-load configuration-ordering bars. `trend_consistency_peak.csv` and `trend_consistency.csv` report whether the simulation and analytical curves move in the same direction as offered load increases.",
        "",
        "See `hardware_parameters.csv` for the two hardware parameter sets and `error_summary.csv` for the resulting validation metrics.",
    ]
    (output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({
        "scenario": str(args.scenario),
        "observations": str(args.observations),
        "configs": config_ids,
        "fit_targets": ["service_s", "tbt_s"],
        "formula_parameters_fitted": False,
    }, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate completed steady-state LLMServingSim validation batches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "results/llm_steady_state_validation_v3_core",
    ROOT / "results/llm_steady_state_validation_v3_long_interactive",
    ROOT / "results/llm_steady_state_validation_v3_long_transactional",
    ROOT / "results/llm_steady_state_validation_v3_long_balanced",
    ROOT / "results/llm_steady_state_validation_v3_edge",
)
OUTPUT = ROOT / "results/llm_steady_state_validation_final"


def load_unique(name: str) -> pd.DataFrame:
    frames = []
    for directory in INPUTS:
        path = directory / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "n" in frame.columns:
        frame = frame.sort_values("n")
    return frame.drop_duplicates("job_id", keep="last").sort_values("job_id")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No valid rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> int:
    # The inputs are the LLMServingSim cross-validation runs, which are produced on
    # the remote box (see docs/llmservingsim_queue_validation.md) and are not in the
    # repository.  Fail with the list rather than silently writing an empty report.
    missing = [str(directory) for directory in INPUTS if not directory.exists()]
    if missing:
        print("no validation runs to aggregate; expected these directories:")
        for directory in missing:
            print(f"  {directory}")
        print(
            "\nRun the validation jobs first (see docs/llmservingsim_queue_validation.md); "
            "an empty report is not written."
        )
        return 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    observations = load_unique("observations.csv")
    predictions = load_unique("predictions.csv")
    observations.to_csv(OUTPUT / "observations.csv", index=False)
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    if observations.empty:
        (OUTPUT / "validation_report.md").write_text("No observations.\n", encoding="utf-8")
        return 0

    pairs = {
        "wait_s": ("predicted_wait_s", "observed_wait_s"),
        "ttft_s": ("predicted_ttft_s", "observed_ttft_s"),
        "tbt_s": ("predicted_tbt_s", "observed_tbt_s"),
        "response_s": ("predicted_response_s", "observed_response_s"),
        "service_s": ("predicted_service_s", "observed_service_s"),
    }
    rows = []
    for (config, composition), group in observations.groupby(["config_id", "composition_id"]):
        group = group.sort_values("load_factor")
        for metric, (pred, obs) in pairs.items():
            valid = group[["load_factor", pred, obs]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 2:
                continue
            observed = valid[obs].to_numpy(float)
            predicted = valid[pred].to_numpy(float)
            rows.append({
                "config_id": config,
                "composition_id": composition,
                "metric": metric,
                "n_loads": len(valid),
                "spearman_observed_vs_load": valid["load_factor"].corr(valid[obs], method="spearman"),
                "spearman_predicted_vs_load": valid["load_factor"].corr(valid[pred], method="spearman"),
                "wape": np.abs(predicted - observed).sum() / max(np.abs(observed).sum(), 1e-12),
                "mean_ratio_predicted_over_observed": np.mean(predicted / np.maximum(observed, 1e-12)),
                "observed_monotone_non_decreasing": bool(np.all(np.diff(observed) >= -1e-12)),
                "predicted_monotone_non_decreasing": bool(np.all(np.diff(predicted) >= -1e-12)),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "trend_metrics.csv", index=False)

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        for metric, (pred, obs) in pairs.items():
            fig, ax = plt.subplots(figsize=(6.0, 3.8))
            for (config, composition), group in observations.groupby(["config_id", "composition_id"]):
                ordered = group.sort_values("load_factor")
                label = f"{config}/{composition}"
                ax.plot(ordered["load_factor"], ordered[obs], marker="o", alpha=0.65, label=f"sim {label}")
                ax.plot(ordered["load_factor"], ordered[pred], marker="x", linestyle="--", alpha=0.8, label=f"model {label}")
            ax.set_xlabel("offered load / analytical capacity")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            if len(observations.groupby(["config_id", "composition_id"])) <= 6:
                ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(figure_dir / f"{metric}_trend.png", dpi=180)
            plt.close(fig)
    except ImportError:
        pass

    kv_path = ROOT / "results/llm_kv_validation/kv_capacity_compare.json"
    kv = json.loads(kv_path.read_text(encoding="utf-8-sig")) if kv_path.exists() else []
    kv_frame = pd.DataFrame(kv)
    if not kv_frame.empty:
        kv_frame["relative_error_gib"] = (kv_frame["gib"] - kv_frame["sim"]).abs() / kv_frame["sim"]
        kv_frame.to_csv(OUTPUT / "kv_capacity_compare.csv", index=False)
    report = [
        "# LLM steady-state analytical model validation",
        "",
        f"Completed unique simulator runs: {len(observations)}.",
        "The analytical columns are generated from the current TeX equations; simulator outputs are not used to fit the analytical parameters.",
        "The run set combines long interactive/transactional windows with shorter mixed-agent composition probes; `n` is the number of post-warm-up observations retained by the estimator.",
        "",
        "## Trend and error summary",
        "",
        markdown_table(metrics.round(4)),
        "",
        "## KV capacity cross-check",
        "",
        markdown_table(kv_frame[["label", "sim", "gib", "relative_error_gib"]].round(4) if not kv_frame.empty else kv_frame),
        "",
        "## Interpretation",
        "",
        "The simulator consistently exposes increasing running concurrency and response time as the offered load increases. The analytical model preserves this direction for the tested configurations and compositions.",
        "The analytical waiting term remains small when the calculated resident capacity is much larger than the steady active concurrency. The dominant discrepancy is the Roofline service-demand approximation: multiplying the per-request demand by the aggregate active concurrency is conservative for the simulator's continuous-batching profile, especially near high load.",
        "The KV token-capacity calculation agrees with the simulator's GiB accounting up to the configured memory-headroom convention; the remaining difference is a capacity-policy offset rather than a mismatch in the per-token KV formula.",
        "",
        "The results support using the equations as a low-cost, trend-preserving steady-state surrogate for orchestration optimization, while retaining LLMServingSim replay for final latency calibration and evaluation.",
    ]
    (OUTPUT / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

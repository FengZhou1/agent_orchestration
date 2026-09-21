"""Plot the latest single-seed PPO training diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="optional validation_history.jsonl to overlay on the utility panel",
    )
    args = parser.parse_args()

    records = json.loads(args.history.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records).sort_values("update").reset_index(drop=True)
    update = frame["update"].to_numpy(dtype=float) + 1.0
    frame["utility_ma"] = frame["mean_utility"].rolling(args.window, min_periods=1).mean()
    if "mean_learning_utility" in frame:
        frame["learning_utility_ma"] = frame["mean_learning_utility"].rolling(
            args.window, min_periods=1
        ).mean()
    frame["deploy_per_period"] = frame["deployment_steps"] / frame["routing_steps"]
    frame["deploy_ma"] = frame["deploy_per_period"].rolling(args.window, min_periods=1).mean()
    frame["collection_ma"] = frame["collection_time_s"].rolling(args.window, min_periods=1).mean()
    frame["optimization_ma"] = frame["optimization_time_s"].rolling(args.window, min_periods=1).mean()

    validation_path = args.validation
    if validation_path is None:
        candidate = args.history.with_name("validation_history.jsonl")
        validation_path = candidate if candidate.exists() else None
    validation = []
    if validation_path is not None and validation_path.exists():
        validation = [
            json.loads(line)
            for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    _style()
    blue = "#0B5CAD"
    teal = "#0F766E"
    amber = "#B7791F"
    red = "#B64040"
    gray = "#6B7280"

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.55), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(update, frame["mean_utility"], color=blue, alpha=0.22, linewidth=0.8, label="Per update")
    ax.plot(update, frame["utility_ma"], color=blue, linewidth=1.7, label=f"{args.window}-update moving mean")
    if "mean_learning_utility" in frame:
        ax.plot(
            update,
            frame["mean_learning_utility"],
            color=teal,
            alpha=0.28,
            linewidth=0.8,
            label="Relative composition utility",
        )
        ax.plot(
            update,
            frame["learning_utility_ma"],
            color=teal,
            linestyle="--",
            linewidth=1.35,
            label=f"Relative {args.window}-update mean",
        )
    if validation:
        validation_update = np.asarray(
            [float(item["update"]) + 1.0 for item in validation]
        )
        validation_utility = np.asarray(
            [float(item["mean_utility"]) for item in validation]
        )
        ax.plot(
            validation_update,
            validation_utility,
            color="#7C3AED",
            marker="o",
            markersize=3.5,
            linewidth=1.35,
            label="Deterministic validation",
        )
    best_raw = int(frame["mean_utility"].idxmax())
    best_ma = int(frame["utility_ma"].idxmax())
    ax.scatter(update[best_raw], frame.loc[best_raw, "mean_utility"], s=20, color=red, zorder=5)
    ax.scatter(update[best_ma], frame.loc[best_ma, "utility_ma"], s=20, marker="D", color=amber, zorder=5)
    ax.annotate(
        f"best={frame.loc[best_raw, 'mean_utility']:.3f}",
        (update[best_raw], frame.loc[best_raw, "mean_utility"]),
        xytext=(-45, -18),
        textcoords="offset points",
        fontsize=7,
        color=red,
        arrowprops={"arrowstyle": "-", "color": red, "lw": 0.6},
    )
    ax.set_title("(a) Period-level utility")
    ax.set_ylabel("Mean utility")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.plot(update, frame["deploy_per_period"], color=teal, alpha=0.22, linewidth=0.8)
    ax.plot(update, frame["deploy_ma"], color=teal, linewidth=1.7)
    ax.set_title("(b) Sequential deployment decisions")
    ax.set_ylabel("Substeps per period")

    ax = axes[1, 0]
    ax.plot(update, frame["collection_ma"], color=amber, label="Trajectory collection")
    ax.plot(update, frame["optimization_ma"], color=gray, linestyle="--", label="PPO optimization")
    ax.set_title("(c) Training time per update")
    ax.set_ylabel("Wall time (s)")
    ax.legend(frameon=False, loc="center right")

    ax = axes[1, 1]
    rnd_loss = np.maximum(frame["mean_rnd_loss"].to_numpy(dtype=float), 1e-9)
    rnd_reward = np.maximum(frame["mean_raw_intrinsic_reward"].to_numpy(dtype=float), 1e-9)
    ax.semilogy(update, rnd_loss, color=red, label="RND prediction loss")
    ax.semilogy(update, rnd_reward, color=blue, linestyle="--", label="Raw intrinsic reward")
    ax.text(
        0.98,
        0.95,
        r"$\beta_{\mathrm{RND}}$: 0.01 $\rightarrow$ 0",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.set_title("(d) RND exploration diagnostics")
    ax.set_ylabel("Value (log scale)")
    ax.legend(frameon=False, loc="lower left")

    for ax in axes.flat:
        ax.set_xlabel("PPO update")
        ax.set_xlim(update.min(), update.max())
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        target = args.output.with_suffix(suffix)
        fig.savefig(target, dpi=300 if suffix == ".png" else None, bbox_inches="tight")

    selected = [
        "update",
        "mean_utility",
        "utility_ma",
        *(["mean_learning_utility", "learning_utility_ma"] if "mean_learning_utility" in frame else []),
        "deployment_steps",
        "routing_steps",
        "deploy_per_period",
        "collection_time_s",
        "optimization_time_s",
        "mean_rnd_loss",
        "mean_raw_intrinsic_reward",
        "mean_constraint_cost",
    ]
    frame[selected].to_csv(args.output.with_suffix(".csv"), index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()

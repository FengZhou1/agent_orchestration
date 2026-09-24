"""Plot the two from-scratch v4 model-selection runs and held-out results.

Only completed slot_joint_v4 runs are accepted. Evaluation comparisons use
the same trajectory digest, deployment, and 128 mapping samples per trace.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED0 = ROOT / "results/slot_joint_v4_route_shared_dual_s0_2026_09_24"
DEFAULT_SEED1 = ROOT / "results/slot_joint_v4_route_shared_s1_2026_09_24"
DEFAULT_OUTPUT = ROOT / "results/slot_joint_v4_route_visuals_2026_09_24"
TRACE_SEEDS = range(102, 107)
COLORS = {"seed 0": "#147d83", "seed 1": "#d87926", "best heuristic": "#667085"}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def snapshot_path(run: Path, trace_seed: int) -> Path:
    if trace_seed == 102:
        plain = run / "snapshot_update_120_samples_128.json"
        if plain.exists():
            return plain
    return run / f"snapshot_update_120_samples_128_seed_{trace_seed}.json"


def load_run(run: Path, expected_seed: int) -> dict:
    manifest = read_json(run / "manifest.json")
    history = read_json(run / "history.json")
    require(manifest["protocol"] == "slot_joint_v4", f"Wrong protocol: {run}")
    require(manifest["ppo_seed"] == expected_seed, f"Wrong PPO seed: {run}")
    require(manifest["training_settings"]["phase_schedule"] == [["composition", 120]],
            f"Not a 120-update model-only run: {run}")
    require(manifest["training_settings"].get("shared_model_head") is True,
            f"Not the shared-head run: {run}")
    require(len(history) == 120 and int(history[-1]["update"]) == 119,
            f"Incomplete history: {run}")

    final = {}
    for trace_seed in TRACE_SEEDS:
        snap = read_json(snapshot_path(run, trace_seed))
        require(snap["protocol"] == "slot_joint_v4", f"Wrong snapshot protocol: {run}")
        require(snap["checkpoint_next_update"] == 120, f"Wrong checkpoint: {run}")
        require(snap["evaluation_seed"] == trace_seed, f"Wrong trace seed: {run}")
        require(snap["mapping_samples"] == 128, f"Wrong mapping budget: {run}")
        # A saved policy snapshot can carry the digest of the evaluation trace
        # on which it was first written, rather than the training trace.
        # The run manifest and saved weights establish training provenance.
        require(len(snap["slots"]) == manifest["slots"], f"Wrong slot count: {run}")
        final[trace_seed] = snap

    checkpoints = []
    expression = re.compile(r"snapshot_update_(\d+)_samples_8\.json$")
    for path in run.glob("snapshot_update_*_samples_8.json"):
        match = expression.match(path.name)
        if not match:
            continue
        snap = read_json(path)
        require(snap["evaluation_seed"] == 102 and snap["mapping_samples"] == 8,
                f"Incomparable checkpoint: {path}")
        checkpoints.append((int(match.group(1)), snap))
    checkpoints.sort(key=lambda row: row[0])
    return {"manifest": manifest, "history": history, "final": final,
            "checkpoints": checkpoints}


def load_baselines(run0: Path, finals: dict[int, dict], scenario_hash: str) -> dict:
    baselines = {}
    for trace_seed in TRACE_SEEDS:
        if trace_seed in (102, 104):
            suffix = "" if trace_seed == 102 else f"_seed_{trace_seed}"
            report = read_json(run0 / f"snapshot_update_060_samples_128{suffix}.json")
            policies = report["routing_baselines_same_placement"]
        else:
            report = read_json(run0 / f"routing_baselines_seed_{trace_seed}_samples_128.json")
            require(report["scenario_hash"] == scenario_hash,
                    f"Different scenario in baseline trace {trace_seed}")
            policies = report["policies"]
        require(report["evaluation_seed"] == trace_seed and report["mapping_samples"] == 128,
                f"Different trace or mapping budget in baseline {trace_seed}")
        require(report["trajectory_digest"] == finals[trace_seed]["trajectory_digest"],
                f"Different trajectory in baseline {trace_seed}")
        require(set(policies) == {"greedy", "equal", "least_load", "random"},
                f"Missing heuristic baseline for trace {trace_seed}")
        baselines[trace_seed] = policies
    return baselines


def moving_average(values: np.ndarray, width: int = 10) -> np.ndarray:
    return np.array([np.mean(values[max(0, index - width + 1):index + 1])
                     for index in range(len(values))])


def style_axes(axis, ylabel: str, xlabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.grid(axis="y", color="#d9dde5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def training_figure(runs: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.4), layout="constrained")
    for label, result in runs.items():
        color = COLORS[label]
        values = np.array([row["episode_cumulative_utility"] / row["episode_length"]
                           for row in result["history"]])
        updates = np.arange(1, len(values) + 1)
        axes[0].plot(updates, values, color=color, alpha=0.15, linewidth=0.8)
        axes[0].plot(updates, moving_average(values), color=color, linewidth=2.1,
                     label=label)

        checkpoints = result["checkpoints"]
        x = [update for update, _ in checkpoints]
        utility = [snap["mean_utility"] for _, snap in checkpoints]
        worst = [min(slot["slo_attainment"] for slot in snap["slots"])
                 for _, snap in checkpoints]
        axes[1].plot(x, utility, "o-", color=color, linewidth=2, markersize=4,
                     label=label)
        axes[2].plot(x, np.array(worst) * 100, "o-", color=color, linewidth=2,
                     markersize=4, label=label)

    axes[0].axhline(0, color="#667085", linewidth=0.8)
    axes[1].axhline(0, color="#667085", linewidth=0.8)
    axes[2].axhline(90, color="#aa3d3d", linestyle="--", linewidth=1.2,
                    label="90% SLO target")
    axes[0].set_title("Training trajectory · 10-update moving mean")
    axes[1].set_title("Held-out trace 102 · deterministic policy")
    axes[2].set_title("Held-out trace 102 · weakest slot")
    style_axes(axes[0], "Utility per physical slot", "PPO update")
    style_axes(axes[1], "Mean utility per slot", "PPO update")
    style_axes(axes[2], "Minimum slot SLO attainment (%)", "PPO update")
    for axis in axes:
        axis.set_xlim(0, 122)
    axes[0].legend(frameon=False, loc="lower right")
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Model-selection PPO learning (random initialization; no distillation)",
                 fontsize=14, fontweight="bold")
    fig.savefig(output / "01_training_and_checkpoints.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def final_figure(runs: dict[str, dict], baselines: dict, output: Path) -> None:
    trace_seeds = list(TRACE_SEEDS)
    x = np.arange(len(trace_seeds))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    for label, result in runs.items():
        snaps = result["final"]
        utility = [snaps[seed]["mean_utility"] for seed in trace_seeds]
        attainment = [snaps[seed]["mean_slo_attainment"] * 100 for seed in trace_seeds]
        axes[0].plot(x, utility, "o-", linewidth=2.2, markersize=6,
                     color=COLORS[label], label=label)
        axes[1].plot(x, attainment, "o-", linewidth=2.2, markersize=6,
                     color=COLORS[label], label=label)

    best_utility = [max(policy["mean_utility"] for policy in baselines[seed].values())
                    for seed in trace_seeds]
    best_attainment = [max(np.mean([row["slo_attainment"] for row in policy["slots"]])
                           for policy in baselines[seed].values()) * 100
                       for seed in trace_seeds]
    axes[0].plot(x, best_utility, "s--", linewidth=1.6, markersize=5,
                 color=COLORS["best heuristic"], label="best of 4 heuristics")
    axes[1].plot(x, best_attainment, "s--", linewidth=1.6, markersize=5,
                 color=COLORS["best heuristic"], label="best of 4 heuristics")
    axes[0].axhline(0, color="#667085", linewidth=0.8)
    axes[1].axhline(90, color="#aa3d3d", linestyle=":", linewidth=1.3,
                    label="90% target (not a per-slot test)")
    axes[0].set_title("Held-out utility")
    axes[1].set_title("Held-out mean SLO attainment")
    style_axes(axes[0], "Mean utility per slot", "Unseen trajectory seed")
    style_axes(axes[1], "Mean SLO attainment (%)", "Unseen trajectory seed")
    for axis in axes:
        axis.set_xticks(x, trace_seeds)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Final update 120 · five unseen trajectories · 128 mapping samples",
                 fontsize=14, fontweight="bold")
    fig.savefig(output / "02_final_heldout_comparison.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def slot_figure(runs: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), layout="constrained")
    image = None
    for axis, (label, result) in zip(axes, runs.items()):
        values = np.array([[slot["slo_attainment"] * 100
                            for slot in result["final"][seed]["slots"]]
                           for seed in TRACE_SEEDS])
        image = axis.imshow(values, vmin=88, vmax=98, cmap="RdYlGn", aspect="auto")
        axis.set_title(f"{label}: {int(np.sum(values < 90 - 1e-8))}/40 slots below 90%")
        axis.set_xlabel("Physical slot")
        axis.set_ylabel("Unseen trajectory seed")
        axis.set_xticks(range(values.shape[1]), range(values.shape[1]))
        axis.set_yticks(range(values.shape[0]), list(TRACE_SEEDS))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                axis.text(column, row, f"{value:.1f}", ha="center", va="center",
                          color="#161a23", fontsize=8.2,
                          fontweight="bold" if value < 90 - 1e-8 else "normal")
                if value < 90 - 1e-8:
                    axis.add_patch(plt.Rectangle((column - 0.48, row - 0.48), 0.96,
                                                 0.96, fill=False, edgecolor="#9b1c1c",
                                                 linewidth=2.2))
    fig.colorbar(image, ax=axes, shrink=0.82, label="Slot SLO attainment (%)")
    fig.suptitle("Per-slot SLO check · red frames mark target violations",
                 fontsize=14, fontweight="bold")
    fig.savefig(output / "03_per_slot_slo_heatmap.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0", type=Path, default=DEFAULT_SEED0)
    parser.add_argument("--seed1", type=Path, default=DEFAULT_SEED1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = {"seed 0": load_run(args.seed0, 0), "seed 1": load_run(args.seed1, 1)}
    first, second = (runs["seed 0"]["manifest"], runs["seed 1"]["manifest"])
    require(first["scenario_hash"] == second["scenario_hash"], "Scenarios differ")
    require(first["trajectory_digest"] == second["trajectory_digest"],
            "Training trajectories differ")
    require(first["training_settings"] == second["training_settings"],
            "Training settings differ")
    for seed in TRACE_SEEDS:
        require(runs["seed 0"]["final"][seed]["trajectory_digest"] ==
                runs["seed 1"]["final"][seed]["trajectory_digest"],
                f"Held-out trajectories differ for seed {seed}")
    baselines = load_baselines(args.seed0, runs["seed 0"]["final"],
                               first["scenario_hash"])
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    training_figure(runs, args.output)
    final_figure(runs, baselines, args.output)
    slot_figure(runs, args.output)
    for label, result in runs.items():
        final = result["final"]
        utility = np.mean([final[seed]["mean_utility"] for seed in TRACE_SEEDS])
        attainment = np.mean([final[seed]["mean_slo_attainment"] for seed in TRACE_SEEDS])
        violations = sum(slot["slo_attainment"] < 0.9 - 1e-8
                         for snap in final.values() for slot in snap["slots"])
        print(f"{label}: utility={utility:.6f}, attainment={attainment:.6%}, "
              f"violations={violations}/40")
    print(f"Figures: {args.output.resolve()}")


if __name__ == "__main__":
    main()

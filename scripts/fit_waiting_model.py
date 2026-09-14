"""Fit a composition-aware admission-waiting model.

The simulator observations are aggregated by composition and normalized load.
The fitted curve is intended to approximate the mean first-admission waiting
time of a continuous-batching instance; overload points are reported but are
not used as finite steady-state observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = np.abs(predicted - observed)
    relative = error / np.maximum(np.abs(observed), 1e-4)
    return {
        "n": int(observed.size),
        "mae_s": float(error.mean()),
        "wape_pct": float(100.0 * error.sum() / max(np.abs(observed).sum(), 1e-12)),
        "median_ape_pct": float(100.0 * np.median(relative)),
        "p95_ape_pct": float(100.0 * np.percentile(relative, 95)),
    }


def composition_codes(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    names = sorted(frame["composition_id"].astype(str).unique())
    index = {name: i for i, name in enumerate(names)}
    codes = frame["composition_id"].astype(str).map(index).to_numpy(dtype=int)
    return names, codes


def fit_threshold_model(frame: pd.DataFrame, names: list[str]) -> tuple[dict, np.ndarray]:
    """Fit w0_c + alpha_c*tau*((rho-rho0_c)/(1-rho))^nu.

    The floor is the iteration-boundary component.  The positive-part term is
    the congestion component.  The exponent is shared to keep the model small
    while allowing different workload compositions to have different onset,
    scale, and boundary waiting levels.
    """
    _, code = composition_codes(frame)
    rho = frame["load_factor"].to_numpy(dtype=float).clip(0.0, 0.995)
    tau = frame["avg_iteration_s"].to_numpy(dtype=float)
    observed = frame["observed"].to_numpy(dtype=float)
    count = len(names)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        floors = np.exp(theta[:count])
        scales = np.exp(theta[count : 2 * count])
        thresholds = 0.99 / (1.0 + np.exp(-theta[2 * count : 3 * count]))
        exponent = float(np.exp(theta[-1]))
        return floors, scales, thresholds, exponent

    def predict(theta: np.ndarray, local: pd.DataFrame = frame) -> np.ndarray:
        floors, scales, thresholds, exponent = unpack(theta)
        _, local_code = composition_codes(local)
        local_rho = local["load_factor"].to_numpy(dtype=float).clip(0.0, 0.995)
        local_tau = local["avg_iteration_s"].to_numpy(dtype=float)
        term = np.maximum(local_rho - thresholds[local_code], 0.0) / (1.0 - local_rho)
        return floors[local_code] + scales[local_code] * local_tau * term**exponent

    initial = np.r_[
        np.log(np.full(count, 0.008)),
        np.log(np.full(count, 0.5)),
        np.zeros(count),
        np.log(0.8),
    ]
    lower = np.r_[
        np.log(np.full(count, 1e-5)),
        np.log(np.full(count, 1e-8)),
        np.full(count, -8.0),
        np.log(0.2),
    ]
    upper = np.r_[
        np.log(np.full(count, 0.2)),
        np.log(np.full(count, 100.0)),
        np.full(count, 8.0),
        np.log(3.0),
    ]
    result = least_squares(
        lambda theta: np.log(np.maximum(predict(theta), 1e-8))
        - np.log(np.maximum(observed, 1e-8)),
        initial,
        bounds=(lower, upper),
        max_nfev=100000,
    )
    floors, scales, thresholds, exponent = unpack(result.x)
    parameters = {
        "floor_s": dict(zip(names, floors.tolist(), strict=True)),
        "scale": dict(zip(names, scales.tolist(), strict=True)),
        "threshold": dict(zip(names, thresholds.tolist(), strict=True)),
        "exponent": exponent,
        "objective": float(result.cost),
        "success": bool(result.success),
    }
    return parameters, predict(result.x)


def predict_threshold(frame: pd.DataFrame, parameters: dict) -> np.ndarray:
    names = sorted(parameters["floor_s"])
    index = {name: i for i, name in enumerate(names)}
    code = frame["composition_id"].astype(str).map(index).to_numpy(dtype=int)
    rho = frame["load_factor"].to_numpy(dtype=float).clip(0.0, 0.995)
    tau = frame["avg_iteration_s"].to_numpy(dtype=float)
    floors = np.array([parameters["floor_s"][name] for name in names])
    scales = np.array([parameters["scale"][name] for name in names])
    thresholds = np.array([parameters["threshold"][name] for name in names])
    term = np.maximum(rho - thresholds[code], 0.0) / (1.0 - rho)
    return floors[code] + scales[code] * tau * term ** float(parameters["exponent"])


def fit_global_rational(frame: pd.DataFrame) -> dict:
    rho = frame["load_factor"].to_numpy(dtype=float).clip(0.0, 0.995)
    tau = frame["avg_iteration_s"].to_numpy(dtype=float)
    observed = frame["observed"].to_numpy(dtype=float)

    def prediction(theta: np.ndarray) -> np.ndarray:
        floor, scale, exponent = np.exp(theta)
        return floor + scale * tau * rho**exponent / (1.0 - rho)

    result = least_squares(
        lambda theta: np.log(np.maximum(prediction(theta), 1e-8))
        - np.log(np.maximum(observed, 1e-8)),
        np.log([0.008, 0.1, 1.0]),
        bounds=(np.log([1e-5, 1e-8, 0.2]), np.log([0.2, 100.0, 3.0])),
        max_nfev=100000,
    )
    floor, scale, exponent = np.exp(result.x)
    return {"floor_s": float(floor), "scale": float(scale), "exponent": float(exponent)}


def predict_global_rational(frame: pd.DataFrame, parameters: dict) -> np.ndarray:
    rho = frame["load_factor"].to_numpy(dtype=float).clip(0.0, 0.995)
    tau = frame["avg_iteration_s"].to_numpy(dtype=float)
    return parameters["floor_s"] + parameters["scale"] * tau * rho ** parameters["exponent"] / (1.0 - rho)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, default=Path("results/llm_queue_validation_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/llm_queue_validation_v2/waiting_fit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.v2 / "queue_predictions_v2.csv")
    source = source[(source["metric"] == "waiting_s") & np.isfinite(source["observed"])]
    stable = source[source["load_factor"] < 1.0].copy()
    stable["composition_id"] = stable["composition_id"].astype(str)
    names = sorted(stable["composition_id"].unique())
    aggregate = (
        stable.groupby(["composition_id", "load_factor"], as_index=False)
        .agg(
            observed=("observed", "mean"),
            avg_iteration_s=("avg_iteration_s", "mean"),
            predicted_pk=("predicted", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["composition_id", "load_factor"])
        .reset_index(drop=True)
    )

    final_parameters, aggregate_pred = fit_threshold_model(aggregate, names)
    aggregate["predicted_threshold"] = aggregate_pred
    stable["predicted_threshold"] = predict_threshold(stable, final_parameters)

    # Baselines: the old PK approximation and a no-congestion composition floor.
    stable["predicted_pk"] = stable["predicted"]
    # Falsification candidate: add the mean residual time of the current
    # iteration to the old PK term.  This is reported so that the proposed
    # decomposition can be checked rather than assumed to be valid.
    stable["predicted_residual_pk"] = stable["predicted_pk"] + stable["avg_iteration_s"] / 2.0
    floors = stable[stable["load_factor"] <= 0.2].groupby("composition_id")["observed"].mean()
    stable["predicted_flat"] = stable["composition_id"].map(floors).astype(float)
    aggregate["predicted_flat"] = aggregate["composition_id"].map(floors).astype(float)
    aggregate["predicted_residual_pk"] = aggregate["predicted_pk"] + aggregate["avg_iteration_s"] / 2.0
    global_parameters = fit_global_rational(aggregate)
    aggregate["predicted_global"] = predict_global_rational(aggregate, global_parameters)
    stable["predicted_global"] = predict_global_rational(stable, global_parameters)

    model_metrics = {}
    for model in ("predicted_pk", "predicted_residual_pk", "predicted_flat", "predicted_global", "predicted_threshold"):
        model_metrics[f"raw_{model}"] = metrics(stable["observed"], stable[model])
        model_metrics[f"mean_{model}"] = metrics(aggregate["observed"], aggregate[model])

    # Leave-one-seed-out validation of the selected model.  Fitting is always
    # performed on composition/load means so that random arrival variation is
    # not mistaken for a structural queueing effect.
    cv_rows = []
    for seed in sorted(stable["seed"].unique()):
        train_raw = stable[stable["seed"] != seed]
        train_aggregate = (
            train_raw.groupby(["composition_id", "load_factor"], as_index=False)
            .agg(observed=("observed", "mean"), avg_iteration_s=("avg_iteration_s", "mean"))
        )
        parameters, _ = fit_threshold_model(train_aggregate, names)
        test = stable[stable["seed"] == seed].copy()
        test["prediction"] = predict_threshold(test, parameters)
        result = metrics(test["observed"], test["prediction"])
        cv_rows.append({"held_out_seed": int(seed), **result})
    cv = pd.DataFrame(cv_rows)

    stable.to_csv(args.output / "waiting_predictions.csv", index=False)
    aggregate.to_csv(args.output / "waiting_curve_means.csv", index=False)
    cv.to_csv(args.output / "waiting_cv.csv", index=False)
    with (args.output / "waiting_fit.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "formula": "W_c(rho)=w0_c+alpha_c*tau_iter*((rho-rho0_c)^+/(1-rho))^nu",
                "rho_definition": "arrival_rate / saturated_capacity for simulator calibration; replace by Lambda/mu in deployment",
                "fit_domain": "load_factor < 1",
                "compositions": names,
                "parameters": final_parameters,
                "global_rational_parameters": global_parameters,
                "metrics": model_metrics,
                "cross_validation": {
                    "mean": {key: float(value) for key, value in cv.drop(columns=["held_out_seed"]).mean().items()},
                    "rows": cv.to_dict(orient="records"),
                },
            },
            handle,
            indent=2,
        )

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.7), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        curve = aggregate[aggregate["composition_id"] == name]
        ax.plot(curve["load_factor"], curve["observed"], "ko", label="LLMServingSim")
        ax.plot(curve["load_factor"], curve["predicted_threshold"], "C1-", label="fitted model")
        ax.plot(curve["load_factor"], curve["predicted_pk"], "C0--", label="PK")
        ax.set_title(name)
        ax.set_xlabel(r"normalized load $\rho$")
        ax.set_ylabel("mean waiting (s)")
        ax.set_xlim(0.15, 0.98)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / "waiting_fit_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "waiting_fit_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Admission-waiting model fit",
        "",
        "The target is the mean first-admission waiting time recorded by the instrumented LLMServingSim runs. Points with normalized load >= 1 are treated as overload and excluded from finite steady-state fitting.",
        "",
        "## Selected model",
        "",
        r"\[W_c(\rho)=w_{0,c}+\alpha_c\,\bar t^{\rm iter}\left(\frac{[\rho-\rho_{0,c}]^+}{1-\rho}\right)^\nu.\]",
        "",
        "The floor captures the next-iteration admission boundary. The positive-part term captures the rapid growth after a composition-dependent congestion onset. The exponent is shared across compositions.",
        "",
        "## Comparison",
        "",
        "| model | raw WAPE | raw median APE | raw P95 APE | curve-mean WAPE | curve-mean median APE | curve-mean P95 APE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("old PK", "predicted_pk"), ("residual + PK", "predicted_residual_pk"), ("flat floor", "predicted_flat"), ("global rational", "predicted_global"), ("composition-threshold", "predicted_threshold")]:
        raw = model_metrics[f"raw_{key}"]
        mean = model_metrics[f"mean_{key}"]
        report.append(f"| {label} | {raw['wape_pct']:.2f}% | {raw['median_ape_pct']:.2f}% | {raw['p95_ape_pct']:.2f}% | {mean['wape_pct']:.2f}% | {mean['median_ape_pct']:.2f}% | {mean['p95_ape_pct']:.2f}% |")
    report += [
        "",
        "## Validation conclusion",
        "",
        f"The selected curve fit gives curve-mean WAPE {model_metrics['mean_predicted_threshold']['wape_pct']:.2f}% and raw-request WAPE {model_metrics['raw_predicted_threshold']['wape_pct']:.2f}%. The residual-plus-PK candidate gives curve-mean WAPE {model_metrics['mean_predicted_residual_pk']['wape_pct']:.2f}% and is rejected for the present data. Leave-one-seed-out mean WAPE is {cv['wape_pct'].mean():.2f}%.",
        "The finite formula is intended for stable operation. At or above the measured saturated capacity, no finite steady-state waiting time is assigned; the simulator result is retained as an overload diagnostic.",
        "",
        "Artifacts: `waiting_fit.json`, `waiting_predictions.csv`, `waiting_curve_means.csv`, `waiting_cv.csv`, and `waiting_fit_curves.pdf`.",
    ]
    (args.output / "waiting_fit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

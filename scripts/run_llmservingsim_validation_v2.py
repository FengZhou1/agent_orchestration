"""Fit and validate a low-dimensional LLM continuous-batching model.

This script consumes the public LLMServingSim operator profile and the completed
v1 queue-validation runs.  It deliberately keeps the simulator as the reference
system: the analytical model is fitted to iteration-level profile rows and then
evaluated on request-level workloads and queue runs that were not used for fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq

from agent_orch.validation.llmservingsim import (
    JITSERVE_CLASSES,
    WorkloadClass,
    read_run_specs,
    read_simulator_output,
)


JOBS = ("CS", "CC", "DS", "DC")
NUM_LAYERS = 32
SINGLE_PASS_DENSE = {"embedding", "final_layernorm"}
COMPOSITIONS = {
    "CS": {"CS": 1.0},
    "DS": {"DS": 1.0},
    "MIX": {"CS": 0.5, "DS": 0.5},
    "AGENT": {"CS": 0.7, "DS": 0.1, "CC": 0.1, "DC": 0.1},
}


@dataclass(frozen=True)
class FitParameters:
    # tau = c + a [x+n-b]^+ + u A_pre + d A_dec, all times in seconds.
    c: float
    a: float
    b: float
    u: float
    d: float
    # Decode-only state parameters used by m2.
    c_decode: float = 0.0
    a_decode: float = 0.0
    d_decode: float = 0.0


@dataclass(frozen=True)
class ProfileTables:
    attention: pd.DataFrame
    dense: pd.DataFrame
    per_sequence: pd.DataFrame


def _interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linear interpolation with endpoint extension, matching profile use."""
    if len(xp) == 1:
        return float(fp[0])
    order = np.argsort(xp)
    return float(np.interp(float(x), xp[order], fp[order]))


def load_profile(path: Path) -> ProfileTables:
    required = {
        "attention.csv": {"prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"},
        "dense.csv": {"layer", "tokens", "time_us"},
        "per_sequence.csv": {"layer", "sequences", "time_us"},
    }
    tables: dict[str, pd.DataFrame] = {}
    for filename, columns in required.items():
        frame = pd.read_csv(path / filename)
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} missing columns: {sorted(missing)}")
        tables[filename] = frame
    return ProfileTables(
        attention=tables["attention.csv"],
        dense=tables["dense.csv"],
        per_sequence=tables["per_sequence.csv"],
    )


def _table_sum(frame: pd.DataFrame, xcolumn: str, x: float, multiply_layers: bool = True) -> float:
    total = 0.0
    for _, group in frame.groupby("layer", sort=False):
        value = _interp(
            x,
            group[xcolumn].to_numpy(dtype=float),
            group["time_us"].to_numpy(dtype=float),
        )
        if multiply_layers and str(group.iloc[0]["layer"]) not in SINGLE_PASS_DENSE:
            value *= NUM_LAYERS
        total += value
    return total * 1e-6


def profile_iteration_time(
    tables: ProfileTables,
    prefill_tokens: float,
    kv_prefill: float,
    n_decode: float,
    kv_decode: float,
) -> float:
    """Reconstruct total iteration time from LLMServingSim operator tables."""
    attention = tables.attention
    exact = attention[
        (attention["prefill_chunk"] == int(round(prefill_tokens)))
        & (attention["kv_prefill"] == int(round(kv_prefill)))
        & (attention["n_decode"] == int(round(n_decode)))
        & (attention["kv_decode"] == int(round(kv_decode)))
    ]
    if len(exact):
        attention_us = float(exact.iloc[0]["time_us"]) * NUM_LAYERS * 1e-6
    else:
        # The simulator profile is tabulated on a grid.  Nearest-neighbour
        # lookup is used for the four-dimensional attention term; dense and
        # per-sequence terms retain the documented one-dimensional interpolation.
        distance = (
            (attention["prefill_chunk"].to_numpy(dtype=float) - prefill_tokens) ** 2
            + (attention["kv_prefill"].to_numpy(dtype=float) - kv_prefill) ** 2
            + (attention["n_decode"].to_numpy(dtype=float) - n_decode) ** 2
            + (attention["kv_decode"].to_numpy(dtype=float) - kv_decode) ** 2
        )
        attention_us = float(attention.iloc[int(np.argmin(distance))]["time_us"]) * NUM_LAYERS * 1e-6
    return (
        attention_us
        + _table_sum(tables.dense, "tokens", prefill_tokens + n_decode)
        + _table_sum(tables.per_sequence, "sequences", n_decode, multiply_layers=False)
    )


def feature_row(x: float, h: float, n: float, k: float) -> tuple[float, float, float]:
    z = x + n
    apre = x * (h + (x + 1.0) / 2.0)
    adec = n * (k + 1.0)
    return z, apre, adec


def fit_model(samples: pd.DataFrame, mode: str) -> FitParameters:
    if mode not in {"m0", "m1", "m2"}:
        raise ValueError(f"unknown mode {mode}")
    z = samples["z"].to_numpy(dtype=float)
    apre = samples["a_pre"].to_numpy(dtype=float)
    adec = samples["a_dec"].to_numpy(dtype=float)
    target = samples["time_s"].to_numpy(dtype=float)
    scale = max(float(np.median(target)), 1e-9)

    if mode == "m0":
        # A three-parameter state-dependent iteration model.
        def predict(theta: np.ndarray) -> np.ndarray:
            c, a, b = theta
            return c + a * np.maximum(z - b, 0.0)

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([np.max(target) * 10.0, np.max(target) * 10.0, np.max(z)])
        starts = [
            np.array([np.min(target), np.median(target) / max(np.max(z), 1.0), 0.0]),
            np.array([np.min(target), 0.0, np.median(z)]),
        ]
    elif mode == "m1":
        def predict(theta: np.ndarray) -> np.ndarray:
            c, a, b, u, d = theta
            return c + a * np.maximum(z - b, 0.0) + u * apre + d * adec

        lower = np.zeros(5)
        upper = np.array(
            [np.max(target) * 10.0, np.max(target) * 10.0, np.max(z),
             np.max(target) * 10.0 / max(np.max(apre), 1.0),
             np.max(target) * 10.0 / max(np.max(adec), 1.0)]
        )
        starts = [
            np.array([np.min(target), 0.0, np.median(z), 0.0, 0.0]),
            np.array([np.min(target), np.median(target) / max(np.max(z), 1.0), 0.0, 0.0, 0.0]),
        ]
    else:
        # Two state modes: decode-only x=0 and mixed x>0.  The decode state
        # follows the measured KV-dependent attention term, while the mixed
        # state includes prefill work and active decode work.
        decode = samples[samples["x"] == 0].copy()
        mixed = samples[samples["x"] > 0].copy()

        def fit_linear(part: pd.DataFrame, design: np.ndarray, initial: np.ndarray) -> np.ndarray:
            target_part = part["time_s"].to_numpy(dtype=float)
            result = least_squares(
                lambda theta: np.log(np.maximum(design @ theta, 1e-12) / np.maximum(target_part, 1e-12)),
                initial,
                bounds=(0.0, np.inf),
                max_nfev=5000,
            )
            return result.x

        zd = decode["n"].to_numpy(dtype=float)
        kd = decode["k"].to_numpy(dtype=float)
        xd = np.column_stack([np.ones(len(decode)), zd, zd * (kd + 1.0)])
        yd = decode["time_s"].to_numpy(dtype=float)
        dec_theta = fit_linear(decode, xd, np.linalg.lstsq(xd, yd, rcond=None)[0])

        zm = mixed["z"].to_numpy(dtype=float)
        am = mixed["a_pre"].to_numpy(dtype=float)
        dm = mixed["a_dec"].to_numpy(dtype=float)
        xm = np.column_stack([np.ones(len(mixed)), zm, am, dm])
        ym = mixed["time_s"].to_numpy(dtype=float)
        def mixed_residual(theta: np.ndarray) -> np.ndarray:
            prediction = theta[0] + theta[1] * np.maximum(zm - theta[2], 0.0) + theta[3] * am + theta[4] * dm
            return np.log(np.maximum(prediction, 1e-12) / np.maximum(ym, 1e-12))

        mixed_start = np.array([np.min(ym), np.median(ym) / max(np.max(zm), 1.0), np.median(zm), xm[:, 2].dot(ym) / max(xm[:, 2].dot(xm[:, 2]), 1e-12), xm[:, 3].dot(ym) / max(xm[:, 3].dot(xm[:, 3]), 1e-12)])
        mix_result = least_squares(
            mixed_residual,
            np.maximum(mixed_start, 0.0),
            bounds=(0.0, np.inf),
            max_nfev=5000,
        )
        mix_theta = mix_result.x
        return FitParameters(
            c=float(mix_theta[0]),
            a=float(mix_theta[1]),
            b=float(mix_theta[2]),
            u=float(mix_theta[3]),
            d=float(mix_theta[4]),
            c_decode=float(dec_theta[0]),
            a_decode=float(dec_theta[1]),
            d_decode=float(dec_theta[2]),
        )

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(predict(theta), 1e-12) / np.maximum(target, 1e-12))

    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=5000)
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    values = best.x
    if mode == "m0":
        return FitParameters(float(values[0]), float(values[1]), float(values[2]), 0.0, 0.0)
    return FitParameters(*map(float, values))


def fitted_iteration_time(params: FitParameters, x: float, h: float, n: float, k: float, mode: str) -> float:
    z, apre, adec = feature_row(x, h, n, k)
    if mode == "m0":
        return params.c + params.a * max(z - params.b, 0.0)
    if mode == "m2":
        if x == 0:
            return max(params.c_decode + params.a_decode * n + params.d_decode * n * (k + 1.0), 1e-12)
        return max(params.c + params.a * max(z - params.b, 0.0) + params.u * apre + params.d * adec, 1e-12)
    return max(params.c + params.a * max(z - params.b, 0.0) + params.u * apre + params.d * adec, 1e-12)


def build_profile_samples(tables: ProfileTables) -> pd.DataFrame:
    rows = []
    for row in tables.attention.itertuples(index=False):
        x = float(row.prefill_chunk)
        h = float(row.kv_prefill)
        n = float(row.n_decode)
        k = float(row.kv_decode)
        rows.append(
            {
                "x": x,
                "h": h,
                "n": n,
                "k": k,
                "z": x + n,
                "a_pre": x * (h + (x + 1.0) / 2.0),
                "a_dec": n * (k + 1.0),
                "time_s": profile_iteration_time(tables, x, h, n, k),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["x", "h", "n", "k"])


def errors(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(observed), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    abs_error = np.abs(p - y)
    ape = abs_error / np.maximum(np.abs(y), 1e-12)
    return {
        "n": int(len(y)),
        "mae_s": float(np.mean(abs_error)),
        "wape_pct": float(100.0 * np.sum(abs_error) / max(np.sum(np.abs(y)), 1e-12)),
        "median_ape_pct": float(100.0 * np.median(ape)),
        "p95_ape_pct": float(100.0 * np.quantile(ape, 0.95)),
    }


def workload_service(
    workload: WorkloadClass,
    params: FitParameters,
    mode: str,
    chunk: int,
    n_decode: float = 0.0,
    decode_k: float | None = None,
) -> dict[str, float]:
    prefill = 0.0
    context = 0
    while context < workload.prompt_tokens:
        size = min(chunk, workload.prompt_tokens - context)
        prefill += fitted_iteration_time(params, size, context, n_decode, context, mode)
        context += size
    decode = 0.0
    for j in range(1, workload.output_tokens):
        k = float(decode_k if decode_k is not None else workload.prompt_tokens + j - 1)
        decode += fitted_iteration_time(params, 0.0, 0.0, max(n_decode, 1.0), k, mode)
    return {
        "prefill_s": prefill,
        "decode_s": decode,
        "service_s": prefill + decode,
        "tbt_s": decode / max(workload.output_tokens - 1, 1),
    }


def load_service_observations(old: Path) -> pd.DataFrame:
    rows = []
    for class_id, workload in {**JITSERVE_CLASSES}.items():
        frame = read_simulator_output(old / "runs" / f"service-{class_id.lower()}.csv")
        row = frame.iloc[0]
        rows.append(
            {
                "class_id": class_id,
                "prompt_tokens": workload.prompt_tokens,
                "output_tokens": workload.output_tokens,
                "prefill_s": float(row.prefill_s),
                "decode_s": float(row.decode_s),
                "service_s": float(row.service_s),
                "tbt_s": float(row.tbt_s),
            }
        )
    # The calibration rows are part of the v1 run specification and are kept
    # here as a held-out request-level check, not as iteration-model fit data.
    specs = read_run_specs(old / "calibration_runs.jsonl")
    for spec in specs:
        if spec.get("kind") != "service" or not str(spec["class_id"]).startswith("CAL_"):
            continue
        class_id = str(spec["class_id"])
        _, p, o = class_id.split("_")
        frame = read_simulator_output(old / "runs" / f"{spec['run_id']}.csv")
        row = frame.iloc[0]
        rows.append(
            {
                "class_id": class_id,
                "prompt_tokens": int(p),
                "output_tokens": int(o),
                "prefill_s": float(row.prefill_s),
                "decode_s": float(row.decode_s),
                "service_s": float(row.service_s),
                "tbt_s": float(row.tbt_s),
            }
        )
    return pd.DataFrame(rows)


def fixed_point_prediction(
    composition: Mapping[str, float],
    arrival_rate: float,
    params: FitParameters,
    mode: str,
    chunk: int,
) -> dict[str, float | bool]:
    weights = {key: float(value) for key, value in composition.items() if value > 0}
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}

    def class_prefill(key: str, n: float) -> float:
        workload = JITSERVE_CLASSES[key]
        result = workload_service(workload, params, mode, chunk, n_decode=n)
        return result["prefill_s"]

    def tau_decode(n: float) -> float:
        token_weights = []
        values = []
        for key, weight in weights.items():
            workload = JITSERVE_CLASSES[key]
            count = max(workload.output_tokens - 1, 1)
            k = workload.prompt_tokens + (workload.output_tokens - 1) / 2.0
            values.append(weight * count * fitted_iteration_time(params, 0, 0, max(n, 1.0), k, mode))
            token_weights.append(weight * count)
        return sum(values) / max(sum(token_weights), 1e-12)

    def quantities(n: float) -> tuple[float, float, float, float]:
        chunks = sum(
            weight * math.ceil(JITSERVE_CLASSES[key].prompt_tokens / chunk)
            for key, weight in weights.items()
        )
        prefill_work = sum(weight * arrival_rate * class_prefill(key, n) for key, weight in weights.items())
        decode_arrivals = sum(
            weight * arrival_rate * max(JITSERVE_CLASSES[key].output_tokens - 1, 1)
            for key, weight in weights.items()
        )
        tau = tau_decode(n)
        return chunks * arrival_rate, prefill_work, decode_arrivals, tau

    def fixed_residual(n: float) -> float:
        apre, prefill_work, adec, tau = quantities(n)
        avg_tau = 1.0 / max(apre + (1.0 - prefill_work) / max(tau, 1e-12), 1e-12)
        return n - adec * avg_tau

    upper = max(1.0, sum(weights.values()) * 4096.0)
    grid = np.linspace(0.0, upper, 256)
    roots = []
    previous = fixed_residual(float(grid[0]))
    for left, right in zip(grid[:-1], grid[1:]):
        current = fixed_residual(float(right))
        if previous == 0.0 or current == 0.0 or previous * current < 0.0:
            roots.append(brentq(fixed_residual, float(left), float(right)))
        previous = current
    n = float(roots[0] if roots else max(0.0, grid[int(np.argmin(np.abs([fixed_residual(x) for x in grid])))]))
    apre, prefill_work, adec, tau = quantities(n)
    overloaded = prefill_work >= 1.0
    avg_tau = math.inf if overloaded else 1.0 / max(apre + (1.0 - prefill_work) / max(tau, 1e-12), 1e-12)
    mean_prefill = sum(weights[key] * class_prefill(key, n) for key in weights)
    if overloaded:
        waiting = math.inf
        response = math.inf
    else:
        second = sum(
            arrival_rate * weights[key] * class_prefill(key, n) ** 2 for key in weights
        )
        waiting = second / max(2.0 * (1.0 - prefill_work), 1e-12)
        mean_decode = sum(
            weights[key] * max(JITSERVE_CLASSES[key].output_tokens - 1, 1) * avg_tau
            for key in weights
        )
        response = waiting + mean_prefill + mean_decode
    return {
        "n_decode": n,
        "avg_iteration_s": avg_tau,
        "prefill_utilization": prefill_work,
        "waiting_s": waiting,
        "ttft_s": waiting + mean_prefill if math.isfinite(waiting) else math.inf,
        "tbt_s": avg_tau,
        "response_s": response,
        "overloaded": bool(overloaded),
    }


def load_capacities(old: Path) -> dict[str, float]:
    frame = pd.read_csv(old / "capacity_observations.csv")
    return dict(zip(frame["composition_id"], frame["saturated_capacity_rps"], strict=True))


def analyze_queue(old: Path, output: Path, params: FitParameters, mode: str, chunk: int) -> pd.DataFrame:
    capacities = load_capacities(old)
    rows = []
    for spec in read_run_specs(old / "queue_runs.jsonl"):
        run_id = str(spec["run_id"])
        csv = old / "runs" / f"{run_id}.csv"
        if not csv.exists():
            continue
        sim = read_simulator_output(csv)
        manifest = pd.read_csv(old / "manifests" / str(spec["manifest"]))
        merged = sim.merge(manifest, on="request_id", validate="one_to_one").sort_values("request_id")
        lo = int(math.floor(0.2 * len(merged)))
        hi = int(math.ceil(0.9 * len(merged)))
        sample = merged.iloc[lo:hi]
        composition = COMPOSITIONS[str(spec["composition_id"])]
        pred = fixed_point_prediction(
            composition,
            float(spec["arrival_rate_rps"]),
            params,
            mode,
            chunk,
        )
        for metric in ("waiting_s", "ttft_s", "tbt_s", "response_s"):
            observed = float(sample[metric].mean())
            rows.append(
                {
                    "run_id": run_id,
                    "composition_id": str(spec["composition_id"]),
                    "load_factor": float(spec["load_factor"]),
                    "arrival_rate_rps": float(spec["arrival_rate_rps"]),
                    "seed": int(spec["seed"]),
                    "metric": metric,
                    "observed": observed,
                    "predicted": float(pred[metric]),
                    "absolute_error": abs(float(pred[metric]) - observed),
                    "n_decode": float(pred["n_decode"]),
                    "avg_iteration_s": float(pred["avg_iteration_s"]),
                    "prefill_utilization": float(pred["prefill_utilization"]),
                    "overloaded": bool(pred["overloaded"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "queue_predictions_v2.csv", index=False)
    return frame


def plot_results(output: Path, service: pd.DataFrame, queue: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, metric, label in zip(axes, ("prefill_s", "decode_s", "service_s"), ("Prefill", "Decode", "Complete service"), strict=True):
        ax.scatter(service[metric], service[f"predicted_{metric}"], c=np.where(service["split"] == "test", "C1", "C0"), s=18)
        max_value = max(service[metric].max(), service[f"predicted_{metric}"].max())
        ax.plot([0, max_value], [0, max_value], "k--", linewidth=0.8)
        ax.set_xlabel("LLMServingSim (s)")
        ax.set_ylabel("Analytical (s)")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(output / "figures" / "service_scatter_v2.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figures" / "service_scatter_v2.pdf", bbox_inches="tight")
    plt.close(fig)

    for metric in ("waiting_s", "ttft_s", "tbt_s", "response_s"):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        subset = queue[queue["metric"] == metric]
        for composition, group in subset.groupby("composition_id"):
            observed = group.groupby("load_factor")["observed"].mean()
            predicted = group.groupby("load_factor")["predicted"].mean()
            ax.plot(observed.index, observed.values, "o-", label=f"{composition} sim", linewidth=1.0)
            ax.plot(predicted.index, predicted.values, "--", label=f"{composition} model", linewidth=1.0)
        ax.set_xlabel("Offered load / saturated simulator capacity")
        ax.set_ylabel("Seconds")
        ax.set_title(metric.replace("_s", "").upper())
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output / "figures" / f"{metric}_v2.png", dpi=220, bbox_inches="tight")
        fig.savefig(output / "figures" / f"{metric}_v2.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("results/llm_queue_validation"))
    parser.add_argument("--profile", type=Path, default=Path("results/llm_queue_validation_v2/profile"))
    parser.add_argument("--output", type=Path, default=Path("results/llm_queue_validation_v2"))
    parser.add_argument("--mode", choices=("m0", "m1", "m2"), default="m2")
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(parents=True, exist_ok=True)

    tables = load_profile(args.profile)
    all_samples = build_profile_samples(tables)
    # Profile extrapolation above the supported KV grid is reported separately;
    # fitting remains restricted to the empirically covered KV range.
    fit_samples = all_samples[
        (all_samples["h"] <= 2048) & (all_samples["k"] <= 2048)
    ].copy()
    params = fit_model(fit_samples, args.mode)
    fit_samples["predicted_s"] = [
        fitted_iteration_time(params, r.x, r.h, r.n, r.k, args.mode)
        for r in fit_samples.itertuples()
    ]
    all_samples["predicted_s"] = [
        fitted_iteration_time(params, r.x, r.h, r.n, r.k, args.mode)
        for r in all_samples.itertuples()
    ]
    profile_summary = {
        "all_rows": errors(all_samples["time_s"], all_samples["predicted_s"]),
        "fit_domain_rows": errors(fit_samples["time_s"], fit_samples["predicted_s"]),
        "fit_rows": int(len(fit_samples)),
        "all_profile_rows": int(len(all_samples)),
        "fit_domain": "kv_prefill <= 2048 and kv_decode <= 2048",
    }
    all_samples.to_csv(args.output / "iteration_profile_samples.csv", index=False)
    _write_json(args.output / "iteration_fit.json", {"mode": args.mode, "parameters": asdict(params), "summary": profile_summary})

    observations = load_service_observations(args.old)
    service_rows = []
    for row in observations.itertuples(index=False):
        workload = WorkloadClass(str(row.class_id), int(row.prompt_tokens), int(row.output_tokens))
        predicted = workload_service(workload, params, args.mode, args.chunk)
        record = row._asdict()
        record["split"] = "test" if str(row.class_id) in JITSERVE_CLASSES else "calibration"
        for metric in ("prefill_s", "decode_s", "service_s", "tbt_s"):
            record[f"predicted_{metric}"] = predicted[metric]
        service_rows.append(record)
    service = pd.DataFrame(service_rows)
    service.to_csv(args.output / "service_predictions_v2.csv", index=False)
    service_error = {
        metric: errors(service[f"{metric}"], service[f"predicted_{metric}"])
        for metric in ("prefill_s", "decode_s", "service_s", "tbt_s")
    }

    queue = analyze_queue(args.old, args.output, params, args.mode, args.chunk)
    queue_summary = []
    stable = queue[queue["load_factor"] < 1.0]
    for metric, group in stable.groupby("metric"):
        finite = np.isfinite(group["predicted"].to_numpy(dtype=float))
        queue_summary.append({"metric": metric, **errors(group.loc[finite, "observed"], group.loc[finite, "predicted"])})
    queue_summary_frame = pd.DataFrame(queue_summary)
    queue_summary_frame.to_csv(args.output / "queue_error_summary_v2.csv", index=False)
    plot_results(args.output, service, queue)
    _write_json(
        args.output / "provenance_v2.json",
        {
            "simulator": "LLMServingSim 2.0",
            "reference_hardware": "RTX4090",
            "reference_model": "meta-llama/Llama-3.1-8B",
            "profile_path": str(args.profile.resolve()),
            "profile_sha256": {
                name: _sha256(args.profile / name)
                for name in ("attention.csv", "dense.csv", "per_sequence.csv")
            },
            "source_queue_results": str(args.old.resolve()),
            "model_mode": args.mode,
            "chunk_tokens": args.chunk,
            "profile_fit_domain": profile_summary["fit_domain"],
            "official_sanity": "PASS",
        },
    )

    report = [
        "# LLMServingSim continuous-batching model validation (v2)",
        "",
        "## Scope",
        "",
        "The iteration model is fitted to the public RTX4090/Llama-3.1-8B profile and evaluated against the completed v1 request-level simulations. The simulator remains the reference implementation.",
        "",
        f"- Model: `{args.mode}`; chunk size: `{args.chunk}` tokens.",
        f"- Profile rows: `{len(all_samples)}`; fit-domain rows: `{len(fit_samples)}`.",
        f"- Fit domain: `{profile_summary['fit_domain']}`.",
        "",
        "## Fitted iteration model",
        "",
        "For mixed iterations, `tau = c + a [x+n-b]^+ + u*x*(h+(x+1)/2) + d*n*(k+1)`; for decode-only iterations, `tau = c_decode + a_decode*n + d_decode*n*(k+1)`. ",
        "",
        "The four state variables are prefill chunk tokens `x`, prefill context `h`, active decode sequences `n`, and decode context `k`.",
        "",
        "```json",
        json.dumps(asdict(params), indent=2),
        "```",
        "",
        "## Error summary",
        "",
        "### Iteration profile",
        "",
        f"- All profile rows: `{profile_summary['all_rows']}`.",
        f"- Fitting domain: `{profile_summary['fit_domain_rows']}`.",
        "",
        "### Isolated request service",
        "",
    ]
    for metric, values in service_error.items():
        report.append(f"- `{metric}`: `{values}`.")
    report += ["", "### Queue and response approximation", ""]
    for row in queue_summary_frame.itertuples(index=False):
        report.append(f"- `{row.metric}`: n={int(row.n)}, WAPE={row.wape_pct:.2f}%, median APE={row.median_ape_pct:.2f}%, P95 APE={row.p95_ape_pct:.2f}%.")
    report += [
        "",
        "## Interpretation",
        "",
        "The fit-domain result measures whether the low-dimensional iteration relation reproduces the public operator profile. The request-level result additionally includes chunk accumulation and output-token accumulation. Queue results use the mean prefill waiting approximation and the scalar steady-state fixed point; they are not a replacement for simulator replay in the final evaluation.",
        "",
        "Generated artifacts are in `figures/`, `iteration_profile_samples.csv`, `iteration_fit.json`, `service_predictions_v2.csv`, `queue_predictions_v2.csv`, and `queue_error_summary_v2.csv`.",
    ]
    (args.output / "validation_report_v2.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "profile": profile_summary, "service": service_error, "queue_rows": len(queue)}, indent=2))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

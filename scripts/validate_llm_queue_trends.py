"""Validate the paper's LLM service and queueing model against LLMServingSim.

Calibration and evaluation are separated.  Isolated requests fit the two
effective Roofline rates; a saturated mixed batch fits the effective
concurrency; the load/composition/model-GPU matrix is then evaluated with the
fitted parameters frozen.  The goal is trend validation, not exact emulation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_orch.validation.llmservingsim import (  # noqa: E402
    AnalyticalParameters,
    WorkloadClass,
    allen_cunneen_prediction,
    fit_effective_rates,
    generate_poisson_trace,
    read_simulator_output,
    roofline_service,
    saturated_throughput,
)

REMOTE = "zf@192.168.234.128"
REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
REMOTE_STAGE = "outputs/llm_queue_trends"
CONTAINER = "servingsim_docker"
CONTAINER_REPO = "/app/LLMServingSim"

CLASSES: dict[str, WorkloadClass] = {
    "short": WorkloadClass("short", 128, 64),
    "prefill": WorkloadClass("prefill", 2048, 64),
    "decode": WorkloadClass("decode", 128, 512),
    "balanced": WorkloadClass("balanced", 1024, 256),
}

COMPOSITIONS: dict[str, dict[str, float]] = {
    "short": {"short": 1.0},
    "prefill": {"prefill": 1.0},
    "decode": {"decode": 1.0},
    "mixed": {"short": 0.25, "prefill": 0.25, "decode": 0.25, "balanced": 0.25},
}

GPU_SPECS: dict[str, dict[str, float]] = {
    "A10": {"compute_tflops": 125.0, "bandwidth_gbs": 600.0, "memory_gb": 24.0},
    "L20": {"compute_tflops": 119.5, "bandwidth_gbs": 864.0, "memory_gb": 48.0},
    "H20": {"compute_tflops": 148.0, "bandwidth_gbs": 4000.0, "memory_gb": 96.0},
}

MODEL_SPECS: dict[str, dict[str, float]] = {
    "Qwen3-4B": {"parameter_count": 4.0e9, "layers": 36, "hidden_size": 2560, "weight_bytes": 8.0e9, "kv_bytes_per_token": 147456.0},
    "Qwen3-8B": {"parameter_count": 8.2e9, "layers": 36, "hidden_size": 4096, "weight_bytes": 16.4e9, "kv_bytes_per_token": 147456.0},
    "Qwen3-14B": {"parameter_count": 14.8e9, "layers": 40, "hidden_size": 5120, "weight_bytes": 29.6e9, "kv_bytes_per_token": 163840.0},
    "Qwen3-32B": {"parameter_count": 32.8e9, "layers": 64, "hidden_size": 5120, "weight_bytes": 65.6e9, "kv_bytes_per_token": 262144.0},
}

CONFIGS: list[dict[str, Any]] = [
    {"id": "qwen3-4b-a10", "model": "Qwen3-4B", "gpu": "A10", "tp": 1, "role": "main"},
    {"id": "qwen3-4b-l20", "model": "Qwen3-4B", "gpu": "L20", "tp": 1, "role": "gpu-cross"},
    {"id": "qwen3-4b-h20", "model": "Qwen3-4B", "gpu": "H20", "tp": 1, "role": "main"},
    {"id": "qwen3-8b-l20", "model": "Qwen3-8B", "gpu": "L20", "tp": 1, "role": "main"},
    {"id": "qwen3-8b-h20", "model": "Qwen3-8B", "gpu": "H20", "tp": 1, "role": "gpu-cross"},
    {"id": "qwen3-14b-l20", "model": "Qwen3-14B", "gpu": "L20", "tp": 1, "role": "gpu-cross"},
    {"id": "qwen3-14b-h20", "model": "Qwen3-14B", "gpu": "H20", "tp": 1, "role": "main"},
    {"id": "qwen3-32b-l20-tp2", "model": "Qwen3-32B", "gpu": "L20", "tp": 2, "role": "gpu-cross"},
    {"id": "qwen3-32b-h20", "model": "Qwen3-32B", "gpu": "H20", "tp": 1, "role": "main"},
]

TEST_LOAD_FACTORS = (0.40, 0.70, 0.85, 0.95)
COMPOSITION_LOAD_FACTOR = 0.70
CALIBRATION_REQUESTS = 4
SATURATION_REQUESTS = 24
TEST_REQUESTS = 24
SEED_CAL = 20260911
SEED_TEST = 20260912


def run_checked(command: list[str], timeout: int | None = None) -> str:
    completed = subprocess.run(command, check=False, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-4000:]}")
    return completed.stdout


def analytical_parameters(config: dict[str, Any], effective_flops: float | None = None, effective_bandwidth: float | None = None) -> AnalyticalParameters:
    model = MODEL_SPECS[config["model"]]
    gpu = GPU_SPECS[config["gpu"]]
    flops = effective_flops if effective_flops is not None else gpu["compute_tflops"] * 1e12 * config["tp"]
    bandwidth = effective_bandwidth if effective_bandwidth is not None else gpu["bandwidth_gbs"] * 1e9 * config["tp"]
    return AnalyticalParameters(
        parameter_count=float(model["parameter_count"]),
        layers=int(model["layers"]),
        hidden_size=int(model["hidden_size"]),
        weight_bytes=float(model["weight_bytes"]),
        kv_bytes_per_token=float(model["kv_bytes_per_token"]),
        chunk_tokens=512,
        effective_flops=float(flops),
        effective_bandwidth_bytes_s=float(bandwidth),
    )


def cluster_config(config: dict[str, Any]) -> dict[str, Any]:
    gpu = GPU_SPECS[config["gpu"]]
    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 1000,
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
            "instances": [{
                "model_name": f"Qwen/{config['model']}",
                "hardware": config["gpu"],
                "npu_mem": {"mem_size": gpu["memory_gb"], "mem_bw": gpu["bandwidth_gbs"], "mem_latency": 0, "mem_util": 0.9},
                "num_npus": config["tp"],
                "tp_size": config["tp"],
                "pd_type": None,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 128,
                "long_prefill_token_threshold": 512,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "block_size": 16,
            }],
        }],
    }


def job_id(*parts: Any) -> str:
    return "-".join(str(p) for p in parts).replace("_", "-")


def write_jobs(staging: Path, jobs: list[dict[str, Any]]) -> None:
    (staging / "configs").mkdir(parents=True, exist_ok=True)
    (staging / "workloads").mkdir(parents=True, exist_ok=True)
    (staging / "manifests").mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    for item in jobs:
        cfg = next(c for c in CONFIGS if c["id"] == item["config_id"])
        (staging / "configs" / f"{item['job_id']}.json").write_text(json.dumps(cluster_config(cfg), indent=2), encoding="utf-8", newline="\n")
        generate_poisson_trace(
            staging / "workloads" / f"{item['job_id']}.jsonl",
            staging / "manifests" / f"{item['job_id']}.csv",
            CLASSES,
            item["composition"],
            int(item["num_requests"]),
            None if item.get("simultaneous") else float(item["arrival_rate_rps"]),
            int(item["seed"]),
            simultaneous=bool(item.get("simultaneous", False)),
        )
        commands.append(
            f"docker exec -w {CONTAINER_REPO} {CONTAINER} python -m serving "
            f"--cluster-config {REMOTE_STAGE}/configs/{item['job_id']}.json "
            f"--dtype bfloat16 --block-size 16 "
            f"--dataset {REMOTE_STAGE}/workloads/{item['job_id']}.jsonl "
            f"--num-reqs {int(item['num_requests'])} "
            f"--output {REMOTE_STAGE}/runs/{item['job_id']}.csv "
            f"--run-id {item['job_id']} --log-level WARNING --no-enable-prefix-caching "
            f"> {REMOTE_STAGE}/logs/{item['job_id']}.log 2>&1"
        )
    (staging / "jobs.txt").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (staging / "run_jobs.sh").write_text(
        "#!/bin/bash\nset -u\n"
        f"cd '{REMOTE_REPO}'\n"
        f"mkdir -p {REMOTE_STAGE}/runs {REMOTE_STAGE}/logs\n"
        f"cat {REMOTE_STAGE}/jobs.txt | xargs -P 4 -I CMD bash -lc 'CMD'\n",
        encoding="utf-8", newline="\n",
    )


def stage_and_run(staging: Path, label: str) -> None:
    run_checked(["ssh", REMOTE, "mkdir", "-p", f"{REMOTE_REPO}/outputs"])
    # Copy the staging directory itself under the remote outputs directory.
    run_checked(["scp", "-q", "-r", str(staging), f"{REMOTE}:{REMOTE_REPO}/outputs/"])
    print(f"[{label}] running 4-way parallel remote jobs", flush=True)
    run_checked(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=7200)
    print(f"[{label}] remote jobs finished", flush=True)


def fetch_results(output: Path) -> None:
    local_runs = output / "runs"
    local_runs.mkdir(parents=True, exist_ok=True)
    run_checked(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.", str(local_runs)])


def isolated_jobs() -> list[dict[str, Any]]:
    return [{
        "job_id": job_id("cal", "isolated", cfg["id"]),
        "kind": "isolated",
        "config_id": cfg["id"],
        "composition": COMPOSITIONS["mixed"],
        "num_requests": CALIBRATION_REQUESTS,
        "arrival_rate_rps": 0.01,
        "seed": SEED_CAL,
        "simultaneous": False,
    } for cfg in CONFIGS]


def saturated_jobs() -> list[dict[str, Any]]:
    return [{
        "job_id": job_id("cal", "sat", cfg["id"]),
        "kind": "saturation",
        "config_id": cfg["id"],
        "composition": COMPOSITIONS["mixed"],
        "num_requests": SATURATION_REQUESTS,
        "arrival_rate_rps": 0.0,
        "seed": SEED_CAL,
        "simultaneous": True,
    } for cfg in CONFIGS]


def calibrate(iso_jobs: list[dict[str, Any]], sat_jobs: list[dict[str, Any]], output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for iso, sat in zip(iso_jobs, sat_jobs, strict=True):
        cfg = next(c for c in CONFIGS if c["id"] == iso["config_id"])
        manifest = pd.read_csv(output / "manifests" / f"{iso['job_id']}.csv")
        sim = read_simulator_output(output / "runs" / f"{iso['job_id']}.csv")
        merged = sim.merge(manifest, on="request_id", validate="one_to_one").sort_values("request_id")
        observations = pd.DataFrame({
            "prompt_tokens": merged["prompt_tokens"],
            "output_tokens": merged["output_tokens"],
            "prefill_s": merged["prefill_s"].clip(lower=1e-9),
            "decode_s": merged["decode_s"].clip(lower=1e-9),
        })
        base = analytical_parameters(cfg)
        fitted, fit_info = fit_effective_rates(observations, base)
        sat_sim = read_simulator_output(output / "runs" / f"{sat['job_id']}.csv")
        sat_manifest = pd.read_csv(output / "manifests" / f"{sat['job_id']}.csv")
        sat_merged = sat_sim.merge(sat_manifest, on="request_id", validate="one_to_one")
        observed_service = float(sat_merged["service_s"].mean())
        weights = {name: float((sat_manifest["class_id"] == name).mean()) for name in CLASSES}
        best_nu, best_error, best_pred = 1, math.inf, math.inf
        for nu in range(1, 129):
            predicted = float(sum(weights[name] * roofline_service(CLASSES[name], fitted, concurrency=nu).service_s for name in weights))
            error = abs(predicted - observed_service) / max(observed_service, 1e-12)
            if error < best_error:
                best_nu, best_error, best_pred = nu, error, predicted
        rows.append({
            "config_id": cfg["id"],
            "model": cfg["model"],
            "gpu": cfg["gpu"],
            "tp": cfg["tp"],
            "effective_flops_tflops": fitted.effective_flops / 1e12,
            "effective_bandwidth_gbs": fitted.effective_bandwidth_bytes_s / 1e9,
            "fit_cost": fit_info["cost"],
            "effective_concurrency": best_nu,
            "observed_saturated_service_s": observed_service,
            "predicted_saturated_service_s": best_pred,
            "saturated_capacity_rps": saturated_throughput(sat_sim),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "calibration.csv", index=False)
    return frame


def test_jobs(calibration: pd.DataFrame) -> list[dict[str, Any]]:
    capacity = dict(zip(calibration["config_id"], calibration["saturated_capacity_rps"], strict=True))
    jobs: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        base_rate = float(capacity[cfg["id"]])
        for load in TEST_LOAD_FACTORS:
            jobs.append({
                "job_id": job_id("test", cfg["id"], "mixed", f"load{load:.2f}"),
                "kind": "test",
                "config_id": cfg["id"],
                "composition": COMPOSITIONS["mixed"],
                "composition_name": "mixed",
                "num_requests": TEST_REQUESTS,
                "arrival_rate_rps": load * base_rate,
                "load_factor": load,
                "seed": SEED_TEST,
                "simultaneous": False,
            })
        for name in ("short", "prefill", "decode"):
            jobs.append({
                "job_id": job_id("test", cfg["id"], name, "same-load"),
                "kind": "test",
                "config_id": cfg["id"],
                "composition": COMPOSITIONS[name],
                "composition_name": name,
                "num_requests": TEST_REQUESTS,
                "arrival_rate_rps": COMPOSITION_LOAD_FACTOR * base_rate,
                "load_factor": COMPOSITION_LOAD_FACTOR,
                "seed": SEED_TEST,
                "simultaneous": False,
            })
    return jobs


def summarize_tests(jobs: list[dict[str, Any]], calibration: pd.DataFrame, output: Path) -> pd.DataFrame:
    params = {
        row["config_id"]: analytical_parameters(
            next(c for c in CONFIGS if c["id"] == row["config_id"]),
            effective_flops=float(row["effective_flops_tflops"]) * 1e12,
            effective_bandwidth=float(row["effective_bandwidth_gbs"]) * 1e9,
        )
        for _, row in calibration.iterrows()
    }
    concurrency = {row["config_id"]: int(row["effective_concurrency"]) for _, row in calibration.iterrows()}
    rows: list[dict[str, Any]] = []
    for item in jobs:
        cfg = next(c for c in CONFIGS if c["id"] == item["config_id"])
        manifest = pd.read_csv(output / "manifests" / f"{item['job_id']}.csv")
        sim = read_simulator_output(output / "runs" / f"{item['job_id']}.csv")
        merged = sim.merge(manifest, on="request_id", validate="one_to_one").sort_values("end_s")
        if len(merged) >= 10:
            sample = merged.iloc[int(math.floor(0.1 * len(merged))):int(math.ceil(0.9 * len(merged)))]
        else:
            sample = merged
        weights = {name: float((manifest["class_id"] == name).mean()) for name in CLASSES}
        services = {name: roofline_service(CLASSES[name], params[cfg["id"]], concurrency=concurrency[cfg["id"]]) for name in weights}
        prediction = allen_cunneen_prediction(float(item["arrival_rate_rps"]), weights, services, concurrency[cfg["id"]])
        rows.append({
            "job_id": item["job_id"],
            "config_id": cfg["id"],
            "model": cfg["model"],
            "gpu": cfg["gpu"],
            "tp": cfg["tp"],
            "composition": item["composition_name"],
            "load_factor": float(item["load_factor"]),
            "arrival_rate_rps": float(item["arrival_rate_rps"]),
            "observed_waiting_s": float(sample["waiting_s"].mean()),
            "observed_ttft_s": float(sample["ttft_s"].mean()),
            "observed_tbt_s": float(sample["tbt_s"].mean()),
            "observed_response_s": float(sample["response_s"].mean()),
            "observed_service_s": float(sample["service_s"].mean()),
            "predicted_waiting_s": float(prediction.waiting_s),
            "predicted_ttft_s": float(prediction.ttft_s),
            "predicted_tbt_s": float(prediction.tbt_s),
            "predicted_response_s": float(prediction.response_s),
            "predicted_utilization": float(prediction.utilization),
            "predicted_overloaded": bool(prediction.overloaded),
            "n_requests": int(len(sample)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "queue_trend_predictions.csv", index=False)
    return frame


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3 or len(np.unique(xa[mask])) < 2 or len(np.unique(ya[mask])) < 2:
        return float("nan")
    return float(pd.Series(xa[mask]).corr(pd.Series(ya[mask]), method="spearman"))


def trend_metrics(predictions: pd.DataFrame, output: Path) -> dict[str, Any]:
    finite = predictions.replace([np.inf, -np.inf], np.nan).dropna(subset=["observed_waiting_s", "predicted_waiting_s"])
    metrics: dict[str, Any] = {
        "overall": {
            "n_evaluated": int(len(finite)),
            "waiting_spearman": spearman(finite["observed_waiting_s"], finite["predicted_waiting_s"]),
            "ttft_spearman": spearman(finite["observed_ttft_s"], finite["predicted_ttft_s"]),
            "response_spearman": spearman(finite["observed_response_s"], finite["predicted_response_s"]),
        },
        "load_trend": [],
        "composition_trend": [],
        "model_gpu_trend": {},
    }
    for config_id, group in predictions[predictions["composition"] == "mixed"].groupby("config_id"):
        group = group.sort_values("load_factor")
        metrics["load_trend"].append({
            "config_id": config_id,
            "observed_load_spearman": spearman(group["load_factor"], group["observed_waiting_s"]),
            "predicted_load_spearman": spearman(group["load_factor"], group["predicted_waiting_s"]),
            "observed_monotone": bool(group["observed_waiting_s"].is_monotonic_increasing),
            "predicted_monotone": bool(group["predicted_waiting_s"].is_monotonic_increasing),
        })
    for config_id, group in predictions[predictions["load_factor"] == COMPOSITION_LOAD_FACTOR].groupby("config_id"):
        observed_order = ",".join(group.sort_values("observed_service_s")["composition"].tolist())
        predicted_order = ",".join(group.sort_values("predicted_response_s")["composition"].tolist())
        metrics["composition_trend"].append({
            "config_id": config_id,
            "observed_service_order": observed_order,
            "predicted_response_order": predicted_order,
            "service_spearman": spearman(group["observed_service_s"], group["predicted_response_s"]),
            "response_spearman": spearman(group["observed_response_s"], group["predicted_response_s"]),
        })
    low = predictions[(predictions["composition"] == "mixed") & (predictions["load_factor"] == TEST_LOAD_FACTORS[0])]
    metrics["model_gpu_trend"] = {
        "service_spearman": spearman(low["observed_service_s"], low["predicted_response_s"]),
        "response_spearman": spearman(low["observed_response_s"], low["predicted_response_s"]),
    }
    (output / "trend_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    return metrics


def plot_results(predictions: pd.DataFrame, calibration: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    pairs = [("predicted_waiting_s", "observed_waiting_s", "Waiting"), ("predicted_ttft_s", "observed_ttft_s", "TTFT"), ("predicted_response_s", "observed_response_s", "Response")]
    for ax, (pred, obs, label) in zip(axes, pairs, strict=True):
        data = predictions.replace([np.inf, -np.inf], np.nan).dropna(subset=[pred, obs])
        ax.scatter(data[obs] * 1e3, data[pred] * 1e3, s=14, alpha=0.7)
        top = max(float((data[obs] * 1e3).max()), float((data[pred] * 1e3).max()), 1e-3)
        ax.plot([1e-3, top], [1e-3, top], "k--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Observed {label} (ms)")
        ax.set_ylabel(f"Predicted {label} (ms)")
    fig.tight_layout()
    fig.savefig(output / "queue_prediction_scatter.png", dpi=220)
    plt.close(fig)

    selected = ["qwen3-4b-a10", "qwen3-8b-l20", "qwen3-14b-h20", "qwen3-32b-h20"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex=True)
    for ax, config_id in zip(axes.ravel(), selected, strict=True):
        group = predictions[(predictions["config_id"] == config_id) & (predictions["composition"] == "mixed")].sort_values("load_factor")
        if group.empty:
            continue
        finite = group.replace([np.inf, -np.inf], np.nan)
        ax.plot(finite["load_factor"], finite["observed_waiting_s"] * 1e3, "o-", label="Simulator")
        ax.plot(finite["load_factor"], finite["predicted_waiting_s"] * 1e3, "s--", label="Paper model")
        ax.set_title(config_id)
        ax.set_ylabel("Waiting (ms)")
        ax.set_xlabel("Load factor")
        if config_id == selected[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "load_trend_waiting.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ordered = calibration.sort_values("saturated_capacity_rps")
    x = np.arange(len(ordered))
    ax.bar(x, ordered["saturated_capacity_rps"])
    ax.set_xticks(x, ordered["config_id"], rotation=55, ha="right")
    ax.set_ylabel("Saturated capacity (req/s)")
    ax.set_title("Simulator saturated mixed-workload capacity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "capacity_by_config.png", dpi=220)
    plt.close(fig)


def write_report(predictions: pd.DataFrame, calibration: pd.DataFrame, metrics: dict[str, Any], output: Path) -> None:
    overall = metrics["overall"]
    lines = [
        "# LLM queueing model trend validation",
        "",
        "The paper model is evaluated after freezing the fitted effective Roofline rates and effective concurrency.  The comparison is trend-oriented.",
        "",
        "## Calibration",
        "",
        "| Config | Effective FLOPs (TFLOP/s) | Effective BW (GB/s) | Effective concurrency | Saturated capacity (req/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in calibration.sort_values("config_id").itertuples(index=False):
        lines.append(f"| {row.config_id} | {row.effective_flops_tflops:.2f} | {row.effective_bandwidth_gbs:.1f} | {row.effective_concurrency} | {row.saturated_capacity_rps:.3f} |")
    lines += [
        "",
        "## Overall rank agreement",
        "",
        f"- Waiting: Spearman {overall['waiting_spearman']:.3f}",
        f"- TTFT: Spearman {overall['ttft_spearman']:.3f}",
        f"- Complete response: Spearman {overall['response_spearman']:.3f}",
        f"- Evaluated runs: {overall['n_evaluated']}",
        "",
        "## Load trend (mixed composition)",
        "",
        "| Config | Observed Spearman | Predicted Spearman | Observed increasing | Predicted increasing |",
        "|---|---:|---:|---|---|",
    ]
    for row in metrics["load_trend"]:
        lines.append(f"| {row['config_id']} | {row['observed_load_spearman']:.3f} | {row['predicted_load_spearman']:.3f} | {row['observed_monotone']} | {row['predicted_monotone']} |")
    lines += [
        "",
        "## Composition trend at a common arrival rate",
        "",
        "| Config | Observed service order | Predicted response order | Service Spearman | Response Spearman |",
        "|---|---|---|---:|---:|",
    ]
    for row in metrics["composition_trend"]:
        lines.append(f"| {row['config_id']} | {row['observed_service_order']} | {row['predicted_response_order']} | {row['service_spearman']:.3f} | {row['response_spearman']:.3f} |")
    lines += [
        "",
        "## Scope",
        "",
        "- The simulator uses the generated per-kernel LLMServingSim profiles for A10, L20, and H20.",
        "- Qwen3-4B/8B/14B/32B are dense models covered by the paper's dense Roofline equations; MoE is excluded.",
        "- The analytical model is a macroscopic trend model and is not expected to reproduce every scheduler iteration or kernel launch.",
    ]
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "llm_queue_trends")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-calibration-run", action="store_true")
    args = parser.parse_args()
    output = args.output
    staging_root = output / "staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    iso_jobs = isolated_jobs()
    sat_jobs = saturated_jobs()
    cal_staging = staging_root / REMOTE_STAGE
    cal_staging.mkdir(parents=True, exist_ok=True)
    write_jobs(cal_staging, iso_jobs + sat_jobs)
    for sub in ("configs", "workloads", "manifests"):
        target = output / sub
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(cal_staging / sub, target)
    if not args.skip_run and not args.skip_calibration_run:
        stage_and_run(cal_staging, "calibration")
        fetch_results(output)
    calibration = calibrate(iso_jobs, sat_jobs, output)
    print(calibration.to_string(index=False), flush=True)

    jobs = test_jobs(calibration)
    test_staging = staging_root / REMOTE_STAGE
    if test_staging.exists():
        shutil.rmtree(test_staging)
    test_staging.mkdir(parents=True, exist_ok=True)
    write_jobs(test_staging, jobs)
    for sub in ("configs", "workloads", "manifests"):
        target = output / sub
        target.mkdir(parents=True, exist_ok=True)
        for path in (test_staging / sub).glob("*"):
            shutil.copy2(path, target / path.name)
    if not args.skip_run:
        stage_and_run(test_staging, "test")
        fetch_results(output)

    predictions = summarize_tests(jobs, calibration, output)
    metrics = trend_metrics(predictions, output)
    plot_results(predictions, calibration, output)
    write_report(predictions, calibration, metrics, output)
    print(json.dumps(metrics, indent=2), flush=True)
    print(output / "validation_report.md", flush=True)


if __name__ == "__main__":
    main()

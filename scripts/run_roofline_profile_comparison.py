"""Compare a synthetic Roofline hardware profile with measured LLMServingSim data.

The synthetic bundle follows LLMServingSim's public CSV contract.  Its rows use
the same axes as the measured RTX4090/Llama-3.1-8B profile, but every latency is
computed from analytical FLOP and compulsory-memory estimates.  The script then
runs identical isolated requests through the synthetic profile and compares the
results with the measured RTX4090 profile and the paper's request-level Roofline
equations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_orch.validation.llmservingsim import (
    AnalyticalParameters,
    JITSERVE_CLASSES,
    WorkloadClass,
    read_run_specs,
    read_simulator_output,
    roofline_service,
)


HARDWARE = "RTX4090_ROOFLINE_BF16"
MODEL = "meta-llama/Llama-3.1-8B"
VARIANT = "bf16"
REMOTE_HOST = "zf@192.168.234.128"
CONTAINER = "servingsim_docker"
CONTAINER_REPO = "/app/LLMServingSim"
REMOTE_STAGE = "/tmp/llmservingsim_roofline_comparison"

# Llama-3.1-8B dimensions.
LAYERS = 32
HIDDEN = 4096
INTERMEDIATE = 14336
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
KV_DIM = KV_HEADS * HEAD_DIM
VOCAB = 128256
DTYPE_BYTES = 2

# The synthetic profile uses the datasheet BF16 Tensor Core peak without
# sparsity.  The original paper-side model used 82.6 TFLOP/s and is retained as
# a separate request-level baseline below.
SYNTHETIC_FLOPS = 165.2e12
OLD_ROOFLINE_FLOPS = 82.6e12
PEAK_BANDWIDTH = 1008e9

SITE_CUSTOMIZE = '''"""Metric-only first-admission instrumentation."""
from serving.core.request import Request

_original_set_que_delay = Request.set_que_delay

def _set_first_admission_delay(self, current):
    if self.queuing_delay < 0:
        _original_set_que_delay(self, current)

Request.set_que_delay = _set_first_admission_delay
'''


def run_checked(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout


def roofline_us(flops: float, memory_bytes: float) -> float:
    return 1e6 * max(float(flops) / SYNTHETIC_FLOPS, float(memory_bytes) / PEAK_BANDWIDTH)


def gemm_cost(tokens: float, k: int, n: int) -> tuple[float, float]:
    flops = 2.0 * tokens * k * n
    memory = DTYPE_BYTES * (k * n + tokens * (k + n))
    return flops, memory


def dense_cost(layer: str, tokens: int) -> tuple[float, float]:
    t = float(tokens)
    if layer == "embedding":
        return 0.0, DTYPE_BYTES * t * HIDDEN * 2.0
    if layer in {"layernorm", "final_layernorm"}:
        return 5.0 * t * HIDDEN, DTYPE_BYTES * t * HIDDEN * 2.0
    if layer == "qkv_proj":
        return gemm_cost(t, HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM)
    if layer == "rotary_emb":
        elements = t * (HIDDEN + KV_DIM)
        return 6.0 * elements, DTYPE_BYTES * elements * 2.0
    if layer == "o_proj":
        return gemm_cost(t, HIDDEN, HIDDEN)
    if layer == "gate_up_proj":
        return gemm_cost(t, HIDDEN, 2 * INTERMEDIATE)
    if layer == "act_fn":
        return 8.0 * t * INTERMEDIATE, DTYPE_BYTES * t * 3.0 * INTERMEDIATE
    if layer == "down_proj":
        return gemm_cost(t, INTERMEDIATE, HIDDEN)
    raise KeyError(f"unsupported dense layer: {layer}")


def per_sequence_cost(layer: str, sequences: int) -> tuple[float, float]:
    n = float(sequences)
    if layer == "lm_head":
        return gemm_cost(n, HIDDEN, VOCAB)
    if layer == "sampler":
        return 5.0 * n * VOCAB, 4.0 * n * VOCAB
    raise KeyError(f"unsupported per-sequence layer: {layer}")


def attention_cost(prefill_chunk: int, kv_prefill: int, n_decode: int, kv_decode: int) -> tuple[float, float]:
    x = float(prefill_chunk)
    hp = float(kv_prefill)
    n = float(n_decode)
    kd = float(kv_decode)
    prefill_pairs = x * (hp + (x + 1.0) / 2.0)
    decode_pairs = n * (kd + 1.0)
    flops = 4.0 * HIDDEN * (prefill_pairs + decode_pairs)

    # Compulsory FlashAttention traffic: Q/output vectors and the K/V history.
    kv_pair_bytes = 2.0 * KV_DIM * DTYPE_BYTES
    prefill_memory = 2.0 * x * HIDDEN * DTYPE_BYTES + (hp + x) * kv_pair_bytes
    decode_memory = 2.0 * n * HIDDEN * DTYPE_BYTES + n * (kd + 1.0) * kv_pair_bytes
    return flops, prefill_memory + decode_memory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output: Path, measured_profile: Path, reference_results: Path) -> None:
    profile_root = output / "synthetic_profile" / HARDWARE / "meta-llama" / "Llama-3.1-8B" / VARIANT
    tp = profile_root / "tp1"
    tp.mkdir(parents=True, exist_ok=True)

    dense_axes = pd.read_csv(measured_profile / "dense.csv")[["layer", "tokens"]].drop_duplicates()
    dense_rows = []
    for row in dense_axes.itertuples(index=False):
        flops, memory = dense_cost(str(row.layer), int(row.tokens))
        dense_rows.append({"layer": row.layer, "tokens": int(row.tokens), "time_us": roofline_us(flops, memory)})
    pd.DataFrame(dense_rows).to_csv(tp / "dense.csv", index=False)

    sequence_axes = pd.read_csv(measured_profile / "per_sequence.csv")[["layer", "sequences"]].drop_duplicates()
    sequence_rows = []
    for row in sequence_axes.itertuples(index=False):
        flops, memory = per_sequence_cost(str(row.layer), int(row.sequences))
        sequence_rows.append({"layer": row.layer, "sequences": int(row.sequences), "time_us": roofline_us(flops, memory)})
    pd.DataFrame(sequence_rows).to_csv(tp / "per_sequence.csv", index=False)

    attention_axes = pd.read_csv(measured_profile / "attention.csv")[
        ["prefill_chunk", "kv_prefill", "n_decode", "kv_decode"]
    ].drop_duplicates()
    attention_rows = []
    for row in attention_axes.itertuples(index=False):
        flops, memory = attention_cost(
            int(row.prefill_chunk), int(row.kv_prefill), int(row.n_decode), int(row.kv_decode)
        )
        attention_rows.append(
            {
                "prefill_chunk": int(row.prefill_chunk),
                "kv_prefill": int(row.kv_prefill),
                "n_decode": int(row.n_decode),
                "kv_decode": int(row.kv_decode),
                "time_us": roofline_us(flops, memory),
            }
        )
    pd.DataFrame(attention_rows).to_csv(tp / "attention.csv", index=False)

    meta = f"""profiler_version: synthetic-roofline-v1
vllm_version: n/a
gpu: NVIDIA GeForce RTX 4090 (analytical Roofline)
hardware: {HARDWARE}
profiled_at: '{datetime.now(timezone.utc).isoformat()}'
architecture: llama
architecture_sha256: 19d708000ec8a89edf1ca12dd388235b2cfe59007aaec6ea48702fb856ff2cfb
model: {MODEL}
variant: {VARIANT}
tp_degrees: [1]
engine_effective:
  max_num_batched_tokens: 2048
  max_num_seqs: 256
attention_grid:
  max_kv: 16384
  chunk_factor: 2.0
  kv_factor: 2.0
  chunks: 0, 16-2048 x2
  n_decode: 0, 1-256 x2
  kv: 0, 16-16384 x2
skew_fit:
  enabled: false
"""
    (profile_root / "meta.yaml").write_text(meta, encoding="utf-8")

    cluster = {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 1000,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {"mem_size": 64, "mem_bw": 50, "mem_latency": 0},
                "instances": [
                    {
                        "model_name": MODEL,
                        "hardware": HARDWARE,
                        "npu_mem": {"mem_size": 24, "mem_bw": 1008, "mem_latency": 0, "mem_util": 0.833919},
                        "num_npus": 1,
                        "tp_size": 1,
                        "pd_type": None,
                    }
                ],
            }
        ],
    }
    (output / "cluster_config.json").write_text(json.dumps(cluster, indent=2), encoding="utf-8")

    workloads = output / "workloads"
    workloads.mkdir(parents=True, exist_ok=True)
    instrument = output / "instrument"
    instrument.mkdir(parents=True, exist_ok=True)
    (instrument / "sitecustomize.py").write_text(SITE_CUSTOMIZE, encoding="utf-8")
    specs = [row for row in read_run_specs(reference_results / "calibration_runs.jsonl") if row.get("kind") == "service"]
    for spec in specs:
        shutil.copy2(reference_results / "workloads" / str(spec["workload"]), workloads / str(spec["workload"]))
    (output / "service_runs.jsonl").write_text(
        "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs), encoding="utf-8"
    )
    provenance = {
        "hardware_label": HARDWARE,
        "model": MODEL,
        "synthetic_bf16_tensor_flops": SYNTHETIC_FLOPS,
        "old_request_roofline_flops": OLD_ROOFLINE_FLOPS,
        "effective_bandwidth_bytes_s": PEAK_BANDWIDTH,
        "model_dimensions": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab": VOCAB,
            "dtype_bytes": DTYPE_BYTES,
        },
        "profile_hashes": {name: sha256(tp / name) for name in ("dense.csv", "per_sequence.csv", "attention.csv")},
        "instrumentation_sha256": sha256(instrument / "sitecustomize.py"),
    }
    (output / "synthetic_profile.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"prepared synthetic profile: {profile_root}")


def install(output: Path, host: str, container: str, container_repo: str) -> None:
    local_hw = output / "synthetic_profile" / HARDWARE
    run_checked(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", REMOTE_STAGE])
    run_checked(["scp", "-q", "-r", str(local_hw), f"{host}:{REMOTE_STAGE}/"])
    run_checked(["scp", "-q", str(output / "cluster_config.json"), f"{host}:{REMOTE_STAGE}/cluster_config.json"])
    run_checked(["scp", "-q", "-r", str(output / "workloads"), f"{host}:{REMOTE_STAGE}/"])
    run_checked(["scp", "-q", "-r", str(output / "instrument"), f"{host}:{REMOTE_STAGE}/"])
    target_profile = f"{container_repo}/profiler/perf/{HARDWARE}"
    target_output = f"{container_repo}/outputs/roofline_profile_comparison"
    target_config = f"{container_repo}/configs/cluster/rtx4090_roofline_single.json"
    run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "exec", container, "mkdir", "-p", target_profile, target_output + "/runs", target_output + "/workloads", target_output + "/instrument"])
    run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "cp", f"{REMOTE_STAGE}/{HARDWARE}/.", f"{container}:{target_profile}/"])
    run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "cp", f"{REMOTE_STAGE}/cluster_config.json", f"{container}:{target_config}"])
    run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "cp", f"{REMOTE_STAGE}/workloads/.", f"{container}:{target_output}/workloads/"])
    run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "cp", f"{REMOTE_STAGE}/instrument/.", f"{container}:{target_output}/instrument/"])
    print(f"installed {HARDWARE} in {container}:{target_profile}")


def run_simulations(output: Path, host: str, container: str, container_repo: str, force: bool) -> None:
    runs = output / "runs"
    logs = output / "logs"
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    specs = read_run_specs(output / "service_runs.jsonl")
    for index, spec in enumerate(specs, start=1):
        run_id = f"roofline-{spec['run_id']}"
        local_csv = runs / f"{run_id}.csv"
        if local_csv.exists() and not force:
            print(f"[{index}/{len(specs)}] {run_id}: skipped")
            continue
        container_csv = f"outputs/roofline_profile_comparison/runs/{run_id}.csv"
        container_workload = f"outputs/roofline_profile_comparison/workloads/{spec['workload']}"
        command = [
            "ssh", "-o", "BatchMode=yes", host,
            "docker", "exec", "-e", f"PYTHONPATH={container_repo}/outputs/roofline_profile_comparison/instrument:{container_repo}", "-w", container_repo, container,
            "python", "-m", "serving",
            "--cluster-config", "configs/cluster/rtx4090_roofline_single.json",
            "--dtype", "bfloat16",
            "--block-size", "16",
            "--max-num-batched-tokens", "2048",
            "--max-num-seqs", "128",
            "--long-prefill-token-threshold", "512",
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--dataset", container_workload,
            "--num-reqs", "1",
            "--output", container_csv,
            "--run-id", run_id,
            "--log-level", "WARNING",
        ]
        print(f"[{index}/{len(specs)}] {run_id}: running", flush=True)
        text = run_checked(command)
        (logs / f"{run_id}.log").write_text(text, encoding="utf-8")
        remote_csv = f"{REMOTE_STAGE}/{run_id}.csv"
        run_checked(["ssh", "-o", "BatchMode=yes", host, "docker", "cp", f"{container}:{container_repo}/{container_csv}", remote_csv])
        run_checked(["scp", "-q", f"{host}:{remote_csv}", str(local_csv)])
        print(f"[{index}/{len(specs)}] {run_id}: completed", flush=True)


def error_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    obs = observed.to_numpy(dtype=float)
    pred = predicted.to_numpy(dtype=float)
    error = np.abs(pred - obs)
    relative = error / np.maximum(np.abs(obs), 1e-12)
    return {
        "n": int(len(obs)),
        "mae_s": float(error.mean()),
        "wape_pct": float(100.0 * error.sum() / max(np.abs(obs).sum(), 1e-12)),
        "median_ape_pct": float(100.0 * np.median(relative)),
        "p95_ape_pct": float(100.0 * np.percentile(relative, 95)),
    }


def analyze(output: Path, reference_results: Path) -> None:
    calibration = json.loads((reference_results / "calibration.json").read_text(encoding="utf-8"))
    fitted_params = AnalyticalParameters(**calibration["analytical_parameters"])
    peak_params = AnalyticalParameters(effective_flops=OLD_ROOFLINE_FLOPS, effective_bandwidth_bytes_s=PEAK_BANDWIDTH)
    rows: list[dict[str, Any]] = []
    for spec in read_run_specs(output / "service_runs.jsonl"):
        class_id = str(spec["class_id"])
        actual = read_simulator_output(reference_results / "runs" / f"{spec['run_id']}.csv").iloc[0]
        synthetic = read_simulator_output(output / "runs" / f"roofline-{spec['run_id']}.csv").iloc[0]
        if class_id.startswith("CAL_"):
            _, prompt, generated = class_id.split("_")
            workload = WorkloadClass(class_id, int(prompt), int(generated))
            split = "calibration"
        else:
            workload = JITSERVE_CLASSES[class_id]
            split = "JITServe"
        peak = roofline_service(workload, peak_params, concurrency=1)
        fitted = roofline_service(workload, fitted_params, concurrency=1)
        for method, value in (
            ("LLMServingSim measured profile", actual),
            ("LLMServingSim synthetic Roofline profile", synthetic),
            ("request-level Roofline (peak rates)", peak),
            ("request-level Roofline (fitted rates)", fitted),
        ):
            rows.append(
                {
                    "class_id": class_id,
                    "split": split,
                    "prompt_tokens": workload.prompt_tokens,
                    "output_tokens": workload.output_tokens,
                    "method": method,
                    "prefill_s": float(value.prefill_s),
                    "decode_s": float(value.decode_s),
                    "service_s": float(value.service_s),
                    "tbt_s": float(value.tbt_s),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    reference = frame[frame["method"] == "LLMServingSim measured profile"].set_index("class_id")
    summary_rows = []
    for method, group in frame.groupby("method"):
        if method == "LLMServingSim measured profile":
            continue
        aligned = group.set_index("class_id").loc[reference.index]
        for metric in ("prefill_s", "decode_s", "service_s", "tbt_s"):
            summary_rows.append({"method": method, "metric": metric, **error_metrics(reference[metric], aligned[metric])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "error_summary.csv", index=False)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    metrics = [("prefill_s", "Prefill"), ("tbt_s", "TBT"), ("service_s", "Complete service")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for method, group in frame.groupby("method"):
            if method == "LLMServingSim measured profile":
                continue
            aligned = group.set_index("class_id").loc[reference.index]
            ax.scatter(reference[metric], aligned[metric], s=16, label=method)
        maximum = max(frame[metric].max(), reference[metric].max())
        ax.plot([0, maximum], [0, maximum], "k--", linewidth=0.8)
        if metric == "tbt_s":
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("measured profile (s)")
        ax.set_ylabel("alternative (s)")
        ax.set_title(title)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "roofline_profile_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "roofline_profile_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Synthetic Roofline profile comparison",
        "",
        "Reference: LLMServingSim measured RTX4090/Llama-3.1-8B profile. The synthetic hardware uses the same profile axes and Llama architecture catalog, while all kernel times are generated by a Roofline model with the 165.2 TFLOP/s BF16 Tensor Core peak (FP32 accumulate, no sparsity) and 1008 GB/s. The old request-level baseline retains its original 82.6 TFLOP/s assumption.",
        "",
        "| method | metric | WAPE | median APE | P95 APE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.method} | {row.metric} | {row.wape_pct:.2f}% | {row.median_ape_pct:.2f}% | {row.p95_ape_pct:.2f}% |")
    report += [
        "",
        "The synthetic profile and the request-level peak Roofline are both parameter-free timing estimates after model and hardware specifications are fixed. The fitted request-level Roofline is reported separately because its effective FLOP/s and bandwidth were calibrated against measured-profile request latencies.",
    ]
    (output / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "install", "run", "analyze", "all"))
    parser.add_argument("--output", type=Path, default=Path("results/roofline_profile_comparison"))
    parser.add_argument("--measured-profile", type=Path, default=Path("results/llm_queue_validation_v2/profile"))
    parser.add_argument("--reference-results", type=Path, default=Path("results/llm_queue_validation"))
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument("--container-repo", default=CONTAINER_REPO)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command in {"prepare", "all"}:
        prepare(args.output, args.measured_profile, args.reference_results)
    if args.command in {"install", "all"}:
        install(args.output, args.host, args.container, args.container_repo)
    if args.command in {"run", "all"}:
        run_simulations(args.output, args.host, args.container, args.container_repo, args.force)
    if args.command in {"analyze", "all"}:
        analyze(args.output, args.reference_results)


if __name__ == "__main__":
    main()

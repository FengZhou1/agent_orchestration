"""Long-run validation of the paper's steady-state LLM model.

The generator uses the current preconstructed Agent workloads, while the
remote simulator is kept as an independent execution backend.  The analytical
side uses only model/GPU parameters and the equations implemented by
``AnalyticalBackend``; no simulator result is used to fit a queue parameter.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_orch.performance.analytical import evaluate_llm_instance  # noqa: E402
from agent_orch.performance.llm import (  # noqa: E402
    mean_decode_context,
    residency_capacity,
    service_curve,
    throughput_capacity,
)
from agent_orch.schema.loader import ScenarioLoader  # noqa: E402
from agent_orch.schema.models import NodeType  # noqa: E402
from agent_orch.validation.llmservingsim import (  # noqa: E402
    WorkloadClass,
    generate_poisson_trace,
    read_simulator_output,
)

REMOTE = "zf@192.168.234.128"
REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
REMOTE_STAGE = "outputs/llm_steady_state_v3"
CONTAINER = "servingsim_docker"
CONTAINER_REPO = "/app/LLMServingSim"
CHUNK = 512
MAX_SEQS = 128
LOAD_FACTORS = (0.40, 0.70, 0.85, 0.95)
DEFAULT_CONFIGS = (
    "qwen3-4b-a10",
    "qwen3-8b-h20",
    "qwen3-14b-h20",
    "qwen3-32b-h20",
)
DEFAULT_COMPOSITIONS = ("interactive", "transactional", "deep_research", "coding", "balanced")

SITE_CUSTOMIZE = '''"""Metric-only first-admission instrumentation."""
from serving.core.request import Request

_original_set_que_delay = Request.set_que_delay

def _set_first_admission_delay(self, current):
    if self.queuing_delay < 0:
        _original_set_que_delay(self, current)

Request.set_que_delay = _set_first_admission_delay
'''

GPU_SPECS = {
    "A10": {"memory_gb": 24.0, "bandwidth_gbs": 600.0},
    "L20": {"memory_gb": 48.0, "bandwidth_gbs": 864.0},
    "H20": {"memory_gb": 96.0, "bandwidth_gbs": 4000.0},
}


def run_checked(command: list[str], timeout: int | None = None) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout[-4000:]}")
    return result.stdout


def family_workloads(scenario, model_id: str) -> dict[str, WorkloadClass]:
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for app in scenario.applications.values():
        family = app.family if app.family != "unspecified" else app.id
        request_rate = sum(app.ingress_rates.values())
        for flow in app.pattern_flows:
            llm_nodes = [app.nodes[node] for node in flow.nodes if app.nodes[node].type is NodeType.LLM]
            for node in llm_nodes:
                weight = request_rate * flow.probability
                grouped.setdefault(family, []).append(
                    (weight, node.prompt_tokens[model_id], node.output_tokens[model_id])
                )
    workloads: dict[str, WorkloadClass] = {}
    for family, values in grouped.items():
        total = sum(v[0] for v in values)
        if total <= 0:
            continue
        prompt = sum(w * p for w, p, _ in values) / total
        output = sum(w * o for w, _, o in values) / total
        workloads[family] = WorkloadClass(family, max(1, round(prompt)), max(1, round(output)))
    return workloads


def gpu_config(config_id: str) -> dict[str, Any]:
    parts = config_id.split("-")
    gpu = "H20" if "h20" in config_id else "L20" if "l20" in config_id else "A10"
    tp = 2 if "2xl20" in config_id else 1
    model = "Qwen3-32B" if "32b" in config_id else "Qwen3-14B" if "14b" in config_id else "Qwen3-8B" if "8b" in config_id else "Qwen3-4B"
    spec = GPU_SPECS[gpu]
    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 1000,
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
            "instances": [{
                "model_name": f"Qwen/{model}",
                "hardware": gpu,
                "npu_mem": {"mem_size": spec["memory_gb"], "mem_bw": spec["bandwidth_gbs"], "mem_latency": 0, "mem_util": 0.9},
                "num_npus": tp,
                "tp_size": tp,
                "pd_type": None,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": MAX_SEQS,
                "long_prefill_token_threshold": CHUNK,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "block_size": 16,
            }],
        }],
    }


def job_id(*parts: object) -> str:
    return "-".join(str(p).replace("_", "-") for p in parts)


def model_id_for(config_id: str) -> str:
    return "qwen3-32b" if "32b" in config_id else "qwen3-14b" if "14b" in config_id else "qwen3-8b" if "8b" in config_id else "qwen3-4b"


def _instance_inputs(scenario, config_id: str, workloads, composition):
    config = scenario.llm_configs[config_id]
    model = scenario.models[config.model]
    weights = {
        key: value / sum(composition.values())
        for key, value in composition.items()
        if value > 0
    }
    classes = [workloads[key] for key in weights]
    call_classes = [(w.prompt_tokens, w.output_tokens) for w in classes]
    class_weights = [weights[w.id] for w in classes]
    return config, model, call_classes, class_weights


def _instance_metrics(
    scenario, config_id: str, workloads, composition, rate: float
) -> dict[str, float | bool]:
    config, model, call_classes, class_weights = _instance_inputs(
        scenario, config_id, workloads, composition
    )
    instance, per_class = evaluate_llm_instance(
        model, config, call_classes, class_weights, rate, CHUNK
    )
    total = sum(class_weights)
    return {
        "predicted_active_concurrency": instance.active_concurrency,
        "predicted_resident_capacity": float(instance.resident_capacity),
        "predicted_utilization": instance.utilization,
        "predicted_capacity_rps": instance.throughput_capacity_rps,
        "predicted_wait_s": 0.0,
        "predicted_ttft_s": sum(
            w * p.ttft_s for w, p in zip(class_weights, per_class)
        )
        / total,
        "predicted_tbt_s": sum(
            w * p.tbt_s for w, p in zip(class_weights, per_class)
        )
        / total,
        "predicted_response_s": sum(
            w * p.response_s for w, p in zip(class_weights, per_class)
        )
        / total,
        "predicted_service_s": sum(
            w * p.service_s for w, p in zip(class_weights, per_class)
        )
        / total,
        "predicted_stable": instance.stable,
    }


def analytical_capacity(scenario, config_id: str, workloads, composition) -> float:
    """Largest call rate the instance sustains below its residency limit."""
    config, model, call_classes, class_weights = _instance_inputs(
        scenario, config_id, workloads, composition
    )
    residency = residency_capacity(
        call_classes,
        class_weights,
        config.kv_token_capacity,
        config.max_num_seqs,
        CHUNK,
    )
    peer_context = mean_decode_context(call_classes, class_weights)
    curves = [
        service_curve(
            model, config, prompt, output, CHUNK, peer_decode_context=peer_context
        )
        for prompt, output in call_classes
    ]
    capacity, _ = throughput_capacity(curves, class_weights, residency.capacity)
    return max(capacity, 1.0e-6)


def analytical_metrics(scenario, config_id: str, workloads, composition, rate: float):
    return _instance_metrics(scenario, config_id, workloads, composition, rate)

def write_jobs(staging: Path, jobs: list[dict[str, Any]], scenario, workers: int = 4) -> None:
    for name in ("configs", "workloads", "manifests", "runs", "logs", "instrument"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    commands = []
    for job in jobs:
        cfg = gpu_config(job["config_id"])
        (staging / "configs" / f"{job['job_id']}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8", newline="\n")
        workloads = family_workloads(scenario, model_id_for(job["config_id"]))
        generate_poisson_trace(
            staging / "workloads" / f"{job['job_id']}.jsonl",
            staging / "manifests" / f"{job['job_id']}.csv",
            workloads,
            job["composition"],
            int(job["requests"]),
            float(job["arrival_rate_rps"]),
            int(job["seed"]),
        )
        commands.append(
            f"docker exec -e PYTHONPATH={CONTAINER_REPO}/{REMOTE_STAGE}/instrument:{CONTAINER_REPO} "
            f"-w {CONTAINER_REPO} {CONTAINER} python -m serving "
            f"--cluster-config {REMOTE_STAGE}/configs/{job['job_id']}.json "
            f"--dtype bfloat16 --block-size 16 "
            f"--dataset {REMOTE_STAGE}/workloads/{job['job_id']}.jsonl "
            f"--num-reqs {int(job['requests'])} --output {REMOTE_STAGE}/runs/{job['job_id']}.csv "
            f"--run-id {job['job_id']} --log-level WARNING --no-enable-prefix-caching "
            f"> {REMOTE_STAGE}/logs/{job['job_id']}.log 2>&1"
        )
    (staging / "instrument" / "sitecustomize.py").write_text(SITE_CUSTOMIZE, encoding="utf-8", newline="\n")
    (staging / "jobs.txt").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (staging / "run_jobs.sh").write_text(
        "#!/bin/bash\nset -u\n"
        f"cd '{REMOTE_REPO}'\nmkdir -p {REMOTE_STAGE}/runs {REMOTE_STAGE}/logs\n"
        f"cat {REMOTE_STAGE}/jobs.txt | xargs -P {max(1, int(workers))} -I CMD bash -lc 'CMD'\n",
        encoding="utf-8",
        newline="\n",
    )


def summarize(output: Path, jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        path = output / "runs" / f"{job['job_id']}.csv"
        if not path.exists():
            continue
        frame = read_simulator_output(path)
        arrivals = frame["arrival_ns"].to_numpy(dtype=float) / 1e9
        start = float(np.quantile(arrivals, 0.2))
        end = float(np.quantile(arrivals, 0.9))
        sample = frame[(arrivals >= start) & (arrivals <= end)]
        log_path = output / "logs" / f"{job['job_id']}.log"
        running, memory = parse_runtime_log(log_path)
        rows.append({
            **{key: job[key] for key in ("job_id", "config_id", "composition_id", "load_factor", "arrival_rate_rps", "predicted_capacity_rps", "seed", "predicted_active_concurrency", "predicted_resident_capacity", "predicted_utilization", "predicted_wait_s", "predicted_ttft_s", "predicted_tbt_s", "predicted_response_s", "predicted_service_s", "predicted_stable")},
            "n": int(len(sample)),
            "observed_wait_s": float(sample["first_waiting_s"].mean()),
            "observed_ttft_s": float(sample["ttft_s"].mean()),
            "observed_tbt_s": float(sample["tbt_s"].mean()),
            "observed_response_s": float(sample["response_s"].mean()),
            "observed_service_s": float(sample["service_s"].mean()),
            "observed_throughput_rps": float(len(frame) / max(frame["end_s"].max() - frame["arrival_ns"].min() / 1e9, 1e-9)),
            "observed_mean_running_reqs": running[0] if running else math.nan,
            "observed_peak_running_reqs": running[1] if running else math.nan,
            "observed_mean_gpu_memory_mb": memory[0] if memory else math.nan,
            "observed_peak_gpu_memory_mb": memory[1] if memory else math.nan,
        })
    return pd.DataFrame(rows)


def parse_runtime_log(path: Path) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if not path.exists():
        return None, None
    running: list[float] = []
    memory: list[float] = []
    pattern = re.compile(r"Running Instance\[0\]:\s+(\d+) reqs,.*?Memory Usage\s+([0-9.]+) MB")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            running.append(float(match.group(1)))
            memory.append(float(match.group(2)))
    if not running:
        return None, None
    return (float(np.mean(running)), float(np.max(running))), (float(np.mean(memory)), float(np.max(memory)))


def analyze(output: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metric_pairs = {
        "wait_s": ("predicted_wait_s", "observed_wait_s"),
        "ttft_s": ("predicted_ttft_s", "observed_ttft_s"),
        "tbt_s": ("predicted_tbt_s", "observed_tbt_s"),
        "response_s": ("predicted_response_s", "observed_response_s"),
        "service_s": ("predicted_service_s", "observed_service_s"),
    }
    rows = []
    for (config, composition), group in summary.groupby(["config_id", "composition_id"]):
        for metric, (pred, obs) in metric_pairs.items():
            frame = group[["load_factor", pred, obs]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 2:
                continue
            scale = max(float(frame[obs].mean()), 1e-9)
            rows.append({
                "config_id": config,
                "composition_id": composition,
                "metric": metric,
                "n": len(frame),
                "spearman_load_observed": float(frame["load_factor"].corr(frame[obs], method="spearman")),
                "spearman_load_predicted": float(frame["load_factor"].corr(frame[pred], method="spearman")),
                "wape": float((frame[pred] - frame[obs]).abs().sum() / max(frame[obs].abs().sum(), 1e-9)),
                "mae_over_mean_observed": float((frame[pred] - frame[obs]).abs().mean() / scale),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "trend_metrics.csv", index=False)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        for metric, (pred, obs) in metric_pairs.items():
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            for (config, composition), group in summary.groupby(["config_id", "composition_id"]):
                ordered = group.sort_values("load_factor")
                ax.plot(ordered["load_factor"], ordered[obs], marker="o", alpha=0.55, label=f"sim {config}/{composition}")
                ax.plot(ordered["load_factor"], ordered[pred], marker="x", linestyle="--", alpha=0.8, label=f"model {config}/{composition}")
            ax.set_xlabel("offered load / analytical capacity")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            if len(summary.groupby(["config_id", "composition_id"])) <= 5:
                ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(figures / f"{metric}_trend.png", dpi=180)
            plt.close(fig)
    except ImportError:
        pass
    if not metrics.empty:
        # Keep report generation independent of the optional ``tabulate``
        # package used by pandas.DataFrame.to_markdown().
        headers = list(metrics.columns)
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in metrics.itertuples(index=False, name=None):
            table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
        metrics_table = "\n".join(table_lines)
    else:
        metrics_table = "No valid metric groups."
    lines = [
        "# LLM steady-state model validation",
        "",
        f"Completed observations: {len(summary)}",
        "",
        "The analytical series is generated from the current TeX equations; no simulator output is used to fit parameters.",
        "",
        metrics_table,
    ]
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/benchmarks/main_abilene_revised.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/llm_steady_state_validation_v3")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--compositions", default=",".join(DEFAULT_COMPOSITIONS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--min-requests", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--load-factors", default=",".join(f"{x:.2f}" for x in LOAD_FACTORS))
    parser.add_argument("--include-overload", action="store_true")
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()
    scenario = ScenarioLoader.load(args.scenario)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in ("staging", "runs", "configs", "workloads", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)

    configs = [x for x in args.configs.split(",") if x]
    compositions = [x for x in args.compositions.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    all_workloads = {cfg: family_workloads(scenario, model_id_for(cfg)) for cfg in configs}
    composition_map = {
        "interactive": {"interactive_retrieval": 1.0},
        "transactional": {"transactional_tool": 1.0},
        "deep_research": {"deep_research": 1.0},
        "coding": {"coding_agent": 1.0},
        "balanced": {name: 0.25 for name in ("interactive_retrieval", "transactional_tool", "deep_research", "coding_agent")},
    }
    load_factors = tuple(float(x) for x in args.load_factors.split(",") if x)
    if args.include_overload and 1.05 not in load_factors:
        load_factors += (1.05,)
    jobs = []
    predictions = []
    for cfg in configs:
        model_workloads = all_workloads[cfg]
        for comp_id in compositions:
            comp = {k: v for k, v in composition_map[comp_id].items() if k in model_workloads}
            if not comp:
                raise ValueError(f"composition {comp_id} has no classes for {cfg}")
            capacity = analytical_capacity(scenario, cfg, model_workloads, comp)
            for load in load_factors:
                for seed in seeds:
                    jid = job_id("steady", cfg, comp_id, f"l{load:.2f}", f"s{seed}")
                    metrics = analytical_metrics(scenario, cfg, model_workloads, comp, load * capacity)
                    rate = load * capacity
                    # Keep the virtual arrival horizon bounded for low-capacity
                    # long-context workloads while retaining enough requests to
                    # estimate a steady mean.
                    requests = min(args.requests, max(args.min_requests, int(rate * 600.0)))
                    job = {"job_id": jid, "config_id": cfg, "composition": comp, "composition_id": comp_id, "load_factor": load, "arrival_rate_rps": rate, "predicted_capacity_rps": capacity, "requests": requests, "seed": seed, **metrics}
                    jobs.append(job)
                    predictions.append({key: value for key, value in job.items() if key != "composition"})
    (output / "predictions.csv").write_text(pd.DataFrame(predictions).to_csv(index=False), encoding="utf-8", newline="\n")
    (output / "experiment_config.json").write_text(json.dumps({"scenario": str(args.scenario), "configs": configs, "compositions": compositions, "seeds": seeds, "requests": args.requests, "min_requests": args.min_requests, "workers": args.workers, "load_factors": load_factors, "chunk_tokens": CHUNK, "max_num_seqs": MAX_SEQS}, indent=2), encoding="utf-8", newline="\n")
    staging = output / "staging" / REMOTE_STAGE
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    write_jobs(staging, jobs, scenario, workers=args.workers)
    if not args.no_remote:
        run_checked(["ssh", REMOTE, "mkdir", "-p", f"{REMOTE_REPO}/outputs"], timeout=30)
        run_checked(["scp", "-q", "-r", str(staging), f"{REMOTE}:{REMOTE_REPO}/outputs/"], timeout=300)
        started = time.time()
        run_checked(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=24 * 3600)
        print(f"remote simulation completed in {time.time() - started:.1f}s", flush=True)
        run_checked(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.", str(output / "runs")], timeout=600)
        run_checked(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/logs/.", str(output / "logs")], timeout=600)
    summary = summarize(output, jobs)
    summary.to_csv(output / "observations.csv", index=False)
    analyze(output, summary)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

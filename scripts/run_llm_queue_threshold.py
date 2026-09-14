"""Threshold sweep in long-run steady state.

The earlier load grid used lambda / mu_sat, where mu_sat is the saturated
throughput. At lambda = 0.95 mu_sat the pool is only half full, so that grid
never approached the queueing threshold. This sweep pushes lambda through the
threshold and measures the admission wait, the pool occupancy and whether the
wait accumulates over the run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_llm_queue_trends as base  # noqa: E402
from agent_orch.validation.llmservingsim import read_simulator_output  # noqa: E402

REMOTE = "zf@192.168.234.128"
REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
REMOTE_STAGE = "outputs/llm_queue_threshold"
REQUESTS = 2048
SEED = 20260919
LOAD_GRID = (0.80, 0.90, 0.95, 0.98, 1.00, 1.03, 1.08)
CONFIG_IDS = ("qwen3-4b-a10", "qwen3-8b-l20", "qwen3-14b-l20", "qwen3-32b-h20")


def run(cmd, timeout=None):
    completed = subprocess.run(
        cmd, check=False, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    return completed.stdout


def main() -> int:
    result = ROOT / "results" / "llm_queue_threshold"
    result.mkdir(parents=True, exist_ok=True)
    capacity = pd.read_csv(ROOT / "results" / "llm_queue_steady" / "steady_capacity.csv")
    cap = dict(zip(capacity.config_id, capacity.steady_state_throughput_rps, strict=True))

    jobs = []
    for config_id in CONFIG_IDS:
        for load in LOAD_GRID:
            jobs.append(
                {
                    "job_id": base.job_id("thr", config_id, f"{load:.2f}"),
                    "kind": "threshold",
                    "config_id": config_id,
                    "composition": base.COMPOSITIONS["mixed"],
                    "composition_name": "mixed",
                    "num_requests": REQUESTS,
                    "arrival_rate_rps": load * float(cap[config_id]),
                    "load_factor": load,
                    "seed": SEED,
                    "simultaneous": False,
                }
            )

    staging_root = result / "staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / REMOTE_STAGE
    staging.mkdir(parents=True, exist_ok=True)
    base.REMOTE_STAGE = REMOTE_STAGE
    base.write_jobs(staging, jobs)
    for sub in ("configs", "workloads", "manifests"):
        target = result / sub
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staging / sub, target)
    run(["ssh", REMOTE, "mkdir", "-p", f"{REMOTE_REPO}/outputs"])
    run(["scp", "-q", "-r", str(staging), f"{REMOTE}:{REMOTE_REPO}/outputs/"])
    print(f"[threshold] {len(jobs)} jobs of {REQUESTS} requests", flush=True)
    run(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=28800)
    runs = result / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.", str(runs)])

    rows = []
    for item in jobs:
        frame = read_simulator_output(runs / f"{item['job_id']}.csv")
        arrivals = frame.arrival_ns.to_numpy(dtype=float) / 1e9
        ends = frame.end_s.to_numpy(dtype=float)
        gate = float(arrivals.min() + frame.service_s.mean())
        steady = frame[(arrivals >= gate) & (arrivals <= arrivals.max())]
        span = float(arrivals.max() - arrivals.min())
        half = float(arrivals.min() + 0.5 * span)
        first = frame[arrivals <= half]
        second = frame[arrivals > half]
        rows.append(
            {
                "config_id": item["config_id"],
                "load_factor": item["load_factor"],
                "offered_rps": item["arrival_rate_rps"],
                "capacity_rps": cap[item["config_id"]],
                "steady": span >= 10.0 * float(frame.service_s.mean()),
                "arrival_span_s": span,
                "mean_service_s": float(frame.service_s.mean()),
                "mean_wait_s": float(steady.waiting_s.mean()),
                "p95_wait_s": float(steady.waiting_s.quantile(0.95)),
                "max_wait_s": float(steady.waiting_s.max()),
                "mean_response_s": float(steady.response_s.mean()),
                "occupancy": float(item["arrival_rate_rps"] * frame.service_s.mean()) / 128.0,
                "wait_first_half_s": float(first.waiting_s.mean()),
                "wait_second_half_s": float(second.waiting_s.mean()),
                "achieved_rps": len(frame) / (float(ends.max()) - float(arrivals.min())),
                "n": int(len(frame)),
            }
        )
    table = pd.DataFrame(rows)
    table["wait_growth"] = table.wait_second_half_s / table.wait_first_half_s
    table.to_csv(result / "threshold_observations.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:10.4f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
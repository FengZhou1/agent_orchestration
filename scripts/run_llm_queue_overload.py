"""Run overload points and fit a low-dimensional average waiting model."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_llm_queue_trends as base  # noqa: E402
from agent_orch.validation.llmservingsim import read_simulator_output  # noqa: E402


REMOTE = "zf@192.168.234.128"
REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
REMOTE_STAGE = "outputs/llm_queue_overload"
CONTAINER = "servingsim_docker"
CONTAINER_REPO = "/app/LLMServingSim"
LOAD_FACTORS = (1.00, 1.05, 1.15, 1.30)
REQUESTS = 20
SEED = 20260913


def run(command: list[str], timeout: int | None = None) -> str:
    p = subprocess.run(command, check=False, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"failed: {' '.join(command)}\n{p.stdout[-3000:]}")
    return p.stdout


def main() -> None:
    result = ROOT / "results" / "llm_queue_overload"
    staging_root = result / "staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(ROOT / "results" / "llm_queue_trends" / "calibration.csv")
    capacity = dict(zip(calibration["config_id"], calibration["saturated_capacity_rps"], strict=True))

    # Point the imported job writer at the overload stage.
    base.REMOTE_STAGE = REMOTE_STAGE
    jobs = []
    for cfg in base.CONFIGS:
        for load in LOAD_FACTORS:
            jobs.append({
                "job_id": base.job_id("overload", cfg["id"], f"load{load:.2f}"),
                "kind": "overload",
                "config_id": cfg["id"],
                "composition": base.COMPOSITIONS["mixed"],
                "composition_name": "mixed",
                "num_requests": REQUESTS,
                "arrival_rate_rps": load * float(capacity[cfg["id"]]),
                "load_factor": load,
                "seed": SEED,
                "simultaneous": False,
            })
    staging = staging_root / REMOTE_STAGE
    staging.mkdir(parents=True, exist_ok=True)
    base.write_jobs(staging, jobs)
    for sub in ("configs", "workloads", "manifests"):
        target = result / sub
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staging / sub, target)
    # Run the staged jobs under the overload stage name.
    run(["ssh", REMOTE, "mkdir", "-p", f"{REMOTE_REPO}/outputs"])
    run(["scp", "-q", "-r", str(staging), f"{REMOTE}:{REMOTE_REPO}/outputs/"])
    run(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=7200)
    local_runs = result / "runs"
    local_runs.mkdir(parents=True, exist_ok=True)
    run(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.", str(local_runs)])

    rows = []
    for item in jobs:
        cfg = next(c for c in base.CONFIGS if c["id"] == item["config_id"])
        manifest = pd.read_csv(result / "manifests" / f"{item['job_id']}.csv")
        sim = read_simulator_output(result / "runs" / f"{item['job_id']}.csv")
        merged = sim.merge(manifest, on="request_id", validate="one_to_one").sort_values("end_s")
        sample = merged.iloc[int(math.floor(0.1 * len(merged))):int(math.ceil(0.9 * len(merged)))]
        rows.append({
            "job_id": item["job_id"],
            "config_id": cfg["id"],
            "model": cfg["model"],
            "gpu": cfg["gpu"],
            "tp": cfg["tp"],
            "composition": "mixed",
            "load_factor": float(item["load_factor"]),
            "arrival_rate_rps": float(item["arrival_rate_rps"]),
            "capacity_rps": float(capacity[cfg["id"]]),
            "observed_waiting_s": float(sample["waiting_s"].mean()),
            "observed_ttft_s": float(sample["ttft_s"].mean()),
            "observed_response_s": float(sample["response_s"].mean()),
            "observed_service_s": float(sample["service_s"].mean()),
            "n_requests": int(len(sample)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(result / "overload_observations.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
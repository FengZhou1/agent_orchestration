"""Steady-state saturated capacity and admission-wait measurements.

The earlier ``capacity_long`` / ``longstable`` traces delivered 128 requests in
less than one service time, so the completion stream was the drain of a single
admitted wave and ``saturated_throughput`` measured the arrival burst instead of
the service rate. This script keeps arrivals flowing for many service times and
measures throughput on the window [first arrival + S, last arrival].
"""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_llm_queue_trends as base  # noqa: E402
from agent_orch.validation.llmservingsim import (  # noqa: E402
    read_simulator_output,
    steady_state_throughput,
)

REMOTE = "zf@192.168.234.128"
REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
REMOTE_STAGE = "outputs/llm_queue_steady"
CAPACITY_REQUESTS = 2048
LOAD_REQUESTS = 1536
LOADS = (0.5, 0.7, 0.85, 0.95)
SEED = 20260918


def run(cmd, timeout=None):
    completed = subprocess.run(
        cmd, check=False, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    return completed.stdout


def stage_jobs(result: Path, jobs: list[dict]) -> Path:
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
    return staging


def fetch_runs(result: Path, jobs: list[dict]) -> None:
    local = result / "runs"
    local.mkdir(parents=True, exist_ok=True)
    run(["scp", "-q", "-r", f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.", str(local)])
    missing = [j["job_id"] for j in jobs if not (local / f"{j['job_id']}.csv").exists()]
    if missing:
        raise RuntimeError(f"missing runs: {missing[:5]}")


def main() -> int:
    global REMOTE_STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "llm_queue_steady",
        help="local result directory (default: results/llm_queue_steady)",
    )
    parser.add_argument(
        "--remote-stage", default=REMOTE_STAGE,
        help="remote outputs stage used for this run",
    )
    args = parser.parse_args()
    REMOTE_STAGE = args.remote_stage
    result = args.output
    result.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(ROOT / "results" / "llm_queue_trends" / "calibration.csv")
    naive = dict(zip(calibration.config_id, calibration.saturated_capacity_rps, strict=True))

    capacity_jobs = [
        {
            "job_id": base.job_id("steadycap", cfg["id"]),
            "kind": "capacity",
            "config_id": cfg["id"],
            "composition": base.COMPOSITIONS["mixed"],
            "composition_name": "mixed",
            "num_requests": CAPACITY_REQUESTS,
            "arrival_rate_rps": 8.0 * float(naive[cfg["id"]]),
            "load_factor": 8.0,
            "seed": SEED,
            "simultaneous": False,
        }
        for cfg in base.CONFIGS
    ]
    print(f"[capacity] {len(capacity_jobs)} jobs of {CAPACITY_REQUESTS} requests", flush=True)
    stage_jobs(result, capacity_jobs)
    run(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=28800)
    fetch_runs(result, capacity_jobs)

    capacity_rows = []
    for item in capacity_jobs:
        frame = read_simulator_output(result / "runs" / f"{item['job_id']}.csv")
        row = {"config_id": item["config_id"], "offered_rps": item["arrival_rate_rps"]}
        try:
            row.update(steady_state_throughput(frame))
            row["steady"] = True
        except ValueError as error:
            row["steady"] = False
            row["error"] = str(error)
        capacity_rows.append(row)
    capacity = pd.DataFrame(capacity_rows)
    capacity.to_csv(result / "steady_capacity.csv", index=False)
    print(capacity.to_string(index=False), flush=True)

    measured = dict(
        zip(
            capacity.loc[capacity["steady"], "config_id"],
            capacity.loc[capacity["steady"], "steady_state_throughput_rps"],
            strict=True,
        )
    )
    capacity_map = {cfg["id"]: measured.get(cfg["id"]) for cfg in base.CONFIGS}

    load_jobs = []
    for cfg in base.CONFIGS:
        cap = capacity_map[cfg["id"]]
        if cap is None:
            continue
        for load in LOADS:
            load_jobs.append(
                {
                    "job_id": base.job_id("steadyload", cfg["id"], f"{load:.2f}"),
                    "kind": "load",
                    "config_id": cfg["id"],
                    "composition": base.COMPOSITIONS["mixed"],
                    "composition_name": "mixed",
                    "num_requests": LOAD_REQUESTS,
                    "arrival_rate_rps": load * float(cap),
                    "load_factor": load,
                    "seed": SEED,
                    "simultaneous": False,
                }
            )
    print(f"[load] {len(load_jobs)} jobs of {LOAD_REQUESTS} requests", flush=True)
    stage_jobs(result, load_jobs)
    run(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=28800)
    fetch_runs(result, load_jobs)

    wait_rows = []
    for item in load_jobs:
        frame = read_simulator_output(result / "runs" / f"{item['job_id']}.csv")
        mean_service = float(frame["service_s"].mean())
        arrivals = frame["arrival_ns"].to_numpy(dtype=float) / 1e9
        span = float(arrivals.max() - arrivals.min())
        sample = frame[(arrivals >= arrivals.min() + mean_service) & (arrivals <= arrivals.max())]
        wait_rows.append(
            {
                "job_id": item["job_id"],
                "config_id": item["config_id"],
                "load_factor": item["load_factor"],
                "arrival_rate_rps": item["arrival_rate_rps"],
                "capacity_rps": capacity_map[item["config_id"]],
                "steady": span >= 4.0 * mean_service,
                "arrival_span_s": span,
                "mean_service_s": mean_service,
                "mean_waiting_s": float(sample["waiting_s"].mean()),
                "mean_first_waiting_s": float(sample["first_waiting_s"].mean()),
                "mean_response_s": float(sample["response_s"].mean()),
                "mean_ttft_s": float(sample["ttft_s"].mean()),
                "mean_tbt_s": float(sample["tbt_s"].mean()),
                "preemptions": int(frame["preemptions"].sum()) if "preemptions" in frame else 0,
                "n": int(len(sample)),
            }
        )
    waits = pd.DataFrame(wait_rows)
    waits.to_csv(result / "steady_wait_observations.csv", index=False)
    (result / "steady_capacity.json").write_text(
        json.dumps(capacity_map, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(waits.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

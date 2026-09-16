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
import math
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


def steady_window_metrics(frame: pd.DataFrame) -> dict[str, float | bool | str]:
    """Measure an open Poisson run without using the drain as steady state.

    The window is taken from the interior of the arrival stream.  Stability is
    determined from flow balance and the change in waiting time between the
    first and second half of the observation window.  A finite completion
    trace is therefore not called stable merely because all requests finish.
    """
    frame = frame.sort_values("arrival_ns").reset_index(drop=True)
    arrivals = frame["arrival_ns"].to_numpy(dtype=float) / 1e9
    ends = frame["end_s"].to_numpy(dtype=float)
    if len(frame) < 100:
        return {"steady": False, "stability_reason": "too_few_requests"}

    mean_service = float(frame["service_s"].mean())
    arrival_start = float(arrivals.min())
    arrival_end = float(arrivals.max())
    arrival_span = arrival_end - arrival_start
    if arrival_span <= 10.0 * max(mean_service, 1.0e-9):
        return {
            "steady": False,
            "stability_reason": "arrival_window_shorter_than_ten_service_times",
            "arrival_span_s": arrival_span,
            "mean_service_s": mean_service,
        }

    # Exclude initial filling and the final drain.  The observation interval
    # remains inside the active arrival stream, so it cannot be mistaken for a
    # finite-batch completion interval.
    warmup = max(10.0 * mean_service, 0.10 * arrival_span)
    drain = max(10.0 * mean_service, 0.10 * arrival_span)
    start = arrival_start + warmup
    stop = arrival_end - drain
    if stop <= start:
        return {
            "steady": False,
            "stability_reason": "empty_interior_window",
            "arrival_span_s": arrival_span,
            "mean_service_s": mean_service,
        }

    arrival_mask = (arrivals >= start) & (arrivals <= stop)
    completed_mask = (ends >= start) & (ends <= stop)
    n_arrivals = int(arrival_mask.sum())
    n_completed = int(completed_mask.sum())
    window = float(stop - start)
    input_rate = n_arrivals / window
    output_rate = n_completed / window

    waiting = frame["waiting_s"].to_numpy(dtype=float)
    in_window = frame.loc[arrival_mask, "waiting_s"].to_numpy(dtype=float)
    if len(in_window) >= 20:
        half = len(in_window) // 2
        first_wait = float(np.mean(in_window[:half]))
        second_wait = float(np.mean(in_window[half:]))
        wait_growth = second_wait - first_wait
        time_values = arrivals[arrival_mask]
        wait_slope = float(np.polyfit(time_values, in_window, 1)[0])
    else:
        first_wait = second_wait = wait_growth = wait_slope = float("nan")

    # Flow balance is the primary condition.  The waiting-time condition
    # prevents a long but still finite queue from being called stable.
    flow_ratio = output_rate / max(input_rate, 1.0e-12)
    stable_wait = (
        not np.isfinite(wait_growth)
        or wait_growth <= max(0.05 * mean_service, 0.02)
    )
    stable_flow = flow_ratio >= 0.95
    stable = bool(stable_flow and stable_wait)
    reason = "stable" if stable else (
        "output_below_input" if not stable_flow else "waiting_time_growing"
    )
    return {
        "steady": stable,
        "stability_reason": reason,
        "arrival_span_s": arrival_span,
        "window_start_s": start,
        "window_end_s": stop,
        "window_seconds": window,
        "input_requests": n_arrivals,
        "completed_requests": n_completed,
        "input_rate_rps": input_rate,
        "output_rate_rps": output_rate,
        "flow_ratio": flow_ratio,
        "mean_service_s": mean_service,
        "mean_waiting_s": float(np.mean(in_window)) if len(in_window) else float("nan"),
        "p95_waiting_s": float(np.quantile(in_window, 0.95)) if len(in_window) else float("nan"),
        "first_half_waiting_s": first_wait,
        "second_half_waiting_s": second_wait,
        "waiting_growth_s": wait_growth,
        "waiting_slope_s_per_s": wait_slope,
        "mean_response_s": float(frame.loc[completed_mask, "response_s"].mean()) if n_completed else float("nan"),
        "mean_ttft_s": float(frame.loc[completed_mask, "ttft_s"].mean()) if n_completed else float("nan"),
        "mean_tbt_s": float(frame.loc[completed_mask, "tbt_s"].mean()) if n_completed else float("nan"),
        "peak_concurrency": float(frame.get("concurrency", pd.Series(dtype=float)).max()) if "concurrency" in frame else float("nan"),
    }


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
    parser.add_argument(
        "--duration-s", type=float, default=1200.0,
        help="target arrival-stream duration for each open-system run",
    )
    parser.add_argument(
        "--capacity-factor", type=float, default=1.25,
        help="offered-rate multiplier over the previous capacity estimate",
    )
    parser.add_argument(
        "--min-requests", type=int, default=2048,
        help="minimum requests per capacity run",
    )
    parser.add_argument(
        "--config-ids", default="",
        help="comma-separated subset of configuration IDs; empty means all",
    )
    parser.add_argument(
        "--token-budget", type=int, default=None,
        help="override max_num_batched_tokens in the generated simulator configs",
    )
    parser.add_argument(
        "--capacity-only", action="store_true",
        help="stop after the long-run capacity measurement",
    )
    args = parser.parse_args()
    if args.token_budget is not None:
        if args.token_budget <= 0:
            raise SystemExit("--token-budget must be positive")
        base.SIM_MAX_NUM_BATCHED_TOKENS = int(args.token_budget)
    REMOTE_STAGE = args.remote_stage
    result = args.output
    result.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(ROOT / "results" / "llm_queue_trends" / "calibration.csv")
    naive = dict(zip(calibration.config_id, calibration.saturated_capacity_rps, strict=True))
    selected_ids = {
        value.strip() for value in args.config_ids.split(",") if value.strip()
    }
    configs = [
        cfg for cfg in base.CONFIGS
        if not selected_ids or cfg["id"] in selected_ids
    ]
    if not configs:
        raise SystemExit("--config-ids did not match any known configuration")

    capacity_jobs = []
    for cfg in configs:
        offered = args.capacity_factor * float(naive[cfg["id"]])
        requests = max(args.min_requests, int(math.ceil(offered * args.duration_s)))
        capacity_jobs.append(
            {
                "job_id": base.job_id("steadycap", cfg["id"]),
                "kind": "capacity",
                "config_id": cfg["id"],
                "composition": base.COMPOSITIONS["mixed"],
                "composition_name": "mixed",
                "num_requests": requests,
                "arrival_rate_rps": offered,
                "load_factor": args.capacity_factor,
                "seed": SEED,
                "simultaneous": False,
            }
        )
    print(
        f"[capacity] {len(capacity_jobs)} long open-system jobs, "
        f"target duration={args.duration_s:.0f}s, factor={args.capacity_factor:.2f}",
        flush=True,
    )
    stage_jobs(result, capacity_jobs)
    run(["ssh", REMOTE, "bash", f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"], timeout=28800)
    fetch_runs(result, capacity_jobs)

    capacity_rows = []
    for item in capacity_jobs:
        frame = read_simulator_output(result / "runs" / f"{item['job_id']}.csv")
        row = {
            "config_id": item["config_id"],
            "offered_rps": item["arrival_rate_rps"],
            "num_requests": item["num_requests"],
        }
        row.update(steady_window_metrics(frame))
        capacity_rows.append(row)
    capacity = pd.DataFrame(capacity_rows)
    capacity.to_csv(result / "steady_capacity.csv", index=False)
    print(capacity.to_string(index=False), flush=True)

    # A deliberately overloaded open run is used to expose the saturated
    # output rate.  It is a capacity estimate, not a stable operating point;
    # the stability column above remains the criterion for admissible loads.
    measured = dict(
        zip(
            capacity["config_id"],
            capacity["output_rate_rps"],
            strict=True,
        )
    )
    capacity_map = {cfg["id"]: measured.get(cfg["id"]) for cfg in base.CONFIGS}

    if args.capacity_only:
        print("[capacity] capacity-only run complete", flush=True)
        return 0

    load_jobs = []
    for cfg in configs:
        cap = capacity_map[cfg["id"]]
        if cap is None or not np.isfinite(cap) or cap <= 0.0:
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

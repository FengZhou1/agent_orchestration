from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from agent_orch.baselines import make_policy
from agent_orch.metrics import summarize_slot_metrics
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


def _parquet_record(metrics, record_level: str = "metrics") -> dict:
    record = asdict(metrics)
    if record_level != "full":
        record.pop("diagnostics", None)
    if record_level == "scalar":
        for key in (
            "app_latency_s",
            "llm_utilization",
            "tool_utilization",
            "link_utilization",
        ):
            record.pop(key, None)
    return {
        key: (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
        for key, value in record.items()
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_levels(path: str) -> list[tuple[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (str(level["name"]), float(level["rate_scale"]))
        for level in payload["levels"]
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"baseline:{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_one(
    scenario,
    seed: int,
    policy_name: str,
    slots: int,
    evaluation_steps: int,
    rate_scale: float,
    record_level: str,
    on_step=None,
) -> list[dict]:
    trace = ArrivalTrace.stationary_poisson_intensity(
        scenario, slots, rate_scale=rate_scale
    )
    policy = make_policy(policy_name, scenario, seed)
    deployment = policy.deployment()
    simulator = Simulator(scenario)
    simulator.set_arrival_trace(trace)
    simulator.reset(seed)
    records = []
    for slot in range(evaluation_steps):
        routing = policy.routing(deployment, simulator.last_metrics)
        metrics = simulator.step(deployment, routing).metrics
        records.append(
            {
                "scenario": scenario.id,
                "policy": policy_name,
                "seed": seed,
                "arrival_scale": rate_scale,
                **_parquet_record(metrics, record_level),
            }
        )
        if on_step is not None:
            on_step(slot + 1)
    if evaluation_steps < slots:
        if policy_name == "random" and evaluation_steps > 1:
            recurring_weight = (slots - 1.0) / (evaluation_steps - 1.0)
            for index, record in enumerate(records):
                record["slot_weight"] = 1.0 if index == 0 else recurring_weight
        else:
            for record in records:
                record["slot_weight"] = 1.0
            records[-1]["slot_weight"] += slots - evaluation_steps
    else:
        for record in records:
            record["slot_weight"] = 1.0
    return records


def _evaluation_steps(
    policy_name: str,
    slots: int,
    evaluation_mode: str,
    adaptive_steps: int,
    random_samples: int,
) -> int:
    if evaluation_mode == "full":
        return slots
    if policy_name in {"static", "equal", "greedy"}:
        return min(slots, 2)
    if policy_name == "least_load":
        return min(slots, max(2, adaptive_steps))
    if policy_name == "random":
        return min(slots, max(2, random_samples))
    return slots


def _write(records, output: Path, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    _atomic_parquet(frame, output)
    group_columns = ["scenario", "policy", "seed", "arrival_scale"]
    summary_rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                **summarize_slot_metrics(group.to_dict("records")),
            }
        )
    _atomic_parquet(
        pd.DataFrame(summary_rows),
        output.with_name(f"{output.stem}.summary.parquet"),
    )
    _atomic_json(
        output.with_name(f"{output.stem}.manifest.json"), manifest
    )
    print(output.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slots", type=int, default=600)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--policies", default="static,equal,least_load,random,greedy")
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the scenario arrival intensities",
    )
    parser.add_argument(
        "--load-levels",
        help="Load-level JSON emitted by calibrate_load_levels.py",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed seed-policy runs from the per-run cache",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--status-interval-slots", type=int, default=10)
    parser.add_argument(
        "--evaluation-mode",
        choices=("auto", "full"),
        default="auto",
        help=(
            "auto collapses repeated stationary evaluations; full evaluates every "
            "reported slot"
        ),
    )
    parser.add_argument(
        "--adaptive-steps",
        type=int,
        default=20,
        help="Sequential evaluations retained for the deterministic least-load policy",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=100,
        help="Monte Carlo routing samples per seed in auto mode",
    )
    parser.add_argument(
        "--record-level",
        choices=("scalar", "metrics", "full"),
        default="metrics",
        help="Per-evaluation detail written to Parquet",
    )
    parser.add_argument("--output", default="results/baseline_matrix.parquet")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = ScenarioLoader.load(scenario_path)
    seeds = [int(value) for value in _parse_csv_list(args.seeds)]
    policies = _parse_csv_list(args.policies)
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()[:16]
    levels = _load_levels(args.load_levels) if args.load_levels else [("", args.arrival_scale)]
    output_argument = Path(args.output).resolve()
    experiment_root = output_argument if args.load_levels else output_argument.parent
    experiment_root.mkdir(parents=True, exist_ok=True)
    logger = _logger(experiment_root / "baseline_experiment.log")
    status_path = experiment_root / "baseline_status.json"
    total_runs = len(levels) * len(seeds) * len(policies)
    evaluations_per_level = sum(
        _evaluation_steps(
            policy,
            args.slots,
            args.evaluation_mode,
            args.adaptive_steps,
            args.random_samples,
        )
        * (len(seeds) if policy == "random" else 1)
        for policy in policies
    )
    total_slots = len(levels) * evaluations_per_level
    total_reporting_slots = total_runs * args.slots
    completed_runs = 0
    completed_slots = 0
    started = time.perf_counter()
    progress = tqdm(
        total=total_slots,
        unit="eval",
        desc="baseline analytical evaluations",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    def write_status(status: str, current: str | None = None, error: str | None = None) -> None:
        elapsed = time.perf_counter() - started
        rate = completed_slots / elapsed if elapsed > 0.0 else 0.0
        payload = {
            "status": status,
            "current_run": current,
            "completed_runs": completed_runs,
            "total_runs": total_runs,
            "completed_slots": completed_slots,
            "total_slots": total_slots,
            "completed_evaluations": completed_slots,
            "total_evaluations": total_slots,
            "reporting_slots": total_reporting_slots,
            "progress_percent": 100.0 * completed_slots / max(total_slots, 1),
            "elapsed_time_s": elapsed,
            "eta_seconds": (total_slots - completed_slots) / rate if rate > 0 else None,
            "updated_at_utc": _timestamp(),
        }
        if error is not None:
            payload["error"] = error
        _atomic_json(status_path, payload)

    write_status("running")
    try:
        for name, rate_scale in levels:
            if args.load_levels:
                output = output_argument / f"baseline_load_{name}.parquet"
            else:
                output = output_argument
            cache = output.with_name(f"{output.stem}.runs")
            cache.mkdir(parents=True, exist_ok=True)
            level_records: list[dict] = []
            seed_invariant_records: dict[str, list[dict]] = {}
            for seed in seeds:
                for policy_name in policies:
                    evaluation_steps = _evaluation_steps(
                        policy_name,
                        args.slots,
                        args.evaluation_mode,
                        args.adaptive_steps,
                        args.random_samples,
                    )
                    run_id = (
                        f"{scenario_hash}-{name or 'fixed'}-{policy_name}-s{seed}"
                        f"-r{rate_scale:.10g}-{args.evaluation_mode}"
                        f"-a{args.adaptive_steps}-n{args.random_samples}"
                        f"-{args.record_level}"
                    )
                    run_path = cache / f"{run_id}.parquet"
                    if policy_name != "random" and policy_name in seed_invariant_records:
                        records = [
                            {**record, "seed": seed}
                            for record in seed_invariant_records[policy_name]
                        ]
                        _atomic_parquet(pd.DataFrame(records), run_path)
                        level_records.extend(records)
                        completed_runs += 1
                        logger.info(
                            "reused seed-invariant run %s from seed %s",
                            run_id,
                            seeds[0],
                        )
                        write_status("running", run_id)
                        continue
                    if args.resume and run_path.exists():
                        cached = pd.read_parquet(run_path)
                        cached_weight = float(
                            cached.get("slot_weight", pd.Series(np.ones(len(cached)))).sum()
                        )
                        if len(cached) == evaluation_steps and np.isclose(
                            cached_weight, args.slots
                        ):
                            records = cached.to_dict("records")
                            level_records.extend(records)
                            if policy_name != "random":
                                seed_invariant_records[policy_name] = records
                            completed_runs += 1
                            completed_slots += evaluation_steps
                            progress.update(evaluation_steps)
                            logger.info("resumed completed run %s", run_id)
                            write_status("running", run_id)
                            continue
                    logger.info("starting run %s", run_id)
                    run_started = time.perf_counter()
                    local_slots = 0

                    def on_step(value: int) -> None:
                        nonlocal completed_slots, local_slots
                        delta = value - local_slots
                        local_slots = value
                        completed_slots += delta
                        progress.update(delta)
                        if value % max(1, args.status_interval_slots) == 0:
                            write_status("running", run_id)

                    records = _run_one(
                        scenario,
                        seed,
                        policy_name,
                        args.slots,
                        evaluation_steps,
                        rate_scale,
                        args.record_level,
                        on_step,
                    )
                    _atomic_parquet(pd.DataFrame(records), run_path)
                    level_records.extend(records)
                    if policy_name != "random":
                        seed_invariant_records[policy_name] = records
                    completed_runs += 1
                    logger.info(
                        "completed run %s in %.3fs", run_id, time.perf_counter() - run_started
                    )
                    write_status("running", run_id)
            manifest = {
                "scenario": str(scenario_path),
                "scenario_hash": scenario_hash,
                "slots": args.slots,
                "evaluation_mode": args.evaluation_mode,
                "adaptive_steps": args.adaptive_steps,
                "random_samples": args.random_samples,
                "record_level": args.record_level,
                "seed_invariant_policies_reused": [
                    policy for policy in policies if policy != "random"
                ],
                "evaluated_rows": len(level_records),
                "represented_slots": float(
                    sum(record.get("slot_weight", 1.0) for record in level_records)
                ),
                "seeds": seeds,
                "policies": policies,
                "arrival_process": "stationary_intensity",
                "arrival_scale": rate_scale,
                "load_level": name or None,
                "load_levels_source": (
                    str(Path(args.load_levels).resolve()) if args.load_levels else None
                ),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "resume_enabled": args.resume,
                "completed_runs": len(seeds) * len(policies),
            }
            _write(level_records, output, manifest)
        write_status("completed")
        logger.info("completed baseline matrix in %.3fs", time.perf_counter() - started)
    except BaseException as exc:
        logger.exception("baseline matrix failed")
        write_status("failed", error=repr(exc))
        raise
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

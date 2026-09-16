from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agent_orch.validation.llmservingsim import (
    CALIBRATION_CLASSES,
    COMPOSITIONS,
    JITSERVE_CLASSES,
    AnalyticalParameters,
    ServiceMetrics,
    WorkloadClass,
    allen_cunneen_prediction,
    calibrate_effective_concurrency,
    fit_effective_rates,
    generate_poisson_trace,
    normalize_composition,
    read_run_specs,
    read_simulator_output,
    roofline_service,
    saturated_throughput,
    write_run_specs,
)


DEFAULT_OUTPUT = Path("results/llm_queue_validation")
DEFAULT_HOST = "zf@192.168.234.128"
DEFAULT_REMOTE_REPO = "/home/zf/桌面/LLMServingSim"
DEFAULT_CONTAINER_REPO = "/app/LLMServingSim"
DEFAULT_CONTAINER = "servingsim_docker"
CLUSTER_CONFIG = "configs/cluster/rtx4090_single_instance.json"

SITE_CUSTOMIZE = '''"""Metric-only instrumentation for LLMServingSim queue validation."""
from serving.core.request import Request

_original_set_que_delay = Request.set_que_delay

def _set_first_admission_delay(self, current):
    if self.queuing_delay < 0:
        _original_set_que_delay(self, current)

Request.set_que_delay = _set_first_admission_delay
'''

_PRINT_LOCK = threading.Lock()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(args)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def _remote_command(host: str, command: list[str]) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", host, *command]


def _prepare_layout(output: Path) -> None:
    for name in ("workloads", "manifests", "instrument", "runs", "logs", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    site = output / "instrument" / "sitecustomize.py"
    site.write_text(SITE_CUSTOMIZE, encoding="utf-8")


def prepare_calibration(output: Path) -> None:
    _prepare_layout(output)
    specs: list[dict[str, Any]] = []
    all_classes = {**CALIBRATION_CLASSES, **JITSERVE_CLASSES}
    for class_id, workload in all_classes.items():
        run_id = f"service-{class_id.lower()}"
        workload_path = output / "workloads" / f"{run_id}.jsonl"
        manifest_path = output / "manifests" / f"{run_id}.csv"
        generate_poisson_trace(
            workload_path,
            manifest_path,
            {class_id: workload},
            {class_id: 1.0},
            num_requests=1,
            arrival_rate_rps=None,
            seed=2026,
            simultaneous=True,
        )
        specs.append(
            {
                "stage": "calibration",
                "kind": "service",
                "run_id": run_id,
                "workload": workload_path.name,
                "manifest": manifest_path.name,
                "class_id": class_id,
                "num_requests": 1,
            }
        )

    for composition_id, composition in COMPOSITIONS.items():
        num_requests = 80 if composition_id == "AGENT" else 300
        run_id = f"capacity-{composition_id.lower()}"
        workload_path = output / "workloads" / f"{run_id}.jsonl"
        manifest_path = output / "manifests" / f"{run_id}.csv"
        generate_poisson_trace(
            workload_path,
            manifest_path,
            JITSERVE_CLASSES,
            composition,
            num_requests=num_requests,
            arrival_rate_rps=None,
            seed=2026,
            simultaneous=True,
        )
        specs.append(
            {
                "stage": "calibration",
                "kind": "capacity",
                "run_id": run_id,
                "workload": workload_path.name,
                "manifest": manifest_path.name,
                "composition_id": composition_id,
                "num_requests": num_requests,
            }
        )

    write_run_specs(output / "calibration_runs.jsonl", specs)
    _write_json(
        output / "experiment_config.json",
        {
            "model": "meta-llama/Llama-3.1-8B",
            "hardware": "RTX4090",
            "dtype": "bfloat16",
            "cluster_config": CLUSTER_CONFIG,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 256,
            "long_prefill_token_threshold": 512,
            "block_size": 16,
            "chunked_prefill": True,
            "prefix_caching": False,
            "calibration_classes": {k: asdict(v) for k, v in CALIBRATION_CLASSES.items()},
            "jitserve_classes": {k: asdict(v) for k, v in JITSERVE_CLASSES.items()},
            "compositions": COMPOSITIONS,
            "instrumentation_sha256": _sha256(output / "instrument" / "sitecustomize.py"),
        },
    )
    print(f"Prepared calibration inputs in {output.resolve()}")


def _sync_inputs(
    output: Path,
    host: str,
    remote_repo: str,
) -> None:
    remote_base = f"{remote_repo}/outputs/queue_validation"
    _run_checked(_remote_command(host, ["mkdir", "-p", remote_base]))
    for directory in ("workloads", "instrument"):
        source = output / directory
        _run_checked(["scp", "-q", "-r", str(source), f"{host}:{remote_base}/"])


def collect_provenance(
    output: Path,
    host: str,
    remote_repo: str,
    container: str,
    container_repo: str,
) -> None:
    commit = _run_checked(
        _remote_command(host, ["git", "-C", remote_repo, "rev-parse", "HEAD"])
    ).strip()
    profile = _run_checked(
        _remote_command(
            host,
            [
                "docker",
                "exec",
                container,
                "cat",
                f"{container_repo}/profiler/perf/RTX4090/meta-llama/Llama-3.1-8B/bf16/meta.yaml",
            ],
        )
    )
    cluster = _run_checked(
        _remote_command(
            host,
            ["docker", "exec", container, "cat", f"{container_repo}/{CLUSTER_CONFIG}"],
        )
    )
    (output / "profile_meta.yaml").write_text(profile, encoding="utf-8")
    (output / "cluster_config.json").write_text(cluster, encoding="utf-8")
    _write_json(
        output / "provenance.json",
        {
            "simulator_commit": commit,
            "host": host,
            "container": container,
            "container_repo": container_repo,
            "profile_meta_sha256": hashlib.sha256(profile.encode()).hexdigest(),
            "cluster_config_sha256": hashlib.sha256(cluster.encode()).hexdigest(),
            "instrumentation_sha256": _sha256(output / "instrument" / "sitecustomize.py"),
            "collected_at_unix_s": time.time(),
        },
    )


def run_sanity(
    output: Path,
    host: str,
    container: str,
    container_repo: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    command = _remote_command(
        host,
        [
            "docker",
            "exec",
            "-w",
            container_repo,
            container,
            "bash",
            "serving/validate.sh",
            "--clocks-only",
            "rtx4090_single",
        ],
    )
    log = _run_checked(command)
    (output / "sanity.log").write_text(log, encoding="utf-8")
    summary = _run_checked(
        _remote_command(
            host,
            [
                "docker",
                "exec",
                container,
                "cat",
                f"{container_repo}/bench/examples/RTX4090/Llama-3.1-8B/validation/summary.txt",
            ],
        )
    )
    (output / "reference_accuracy_summary.txt").write_text(summary, encoding="utf-8")
    if "PASS" not in log:
        raise RuntimeError("LLMServingSim RTX4090 behavior validation did not pass")


def _run_one(
    spec: dict[str, Any],
    output: Path,
    host: str,
    remote_repo: str,
    container: str,
    container_repo: str,
    force: bool,
) -> dict[str, Any]:
    run_id = str(spec["run_id"])
    local_csv = output / "runs" / f"{run_id}.csv"
    local_log = output / "logs" / f"{run_id}.log"
    if local_csv.exists() and not force:
        return {"run_id": run_id, "status": "skipped", "elapsed_s": 0.0}
    remote_csv = f"{remote_repo}/outputs/queue_validation/runs/{run_id}.csv"
    container_csv = f"outputs/queue_validation/runs/{run_id}.csv"
    # LLMServingSim's router resolves datasets relative to its astra-sim
    # working directory by prepending "../". Pass a repository-relative path.
    container_workload = f"outputs/queue_validation/workloads/{spec['workload']}"
    command = _simulator_command(
        host=host,
        container=container,
        container_repo=container_repo,
        container_workload=container_workload,
        container_csv=container_csv,
        num_requests=int(spec["num_requests"]),
        run_id=f"qv-{run_id}",
        instrument=True,
    )
    started = time.time()
    local_log.parent.mkdir(parents=True, exist_ok=True)
    with local_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            if "ERROR" in line or "Traceback" in line:
                with _PRINT_LOCK:
                    print(f"[{run_id}] {line.rstrip()}", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{run_id} failed; see {local_log}")
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(["scp", "-q", f"{host}:{remote_csv}", str(local_csv)])
    elapsed = time.time() - started
    with _PRINT_LOCK:
        print(f"[{run_id}] completed in {elapsed:.1f}s", flush=True)
    return {"run_id": run_id, "status": "completed", "elapsed_s": elapsed}


def _simulator_command(
    host: str,
    container: str,
    container_repo: str,
    container_workload: str,
    container_csv: str,
    num_requests: int,
    run_id: str,
    instrument: bool,
) -> list[str]:
    docker_args = ["docker", "exec"]
    if instrument:
        container_base = f"{container_repo}/outputs/queue_validation"
        docker_args.extend(
            ["-e", f"PYTHONPATH={container_base}/instrument:{container_repo}"]
        )
    docker_args.extend(
        [
            "-w",
            container_repo,
            container,
            "python",
            "-m",
            "serving",
            "--cluster-config",
            CLUSTER_CONFIG,
            "--dtype",
            "bfloat16",
            "--block-size",
            "16",
            "--max-num-batched-tokens",
            "2048",
            "--max-num-seqs",
            "256",
            "--long-prefill-token-threshold",
            "512",
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--dataset",
            container_workload,
            "--num-reqs",
            str(num_requests),
            "--output",
            container_csv,
            "--run-id",
            run_id,
            "--log-level",
            "WARNING",
        ]
    )
    return _remote_command(host, docker_args)


def run_instrumentation_check(
    output: Path,
    host: str,
    remote_repo: str,
    container: str,
    container_repo: str,
) -> None:
    """Verify that first-admission instrumentation changes metrics, not scheduling."""
    _prepare_layout(output)
    workload_name = "instrument-probe.jsonl"
    generate_poisson_trace(
        output / "workloads" / workload_name,
        output / "manifests" / "instrument-probe.csv",
        {"probe": WorkloadClass("probe", 2048, 128)},
        {"probe": 1.0},
        num_requests=1,
        arrival_rate_rps=None,
        seed=2026,
        simultaneous=True,
    )
    _sync_inputs(output, host, remote_repo)
    remote_runs = f"{remote_repo}/outputs/queue_validation/runs"
    _run_checked(_remote_command(host, ["mkdir", "-p", remote_runs]))
    local_paths: dict[str, Path] = {}
    for label, instrument in (("raw", False), ("first_admission", True)):
        filename = f"instrument-{label}.csv"
        command = _simulator_command(
            host=host,
            container=container,
            container_repo=container_repo,
            container_workload=f"outputs/queue_validation/workloads/{workload_name}",
            container_csv=f"outputs/queue_validation/runs/{filename}",
            num_requests=1,
            run_id=f"qv-instrument-{label}",
            instrument=instrument,
        )
        _run_checked(command)
        local_path = output / "instrument" / filename
        _run_checked(["scp", "-q", f"{host}:{remote_runs}/{filename}", str(local_path)])
        local_paths[label] = local_path

    raw = pd.read_csv(local_paths["raw"])
    patched = pd.read_csv(local_paths["first_admission"])
    behavior_columns = (
        "request id",
        "arrival",
        "end_time",
        "latency",
        "TTFT",
        "TPOT",
        "ITL",
    )
    unchanged = {
        column: bool(raw[column].equals(patched[column])) for column in behavior_columns
    }
    result = {
        "behavior_columns_unchanged": unchanged,
        "scheduling_behavior_unchanged": bool(all(unchanged.values())),
        "raw_queuing_delay_ns": float(raw.loc[0, "queuing_delay"]),
        "first_admission_delay_ns": float(patched.loc[0, "queuing_delay"]),
        "queue_metric_changed": bool(
            raw.loc[0, "queuing_delay"] != patched.loc[0, "queuing_delay"]
        ),
    }
    _write_json(output / "instrumentation_check.json", result)
    if not result["scheduling_behavior_unchanged"]:
        raise RuntimeError("queue instrumentation changed simulator scheduling behavior")


def run_stage(
    output: Path,
    stage: str,
    host: str,
    remote_repo: str,
    container: str,
    container_repo: str,
    workers: int,
    force: bool,
) -> None:
    spec_file = output / ("calibration_runs.jsonl" if stage == "calibration" else "queue_runs.jsonl")
    if not spec_file.exists():
        raise FileNotFoundError(f"missing run specification: {spec_file}")
    _sync_inputs(output, host, remote_repo)
    _run_checked(
        _remote_command(
            host,
            ["mkdir", "-p", f"{remote_repo}/outputs/queue_validation/runs"],
        )
    )
    specs = [spec for spec in read_run_specs(spec_file) if spec["stage"] == stage]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as executor:
        futures = {
            executor.submit(
                _run_one,
                spec,
                output,
                host,
                remote_repo,
                container,
                container_repo,
                force,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                spec = futures[future]
                records.append({"run_id": spec["run_id"], "status": "failed", "error": str(exc)})
                _write_json(output / f"{stage}_status.json", records)
                raise
            _write_json(output / f"{stage}_status.json", records)


def _load_service_observations(output: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in read_run_specs(output / "calibration_runs.jsonl"):
        if spec["kind"] != "service":
            continue
        frame = read_simulator_output(output / "runs" / f"{spec['run_id']}.csv")
        if len(frame) != 1:
            raise ValueError(f"service run {spec['run_id']} did not produce exactly one request")
        workload = {**CALIBRATION_CLASSES, **JITSERVE_CLASSES}[str(spec["class_id"])]
        row = frame.iloc[0]
        records.append(
            {
                "class_id": spec["class_id"],
                "prompt_tokens": workload.prompt_tokens,
                "output_tokens": workload.output_tokens,
                "prefill_s": float(row.prefill_s),
                "decode_s": float(row.decode_s),
                "service_s": float(row.service_s),
                "tbt_s": float(row.tbt_s),
                "waiting_s": float(row.waiting_s),
            }
        )
    return pd.DataFrame.from_records(records)


def _realized_composition(manifest: pd.DataFrame) -> dict[str, float]:
    frequencies = manifest["class_id"].value_counts(normalize=True)
    return {str(key): float(value) for key, value in frequencies.items()}


def calibrate(output: Path) -> None:
    observations = _load_service_observations(output)
    calibration = observations[observations["class_id"].str.startswith("CAL_")].copy()
    fitted, fit_info = fit_effective_rates(calibration)

    service_predictions: list[dict[str, Any]] = []
    for row in observations.itertuples(index=False):
        workload = WorkloadClass(str(row.class_id), int(row.prompt_tokens), int(row.output_tokens))
        predicted = roofline_service(workload, fitted, concurrency=1)
        record = row._asdict()
        record["split"] = "calibration" if str(row.class_id).startswith("CAL_") else "test"
        for metric in ("prefill_s", "decode_s", "service_s", "tbt_s"):
            predicted_value = float(getattr(predicted, metric))
            observed_value = float(record[metric])
            record[f"predicted_{metric}"] = predicted_value
            record[f"ape_{metric}"] = abs(predicted_value - observed_value) / max(observed_value, 1e-12)
        service_predictions.append(record)
    pd.DataFrame(service_predictions).to_csv(output / "service_predictions.csv", index=False)

    capacities: list[dict[str, Any]] = []
    realized: dict[str, dict[str, float]] = {}
    for spec in read_run_specs(output / "calibration_runs.jsonl"):
        if spec["kind"] != "capacity":
            continue
        run_id = str(spec["run_id"])
        sim = read_simulator_output(output / "runs" / f"{run_id}.csv")
        manifest = pd.read_csv(output / "manifests" / str(spec["manifest"]))
        composition_id = str(spec["composition_id"])
        realized[composition_id] = _realized_composition(manifest)
        capacities.append(
            {
                "composition_id": composition_id,
                "num_requests": len(sim),
                "saturated_capacity_rps": saturated_throughput(sim),
            }
        )
    capacity_frame = pd.DataFrame(capacities)
    capacity_frame.to_csv(output / "capacity_observations.csv", index=False)
    mix_capacity = float(
        capacity_frame.loc[capacity_frame["composition_id"] == "MIX", "saturated_capacity_rps"].iloc[0]
    )
    effective_concurrency, candidates = calibrate_effective_concurrency(
        mix_capacity,
        realized["MIX"],
        JITSERVE_CLASSES,
        fitted,
        maximum=128,
    )
    candidates.to_csv(output / "effective_concurrency_candidates.csv", index=False)
    empirical_services = {
        str(row.class_id): {
            "prefill_s": float(row.prefill_s),
            "decode_s": float(row.decode_s),
            "service_s": float(row.service_s),
            "tbt_s": float(row.tbt_s),
        }
        for row in observations.itertuples(index=False)
        if row.class_id in JITSERVE_CLASSES
    }
    calibration_result = {
        "analytical_parameters": asdict(fitted),
        "fit": fit_info,
        "effective_concurrency": effective_concurrency,
        "capacities_rps": {
            str(row.composition_id): float(row.saturated_capacity_rps)
            for row in capacity_frame.itertuples(index=False)
        },
        "realized_capacity_compositions": realized,
        "empirical_services": empirical_services,
    }
    _write_json(output / "calibration.json", calibration_result)
    prepare_queue_runs(output, calibration_result)


def prepare_queue_runs(output: Path, calibration: dict[str, Any]) -> None:
    specs: list[dict[str, Any]] = []
    capacities = calibration["capacities_rps"]
    for composition_id in ("CS", "DS", "MIX"):
        for load_factor in (0.2, 0.4, 0.6, 0.75, 0.85, 0.95):
            for seed in range(5):
                run_id = f"queue-{composition_id.lower()}-l{str(load_factor).replace('.', 'p')}-s{seed}"
                workload_path = output / "workloads" / f"{run_id}.jsonl"
                manifest_path = output / "manifests" / f"{run_id}.csv"
                arrival_rate = load_factor * float(capacities[composition_id])
                generate_poisson_trace(
                    workload_path,
                    manifest_path,
                    JITSERVE_CLASSES,
                    COMPOSITIONS[composition_id],
                    num_requests=200,
                    arrival_rate_rps=arrival_rate,
                    seed=seed,
                )
                specs.append(
                    {
                        "stage": "queue",
                        "kind": "queue",
                        "run_id": run_id,
                        "workload": workload_path.name,
                        "manifest": manifest_path.name,
                        "composition_id": composition_id,
                        "load_factor": load_factor,
                        "arrival_rate_rps": arrival_rate,
                        "seed": seed,
                        "num_requests": 200,
                    }
                )
    composition_id = "AGENT"
    for load_factor in (0.2, 0.6, 0.85, 1.05):
        for seed in range(3):
            run_id = f"queue-agent-l{str(load_factor).replace('.', 'p')}-s{seed}"
            workload_path = output / "workloads" / f"{run_id}.jsonl"
            manifest_path = output / "manifests" / f"{run_id}.csv"
            arrival_rate = load_factor * float(capacities[composition_id])
            generate_poisson_trace(
                workload_path,
                manifest_path,
                JITSERVE_CLASSES,
                COMPOSITIONS[composition_id],
                num_requests=80,
                arrival_rate_rps=arrival_rate,
                seed=seed,
            )
            specs.append(
                {
                    "stage": "queue",
                    "kind": "queue",
                    "run_id": run_id,
                    "workload": workload_path.name,
                    "manifest": manifest_path.name,
                    "composition_id": composition_id,
                    "load_factor": load_factor,
                    "arrival_rate_rps": arrival_rate,
                    "seed": seed,
                    "num_requests": 80,
                }
            )
    write_run_specs(output / "queue_runs.jsonl", specs)
    print(f"Prepared {len(specs)} queue-validation runs")


def _service_from_dict(value: dict[str, float]) -> ServiceMetrics:
    return ServiceMetrics(
        prefill_s=float(value["prefill_s"]),
        decode_s=float(value["decode_s"]),
        service_s=float(value["service_s"]),
        tbt_s=float(value["tbt_s"]),
    )


def _capacity_matched_concurrency(
    capacity_rps: float,
    composition: dict[str, float],
    services: dict[str, ServiceMetrics],
) -> int:
    weights = normalize_composition(composition)
    mean_service = sum(weights[key] * services[key].service_s for key in weights)
    candidates = range(1, 129)
    return min(
        candidates,
        key=lambda concurrency: abs(concurrency / mean_service - capacity_rps),
    )


def _is_nondecreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(left <= right for left, right in zip(sequence, sequence[1:]))


def _trim(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("request_id").reset_index(drop=True)
    lo = int(math.floor(0.2 * len(ordered)))
    hi = int(math.ceil(0.9 * len(ordered)))
    return ordered.iloc[lo:hi].copy()


def _bootstrap_ci(values: Iterable[float], seed: int = 2026) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=float)
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(data, size=(2000, len(data)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(output: Path) -> None:
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    params = AnalyticalParameters(**calibration["analytical_parameters"])
    fixed_b = int(calibration["effective_concurrency"])
    empirical = {
        key: _service_from_dict(value) for key, value in calibration["empirical_services"].items()
    }
    records: list[dict[str, Any]] = []
    integrity_records: list[dict[str, Any]] = []
    for spec in read_run_specs(output / "queue_runs.jsonl"):
        csv_path = output / "runs" / f"{spec['run_id']}.csv"
        if not csv_path.exists():
            continue
        simulator = read_simulator_output(csv_path)
        metric_columns = ("waiting_s", "ttft_s", "tbt_s", "response_s", "end_s")
        integrity_records.append(
            {
                "run_id": spec["run_id"],
                "expected_requests": int(spec["num_requests"]),
                "completed_requests": len(simulator),
                "request_count_matches": len(simulator) == int(spec["num_requests"]),
                "has_nan": bool(simulator[list(metric_columns)].isna().any().any()),
            }
        )
        manifest = pd.read_csv(output / "manifests" / str(spec["manifest"]))
        merged = simulator.merge(manifest, on="request_id", validate="one_to_one")
        sample = _trim(merged)
        composition = _realized_composition(sample)
        current_services = {
            key: roofline_service(JITSERVE_CLASSES[key], params, fixed_b) for key in composition
        }
        empirical_services = {key: empirical[key] for key in composition}
        capacity = float(calibration["capacities_rps"][str(spec["composition_id"])])
        matched_b = _capacity_matched_concurrency(capacity, composition, empirical_services)
        variants = {
            "Current": (current_services, fixed_b),
            "Empirical-Service": (empirical_services, fixed_b),
            "Capacity-Matched": (empirical_services, matched_b),
        }
        observed = {
            "waiting_s": float(sample["waiting_s"].mean()),
            "ttft_s": float(sample["ttft_s"].mean()),
            "tbt_s": float(sample["tbt_s"].mean()),
            "response_s": float(sample["response_s"].mean()),
        }
        for variant, (services, concurrency) in variants.items():
            prediction = allen_cunneen_prediction(
                float(spec["arrival_rate_rps"]), composition, services, concurrency
            )
            record: dict[str, Any] = {
                "run_id": spec["run_id"],
                "composition_id": spec["composition_id"],
                "load_factor": float(spec["load_factor"]),
                "arrival_rate_rps": float(spec["arrival_rate_rps"]),
                "seed": int(spec["seed"]),
                "variant": variant,
                "effective_concurrency": concurrency,
                "predicted_utilization": prediction.utilization,
                "predicted_overloaded": prediction.overloaded,
            }
            for metric, observed_value in observed.items():
                predicted_value = float(getattr(prediction, metric))
                record[f"observed_{metric}"] = observed_value
                record[f"predicted_{metric}"] = predicted_value
                record[f"absolute_error_{metric}"] = abs(predicted_value - observed_value)
            records.append(record)
    predictions = pd.DataFrame.from_records(records)
    if predictions.empty:
        raise RuntimeError("no completed queue runs were found")
    predictions.to_csv(output / "queue_predictions.csv", index=False)
    integrity = pd.DataFrame.from_records(integrity_records)
    integrity.to_csv(output / "run_integrity.csv", index=False)

    capacity_rows: list[dict[str, Any]] = []
    for composition_id, observed_capacity in calibration["capacities_rps"].items():
        composition = calibration["realized_capacity_compositions"][composition_id]
        weights = normalize_composition(composition)
        current_services = {
            key: roofline_service(JITSERVE_CLASSES[key], params, fixed_b) for key in weights
        }
        empirical_services = {key: empirical[key] for key in weights}
        matched_b = _capacity_matched_concurrency(
            float(observed_capacity), composition, empirical_services
        )
        for variant, services, concurrency in (
            ("Current", current_services, fixed_b),
            ("Empirical-Service", empirical_services, fixed_b),
            ("Capacity-Matched", empirical_services, matched_b),
        ):
            mean_service = sum(
                weights[key] * services[key].service_s for key in weights
            )
            predicted_capacity = concurrency / mean_service
            capacity_rows.append(
                {
                    "composition_id": composition_id,
                    "variant": variant,
                    "effective_concurrency": concurrency,
                    "observed_capacity_rps": float(observed_capacity),
                    "predicted_capacity_rps": predicted_capacity,
                    "relative_error_pct": 100.0
                    * abs(predicted_capacity - float(observed_capacity))
                    / float(observed_capacity),
                }
            )
    capacity_predictions = pd.DataFrame.from_records(capacity_rows)
    capacity_predictions.to_csv(output / "capacity_predictions.csv", index=False)

    metrics = ("waiting_s", "ttft_s", "tbt_s", "response_s")
    summary_rows: list[dict[str, Any]] = []
    stable = predictions[predictions["load_factor"] < 1.0]
    for (variant, region), group in stable.assign(
        region=np.where(stable["load_factor"] <= 0.8, "rho_le_0p8", "rho_gt_0p8")
    ).groupby(["variant", "region"]):
        for metric in metrics:
            observed = group[f"observed_{metric}"].to_numpy(dtype=float)
            predicted = group[f"predicted_{metric}"].to_numpy(dtype=float)
            finite = np.isfinite(predicted)
            observed = observed[finite]
            predicted = predicted[finite]
            response_scale = float(
                group.loc[finite, "observed_response_s"].mean()
            )
            mae = float(np.mean(np.abs(predicted - observed)))
            summary_rows.append(
                {
                    "variant": variant,
                    "region": region,
                    "metric": metric,
                    "n": len(observed),
                    "total_runs": len(group),
                    "finite_coverage_pct": 100.0 * len(observed) / len(group),
                    "observed_mean_s": float(np.mean(observed)),
                    "predicted_mean_s": float(np.mean(predicted)),
                    "mae": mae,
                    "mae_ms": 1000.0 * mae,
                    "mae_over_response_pct": 100.0 * mae / max(response_scale, 1e-12),
                    "wape_pct": 100.0
                    * float(np.sum(np.abs(predicted - observed)) / max(np.sum(np.abs(observed)), 1e-12)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "queue_error_summary.csv", index=False)

    aggregate_rows: list[dict[str, Any]] = []
    observed_once = predictions[predictions["variant"] == "Current"]
    for (composition, load), group in observed_once.groupby(["composition_id", "load_factor"]):
        for metric in metrics:
            values = group[f"observed_{metric}"].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values)
            aggregate_rows.append(
                {
                    "composition_id": composition,
                    "load_factor": load,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "seeds": len(values),
                }
            )
    pd.DataFrame(aggregate_rows).to_csv(output / "observed_confidence_intervals.csv", index=False)
    monotonicity_rows: list[dict[str, Any]] = []
    current = predictions[predictions["variant"] == "Current"]
    for composition_id, group in current.groupby("composition_id"):
        for metric in ("waiting_s", "ttft_s", "response_s"):
            for source in ("observed", "predicted"):
                curve = (
                    group.groupby("load_factor")[f"{source}_{metric}"]
                    .mean()
                    .sort_index()
                )
                values = curve.to_numpy(dtype=float)
                finite = np.isfinite(values)
                correlation = (
                    float(pd.Series(curve.index.to_numpy()[finite]).corr(
                        pd.Series(values[finite]), method="spearman"
                    ))
                    if finite.sum() >= 2
                    else math.nan
                )
                monotonicity_rows.append(
                    {
                        "composition_id": composition_id,
                        "metric": metric,
                        "source": source,
                        "nondecreasing": _is_nondecreasing(values),
                        "spearman_rho": correlation,
                    }
                )
    monotonicity = pd.DataFrame.from_records(monotonicity_rows)
    monotonicity.to_csv(output / "monotonicity.csv", index=False)
    _plot_results(output, predictions)
    _write_validity(output, summary, capacity_predictions, monotonicity, integrity)
    _write_validation_report(output)


def _plot_results(output: Path, predictions: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
        }
    )
    service = pd.read_csv(output / "service_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for axis, metric, label in zip(
        axes,
        ("prefill_s", "decode_s", "service_s"),
        ("Prefill", "Decode", "Complete service"),
        strict=True,
    ):
        axis.scatter(service[metric], service[f"predicted_{metric}"], c=np.where(service["split"] == "test", "C1", "C0"))
        maximum = max(service[metric].max(), service[f"predicted_{metric}"].max())
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=0.8)
        axis.set_xlabel("LLMServingSim (s)")
        axis.set_ylabel("Analytical (s)")
        axis.set_title(label)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"service_scatter.{suffix}", bbox_inches="tight")
    plt.close(fig)

    colors = {"Current": "C0", "Empirical-Service": "C1", "Capacity-Matched": "C2"}
    for metric, title in (
        ("waiting_s", "Mean queueing delay"),
        ("ttft_s", "Mean TTFT"),
        ("tbt_s", "Mean TBT"),
        ("response_s", "Mean response time"),
    ):
        compositions = list(dict.fromkeys(predictions["composition_id"]))
        fig, axes = plt.subplots(1, len(compositions), figsize=(3.0 * len(compositions), 2.7), squeeze=False)
        for axis, composition in zip(axes[0], compositions, strict=True):
            subset = predictions[predictions["composition_id"] == composition]
            observed = subset[subset["variant"] == "Current"].groupby("load_factor")[f"observed_{metric}"].mean()
            axis.plot(observed.index, observed.values, "ko-", label="LLMServingSim", linewidth=1.2, markersize=3)
            for variant, group in subset.groupby("variant"):
                predicted = group.groupby("load_factor")[f"predicted_{metric}"].mean()
                axis.plot(predicted.index, predicted.values, "--", color=colors[variant], label=variant, linewidth=1.0)
            axis.set_title(composition)
            axis.set_xlabel("Offered load / simulated capacity")
            axis.set_ylabel("Seconds")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / "figures" / f"{metric}_curves.{suffix}", bbox_inches="tight")
        plt.close(fig)

    capacity = pd.read_csv(output / "capacity_predictions.csv")
    compositions = list(dict.fromkeys(capacity["composition_id"]))
    positions = np.arange(len(compositions), dtype=float)
    width = 0.2
    observed = (
        capacity[capacity["variant"] == "Current"]
        .set_index("composition_id")
        .loc[compositions, "observed_capacity_rps"]
    )
    fig, axis = plt.subplots(figsize=(5.2, 2.8))
    axis.bar(positions - 1.5 * width, observed, width, label="LLMServingSim", color="k")
    for offset, variant in enumerate(colors):
        values = (
            capacity[capacity["variant"] == variant]
            .set_index("composition_id")
            .loc[compositions, "predicted_capacity_rps"]
        )
        axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=variant,
            color=colors[variant],
        )
    axis.set_xticks(positions, compositions)
    axis.set_yscale("log")
    axis.set_ylabel("Saturated throughput (request/s, log scale)")
    axis.set_xlabel("Workload composition")
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"capacity_by_composition.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _write_validity(
    output: Path,
    summary: pd.DataFrame,
    capacity: pd.DataFrame,
    monotonicity: pd.DataFrame,
    integrity: pd.DataFrame,
) -> None:
    service = pd.read_csv(output / "service_predictions.csv")
    test = service[service["split"] == "test"]
    service_checks = {}
    for metric in ("prefill_s", "decode_s", "service_s", "tbt_s"):
        ape = 100.0 * test[f"ape_{metric}"].to_numpy(dtype=float)
        service_checks[metric] = {
            "median_ape_pct": float(np.median(ape)),
            "p95_ape_pct": float(np.quantile(ape, 0.95)),
            "pass": bool(np.median(ape) <= 10.0 and np.quantile(ape, 0.95) <= 20.0),
        }
    queue_checks = []
    for row in summary[summary["variant"] == "Current"].itertuples(index=False):
        if row.metric == "waiting_s" and row.observed_mean_s < 0.05:
            criterion = "mae_s"
            value = float(row.mae)
            threshold = 0.05
        else:
            criterion = "wape_pct"
            value = float(row.wape_pct)
            threshold = (
                15.0
                if row.metric != "waiting_s" or row.region == "rho_le_0p8"
                else 25.0
            )
        queue_checks.append(
            {
                "region": row.region,
                "metric": row.metric,
                "criterion": criterion,
                "value": value,
                "threshold": threshold,
                "observed_mean_s": float(row.observed_mean_s),
                "mae_s": float(row.mae),
                "wape_pct": float(row.wape_pct),
                "finite_coverage_pct": float(row.finite_coverage_pct),
                "pass": bool(value <= threshold and row.finite_coverage_pct == 100.0),
            }
        )
    capacity_checks = []
    for row in capacity[capacity["variant"] == "Current"].itertuples(index=False):
        capacity_checks.append(
            {
                "composition_id": row.composition_id,
                "relative_error_pct": float(row.relative_error_pct),
                "threshold_pct": 10.0,
                "pass": bool(row.relative_error_pct <= 10.0),
            }
        )
    trend_checks = []
    for row in monotonicity[
        (monotonicity["source"] == "predicted")
        & monotonicity["metric"].isin(("waiting_s", "response_s"))
    ].itertuples(index=False):
        trend_checks.append(
            {
                "composition_id": row.composition_id,
                "metric": row.metric,
                "nondecreasing": bool(row.nondecreasing),
                "pass": bool(row.nondecreasing),
            }
        )
    integrity_check = {
        "runs": len(integrity),
        "all_request_counts_match": bool(integrity["request_count_matches"].all()),
        "no_nan": bool(not integrity["has_nan"].any()),
    }
    instrumentation_path = output / "instrumentation_check.json"
    instrumentation = (
        json.loads(instrumentation_path.read_text(encoding="utf-8"))
        if instrumentation_path.exists()
        else {"status": "not_run"}
    )
    _write_json(
        output / "validity.json",
        {
            "service": service_checks,
            "capacity": capacity_checks,
            "queue": queue_checks,
            "monotonicity": trend_checks,
            "integrity": integrity_check,
            "instrumentation": instrumentation,
        },
    )


def _write_validation_report(output: Path) -> None:
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    validity = json.loads((output / "validity.json").read_text(encoding="utf-8"))
    service = pd.read_csv(output / "service_predictions.csv")
    capacity = pd.read_csv(output / "capacity_predictions.csv")
    summary = pd.read_csv(output / "queue_error_summary.csv")
    predictions = pd.read_csv(output / "queue_predictions.csv")

    test = service[service["split"] == "test"]
    current = summary[summary["variant"] == "Current"]
    observed = predictions[predictions["variant"] == "Current"]
    observed_curves = (
        observed.groupby(["composition_id", "load_factor"])[
            ["observed_waiting_s", "observed_tbt_s", "observed_response_s"]
        ]
        .mean()
        .reset_index()
    )

    lines = [
        "# LLM 排队模型与 LLMServingSim 对比验证报告",
        "",
        "## 实验结论",
        "",
        "当前 Roofline–固定有效并发度–Allen–Cunneen 模型未达到预设精度，不能作为 continuous batching 下请求时延的定量模型。仿真结果表明，请求在中低负载下通常很快进入运行集合，负载升高主要延长运行阶段的 iteration 时长、TBT 和完整响应时延，而不是形成经典多服务台队列所描述的长准入等待。",
        "",
        "## 校准结果",
        "",
        f"- 有效计算速率：{calibration['analytical_parameters']['effective_flops'] / 1e12:.2f} TFLOP/s。",
        f"- 有效显存带宽：{calibration['analytical_parameters']['effective_bandwidth_bytes_s'] / 1e12:.3f} TB/s。",
        f"- 在 MIX 工作负载上校准得到的固定有效并发度：{calibration['effective_concurrency']}。",
        "",
        "留出工作负载的误差如下：",
        "",
        "| 指标 | Median APE | P95 APE |",
        "|---|---:|---:|",
    ]
    for metric, label in (
        ("prefill_s", "Prefill"),
        ("decode_s", "Decode"),
        ("service_s", "完整处理时延"),
        ("tbt_s", "TBT"),
    ):
        ape = 100.0 * test[f"ape_{metric}"].to_numpy(dtype=float)
        lines.append(
            f"| {label} | {np.median(ape):.2f}% | {np.quantile(ape, 0.95):.2f}% |"
        )

    lines.extend(
        [
            "",
            "各工作负载组成的饱和吞吐及当前模型预测如下：",
            "",
            "| 组成 | LLMServingSim (req/s) | Current (req/s) | 相对误差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in capacity[capacity["variant"] == "Current"].itertuples(index=False):
        lines.append(
            f"| {row.composition_id} | {row.observed_capacity_rps:.3f} | "
            f"{row.predicted_capacity_rps:.3f} | {row.relative_error_pct:.2f}% |"
        )

    lines.extend(
        [
            "",
            "Current 模型在稳定负载区间的聚合误差如下。等待时延接近零时同时报告 MAE，避免 WAPE 被很小的分母放大。",
            "",
            "| 负载区间 | 指标 | MAE | WAPE | 有限预测覆盖率 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in current.itertuples(index=False):
        lines.append(
            f"| {row.region} | {row.metric} | {row.mae:.4f} s | "
            f"{row.wape_pct:.2f}% | {row.finite_coverage_pct:.1f}% |"
        )

    lines.extend(
        [
            "",
            "误差来源诊断如下。Empirical-Service 仅替换为空载实测处理时延，Capacity-Matched 进一步按各组成的饱和吞吐匹配并发度。",
            "",
            "| 模型 | 负载区间 | 等待 MAE | TBT WAPE | 响应时延 WAPE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for (variant, region), group in summary.groupby(["variant", "region"]):
        by_metric = group.set_index("metric")
        lines.append(
            f"| {variant} | {region} | {by_metric.loc['waiting_s', 'mae']:.4f} s | "
            f"{by_metric.loc['tbt_s', 'wape_pct']:.2f}% | "
            f"{by_metric.loc['response_s', 'wape_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "空载服务时间替换后误差仍然存在，说明偏差不只来自 Roofline。按组成匹配饱和吞吐也未恢复高负载精度，表明单个固定并发度和状态无关服务时间不足以描述 continuous batching 的运行阶段。",
        ]
    )

    lines.extend(
        [
            "",
            "## 负载效应",
            "",
            "| 组成 | 负载范围 | 准入等待变化 | TBT 变化 | 完整响应时延变化 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for composition_id, group in observed_curves.groupby("composition_id"):
        ordered = group.sort_values("load_factor")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        lines.append(
            f"| {composition_id} | {first.load_factor:.2f}–{last.load_factor:.2f} | "
            f"{first.observed_waiting_s:.4f}→{last.observed_waiting_s:.4f} s | "
            f"{first.observed_tbt_s:.4f}→{last.observed_tbt_s:.4f} s | "
            f"{first.observed_response_s:.3f}→{last.observed_response_s:.3f} s |"
        )

    all_integrity = validity["integrity"]
    lines.extend(
        [
            "",
            "## 有效性与实现检查",
            "",
            f"- 已完成 {all_integrity['runs']} 个排队实验；请求数全部匹配：{all_integrity['all_request_counts_match']}；无 NaN：{all_integrity['no_nan']}。",
            "- 固定配置关闭 prefix caching，未启用 P/D 分离和多实例路由。",
            "- 插桩只记录第一次准入等待；其对完成时间和 token 间隔的无扰动检查见 `instrumentation_check.json`。",
            "",
            "## 对论文模型的建议",
            "",
            "1. Roofline 公式保留为请求计算量与显存访问需求的结构分析，不再直接声称可准确给出 vLLM 请求时延。",
            "2. LLM 响应性能采用提前建立的、配置相关的工作负载响应表；输入至少包含调用率、输入/输出长度类别及其组成，输出 TTFT、TBT、完整响应时延和稳定容量。",
            "3. 准入等待只表示 token budget、KV cache 或最大运行序列数受限时进入 waiting queue 的时间；continuous batching 中运行集合内的负载相关减速计入处理阶段。",
            "4. 若训练阶段需要低成本解析近似，应在经验验证通过的负载区间内拟合状态相关处理率或分段响应函数，最终评估继续采用 LLMServingSim 回放。",
            "",
            "该结论针对本实验的 Llama-3.1-8B、RTX4090、2048 token budget、512-token chunk、FCFS continuous batching 配置。其他模型、GPU 和 token budget 需要重新建立性能表。",
            "",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the analytical LLM queue model with LLMServingSim")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "sanity",
            "instrument-check",
            "run-calibration",
            "calibrate",
            "run-queue",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--container-repo", default=DEFAULT_CONTAINER_REPO)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.command in ("prepare", "all"):
        prepare_calibration(output)
        collect_provenance(output, args.host, args.remote_repo, args.container, args.container_repo)
    if args.command in ("sanity", "all"):
        run_sanity(output, args.host, args.container, args.container_repo)
    if args.command in ("instrument-check", "all"):
        run_instrumentation_check(
            output,
            args.host,
            args.remote_repo,
            args.container,
            args.container_repo,
        )
    if args.command in ("run-calibration", "all"):
        run_stage(output, "calibration", args.host, args.remote_repo, args.container, args.container_repo, args.workers, args.force)
    if args.command in ("calibrate", "all"):
        calibrate(output)
    if args.command in ("run-queue", "all"):
        run_stage(output, "queue", args.host, args.remote_repo, args.container, args.container_repo, args.workers, args.force)
    if args.command in ("analyze", "all"):
        analyze(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

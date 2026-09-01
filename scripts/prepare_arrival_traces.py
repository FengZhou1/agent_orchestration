from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent_orch.data import DatasetManifest, file_sha256
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import Scenario
from agent_orch.workload import ArrivalTrace


SOURCE_FIELDS = {
    "burstgpt": {
        "timestamp": ("timestamp",),
        "prompt": ("requesttokens", "prompttokens", "contexttokens"),
        "output": ("responsetokens", "outputtokens", "generatedtokens"),
        "session": ("sessionid",),
        "model": ("model",),
        "log_type": ("logtype",),
        "url": "https://github.com/HPMLL/BurstGPT",
        "license": "CC-BY-4.0",
    },
    "azure2024": {
        "timestamp": ("timestamp",),
        "prompt": ("contexttokens", "prompttokens", "requesttokens"),
        "output": ("generatedtokens", "outputtokens", "responsetokens"),
        "session": (),
        "model": ("model",),
        "log_type": (),
        "url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
        "license": "CC BY",
    },
}


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    columns = {_canonical(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    if required:
        raise ValueError(f"None of the required columns {candidates} is present")
    return None


def normalize_events(
    frame: pd.DataFrame,
    source: str,
    *,
    preserve_timestamp_origin: bool = False,
    source_file: str = "in-memory",
    row_offset: int = 0,
) -> pd.DataFrame:
    spec = SOURCE_FIELDS[source]
    timestamp_column = _find_column(frame, spec["timestamp"])
    prompt_column = _find_column(frame, spec["prompt"])
    output_column = _find_column(frame, spec["output"])
    session_column = _find_column(frame, spec["session"], required=False)
    model_column = _find_column(frame, spec.get("model", ()), required=False)
    log_type_column = _find_column(frame, spec.get("log_type", ()), required=False)
    timestamp = frame[timestamp_column]
    numeric = pd.to_numeric(timestamp, errors="coerce")
    if numeric.notna().mean() >= 0.99:
        timestamp_seconds = numeric.astype(float)
        if not preserve_timestamp_origin:
            timestamp_seconds = timestamp_seconds - timestamp_seconds.min()
    else:
        parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("Timestamp column contains values that cannot be parsed")
        timestamp_seconds = parsed.astype("int64") / 1e9
        if not preserve_timestamp_origin:
            timestamp_seconds = timestamp_seconds - timestamp_seconds.min()
    source_rows = np.arange(row_offset, row_offset + len(frame), dtype=np.int64)
    file_prefix = int(hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:7], 16)
    generated_sessions = pd.Series(
        (np.int64(file_prefix) << np.int64(32)) + source_rows, index=frame.index
    )
    normalized = pd.DataFrame(
        {
            "timestamp_s": timestamp_seconds.astype(float),
            "prompt_tokens": pd.to_numeric(frame[prompt_column], errors="coerce"),
            "output_tokens": pd.to_numeric(frame[output_column], errors="coerce"),
            "session_id": (
                frame[session_column].astype(str)
                if session_column is not None
                else generated_sessions
            ),
            "source_model": (
                frame[model_column].astype(str)
                if model_column is not None
                else "unknown"
            ),
            "log_type": (
                frame[log_type_column].astype(str)
                if log_type_column is not None
                else "unknown"
            ),
            "source_file": source_file,
            "source_row": source_rows,
        }
    )
    normalized = normalized.dropna()
    normalized = normalized[
        (normalized["prompt_tokens"] > 0) & (normalized["output_tokens"] > 0)
    ]
    normalized = normalized.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("No valid request events remain after normalization")
    return normalized


def load_event_files(
    paths: list[Path], source: str, chunksize: int = 500_000
) -> pd.DataFrame:
    """Load consecutive source files without resetting each file's clock."""
    normalized_chunks: list[pd.DataFrame] = []
    for path in paths:
        row_offset = 0
        for frame in pd.read_csv(path, chunksize=chunksize):
            normalized_chunks.append(
                normalize_events(
                    frame,
                    source,
                    preserve_timestamp_origin=True,
                    source_file=path.name,
                    row_offset=row_offset,
                )
            )
            row_offset += len(frame)
    events = pd.concat(normalized_chunks, ignore_index=True)
    events["timestamp_s"] -= events["timestamp_s"].min()
    events = events.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
    events.insert(0, "request_id", np.arange(len(events), dtype=np.int64))
    for column in ("source_model", "log_type", "source_file"):
        events[column] = events[column].astype("category")
    return events


def _joint_length_classes(
    events: pd.DataFrame, seed: int = 2026, sample_size: int = 200_000
) -> tuple[pd.Series, pd.DataFrame]:
    """Stratify paired prompt/output lengths with an ordered joint percentile score."""
    del seed, sample_size
    log_prompt = pd.Series(np.log1p(events["prompt_tokens"].to_numpy(dtype=float)))
    log_output = pd.Series(np.log1p(events["output_tokens"].to_numpy(dtype=float)))
    joint_score = 0.5 * (
        log_prompt.rank(method="average", pct=True)
        + log_output.rank(method="average", pct=True)
    )
    p50, p90 = joint_score.quantile([0.50, 0.90])
    ordered_labels = np.where(
        joint_score <= p50, 0, np.where(joint_score <= p90, 1, 2)
    ).astype(np.int8)
    names = np.asarray(["short", "medium", "long"], dtype=object)
    series = pd.Series(names[ordered_labels], index=events.index, name="length_class")
    records = []
    for index, name in enumerate(names):
        selected = ordered_labels == index
        records.append(
            {
                "length_class": name,
                "request_count": int(selected.sum()),
                "request_fraction": float(selected.mean()),
                "prompt_tokens_mean": float(events.loc[selected, "prompt_tokens"].mean()),
                "output_tokens_mean": float(events.loc[selected, "output_tokens"].mean()),
                "total_tokens_mean": float(
                    (events.loc[selected, "prompt_tokens"] + events.loc[selected, "output_tokens"]).mean()
                ),
            }
        )
    return series, pd.DataFrame.from_records(records)


def _length_classes(events: pd.DataFrame) -> pd.Series:
    return _joint_length_classes(events)[0]


def _parse_family_mix(raw: str, scenario: Scenario) -> dict[str, float]:
    families = sorted({app.family for app in scenario.applications.values()})
    if raw == "balanced":
        return {family: 1.0 / len(families) for family in families}
    parts = dict(item.split("=", 1) for item in raw.split(","))
    mix = {family: float(parts.get(family, 0.0)) for family in families}
    if any(value < 0.0 for value in mix.values()) or sum(mix.values()) <= 0.0:
        raise ValueError("Family mix must contain non-negative weights")
    total = sum(mix.values())
    return {family: value / total for family, value in mix.items()}


def assign_applications(
    events: pd.DataFrame,
    scenario: Scenario,
    family_mix: dict[str, float],
    seed: int,
    length_classes: pd.Series | None = None,
) -> pd.DataFrame:
    result = events.copy()
    result["length_class"] = (
        _length_classes(result) if length_classes is None else length_classes.to_numpy()
    )
    families = list(family_mix)
    probabilities = np.asarray([family_mix[family] for family in families], dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    session_codes, sessions = pd.factorize(result["session_id"], sort=False)
    session_family = rng.choice(families, size=len(sessions), p=probabilities)
    result["family"] = session_family[session_codes]
    groups: dict[tuple[str, str], list[str]] = {}
    fallback: dict[str, list[str]] = {}
    ingress: dict[str, str] = {}
    for app in scenario.applications.values():
        groups.setdefault((app.family, app.length_class), []).append(app.id)
        fallback.setdefault(app.family, []).append(app.id)
        ingress[app.id] = next(iter(app.ingress_rates))
    assigned = np.empty(len(result), dtype=object)
    total_tokens = (result["prompt_tokens"] + result["output_tokens"]).to_numpy()
    family_values = result["family"].to_numpy()
    length_values = result["length_class"].to_numpy()
    for family in families:
        for length_class in ("short", "medium", "long"):
            indices = np.flatnonzero(
                (family_values == family) & (length_values == length_class)
            )
            if not len(indices):
                continue
            candidates = sorted(groups.get((family, length_class)) or fallback[family])
            order = np.argsort(total_tokens[indices], kind="stable")
            candidate_index = np.minimum(
                len(candidates) - 1,
                np.floor(np.arange(len(indices)) * len(candidates) / len(indices)).astype(int),
            )
            assigned[indices[order]] = np.asarray(candidates, dtype=object)[candidate_index]
    result["application"] = assigned
    result["ingress"] = result["application"].map(ingress)
    return result


def token_characteristics(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for application, group in events.groupby("application", sort=True):
        prompt = group["prompt_tokens"]
        output = group["output_tokens"]
        total = prompt + output
        records.append(
            {
                "application": application,
                "family": str(group["family"].iloc[0]),
                "length_class": str(group["length_class"].iloc[0]),
                "request_count": int(len(group)),
                "prompt_tokens_mean": float(prompt.mean()),
                "prompt_tokens_p50": float(prompt.quantile(0.50)),
                "prompt_tokens_p90": float(prompt.quantile(0.90)),
                "prompt_tokens_p99": float(prompt.quantile(0.99)),
                "output_tokens_mean": float(output.mean()),
                "output_tokens_p50": float(output.quantile(0.50)),
                "output_tokens_p90": float(output.quantile(0.90)),
                "output_tokens_p99": float(output.quantile(0.99)),
                "total_tokens_mean": float(total.mean()),
                "total_tokens_p50": float(total.quantile(0.50)),
                "total_tokens_p90": float(total.quantile(0.90)),
                "total_tokens_p99": float(total.quantile(0.99)),
                "prompt_output_correlation": float(prompt.corr(output)),
            }
        )
    return pd.DataFrame.from_records(records)


def write_token_calibrated_scenario(
    scenario_path: Path,
    output_path: Path,
    characteristics: pd.DataFrame,
    source_version: str,
) -> None:
    """Calibrate application ingress/final token characteristics and retain stage ratios."""
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    targets = characteristics.set_index("application")
    for application in raw["applications"]:
        target = targets.loc[application["id"]]
        nodes = {node["id"]: node for node in application["nodes"]}
        llm_nodes = {node_id for node_id, node in nodes.items() if node["type"] == "llm"}
        incoming = {
            target_node
            for flow in application["pattern_flows"]
            for chain in flow["chains"]
            for target_node in chain[1:]
        }
        source_nodes = [node_id for node_id in llm_nodes if node_id not in incoming]
        final_nodes = list(dict.fromkeys(flow["final_node"] for flow in application["pattern_flows"]))
        if len(source_nodes) != 1 or len(final_nodes) != 1 or final_nodes[0] not in llm_nodes:
            raise ValueError(
                f"Application {application['id']} must have one entry and one final LLM node"
            )
        entry_node = nodes[source_nodes[0]]
        final_node = nodes[final_nodes[0]]
        models = list(entry_node["prompt_tokens"])
        prompt_scale = {
            model: float(target["prompt_tokens_mean"]) / float(entry_node["prompt_tokens"][model])
            for model in models
        }
        output_scale = {
            model: float(target["output_tokens_mean"]) / float(final_node["output_tokens"][model])
            for model in models
        }
        for node in nodes.values():
            if node["type"] != "llm":
                continue
            node["prompt_tokens"] = {
                model: round(float(value) * prompt_scale[model], 3)
                for model, value in node["prompt_tokens"].items()
            }
            node["output_tokens"] = {
                model: round(float(value) * output_scale[model], 3)
                for model, value in node["output_tokens"].items()
            }
        application["entry_data_mb"] = {
            model: round(4.0 * float(entry_node["prompt_tokens"][model]) / 1e6, 9)
            for model in models
        }
        application["exit_data_mb"] = {
            model: round(4.0 * float(final_node["output_tokens"][model]) / 1e6, 9)
            for model in models
        }
        edge_data = application.get("edge_data_mb", {})
        for key in list(edge_data):
            model, edge = key.split(":", 1)
            source_node, _ = edge.split("->", 1)
            if source_node in llm_nodes:
                edge_data[key] = round(
                    4.0 * float(nodes[source_node]["output_tokens"][model]) / 1e6, 9
                )
    metadata = raw.setdefault("metadata", {})
    metadata["token_status"] = "calibrated from paired BurstGPT input-output token samples"
    metadata["token_source_version"] = source_version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _bundle_checksum(paths: list[Path]) -> tuple[str, dict[str, str]]:
    checksums = {path.name: file_sha256(path) for path in paths}
    digest = hashlib.sha256()
    for name, checksum in sorted(checksums.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest(), checksums


def aggregate_trace(events: pd.DataFrame, scenario: Scenario) -> ArrivalTrace:
    seconds = scenario.simulation.slot_seconds
    working = events.copy()
    working["slot"] = np.floor(working["timestamp_s"] / seconds).astype(int)
    frame = (
        working.groupby(["slot", "application", "ingress"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    frame["rate_rps"] = frame["count"] / seconds
    rates: dict[int, dict[tuple[str, str], float]] = {}
    for row in frame.itertuples(index=False):
        rates.setdefault(int(row.slot), {})[(row.application, row.ingress)] = float(row.rate_rps)
    for slot in range(max(rates, default=0) + 1):
        rates.setdefault(slot, {})
    return ArrivalTrace(rates)


def time_scaled_trace(
    events: pd.DataFrame, scenario: Scenario, intensity_factor: float
) -> tuple[ArrivalTrace, int]:
    """Scale all timestamps by one factor while preserving request order and pairs."""
    if intensity_factor <= 0.0:
        raise ValueError("Arrival intensity factor must be positive")
    scaled = events.copy()
    scaled["timestamp_s"] = scaled["timestamp_s"] / intensity_factor
    trace = aggregate_trace(scaled, scenario)
    slots = max(trace.rates, default=0) + 1
    return trace, slots


def _write_split(
    trace: ArrivalTrace, output: Path, slots: int, scenario: Scenario
) -> dict[str, tuple[int, int]]:
    train_stop = int(0.60 * slots)
    validation_stop = int(0.80 * slots)
    boundaries = {
        "train": (0, train_stop),
        "validation": (train_stop, validation_stop),
        "test": (validation_stop, slots),
    }
    for name, (start, stop) in boundaries.items():
        path = output / f"{name}.csv"
        trace.window(start, stop).to_frame(scenario).to_csv(path, index=False)
    return boundaries


def trace_statistics(events: pd.DataFrame, trace: ArrivalTrace, slots: int) -> dict[str, Any]:
    total_tokens = events["prompt_tokens"] + events["output_tokens"]
    per_slot = np.asarray(
        [sum(trace.rates.get(slot, {}).values()) for slot in range(slots)], dtype=float
    )
    positive_slots = np.flatnonzero(per_slot > 0.0)
    gaps = np.diff(positive_slots) if len(positive_slots) > 1 else np.asarray([0.0])
    return {
        "requests": int(len(events)),
        "slots": slots,
        "mean_rps": float(per_slot.mean()),
        "peak_to_mean": float(per_slot.max() / max(per_slot.mean(), 1e-12)),
        "nonzero_gap_p50_slots": float(np.quantile(gaps, 0.50)),
        "nonzero_gap_p95_slots": float(np.quantile(gaps, 0.95)),
        "prompt_tokens": {f"p{p}": float(events["prompt_tokens"].quantile(p / 100)) for p in (50, 90, 99)},
        "output_tokens": {f"p{p}": float(events["output_tokens"].quantile(p / 100)) for p in (50, 90, 99)},
        "total_tokens": {f"p{p}": float(total_tokens.quantile(p / 100)) for p in (50, 90, 99)},
    }


def _source_statistics(events: pd.DataFrame, cluster_frame: pd.DataFrame) -> dict[str, Any]:
    prompt = events["prompt_tokens"]
    output = events["output_tokens"]
    total = prompt + output
    duration = float(events["timestamp_s"].max()) + 1.0
    timestamp = events["timestamp_s"].to_numpy(dtype=float)
    per_second = np.unique(np.floor(timestamp).astype(np.int64), return_counts=True)[1]
    interarrival = np.diff(timestamp)
    return {
        "requests": int(len(events)),
        "duration_s": duration,
        "mean_rps": float(len(events) / duration),
        "peak_rps": float(per_second.max()),
        "peak_to_mean": float(per_second.max() / (len(events) / duration)),
        "interarrival_s": {
            "p50": float(np.quantile(interarrival, 0.50)),
            "p95": float(np.quantile(interarrival, 0.95)),
        },
        "prompt_output_correlation": float(prompt.corr(output)),
        "prompt_tokens": {
            f"p{p}": float(prompt.quantile(p / 100)) for p in (50, 90, 99)
        },
        "output_tokens": {
            f"p{p}": float(output.quantile(p / 100)) for p in (50, 90, 99)
        },
        "total_tokens": {
            f"p{p}": float(total.quantile(p / 100)) for p in (50, 90, 99)
        },
        "length_classes": cluster_frame.to_dict(orient="records"),
    }


def _chronological_boundaries(slots: int) -> dict[str, tuple[int, int]]:
    train_stop = int(0.60 * slots)
    validation_stop = int(0.80 * slots)
    return {
        "train": (0, train_stop),
        "validation": (train_stop, validation_stop),
        "test": (validation_stop, slots),
    }


def _center_window(start: int, stop: int, requested_slots: int) -> tuple[int, int]:
    available = stop - start
    if available <= 0:
        raise ValueError("Chronological split contains no slots")
    length = min(available, requested_slots)
    left = start + (available - length) // 2
    return left, left + length


def _window_trace(
    events: pd.DataFrame,
    scenario: Scenario,
    intensity_factor: float,
    start_slot: int,
    stop_slot: int,
) -> tuple[pd.DataFrame, ArrivalTrace]:
    seconds = scenario.simulation.slot_seconds
    scaled_timestamp = events["timestamp_s"] / intensity_factor
    left = start_slot * seconds
    right = stop_slot * seconds
    selected = events.loc[
        (scaled_timestamp >= left) & (scaled_timestamp < right)
    ].copy()
    selected["timestamp_s"] = scaled_timestamp.loc[selected.index] - left
    trace = aggregate_trace(selected, scenario)
    for slot in range(stop_slot - start_slot):
        trace.rates.setdefault(slot, {})
    return selected, trace


def _write_window_bundle(
    events: pd.DataFrame,
    scenario: Scenario,
    output: Path,
    intensity_factor: float,
    window_slots: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seconds = scenario.simulation.slot_seconds
    total_slots = int(np.floor(events["timestamp_s"].max() / intensity_factor / seconds)) + 1
    source_boundaries = _chronological_boundaries(total_slots)
    mode_frames: dict[str, list[pd.DataFrame]] = {
        "trace": [],
        "nhpp": [],
        "poisson": [],
    }
    manifest: dict[str, Any] = {
        "timestamp_intensity_factor": intensity_factor,
        "source_slots": total_slots,
        "modes": {},
        "windows": {},
    }
    statistics: dict[str, Any] = {}
    full_offset = 0
    for split_index, (name, boundary) in enumerate(source_boundaries.items()):
        start, stop = _center_window(*boundary, window_slots)
        selected, source_trace = _window_trace(
            events, scenario, intensity_factor, start, stop
        )
        slots = stop - start
        modes = {
            "trace": source_trace,
            "nhpp": ArrivalTrace.nhpp_control(
                scenario, source_trace, slots, seed + split_index
            ),
            "poisson": ArrivalTrace.homogeneous_poisson(
                scenario, source_trace, slots, seed + split_index
            ),
        }
        manifest["windows"][name] = {
            "source_split": boundary,
            "selected_source_window": (start, stop),
            "output_slots": slots,
        }
        statistics[name] = trace_statistics(selected, source_trace, slots)
        for mode, trace in modes.items():
            mode_dir = output / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            frame = trace.to_frame(scenario)
            frame.to_csv(mode_dir / f"{name}.csv", index=False)
            shifted = frame.copy()
            shifted["slot"] += full_offset
            mode_frames[mode].append(shifted)
            manifest["modes"].setdefault(mode, {})[name] = (0, slots)
        full_offset += slots
    for mode, frames in mode_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(
            output / mode / "full.csv", index=False
        )
    return manifest, statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--source", choices=sorted(SOURCE_FIELDS), required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", default="data/processed/arrivals")
    parser.add_argument("--calibrated-scenario")
    parser.add_argument("--family-mix", default="balanced")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window-slots", type=int, default=3600)
    parser.add_argument("--reference-capacity-rps", type=float)
    parser.add_argument("--load-levels", default="0.4,0.65,0.85,1.05")
    args = parser.parse_args()

    if args.window_slots <= 0:
        raise ValueError("window-slots must be positive")
    source_paths = [Path(path).resolve() for path in args.input]
    scenario = ScenarioLoader.load(args.scenario)
    events = load_event_files(source_paths, args.source)
    length_classes, cluster_frame = _joint_length_classes(events, args.seed)
    mix = _parse_family_mix(args.family_mix, scenario)
    assigned = assign_applications(
        events, scenario, mix, args.seed, length_classes=length_classes
    )
    base_slots = (
        int(np.floor(assigned["timestamp_s"].max() / scenario.simulation.slot_seconds))
        + 1
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assigned.to_parquet(
        output / "paired_requests.parquet", index=False, compression="zstd"
    )
    assigned.head(10_000).to_csv(output / "paired_requests_sample.csv", index=False)
    characteristics = token_characteristics(assigned)
    characteristics.to_csv(output / "token_characteristics.csv", index=False)
    cluster_frame.to_csv(output / "length_class_characteristics.csv", index=False)
    (output / "source_statistics.json").write_text(
        json.dumps(
            _source_statistics(assigned, cluster_frame), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    if args.calibrated_scenario:
        write_token_calibrated_scenario(
            Path(args.scenario).resolve(),
            Path(args.calibrated_scenario).resolve(),
            characteristics,
            args.source_version,
        )

    raw_mean = len(assigned) / (base_slots * scenario.simulation.slot_seconds)
    levels = [float(item) for item in args.load_levels.split(",")]
    targets = {"raw": 1.0}
    if args.reference_capacity_rps is not None:
        targets = {
            f"load_{level:.2f}".replace(".", "p"): level * args.reference_capacity_rps / max(raw_mean, 1e-12)
            for level in levels
        }
    split_manifest: dict[str, Any] = {}
    window_statistics: dict[str, Any] = {}
    for label, factor in targets.items():
        label_dir = output / label
        label_dir.mkdir(parents=True, exist_ok=True)
        split_manifest[label], window_statistics[label] = _write_window_bundle(
            assigned,
            scenario,
            label_dir,
            factor,
            args.window_slots,
            args.seed,
        )
        (label_dir / "statistics.json").write_text(
            json.dumps(window_statistics[label], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    source_checksum, source_checksums = _bundle_checksum(source_paths)
    manifest = DatasetManifest(
        artifact_type="arrival-trace-bundle",
        source_name=args.source,
        source_url=str(SOURCE_FIELDS[args.source]["url"]),
        source_version=args.source_version,
        license=str(SOURCE_FIELDS[args.source]["license"]),
        source_checksum_sha256=source_checksum,
        preprocessing_command=" ".join(sys.argv),
        random_seed=args.seed,
        split_boundaries=split_manifest,
        parameters={
            "scenario": str(Path(args.scenario).resolve()),
            "source_files": [str(path) for path in source_paths],
            "source_file_checksums_sha256": source_checksums,
            "family_mix": mix,
            "reference_capacity_rps": args.reference_capacity_rps,
            "slot_seconds": scenario.simulation.slot_seconds,
            "window_slots": args.window_slots,
            "length_stratification": "joint percentile score of paired log1p(prompt) and log1p(output), split at p50 and p90",
            "paired_token_samples_preserved": True,
            "token_calibration": "entry prompt and final output means; intermediate LLM stage ratios retained",
            "calibrated_scenario": args.calibrated_scenario,
        },
    )
    manifest.write(output / "manifest.json")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

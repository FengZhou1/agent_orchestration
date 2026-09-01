from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

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
        "url": "https://github.com/HPMLL/BurstGPT",
        "license": "See pinned BurstGPT release",
    },
    "azure2024": {
        "timestamp": ("timestamp",),
        "prompt": ("contexttokens", "prompttokens", "requesttokens"),
        "output": ("generatedtokens", "outputtokens", "responsetokens"),
        "session": (),
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


def normalize_events(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    spec = SOURCE_FIELDS[source]
    timestamp_column = _find_column(frame, spec["timestamp"])
    prompt_column = _find_column(frame, spec["prompt"])
    output_column = _find_column(frame, spec["output"])
    session_column = _find_column(frame, spec["session"], required=False)
    timestamp = frame[timestamp_column]
    numeric = pd.to_numeric(timestamp, errors="coerce")
    if numeric.notna().mean() >= 0.99:
        timestamp_seconds = numeric - numeric.min()
    else:
        parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("Timestamp column contains values that cannot be parsed")
        timestamp_seconds = (parsed - parsed.min()).dt.total_seconds()
    normalized = pd.DataFrame(
        {
            "timestamp_s": timestamp_seconds.astype(float),
            "prompt_tokens": pd.to_numeric(frame[prompt_column], errors="coerce"),
            "output_tokens": pd.to_numeric(frame[output_column], errors="coerce"),
            "session_id": (
                frame[session_column].astype(str)
                if session_column is not None
                else pd.Series(np.arange(len(frame)), index=frame.index).astype(str)
            ),
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


def _length_classes(events: pd.DataFrame) -> pd.Series:
    total = events["prompt_tokens"] + events["output_tokens"]
    p50, p90 = total.quantile([0.50, 0.90])
    return pd.cut(
        total,
        bins=[-np.inf, p50, p90, np.inf],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)


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
) -> pd.DataFrame:
    result = events.copy()
    result["length_class"] = _length_classes(result)
    families = list(family_mix)
    probabilities = np.asarray([family_mix[family] for family in families], dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    session_family = {
        session: str(rng.choice(families, p=probabilities))
        for session in result["session_id"].drop_duplicates()
    }
    result["family"] = result["session_id"].map(session_family)
    groups: dict[tuple[str, str], list[str]] = {}
    fallback: dict[str, list[str]] = {}
    ingress: dict[str, str] = {}
    for app in scenario.applications.values():
        groups.setdefault((app.family, app.length_class), []).append(app.id)
        fallback.setdefault(app.family, []).append(app.id)
        ingress[app.id] = next(iter(app.ingress_rates))
    counters: dict[tuple[str, str], int] = {}
    assigned = []
    for row in result.itertuples(index=False):
        key = (row.family, row.length_class)
        candidates = groups.get(key) or fallback[row.family]
        index = counters.get(key, 0)
        assigned.append(candidates[index % len(candidates)])
        counters[key] = index + 1
    result["application"] = assigned
    result["ingress"] = result["application"].map(ingress)
    return result


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", choices=sorted(SOURCE_FIELDS), required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", default="data/processed/arrivals")
    parser.add_argument("--family-mix", default="balanced")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-capacity-rps", type=float)
    parser.add_argument("--load-levels", default="0.4,0.65,0.85,1.05")
    args = parser.parse_args()

    source_path = Path(args.input).resolve()
    scenario = ScenarioLoader.load(args.scenario)
    events = normalize_events(pd.read_csv(source_path), args.source)
    mix = _parse_family_mix(args.family_mix, scenario)
    assigned = assign_applications(events, scenario, mix, args.seed)
    base_trace, base_slots = time_scaled_trace(assigned, scenario, 1.0)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(output / "assigned_requests.csv.gz", index=False, compression="gzip")

    raw_mean = len(assigned) / (base_slots * scenario.simulation.slot_seconds)
    levels = [float(item) for item in args.load_levels.split(",")]
    targets = {"raw": 1.0}
    if args.reference_capacity_rps is not None:
        targets = {
            f"load_{level:.2f}".replace(".", "p"): level * args.reference_capacity_rps / max(raw_mean, 1e-12)
            for level in levels
        }
    split_manifest: dict[str, Any] = {}
    for label, factor in targets.items():
        label_dir = output / label
        label_dir.mkdir(parents=True, exist_ok=True)
        trace, slots = time_scaled_trace(assigned, scenario, factor)
        modes = {
            "trace": trace,
            "nhpp": ArrivalTrace.nhpp_control(scenario, trace, slots, args.seed),
            "poisson": ArrivalTrace.homogeneous_poisson(scenario, trace, slots, args.seed),
        }
        split_manifest[label] = {
            "timestamp_intensity_factor": factor,
            "slots": slots,
            "modes": {},
        }
        for mode, mode_trace in modes.items():
            mode_dir = label_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            boundaries = _write_split(mode_trace, mode_dir, slots, scenario)
            mode_trace.to_frame(scenario).to_csv(mode_dir / "full.csv", index=False)
            split_manifest[label]["modes"][mode] = boundaries
        (label_dir / "statistics.json").write_text(
            json.dumps(trace_statistics(assigned, trace, slots), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    manifest = DatasetManifest(
        artifact_type="arrival-trace-bundle",
        source_name=args.source,
        source_url=str(SOURCE_FIELDS[args.source]["url"]),
        source_version=args.source_version,
        license=str(SOURCE_FIELDS[args.source]["license"]),
        source_checksum_sha256=file_sha256(source_path),
        preprocessing_command=" ".join(sys.argv),
        random_seed=args.seed,
        split_boundaries=split_manifest,
        parameters={
            "scenario": str(Path(args.scenario).resolve()),
            "family_mix": mix,
            "reference_capacity_rps": args.reference_capacity_rps,
            "slot_seconds": scenario.simulation.slot_seconds,
            "paired_token_samples_preserved": True,
        },
    )
    manifest.write(output / "manifest.json")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

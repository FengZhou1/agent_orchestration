from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

from agent_orch.data import DatasetManifest, file_sha256


GPU_MEMORY_GB = {"A10": 24.0, "L20": 48.0, "H20": 96.0}


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _column(frame: pd.DataFrame, *names: str) -> str:
    mapping = {_canonical(column): column for column in frame.columns}
    for name in names:
        if _canonical(name) in mapping:
            return mapping[_canonical(name)]
    raise ValueError(f"Missing one of the columns {names}")


def build_catalog(frame: pd.DataFrame, count: int, seed: int) -> list[dict]:
    server_column = _column(frame, "server_id", "server")
    gpu_type_column = _column(frame, "gpu_type", "gpu_model")
    gpu_count_column = _column(frame, "gpu_count", "gpu_num")
    cpu_column = _column(frame, "cpu_capacity", "cpu")
    working = frame[[server_column, gpu_type_column, gpu_count_column, cpu_column]].copy()
    working.columns = ["server_id", "gpu_type", "gpu_count", "cpu_cores"]
    working["gpu_type"] = working["gpu_type"].astype(str).str.upper()
    working = working[working["gpu_type"].isin(GPU_MEMORY_GB)]
    working = working.dropna().drop_duplicates("server_id")
    if working.empty:
        raise ValueError("No A10, L20, or H20 servers were found")
    sample_count = min(count, len(working))
    sample = working.sample(n=sample_count, random_state=seed, replace=False)
    rows = []
    for index, row in enumerate(sample.itertuples(index=False)):
        cpu = max(1, int(round(float(row.cpu_cores))))
        rows.append(
            {
                "id": f"trace-node-{index:03d}",
                "source_server_id": str(row.server_id),
                "gpu_type": str(row.gpu_type),
                "gpu_count": max(1, int(round(float(row.gpu_count)))),
                "gpu_memory_gb": GPU_MEMORY_GB[str(row.gpu_type)],
                "cpu_cores": cpu,
                "memory_gb": float(4 * cpu),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--count", type=int, default=22)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/processed/alibaba_gpu_server_catalog.yaml")
    args = parser.parse_args()
    source = Path(args.input).resolve()
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    servers = build_catalog(frame, args.count, args.seed)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            {
                "source": "Alibaba GPU Trace v2026 server-hour table",
                "selection": "joint rows filtered to A10/L20/H20",
                "seed": args.seed,
                "servers": servers,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        artifact_type="infrastructure-catalog",
        source_name="Alibaba GPU Trace v2026",
        source_url="https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026",
        source_version=args.source_version,
        license="See Alibaba Cluster Data repository",
        source_checksum_sha256=file_sha256(source),
        preprocessing_command=" ".join(sys.argv),
        random_seed=args.seed,
        parameters={
            "count": args.count,
            "gpu_filter": sorted(GPU_MEMORY_GB),
            "joint_server_rows_preserved": True,
            "llm_request_fields_used": False,
        },
    )
    manifest.write(output.with_suffix(".manifest.json"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

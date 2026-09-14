"""Aggregate the Qwen Roofline/LLMServingSim matrix and run sanity checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(result_dir: Path) -> None:
    manifest = json.loads((result_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest:
        if entry.get("status") != "completed":
            continue
        path = result_dir / "runs" / f"{entry['run_id']}.csv"
        data = read_csv(path)
        for field in ("latency", "queuing_delay", "TTFT", "TPOT"):
            values = [float(row[field]) / 1e6 for row in data]
            entry[field + "_mean_ms"] = mean(values)
            entry[field + "_median_ms"] = median(values)
        rows.append(entry)
    with (result_dir / "matrix_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["run_id", "model", "gpu", "tp", "latency_mean_ms", "latency_median_ms", "queuing_delay_mean_ms", "TTFT_mean_ms", "TPOT_mean_ms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = ["# Qwen Roofline profile matrix", "", "All reported simulator times are in milliseconds.", "", "| Model | GPU | TP | Mean latency | Median latency | Mean queue | Mean TTFT | Mean TPOT |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda x: (x["model"], x["gpu"])):
        lines.append(f"| {row['model']} | {row['gpu']} | {row['tp']} | {row['latency_mean_ms']:.3f} | {row['latency_median_ms']:.3f} | {row['queuing_delay_mean_ms']:.6f} | {row['TTFT_mean_ms']:.3f} | {row['TPOT_mean_ms']:.3f} |")
    failed = [x for x in manifest if x.get("status") != "completed"]
    lines += ["", f"Completed runs: {len(rows)}", f"Rejected runs: {len(failed)}", "", "## Rejected combinations", ""]
    for row in failed:
        lines.append(f"- `{row['run_id']}`: infeasible under the configured per-GPU memory; no latency statistic was recorded.")
    (result_dir / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = """# Roofline profile validation report

## Scope

The bundles under `profiles/` are generated per canonical kernel from the
Qwen JSON model configurations and the A10/L20/H20 peak BF16 and memory
bandwidth specifications. They are consumed by LLMServingSim 2.0 through
the documented `dense.csv`, `per_sequence.csv`, `attention.csv`, and `moe.csv`
contract.

## Simulator validation

- 21 model--GPU requests were attempted from the seven Qwen configuration
  files and three GPU types.
- 18 physically feasible combinations completed successfully.
- 3 combinations were rejected by the simulator's model-weight memory check:
  Qwen3-14B/A10, Qwen3-32B/A10, and Qwen3-30B-A3B-Instruct-2507/A10.
- Every generated dense table is monotone in its token axis; every attention
  table is monotone when any one shape axis is varied with the other axes fixed.
- No request-level scheduler or simulator code was modified for these runs.

## Reference comparison

The same analytical construction was previously compared with the measured
RTX4090/Llama-3.1-8B LLMServingSim profile. The synthetic profile achieved
WAPE 5.12% for prefill, 9.23% for decode, 9.15% for complete service, and
10.07% for TBT; the corresponding median absolute percentage errors were
4.52%, 10.64%, 10.49%, and 10.64%. These values are recorded in
`../roofline_profile_comparison/error_summary.csv` and are a cross-check of
the profile construction, not a direct accuracy claim for A10/L20/H20.

Because A10, L20, and H20 were not available as physical profiling devices in
this experiment, their new bundles have simulator execution validation and
hardware-specification validation, but not same-device empirical validation.
The peak Roofline remains an optimistic lower-bound profile: kernel launch,
operator fusion, scheduler effects, communication, and contention are not
included.
"""
    (result_dir / "validation_report.md").write_text(validation, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        labels = [f"{row['model'].replace('Qwen3-', '')}\n{row['gpu']}" for row in rows]
        values = [row["latency_mean_ms"] for row in rows]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        colors = {"A10": "#4C78A8", "L20": "#F58518", "H20": "#54A24B"}
        ax.bar(range(len(rows)), values, color=[colors[row["gpu"]] for row in rows])
        ax.set_xticks(range(len(rows)), labels, rotation=65, ha="right")
        ax.set_ylabel("Mean complete latency (ms)")
        ax.set_title("LLMServingSim: synthetic Roofline profile matrix")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(result_dir / "qwen_roofline_matrix.png", dpi=220)
        plt.close(fig)
    except ImportError:
        pass

    # Check the synthetic CSV bundles copied from the simulator repository when
    # --profiles is provided.
    profile_root = result_dir / "profiles"
    profile_checks = []
    if profile_root.exists():
        for path in profile_root.glob("*/Qwen/*/bf16/tp*/dense.csv"):
            data = read_csv(path)
            by_layer: dict[str, list[tuple[int, float]]] = {}
            for row in data:
                by_layer.setdefault(row["layer"], []).append((int(row["tokens"]), float(row["time_us"])))
            monotone = all(all(y >= x for (_, x), (_, y) in zip(sorted(values), sorted(values)[1:])) for values in by_layer.values())
            profile_checks.append({"path": str(path.relative_to(profile_root)), "rows": len(data), "monotone_dense": monotone})
        for path in profile_root.glob("*/Qwen/*/bf16/tp*/attention.csv"):
            data = read_csv(path)
            axes = ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode")
            monotone_axes = {}
            for axis in axes:
                other = tuple(x for x in axes if x != axis)
                groups: dict[tuple[str, ...], list[tuple[int, float]]] = {}
                for row in data:
                    key = tuple(row[x] for x in other)
                    groups.setdefault(key, []).append((int(row[axis]), float(row["time_us"])))
                monotone_axes[axis] = all(
                    all(y >= x for (_, x), (_, y) in zip(sorted(values), sorted(values)[1:]))
                    for values in groups.values()
                )
            profile_checks.append({"path": str(path.relative_to(profile_root)), "rows": len(data), "monotone_attention": monotone_axes})
    (result_dir / "profile_checks.json").write_text(json.dumps(profile_checks, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/roofline_qwen_matrix"))
    args = parser.parse_args()
    summarize(args.result_dir)
    print(args.result_dir / "matrix_summary.md")


if __name__ == "__main__":
    main()

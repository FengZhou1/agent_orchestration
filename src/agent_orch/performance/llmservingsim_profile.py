"""Reader and low-dimensional reducer for LLMServingSim kernel profiles.

The profile is used as a calibration source for the analytical execution
primitive, not as a request-level latency lookup table.  The reducer mirrors
the simulator's dense, attention, and per-sequence layer aggregation for a
single iteration and exposes only the resulting iteration time.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _axis_bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    if len(axis) == 1 or value <= axis[0]:
        return 0, 0, 0.0
    idx = bisect_right(axis.tolist(), float(value))
    if idx >= len(axis):
        lo, hi = len(axis) - 2, len(axis) - 1
    else:
        lo, hi = idx - 1, idx
    gap = float(axis[hi] - axis[lo])
    return lo, hi, 0.0 if gap <= 0.0 else (float(value) - axis[lo]) / gap


def _interp_1d(axis: np.ndarray, values: np.ndarray, value: float) -> float:
    if len(axis) == 1:
        return float(values[0])
    lo, hi, fraction = _axis_bracket(axis, value)
    return float(values[lo] + fraction * (values[hi] - values[lo]))


@dataclass(frozen=True)
class ProfileBundle:
    root: Path
    model_name: str
    gpu: str
    tp: int
    layers: int
    dense: dict[str, tuple[np.ndarray, np.ndarray]]
    per_sequence: dict[str, tuple[np.ndarray, np.ndarray]]
    attention_axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    attention_values: dict[tuple[int, int, int, int], float]

    @classmethod
    def load(
        cls,
        root: str | Path,
        gpu: str,
        model_name: str,
        tp: int,
        layers: int,
    ) -> "ProfileBundle":
        base = Path(root) / gpu / "Qwen" / model_name / "bf16" / f"tp{tp}"
        if not base.exists():
            raise FileNotFoundError(base)
        dense: dict[str, list[tuple[float, float]]] = {}
        with (base / "dense.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                dense.setdefault(row["layer"], []).append((float(row["tokens"]), float(row["time_us"])))
        dense_arrays = {
            name: (np.asarray([x for x, _ in rows]), np.asarray([y for _, y in rows]))
            for name, rows in dense.items()
        }
        per_sequence: dict[str, list[tuple[float, float]]] = {}
        with (base / "per_sequence.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                per_sequence.setdefault(row["layer"], []).append((float(row["sequences"]), float(row["time_us"])))
        per_arrays = {
            name: (np.asarray([x for x, _ in rows]), np.asarray([y for _, y in rows]))
            for name, rows in per_sequence.items()
        }
        rows: list[tuple[int, int, int, int, float]] = []
        with (base / "attention.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((
                    int(row["prefill_chunk"]),
                    int(row["kv_prefill"]),
                    int(row["n_decode"]),
                    int(row["kv_decode"]),
                    float(row["time_us"]),
                ))
        axes = tuple(np.asarray(sorted({row[i] for row in rows}), dtype=float) for i in range(4))
        values = {(a, b, c, d): value for a, b, c, d, value in rows}
        return cls(Path(root), model_name, gpu, int(tp), int(layers), dense_arrays, per_arrays, axes, values)

    def dense_time(self, layer: str, tokens: float) -> float:
        table = self.dense.get(layer)
        if table is None:
            return 0.0
        return _interp_1d(*table, max(0.0, float(tokens)))

    def per_sequence_time(self, layer: str, sequences: float) -> float:
        table = self.per_sequence.get(layer)
        if table is None:
            return 0.0
        return _interp_1d(*table, max(1.0, float(sequences)))

    def attention_time(
        self,
        prefill_chunk: float,
        kv_prefill: float,
        n_decode: float,
        kv_decode: float,
    ) -> float:
        axes = self.attention_axes
        indices: list[tuple[int, int, float]] = [_axis_bracket(axis, value) for axis, value in zip(axes, (prefill_chunk, kv_prefill, n_decode, kv_decode), strict=True)]
        total = 0.0
        for mask in range(16):
            point: list[int] = []
            weight = 1.0
            for axis_index, (lo, hi, fraction) in enumerate(indices):
                if mask & (1 << axis_index):
                    point.append(int(axes[axis_index][hi]))
                    weight *= fraction
                else:
                    point.append(int(axes[axis_index][lo]))
                    weight *= 1.0 - fraction
            total += weight * self.attention_values.get(tuple(point), 0.0)
        return float(total)

    def iteration_time_us(
        self,
        prefill_chunk: float,
        kv_prefill: float,
        n_decode: float,
        kv_decode: float,
        n_prefill: int = 1,
    ) -> float:
        pc = max(0.0, float(prefill_chunk))
        nd = max(0.0, float(n_decode))
        total_tokens = pc + nd
        total = 0.0
        # Qwen3 dense transformer block aggregation follows qwen3.yaml.
        total += self.dense_time("embedding", total_tokens)
        for _ in range(self.layers):
            total += self.dense_time("layernorm", total_tokens)
            total += self.dense_time("qkv_proj", total_tokens)
            total += self.dense_time("qk_norm", total_tokens)
            total += self.dense_time("rotary_emb", total_tokens)
            total += self.attention_time(pc, kv_prefill, nd, kv_decode)
            total += self.dense_time("o_proj", total_tokens)
            total += self.dense_time("layernorm", total_tokens)
            total += self.dense_time("gate_up_proj", total_tokens)
            total += self.dense_time("act_fn", total_tokens)
            total += self.dense_time("down_proj", total_tokens)
        total += self.dense_time("final_layernorm", total_tokens)
        sequences = max(1, int(n_prefill) + int(round(nd)))
        total += self.per_sequence_time("lm_head", sequences)
        total += self.per_sequence_time("sampler", sequences)
        return float(total)


def profile_iteration_grid(
    profile: ProfileBundle,
    prefill_chunks: Iterable[int] = (0, 128, 512, 2048),
    contexts: Iterable[int] = (0, 512, 2048, 8192),
    decode_sequences: Iterable[int] = (0, 1, 8, 32, 64, 128),
    decode_contexts: Iterable[int] = (0, 512, 2048, 8192),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for pc in prefill_chunks:
        for kp in contexts:
            for nd in decode_sequences:
                for kd in decode_contexts:
                    if pc == 0 and nd == 0:
                        continue
                    rows.append({
                        "mode": "mix" if pc > 0 and nd > 0 else "prefill" if pc > 0 else "decode",
                        "prefill_chunk": float(pc),
                        "kv_prefill": float(kp),
                        "n_decode": float(nd),
                        "kv_decode": float(kd),
                        "time_s": profile.iteration_time_us(pc, kp, nd, kd) / 1.0e6,
                    })
    return rows

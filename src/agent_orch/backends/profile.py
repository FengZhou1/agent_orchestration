from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass(frozen=True)
class LLMProfileEstimate:
    ttft_s: float
    tbt_s: float
    response_s: float
    stable_capacity_rps: float
    kv_tokens: float
    interpolation: str


class ProfileBackend:
    """Offline profile lookup with linear interpolation and nearest fallback."""

    INPUT_COLUMNS = (
        "prompt_tokens",
        "output_tokens",
        "arrival_rate_rps",
        "long_request_fraction",
    )
    OUTPUT_COLUMNS = (
        "ttft_s",
        "tbt_s",
        "response_s",
        "stable_capacity_rps",
        "kv_tokens",
    )

    def __init__(self, frame: pd.DataFrame):
        required = {"model", "config", *self.INPUT_COLUMNS, *self.OUTPUT_COLUMNS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Profile table is missing columns: {sorted(missing)}")
        self.frame = frame.copy()
        self._interpolators: dict[tuple[str, str, str], object] = {}

    @classmethod
    def from_csv(cls, path: str | Path) -> "ProfileBackend":
        return cls(pd.read_csv(path))

    def estimate(
        self,
        model: str,
        config: str,
        prompt_tokens: float,
        output_tokens: float,
        arrival_rate_rps: float,
        long_request_fraction: float,
    ) -> LLMProfileEstimate:
        subset = self.frame[
            (self.frame["model"] == model) & (self.frame["config"] == config)
        ]
        if subset.empty:
            raise KeyError(f"No profile for {model}/{config}")
        point = np.asarray(
            [prompt_tokens, output_tokens, arrival_rate_rps, long_request_fraction],
            dtype=float,
        )
        values = []
        modes = []
        for output in self.OUTPUT_COLUMNS:
            linear = self._interpolator(subset, model, config, output, linear=True)
            estimate = float(np.asarray(linear(point)).reshape(-1)[0])
            if np.isnan(estimate):
                nearest = self._interpolator(subset, model, config, output, linear=False)
                estimate = float(np.asarray(nearest(point)).reshape(-1)[0])
                modes.append("nearest")
            else:
                modes.append("linear")
            values.append(max(0.0, estimate))
        mode = "nearest" if "nearest" in modes else "linear"
        return LLMProfileEstimate(*values, interpolation=mode)

    def _interpolator(
        self,
        subset: pd.DataFrame,
        model: str,
        config: str,
        output: str,
        linear: bool,
    ):
        key = (model, config, f"{output}:{'linear' if linear else 'nearest'}")
        if key in self._interpolators:
            return self._interpolators[key]
        points = subset[list(self.INPUT_COLUMNS)].to_numpy(dtype=float)
        values = subset[output].to_numpy(dtype=float)
        if linear and len(subset) >= len(self.INPUT_COLUMNS) + 1:
            try:
                interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
            except Exception:
                interpolator = NearestNDInterpolator(points, values)
        else:
            interpolator = NearestNDInterpolator(points, values)
        self._interpolators[key] = interpolator
        return interpolator


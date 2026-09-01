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

    BASE_INPUT_COLUMNS = (
        "prompt_tokens",
        "output_tokens",
        "arrival_rate_rps",
        "long_request_fraction",
    )
    COMPOSITION_COLUMNS = (
        "interactive_retrieval_fraction",
        "transactional_tool_fraction",
        "deep_research_fraction",
        "coding_agent_fraction",
    )
    OUTPUT_COLUMNS = (
        "ttft_s",
        "tbt_s",
        "response_s",
        "stable_capacity_rps",
        "kv_tokens",
    )

    def __init__(self, frame: pd.DataFrame):
        composition_columns = set(self.COMPOSITION_COLUMNS)
        present_composition = composition_columns & set(frame.columns)
        if present_composition and present_composition != composition_columns:
            missing_composition = sorted(composition_columns - set(frame.columns))
            raise ValueError(
                f"Profile table has an incomplete workload composition: {missing_composition}"
            )
        self.input_columns = self.BASE_INPUT_COLUMNS + (
            self.COMPOSITION_COLUMNS if present_composition else ()
        )
        required = {"model", "config", *self.input_columns, *self.OUTPUT_COLUMNS}
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
        composition: dict[str, float] | None = None,
    ) -> LLMProfileEstimate:
        subset = self.frame[
            (self.frame["model"] == model) & (self.frame["config"] == config)
        ]
        if subset.empty:
            raise KeyError(f"No profile for {model}/{config}")
        point_values = [prompt_tokens, output_tokens, arrival_rate_rps, long_request_fraction]
        if self.COMPOSITION_COLUMNS[0] in self.input_columns:
            composition = composition or {}
            point_values.extend(
                float(composition.get(column.removesuffix("_fraction"), 0.0))
                for column in self.COMPOSITION_COLUMNS
            )
        point = np.asarray(point_values, dtype=float)
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
        points = subset[list(self.input_columns)].to_numpy(dtype=float)
        values = subset[output].to_numpy(dtype=float)
        if linear and len(subset) >= len(self.input_columns) + 1:
            try:
                interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
            except Exception:
                interpolator = NearestNDInterpolator(points, values)
        else:
            interpolator = NearestNDInterpolator(points, values)
        self._interpolators[key] = interpolator
        return interpolator

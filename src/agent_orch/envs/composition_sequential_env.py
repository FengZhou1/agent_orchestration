"""Model composition decided one (application, ingress) group per step.

Reused from 博士答辩第四章 (MTS-PPO), where the decision unit is a single microservice
instance in a queue whose order the DAFG topology fixes, and one decision step places
one instance.  The analogue here is the composition row of one (application, ingress)
group: a slot's composition is built group by group, in a fixed order, and each step
decides exactly one group's model shares.

Three things are taken over deliberately:

* **One decision per step.** ``n_groups`` sub-steps per slot instead of one
  whole-vector action, so each step's choice is a single row over the active models
  rather than a joint choice over 80 numbers.
* **The external reward only arrives when the unit is complete.** The chapter's
  reward is defined at trigger points, and intermediate steps get the intrinsic
  (curiosity) reward instead.  Here the intermediate sub-steps return zero external
  reward and the rollout adds the RND term, and the slot-completing sub-step runs the
  simulator once -- so a slot still costs one simulation, not ``n_groups``.
* **The previous decision's configuration is part of the state**, as in the chapter's
  state item (4): the working composition is appended to the features.

Two deliberate deviations, because the reference chapter has nothing to reuse here:
its reward has three terms whose signs are inconsistent with its own minimisation
objective and whose weights and thresholds are never given, so the differenced reward
below differences *our* four-term utility (weights from the scenario) and is defined
so that an improvement is positive.  No small-change threshold gating is applied: it
exists in the chapter to hand small changes to the intrinsic reward, which the
intermediate sub-steps already do.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .composition_env import CompositionLibraryEnv

ActionMode = Literal["share", "model"]
RewardMode = Literal["cumulative", "delta_step", "delta_round"]


class CompositionSequentialEnv(CompositionLibraryEnv):
    """One model-share decision per (application, ingress) group, in order."""

    def __init__(
        self,
        *args: Any,
        action_mode: ActionMode = "share",
        reward_mode: RewardMode = "cumulative",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if action_mode not in ("share", "model"):
            raise ValueError("action_mode must be 'share' or 'model'")
        if reward_mode not in ("cumulative", "delta_step", "delta_round"):
            raise ValueError(
                "reward_mode must be 'cumulative', 'delta_step' or 'delta_round'"
            )
        self.action_mode: ActionMode = action_mode
        self.reward_mode: RewardMode = reward_mode
        self.n_groups = len(self.layout.model_groups)
        self._sub_step = 0
        self._working_share = np.zeros(
            (self.n_groups, len(self.layout.models)), dtype=np.float64
        )
        self._last_slot_utility: float | None = None
        # Previous round's utility at the same (context, slot), keyed by the context
        # and the slot index: the chapter's baseline is the same slot in the previous
        # training round, not the previous slot.
        self._round_baseline: dict[tuple[int, int, int], float] = {}

    # ------------------------------------------------------------------ helpers

    def _active_model_indices(self) -> list[int]:
        active = self.active_models()
        return [
            index
            for index, model in enumerate(self.layout.models)
            if model in active
        ]

    def _uniform_rows(self) -> np.ndarray:
        rows = np.zeros((self.n_groups, len(self.layout.models)), dtype=np.float64)
        active = self._active_model_indices()
        if active:
            for index in active:
                rows[:, index] = 1.0 / len(active)
        return rows

    def _resolve_row(self, row: np.ndarray) -> np.ndarray:
        """Turn the policy's row into the share vector this group will use."""

        active = self._active_model_indices()
        out = np.zeros_like(row)
        if not active:
            return out
        if self.action_mode == "model":
            # Hard choice: the row carries per-model scores, the best active model
            # takes the whole group.  Kept as an ablation -- the measured optimum is
            # a blend (25% toward uniform beat the best vertex by 0.0054 on one test
            # deployment), so a hard choice gives that back.
            best = max(active, key=lambda index: float(row[index]))
            out[best] = 1.0
            return out
        weights = np.clip(row, 0.0, None)
        total = float(weights[active].sum())
        if total <= 0.0:
            for index in active:
                out[index] = 1.0 / len(active)
            return out
        for index in active:
            out[index] = weights[index] / total
        return out

    def _intermediate_info(self) -> dict[str, Any]:
        """A sub-step that does not complete the slot: no external reward.

        Shaped like the deployment sub-step in the base environment, so the rollout
        treats it the same way (the intrinsic reward is added there).
        """

        return {
            "discount": 1.0,
            "phase": "composition",
            "period_complete": False,
            "model_group": self._sub_step,
            "model_sub_step": self._sub_step,
            "model_sub_steps": self.n_groups,
            "constraint_cost": 0.0,
            "constraint_vector": [0.0] * self.constraint_count,
            "constraint_steps": 0,
            "reward_components": {"utility": 0.0, "delta_step": 0.0, "delta_round": 0.0},
            "episode_slot": self._episode_slots,
            "episode_utility_sum": self._episode_utility_sum,
            "trace_offset": self.trace_offset,
        }

    def _differenced_reward(
        self, utility: float, slot_index: int, info: dict[str, Any]
    ) -> float:
        """The slot's reward under the configured mode; both variants are reported."""

        delta_step = (
            0.0
            if self._last_slot_utility is None
            else utility - self._last_slot_utility
        )
        key = (self._fixed_deployment_index, self.trace_offset, slot_index)
        baseline = self._round_baseline.get(key)
        delta_round = 0.0 if baseline is None else utility - baseline
        self._round_baseline[key] = utility
        self._last_slot_utility = utility
        components = info.setdefault("reward_components", {})
        components["delta_step"] = delta_step
        components["delta_round"] = delta_round
        components["cumulative"] = utility
        if self.reward_mode == "delta_step":
            return delta_step
        if self.reward_mode == "delta_round":
            return delta_round
        return utility

    # ------------------------------------------------------------------ gym api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._sub_step = 0
        self._working_share = self._uniform_rows()
        self._last_slot_utility = None
        info["model_group"] = self._sub_step
        info["model_sub_steps"] = self.n_groups
        return observation, info

    def step(self, action: dict[str, Any]):
        raw = np.asarray(action["model"], dtype=float).reshape(
            self.n_groups, len(self.layout.models)
        )
        group = self._sub_step
        self._working_share[group] = self._resolve_row(raw[group].copy())

        if group < self.n_groups - 1:
            self._sub_step = group + 1
            return self._observation(), 0.0, False, False, self._intermediate_info()

        # Last group of the pass: the composition is complete, run one slot.
        slot_index = self._period_index
        self._sub_step = 0
        observation, _reward, terminated, truncated, info = super().step(
            {"deploy": 0, "model": self._working_share.reshape(-1)}
        )
        info["model_group"] = 0
        info["model_sub_steps"] = self.n_groups
        if info.get("period_complete"):
            reward = self._differenced_reward(
                float(info["utility"]), slot_index, info
            )
        else:  # pragma: no cover - the composition phase always completes a slot
            reward = _reward
        return observation, reward, terminated, truncated, info

    def _observation(self):
        observation = super()._observation()
        # The policy has to know which group it is deciding, and both distributions
        # have to be evaluated for exactly that group, so the index travels with the
        # observation and with every transition.
        observation["model_group"] = self._sub_step
        return observation

    # ------------------------------------------------------------------ features

    def _feature_vector(self) -> np.ndarray:
        # The base constructor sizes the observation space by calling this before the
        # subclass attributes exist, so every read here has to tolerate that.
        base = super()._feature_vector()
        groups = len(self.layout.model_groups)
        width = len(self.layout.models)
        sub_step = int(getattr(self, "_sub_step", 0))
        working = getattr(self, "_working_share", None)
        if working is None or getattr(working, "shape", None) != (groups, width):
            working = np.zeros((groups, width), dtype=np.float64)
        extra = [
            sub_step / max(1, groups),
            float(sub_step == groups - 1),
        ]
        # The configuration produced so far -- the chapter's state item (4): what was
        # decided at the previous step is state for the next one.
        extra.extend(working.reshape(-1).tolist())
        return np.concatenate([base, np.asarray(extra, dtype=np.float32)])

    def action_masks(self) -> dict[str, np.ndarray]:
        masks = super().action_masks()
        return masks

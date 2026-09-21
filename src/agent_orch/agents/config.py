"""PPO hyper-parameters and the constraint-vector bookkeeping around them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal


_CONSTRAINT_VECTOR_FIELDS = (
    "constraint_limits",
    "lagrangian_learning_rates",
    "initial_lagrange_multipliers",
    "max_lagrange_multipliers",
)


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    composition_gamma: float = 0.0
    gae_lambda: float = 0.95
    deployment_gae_lambda: float = 1.0
    clip_ratio: float = 0.2
    learning_rate: float = 3.0e-4
    composition_learning_rate: float = 3.0e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coefficient: float = 0.01
    composition_entropy_coefficient: float = 0.0001
    composition_concentration_min: float = 1.0
    shared_composition_head: bool = False
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    constrained: bool = True
    constraint_limits: tuple[float, float] = (0.0, 0.0)
    lagrangian_learning_rates: tuple[float, float] = (
        0.05,
        0.05,
    )
    initial_lagrange_multipliers: tuple[float, float] = (
        0.0,
        0.0,
    )
    max_lagrange_multipliers: tuple[float, float] = (
        50.0,
        50.0,
    )
    shaping_coefficient: float = 1.0
    exploration_mode: Literal["rnd", "none", "icm"] = "rnd"
    rnd_feature_size: int = 64
    rnd_learning_rate: float = 1.0e-4
    rnd_initial_weight: float = 0.01
    rnd_final_weight: float = 0.0
    rnd_reward_clip: float = 5.0
    rnd_loss_coefficient: float = 1.0
    icm_scale: float = 0.01
    training_phase: Literal["joint", "deployment", "composition"] = "joint"
    composition_group_relative_advantages: bool = False
    composition_group_relative_mode: Literal["center", "standardize"] = "center"
    composition_fixed_concentration: float | None = None
    target_kl: float | None = None
    factorized_credit: bool = False
    composition_group_features: bool = False

    def for_constraint_count(self, count: int) -> "PPOConfig":
        """Resize every constraint vector to ``count`` entries.

        The number of constraints follows the objective profile rather than the
        trainer, so a config written for two constraints has to be widened to
        the three-constraint ``slo_constrained`` profile (and narrowed back when
        the environment reports fewer).  A short vector repeats its last value;
        a config whose four vectors already match is returned unchanged.
        """

        count = max(int(count), 1)
        if all(
            len(getattr(self, field)) == count for field in _CONSTRAINT_VECTOR_FIELDS
        ):
            return self
        return replace(
            self,
            **{
                field: _resize_constraint_vector(getattr(self, field), count)
                for field in _CONSTRAINT_VECTOR_FIELDS
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """The config as a plain mapping, for run logs."""

        return asdict(self)


def _resize_constraint_vector(
    values: tuple[float, ...], count: int
) -> tuple[float, ...]:
    if len(values) >= count:
        return tuple(values[:count])
    filler = values[-1] if values else 0.0
    return tuple(values) + (filler,) * (count - len(values))

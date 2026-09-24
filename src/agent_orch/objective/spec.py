"""Declarative specification of the per-period objective and its constraints.

Two profiles ship here.  ``legacy`` reproduces the four-term weighted sum that
earlier experiment runs used, so existing result files keep their meaning.
``slo_constrained`` moves SLO attainment out of the objective and into the
constraint vector: attainment is what the system must *satisfy*, not what it
should *trade against quality*.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

CostBoundsSource = Literal["library", "theoretical"]
LatencyMap = Literal["clip", "smooth"]
ObjectiveProfile = Literal["legacy", "slo_constrained"]

PROFILES: tuple[str, ...] = ("legacy", "slo_constrained")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Weights, constraint targets and normalisation choices for one period."""

    profile: str = "legacy"
    quality_weight: float = 0.25
    cost_weight: float = 0.25
    latency_weight: float = 0.25
    goodput_weight: float = 0.25
    llm_utilization_target: float = 0.9
    service_utilization_target: float = 0.9
    network_utilization_target: float | None = None
    attainment_target: float | None = None
    cost_bounds: CostBoundsSource = "library"
    latency_map: LatencyMap = "clip"

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown objective profile {self.profile!r}; choose from {PROFILES}")
        if self.cost_bounds not in ("library", "theoretical"):
            raise ValueError(f"Unknown cost_bounds {self.cost_bounds!r}")
        if self.latency_map not in ("clip", "smooth"):
            raise ValueError(f"Unknown latency_map {self.latency_map!r}")
        for name in ("quality_weight", "cost_weight", "latency_weight", "goodput_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.attainment_target is not None and not 0.0 <= self.attainment_target <= 1.0:
            raise ValueError("attainment_target must lie in [0, 1]")
        if self.network_utilization_target is not None and not 0.0 < self.network_utilization_target <= 1.0:
            raise ValueError("network_utilization_target must lie in (0, 1]")

    @property
    def constraint_names(self) -> tuple[str, ...]:
        """Constraint components, in the order the evaluator emits them."""

        names = ["llm", "service"]
        if self.network_utilization_target is not None:
            names.append("network")
        if self.attainment_target is not None:
            names.append("attainment")
        return tuple(names)

    @property
    def constraint_targets(self) -> tuple[float, ...]:
        """The ``limit`` side of every constraint: excess above it is penalised."""

        targets = [self.llm_utilization_target, self.service_utilization_target]
        if self.network_utilization_target is not None:
            targets.append(self.network_utilization_target)
        if self.attainment_target is not None:
            targets.append(self.attainment_target)
        return tuple(targets)

    @property
    def objective_terms(self) -> tuple[str, ...]:
        """Terms with a non-zero weight, in canonical order."""

        pairs = (
            ("goodput_normalized", self.goodput_weight),
            ("quality_normalized", self.quality_weight),
            ("cost_normalized", self.cost_weight),
            ("latency_normalized", self.latency_weight),
        )
        return tuple(name for name, weight in pairs if weight > 0.0)

    def weight_of(self, term: str) -> float:
        return float(getattr(self, term.replace("_normalized", "_weight")))

    @staticmethod
    def legacy() -> "ObjectiveSpec":
        return ObjectiveSpec(profile="legacy")

    @staticmethod
    def slo_constrained(attainment_target: float | None = 0.9) -> "ObjectiveSpec":
        """Quality/cost/latency objective with SLO attainment as a constraint.

        The SLO attainment already aggregates per-application latency and
        deadline satisfaction, so keeping it inside the objective as a fourth
        equally weighted term lets it cancel against the quality it is meant to
        bound.  Here it becomes a constraint instead, and the objective is left
        free to trade quality against cost and latency.

        Pass ``attainment_target=None`` to drop the third constraint.  A stage
        whose deployment is exogenous must do this: attainment is bounded by the
        deployment (a deployment with one weak model cannot reach any target, no
        matter how its traffic is split), so the dual variable would chase a
        violation the stage cannot remove and grow without bound.  The latency
        term still carries the SLO-driven routing pressure.
        """

        return ObjectiveSpec(
            profile="slo_constrained",
            quality_weight=0.5,
            cost_weight=0.25,
            latency_weight=0.25,
            goodput_weight=0.0,
            attainment_target=attainment_target,
        )

    @staticmethod
    def from_mapping(payload: Mapping[str, Any] | None) -> "ObjectiveSpec":
        if not payload:
            return ObjectiveSpec.legacy()
        profile = str(payload.get("profile", "legacy"))
        base = (
            ObjectiveSpec.slo_constrained(
                float(payload.get("attainment_target", 0.9))
            )
            if profile == "slo_constrained"
            else ObjectiveSpec.legacy()
        )
        known = {
            "quality_weight",
            "cost_weight",
            "latency_weight",
            "goodput_weight",
            "llm_utilization_target",
            "service_utilization_target",
            "network_utilization_target",
            "attainment_target",
            "cost_bounds",
            "latency_map",
        }
        overrides = {
            key: payload[key]
            for key in known
            if key in payload and payload[key] is not None
        }
        if "attainment_target" in overrides and base.attainment_target is None:
            # An explicit target on the legacy profile switches the third
            # constraint on without changing the four objective weights.
            base = replace(base, attainment_target=float(overrides["attainment_target"]))
            overrides.pop("attainment_target")
        return replace(base, **overrides) if overrides else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "quality_weight": self.quality_weight,
            "cost_weight": self.cost_weight,
            "latency_weight": self.latency_weight,
            "goodput_weight": self.goodput_weight,
            "llm_utilization_target": self.llm_utilization_target,
            "service_utilization_target": self.service_utilization_target,
            "network_utilization_target": self.network_utilization_target,
            "attainment_target": self.attainment_target,
            "cost_bounds": self.cost_bounds,
            "latency_map": self.latency_map,
            "constraint_names": list(self.constraint_names),
        }

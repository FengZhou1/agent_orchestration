"""Sensitivity audit for the objective.

A term that barely moves across the decisions the agent can actually take
carries no gradient, however large its nominal weight is.  This module measures
that directly: for every objective term it compares the spread *between*
decision labels (policies, deployments, load levels) against the term's weight,
and flags the terms whose influence on the utility is negligible.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

from .evaluator import ObjectiveValue
from .references import ReferenceScales
from .spec import ObjectiveSpec

DEAD_TERM_THRESHOLD = 0.05


@dataclass
class TermSummary:
    """Between-decision spread of one normalised objective term."""

    term: str
    weight: float
    n_observations: int
    minimum: float
    maximum: float
    mean: float
    stdev: float
    between_label_spread: float
    labels: int
    mean_saturation: float | None = None

    @property
    def weighted_spread(self) -> float:
        return abs(self.weight) * self.between_label_spread

    @property
    def weight_scale(self) -> float:
        return abs(self.weight)

    @property
    def influence(self) -> float:
        """Share of the term's nominal weight that survives as decision signal."""

        if self.weight_scale <= 0.0:
            return 0.0
        return self.weighted_spread / self.weight_scale

    @property
    def is_inactive(self) -> bool:
        """True when the profile assigns this term no weight at all."""

        return self.weight == 0.0

    def is_dead(self, threshold: float = DEAD_TERM_THRESHOLD) -> bool:
        return not self.is_inactive and self.influence < threshold

    def to_row(self) -> dict[str, object]:
        return {
            "term": self.term,
            "weight": round(self.weight, 6),
            "n": self.n_observations,
            "labels": self.labels,
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "mean": round(self.mean, 6),
            "stdev": round(self.stdev, 6),
            "between_label_spread": round(self.between_label_spread, 6),
            "weighted_spread": round(self.weighted_spread, 6),
            "influence": round(self.influence, 4),
            "inactive": bool(self.is_inactive),
            "dead": bool(self.is_dead()),
        }


@dataclass
class ConstraintSummary:
    """How often a constraint binds, and by how much."""

    name: str
    target: float
    n_observations: int
    binding_fraction: float
    mean_excess: float
    max_excess: float

    def to_row(self) -> dict[str, object]:
        return {
            "constraint": self.name,
            "target": round(self.target, 6),
            "n": self.n_observations,
            "binding_fraction": round(self.binding_fraction, 4),
            "mean_excess": round(self.mean_excess, 6),
            "max_excess": round(self.max_excess, 6),
        }


@dataclass
class _Accumulator:
    values: list[float] = field(default_factory=list)
    by_label: "OrderedDict[str, list[float]]" = field(default_factory=OrderedDict)
    saturation: list[float] = field(default_factory=list)

    def add(self, label: str, value: float, saturation: float | None) -> None:
        self.values.append(float(value))
        self.by_label.setdefault(label, []).append(float(value))
        if saturation is not None:
            self.saturation.append(float(saturation))

    def between_label_spread(self) -> float:
        if len(self.by_label) < 2:
            return 0.0
        means = [fmean(group) for group in self.by_label.values()]
        return float(max(means) - min(means))


class ObjectiveAudit:
    """Collect objective evaluations and report which terms actually discriminate."""

    def __init__(
        self,
        spec: ObjectiveSpec,
        references: ReferenceScales | None = None,
    ) -> None:
        self.spec = spec
        self.references = references
        self._terms: "OrderedDict[str, _Accumulator]" = OrderedDict(
            (term, _Accumulator()) for term in ("goodput_normalized", "quality_normalized", "cost_normalized", "latency_normalized")
        )
        self._utility = _Accumulator()
        self._constraints: "OrderedDict[str, _Accumulator]" = OrderedDict(
            (name, _Accumulator()) for name in spec.constraint_names
        )
        self._diagnostics: dict[str, list[float]] = {}

    def observe(self, label: str, value: ObjectiveValue) -> None:
        saturation = value.diagnostics.get("latency_saturation_fraction")
        for term, accumulator in self._terms.items():
            if term in value.components:
                accumulator.add(label, value.components[term], saturation if term == "latency_normalized" else None)
        self._utility.add(label, value.utility, None)
        for index, name in enumerate(self.spec.constraint_names):
            if index < len(value.constraints):
                self._constraints[name].add(label, value.constraints[index], None)
        for key, raw in value.diagnostics.items():
            if isinstance(raw, (int, float)):
                self._diagnostics.setdefault(key, []).append(float(raw))

    def observe_many(self, label: str, values: Iterable[ObjectiveValue]) -> None:
        for value in values:
            self.observe(label, value)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._utility.by_label)

    def term_summaries(self) -> list[TermSummary]:
        summaries: list[TermSummary] = []
        for term, accumulator in self._terms.items():
            if not accumulator.values:
                continue
            weight = self.spec.weight_of(term)
            summaries.append(
                TermSummary(
                    term=term,
                    weight=weight,
                    n_observations=len(accumulator.values),
                    minimum=min(accumulator.values),
                    maximum=max(accumulator.values),
                    mean=fmean(accumulator.values),
                    stdev=pstdev(accumulator.values) if len(accumulator.values) > 1 else 0.0,
                    between_label_spread=accumulator.between_label_spread(),
                    labels=len(accumulator.by_label),
                    mean_saturation=fmean(accumulator.saturation) if accumulator.saturation else None,
                )
            )
        return summaries

    def constraint_summaries(self) -> list[ConstraintSummary]:
        targets = dict(zip(self.spec.constraint_names, self.spec.constraint_targets))
        summaries: list[ConstraintSummary] = []
        for name, accumulator in self._constraints.items():
            if not accumulator.values:
                continue
            binding = [value for value in accumulator.values if value > 0.0]
            summaries.append(
                ConstraintSummary(
                    name=name,
                    target=float(targets.get(name, 0.0)),
                    n_observations=len(accumulator.values),
                    binding_fraction=len(binding) / len(accumulator.values),
                    mean_excess=fmean(accumulator.values),
                    max_excess=max(accumulator.values),
                )
            )
        return summaries

    def dead_terms(self, threshold: float = DEAD_TERM_THRESHOLD) -> list[str]:
        return [
            summary.term
            for summary in self.term_summaries()
            if summary.weight > 0.0 and summary.is_dead(threshold)
        ]

    def utility_summary(self) -> TermSummary:
        accumulator = self._utility
        return TermSummary(
            term="utility",
            weight=1.0,
            n_observations=len(accumulator.values),
            minimum=min(accumulator.values) if accumulator.values else 0.0,
            maximum=max(accumulator.values) if accumulator.values else 0.0,
            mean=fmean(accumulator.values) if accumulator.values else 0.0,
            stdev=pstdev(accumulator.values) if len(accumulator.values) > 1 else 0.0,
            between_label_spread=accumulator.between_label_spread(),
            labels=len(accumulator.by_label),
        )

    def label_means(self) -> dict[str, float]:
        return {label: fmean(group) for label, group in self._utility.by_label.items()}

    def rows(self) -> list[dict[str, object]]:
        rows = [summary.to_row() for summary in self.term_summaries()]
        rows.extend(summary.to_row() for summary in self.constraint_summaries())
        return rows

    def to_markdown(self, title: str = "Objective sensitivity") -> str:
        lines = [f"### {title}", ""]
        utility = self.utility_summary()
        lines.append(
            f"utility: n={utility.n_observations}, labels={utility.labels}, "
            f"min={utility.minimum:+.4f}, max={utility.maximum:+.4f}, "
            f"between-label spread={utility.between_label_spread:.4f}"
        )
        lines.append("")
        lines.append("| term | weight | min | max | between-label spread | weighted | influence | status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for summary in self.term_summaries():
            status = "inactive" if summary.is_inactive else ("DEAD" if summary.is_dead() else "")
            lines.append(
                "| {term} | {weight:.3f} | {minimum:+.4f} | {maximum:+.4f} | "
                "{between_label_spread:.4f} | {weighted_spread:.4f} | {influence:.1%} | {status} |".format(
                    term=summary.term,
                    weight=summary.weight,
                    minimum=summary.minimum,
                    maximum=summary.maximum,
                    between_label_spread=summary.between_label_spread,
                    weighted_spread=summary.weighted_spread,
                    influence=summary.influence,
                    status=status,
                )
            )
        constraints = self.constraint_summaries()
        if constraints:
            lines.append("")
            lines.append("| constraint | target | binding | mean excess | max excess |")
            lines.append("|---|---|---|---|---|")
            for summary in constraints:
                lines.append(
                    f"| {summary.name} | {summary.target:.3f} | "
                    f"{summary.binding_fraction:.1%} | {summary.mean_excess:.4f} | "
                    f"{summary.max_excess:.4f} |"
                )
        dead = self.dead_terms()
        lines.append("")
        lines.append(
            "dead terms: " + (", ".join(dead) if dead else "none")
            + f"  (threshold: influence < {DEAD_TERM_THRESHOLD:.0%})"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "utility": self.utility_summary().to_row(),
            "terms": [summary.to_row() for summary in self.term_summaries()],
            "constraints": [summary.to_row() for summary in self.constraint_summaries()],
            "dead_terms": self.dead_terms(),
            "label_means": self.label_means(),
        }


def summarize_terms(
    spec: ObjectiveSpec,
    observations: Mapping[str, Sequence[ObjectiveValue]],
) -> ObjectiveAudit:
    """Convenience wrapper: audit a mapping of label to observed values."""

    audit = ObjectiveAudit(spec)
    for label, values in observations.items():
        audit.observe_many(label, values)
    return audit

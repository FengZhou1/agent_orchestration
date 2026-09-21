"""Composition-only environment: one fixed feasible deployment per episode.

Contexts are drawn from the offline deployment library (``scripts/build_deployment_library.py``)
rather than from a runtime greedy catalogue, so the composition policy sees the
whole range of active-model sets and replica counts it has to condition on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent_orch.deployment import (
    DeploymentLibrary,
    DeploymentLibraryBuilder,
    FixedDeploymentSampler,
    StratifiedSampler,
    deployment_signature,
)
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.models import DeploymentDecision, Scenario
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace

from .base import BaseOrchestrationEnv, _normalized_or_uniform, resolve_deployment_library


class CompositionLibraryEnv(BaseOrchestrationEnv):
    """Composition training under one fixed deployment context per episode."""

    def __init__(
        self,
        scenario: Scenario,
        *args,
        fixed_deployment_count: int = 128,
        fixed_deployment_index: int | None = None,
        sampler_mode: str = "cycle",
        library: DeploymentLibrary | None = None,
        library_path: str | Path | None = None,
        baseline_periods: int = 0,
        baseline_warmup: int = 1,
        use_uniform_baseline: bool = True,
        **kwargs,
    ):
        self.fixed_deployment_count = max(1, int(fixed_deployment_count))
        self.fixed_deployment_index = fixed_deployment_index
        self.sampler_mode = sampler_mode
        self.baseline_periods = max(0, int(baseline_periods))
        self.baseline_warmup = max(0, int(baseline_warmup))
        self.use_uniform_baseline = bool(use_uniform_baseline)
        self._library_override = library
        self._library_path = library_path
        self._sampler: StratifiedSampler | FixedDeploymentSampler | None = None
        self._composition_baseline_utility: float | None = None
        self._fixed_deployment_index = 0
        self._fixed_deployment_catalog: list[DeploymentDecision] = []
        if library is not None:
            # An explicit library object must win over the on-disk default, or a
            # caller-supplied split would be silently replaced by the full
            # catalogue while the environment still reports the split's size.
            kwargs["deployment_library"] = library
            kwargs.pop("deployment_library_path", None)
        super().__init__(scenario, *args, **kwargs)
        self.deployment_library = self._resolve_library()
        self._fixed_deployment_catalog = [
            entry.to_deployment() for entry in self.deployment_library.entries
        ]
        self._sampler = self._build_sampler()

    def _resolve_library(self) -> DeploymentLibrary:
        """Use the library the base environment already resolved, if any.

        The base class resolves it once from ``deployment_library`` /
        ``deployment_library_path``; resolving again here would silently swap in
        the on-disk library and discard a caller-supplied split.
        """

        if self.deployment_library is not None:
            return self.deployment_library
        library = resolve_deployment_library(
            self.scenario, self._library_override, self._library_path
        )
        if library is not None:
            return library
        return DeploymentLibraryBuilder(self.scenario, self.planner).build(
            min_entries=self.fixed_deployment_count,
            max_entries=max(self.fixed_deployment_count, 512),
        )

    def _build_sampler(self):
        if self.fixed_deployment_index is not None:
            return FixedDeploymentSampler(self.deployment_library, self.fixed_deployment_index)
        return StratifiedSampler(
            self.deployment_library, seed=self._seed, mode=self.sampler_mode
        )

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
        self._capacity_plan = self.planner.plan(
            self.simulator.current_arrival_rates(),
            self._planning_model_share,
            self._period_index,
        )
        self._deployment_targets = []
        self._deployment_target_index = 0
        self._current_demand = None
        self._deployment_actions_in_period = 0
        self._period_initial_potential = self._deployment_potential()
        self._last_potential = self._period_initial_potential
        self.phase = self.COMPOSITION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, _ = super().reset(seed=seed, options=options)
        requested = (options or {}).get("fixed_deployment_index")
        if requested is not None:
            self._fixed_deployment_index = int(requested) % len(self._fixed_deployment_catalog)
        else:
            # Advance the sampler; never re-seed it here.  train_ppo calls reset()
            # once per episode with a fresh seed, and re-seeding would rewind the
            # stratified cycle to its first stratum every time -- the composition
            # policy would then only ever see the few deployments of one stratum.
            self._fixed_deployment_index = int(self._sampler.next_index())
        self.current_deployment = self._fixed_deployment_catalog[
            self._fixed_deployment_index
        ].copy()
        self._base_deployment = self.current_deployment.copy()
        self._composition_baseline_utility = (
            self._estimate_uniform_baseline() if self.use_uniform_baseline else None
        )
        self.phase = self.COMPOSITION
        entry = self._entry_at(self._fixed_deployment_index)
        return self._observation(), {
            "discount": self.gamma,
            "phase": "composition",
            "fixed_deployment_index": self._fixed_deployment_index,
            "fixed_deployment_count": len(self._fixed_deployment_catalog),
            "deployment_source_index": entry.index,
            "deployment_stratum": entry.stratum,
            "deployment_active_models": list(entry.active_models(self.scenario)),
        }

    def _entry_at(self, position: int):
        """Entry at a *positional* index in this environment's catalogue.

        Sub-libraries keep their source indices, so ``DeploymentLibrary.entry``
        resolves by index value; the environment indexes its catalogue by
        position and must stay positional for the two to agree.
        """

        return self.deployment_library.entries[position]

    def _normalized_uniform_share(self) -> dict[tuple[str, str, str], float]:
        """Uniform composition over the models this deployment activates.

        ``_uniform_model_share`` spreads mass over every model in the scenario,
        so for a deployment that activates fewer than all of them it does not sum
        to one per group.  The physical router consumes the share as-is, so an
        unnormalised vector under-loads the system and makes the uniform baseline
        far too low -- by up to 0.15, comparable to the whole utility range.
        This mirrors what ``decode_routing`` does with a uniform action.
        """

        active = self.active_models()
        share: dict[tuple[str, str, str], float] = {}
        for group in self.layout.model_groups:
            weights = {model: 1.0 for model in self.layout.models if model in active}
            normalized = _normalized_or_uniform(weights)
            for model in self.layout.models:
                share[(*group, model)] = normalized.get(model, 0.0)
        return share

    def _estimate_uniform_baseline(self) -> float:
        """Utility of the current deployment under a uniform composition.

        Serves as a control variate: subtracting it removes the part of the
        period utility that the deployment fixes regardless of composition, so
        the composition actor sees the composition's marginal effect.

        It must be measured under the *same protocol* as the utility it is
        subtracted from.  A cold-start single period carries the one-off
        ``load_cost`` and no utilization feedback, so it sits far below the
        steady utility the episode reports; subtracting it would leave a
        deployment-dependent bias rather than a control variate.  The scratch
        simulator therefore replays the same number of periods the episode runs,
        with the router fed the utilization it accumulates.
        """

        periods = max(1, self.baseline_periods or self.max_periods)
        simulator = Simulator(
            self.scenario,
            max_mapping_samples=self.simulator.workflow.max_mapping_samples,
        )
        simulator.set_arrival_trace(self.simulator.arrival_trace)
        simulator.reset(self._seed)
        arrival_rates = simulator.current_arrival_rates()
        uniform = self._normalized_uniform_share()
        utilities: list[float] = []
        for period in range(periods):
            previous = None if period == 0 else simulator.last_metrics
            routing = self.physical_router.route(
                self.current_deployment, uniform, previous, arrival_rates
            )
            metrics = simulator.step(self.current_deployment, routing).metrics
            if period >= self.baseline_warmup:
                utilities.append(float(self.objective.evaluate(metrics, arrival_rates).utility))
        if not utilities:
            return float("nan")
        return float(sum(utilities) / len(utilities))

    def stratum_counts(self) -> dict[str, int]:
        return self.deployment_library.strata_counts()

    @staticmethod
    def _deployment_signature(deployment: DeploymentDecision):
        """Historical name for the deployment fingerprint used by tests."""

        return deployment_signature(deployment.llm_active, deployment.tool_replicas)


__all__ = ["CompositionLibraryEnv"]

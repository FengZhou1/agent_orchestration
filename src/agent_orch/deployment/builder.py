from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import CandidateInstance, DeploymentDecision, Scenario

from .library import (
    DeploymentEntry,
    DeploymentLibrary,
    deployment_signature,
)


class DeploymentLibraryBuilder:
    """Build a stratified, auditable feasible deployment library offline.

    Strata and their priority (a deployment belongs to the first stratum that
    produces it)::

        L5_extreme                   4 extreme deployments (largest model, smallest
                                     model, cheapest feasible, capacity-maximal)
        L1_model_subset:<models>     exactly one non-empty proper subset of models
        L2_capacity:cap<percent>     25/50/75/100% of the total candidate GPU budget
        L3_replicas:<replicas>       1/2/4 replicas per service over 1 or 3 servers
        L4_placement:<tier>          full model set, 1 replica per service, rotated
                                     placement in cheap/balanced/expensive order

    ``cost_per_period`` counts steady-state serving cost only: active candidates and
    service replicas priced at ``running_cost_per_slot`` over
    ``simulation.orchestration_period_s``. ``load_cost`` and ``start_cost`` are
    switching costs and are deliberately excluded.
    """

    EXTREME_STRATUM = "L5_extreme"
    SUBSET_PREFIX = "L1_model_subset"
    CAPACITY_PREFIX = "L2_capacity"
    REPLICA_PREFIX = "L3_replicas"
    PLACEMENT_PREFIX = "L4_placement"

    CAPACITY_LEVELS = (0.25, 0.50, 0.75, 1.00)
    REPLICA_LEVELS = (1, 2, 4)
    REPLICA_SERVER_SPREADS = (1, 3)
    PLACEMENT_TIERS = ("cheap", "balanced", "expensive")

    def __init__(
        self, scenario: Scenario, planner: CapacityPlanner | None = None
    ) -> None:
        self.scenario = scenario
        self.planner = planner if planner is not None else CapacityPlanner(scenario)
        self._registry: dict[str, DeploymentEntry] = {}
        self._max_entries = 512
        self._extreme_skipped: list[str] = []

    def build(self, min_entries: int = 128, max_entries: int = 512) -> DeploymentLibrary:
        if min_entries < 1:
            raise ValueError("min_entries must be positive")
        if max_entries < min_entries:
            raise ValueError("max_entries must be at least min_entries")

        self._registry = {}
        self._max_entries = max_entries
        self._extreme_skipped = []

        self._build_extremes()
        self._build_model_subsets()
        self._build_capacity_levels()
        self._build_replica_levels()
        self._build_placement_entries(min_entries)
        if len(self._registry) < min_entries:
            self._extend_placement(min_entries)

        entries = tuple(self._registry.values())
        if len(entries) > max_entries:
            entries = entries[:max_entries]
        entries = tuple(
            DeploymentEntry(
                index=position,
                stratum=entry.stratum,
                llm_active=entry.llm_active,
                tool_replicas=entry.tool_replicas,
                n_models=entry.n_models,
                n_llm=entry.n_llm,
                n_tool_replicas=entry.n_tool_replicas,
                total_gpu=entry.total_gpu,
                cost_per_period=entry.cost_per_period,
                signature=entry.signature,
            )
            for position, entry in enumerate(entries)
        )

        shortfall: str | None = None
        if len(entries) < min_entries:
            shortfall = (
                f"Only {len(entries)} distinct feasible deployments exist for "
                f"{self.scenario.id} under the current strata; the catalog shape is "
                "limited by the candidate/placement space, not by the L4 rotation."
            )
        candidate_models = {
            candidate_id: candidate.model
            for candidate_id, candidate in self.scenario.candidates.items()
        }
        metadata: dict[str, Any] = {
            "builder": "DeploymentLibraryBuilder",
            "scenario_id": self.scenario.id,
            "candidate_models": candidate_models,
            "n_entries": len(entries),
            "min_entries": min_entries,
            "max_entries": max_entries,
            "strata_counts": _strata_counts(entries),
            "n_active_model_sets": len(_active_model_sets(entries, candidate_models)),
            "total_candidate_gpu": self._candidate_gpu_total(),
            "extreme_skipped": list(self._extreme_skipped),
            "shortfall_reason": shortfall,
            "cost_definition": (
                "sum(active running_cost_per_slot) * orchestration_period_s"
                " + sum(replicas * tool running_cost_per_slot) * orchestration_period_s"
            ),
        }
        return DeploymentLibrary(
            scenario_id=self.scenario.id,
            scenario_hash=DeploymentLibrary.scenario_hash_of(self.scenario),
            entries=entries,
            metadata=metadata,
        )

    def _build_extremes(self) -> None:
        parameter_order = sorted(
            self.scenario.models.values(),
            key=lambda item: (-item.parameter_count, item.id),
        )
        for label, model in (
            ("largest_model", parameter_order[0].id),
            ("smallest_model", parameter_order[-1].id),
        ):
            deployment = self._empty_with_tools()
            for candidate in self._candidates_by_flops(model):
                if self.planner.feasible_activation(deployment, candidate.id):
                    deployment.llm_active[candidate.id] = 1
            status = self._register(deployment, self.EXTREME_STRATUM)
            if status != "added":
                self._extreme_skipped.append(f"{label}:{status}")

        try:
            cheapest = self.planner.initial_deployment()
        except ValueError:
            self._extreme_skipped.append("cheapest_feasible:infeasible")
        else:
            status = self._register(cheapest, self.EXTREME_STRATUM)
            if status != "added":
                self._extreme_skipped.append(f"cheapest_feasible:{status}")

        capacity = self._empty_with_tools()
        self._greedy_activate(capacity, self._candidates_by_flops())
        status = self._register(capacity, self.EXTREME_STRATUM)
        if status != "added":
            self._extreme_skipped.append(f"capacity_maximal:{status}")

    def _build_model_subsets(self) -> None:
        models = tuple(sorted(self.scenario.models))
        subsets: list[tuple[str, ...]] = []
        for size in range(1, len(models)):
            subsets.extend(combinations(models, size))
        for subset in subsets:
            deployment = self._empty_with_tools()
            complete = True
            for model in subset:
                if not self._activate_cheapest(deployment, model):
                    complete = False
                    break
            if not complete:
                continue
            self._register(deployment, f"{self.SUBSET_PREFIX}:{'+'.join(subset)}")

    def _build_capacity_levels(self) -> None:
        budget_total = self._candidate_gpu_total()
        order = self._candidates_by_flops()
        fill = self._candidates_by_cost()
        for ratio in self.CAPACITY_LEVELS:
            budget = budget_total * ratio
            deployment = self._empty_with_tools()
            self._greedy_activate(deployment, order, budget=budget)
            # A second, cheap-first pass fills GPU capacity the flops-first order
            # leaves unused, so ``total_gpu`` sits as close to the budget as the
            # physical GPU pool allows.
            self._greedy_activate(deployment, fill, budget=budget)
            self._register(
                deployment, f"{self.CAPACITY_PREFIX}:cap{int(round(ratio * 100))}"
            )

    def _build_replica_levels(self) -> None:
        for replicas in self.REPLICA_LEVELS:
            effective = min(
                replicas,
                self.scenario.simulation.max_tool_replicas_per_server,
                len(self.scenario.servers),
            )
            for spread in self.REPLICA_SERVER_SPREADS:
                if spread > len(self.scenario.servers):
                    continue
                deployment = self._empty_with_tools(replicas=0)
                complete = all(
                    self._activate_cheapest(deployment, model)
                    for model in sorted(self.scenario.models)
                )
                if not complete:
                    continue
                if not self._place_tools(deployment, replicas=effective, spread=spread):
                    continue
                self._register(deployment, f"{self.REPLICA_PREFIX}:{effective}")

    def _build_placement_entries(self, target: int) -> None:
        candidates = self._candidate_list()
        n_servers = max(1, len(self.scenario.servers))
        for tier_index, tier in enumerate(self.PLACEMENT_TIERS):
            order = self._placement_order(tier)
            for offset in range(len(candidates)):
                if len(self._registry) >= min(target, self._max_entries):
                    return
                self._try_placement(
                    order, offset, (offset + tier_index) % n_servers, tier
                )

    def _extend_placement(self, target: int) -> None:
        """Sweep the remaining (LLM offset, tool offset) combinations.

        Used only when the regular strata leave the library below ``min_entries``:
        the extra entries keep the L4 shape (full model set, one replica per
        service) and only vary where the LLM instances and the replicas sit.
        """
        candidates = self._candidate_list()
        servers = self.scenario.servers
        tiers = self.PLACEMENT_TIERS
        orders = {tier: self._placement_order(tier) for tier in tiers}
        index = 0
        for llm_offset in range(len(candidates)):
            for tool_offset in range(len(servers)):
                if len(self._registry) >= target or len(self._registry) >= self._max_entries:
                    return
                tier = tiers[index % len(tiers)]
                self._try_placement(orders[tier], llm_offset, tool_offset, tier)
                index += 1

    def _try_placement(
        self,
        order: Sequence[CandidateInstance],
        llm_offset: int,
        tool_offset: int,
        tier: str,
    ) -> bool:
        deployment = self._empty_with_tools(replicas=0)
        for model_index, model in enumerate(sorted(self.scenario.models)):
            candidates = [item for item in order if item.model == model]
            rotated = _rotate(candidates, llm_offset + model_index)
            selected = next(
                (
                    candidate
                    for candidate in rotated
                    if self.planner.feasible_activation(deployment, candidate.id)
                ),
                None,
            )
            if selected is None:
                return False
            deployment.llm_active[selected.id] = 1
        if not self._place_tools(deployment, replicas=1, offset=tool_offset, stagger=True):
            return False
        return self._register(deployment, f"{self.PLACEMENT_PREFIX}:{tier}") == "added"

    def _empty_with_tools(self, replicas: int = 1) -> DeploymentDecision:
        deployment = self.planner.empty_deployment()
        if replicas > 0:
            self._place_tools(deployment, replicas=replicas)
        return deployment

    def _activate_cheapest(self, deployment: DeploymentDecision, model: str) -> bool:
        for candidate in self._candidates_by_cost(model):
            if self.planner.feasible_activation(deployment, candidate.id):
                deployment.llm_active[candidate.id] = 1
                return True
        return False

    def _greedy_activate(
        self,
        deployment: DeploymentDecision,
        order: Sequence[CandidateInstance],
        budget: float | None = None,
    ) -> None:
        for candidate in order:
            config = self.scenario.llm_configs[candidate.config]
            if budget is not None and self._total_gpu(deployment) + config.gpu_count > budget + 1e-9:
                continue
            if self.planner.feasible_activation(deployment, candidate.id):
                deployment.llm_active[candidate.id] = 1

    def _place_tools(
        self,
        deployment: DeploymentDecision,
        replicas: int,
        spread: int | None = None,
        offset: int = 0,
        stagger: bool = False,
    ) -> bool:
        """Place ``replicas`` copies of every service, fastest server first.

        ``spread`` restricts each service to its own ``spread`` fastest servers, so
        the replicas of one service sit on a small number of machines. ``offset``
        rotates that preference order; with ``stagger`` the rotation is stepped per
        service (the placement-rotation stratum) instead of every service preferring
        the same machine.
        """
        for tool_index, tool_id in enumerate(sorted(self.scenario.tools)):
            rate_order = self._servers_by_tool_rate(tool_id)
            pool = rate_order[:spread] if spread is not None else rate_order
            if not pool:
                return False
            targets = _rotate(pool, offset + (tool_index if stagger else 0))
            for replica in range(replicas):
                preferred = targets[replica % len(targets)]
                order = [preferred] + [
                    server_id for server_id in rate_order if server_id != preferred
                ]
                if not self._place_tool_replica(deployment, tool_id, order):
                    return False
        return True

    def _place_tool_replica(
        self, deployment: DeploymentDecision, tool_id: str, order: Sequence[str]
    ) -> bool:
        for server_id in order:
            if self.planner.feasible_tool_replica(deployment, tool_id, server_id):
                deployment.tool_replicas[(tool_id, server_id)] += 1
                return True
        return False

    def _register(self, deployment: DeploymentDecision, stratum: str) -> str:
        """Register a deployment; returns ``added``, ``infeasible`` or ``duplicate``."""
        if not self.planner.deployment_feasible(deployment):
            return "infeasible"
        signature = deployment_signature(deployment.llm_active, deployment.tool_replicas)
        if signature in self._registry:
            return "duplicate"
        self._registry[signature] = self._make_entry(
            deployment, stratum, len(self._registry)
        )
        return "added"

    def _make_entry(
        self, deployment: DeploymentDecision, stratum: str, index: int
    ) -> DeploymentEntry:
        models = {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in deployment.llm_active.items()
            if active
        }
        n_llm = sum(1 for value in deployment.llm_active.values() if value)
        n_tool_replicas = sum(deployment.tool_replicas.values())
        return DeploymentEntry(
            index=index,
            stratum=stratum,
            llm_active=dict(deployment.llm_active),
            tool_replicas=dict(deployment.tool_replicas),
            n_models=len(models),
            n_llm=n_llm,
            n_tool_replicas=n_tool_replicas,
            total_gpu=self._total_gpu(deployment),
            cost_per_period=self._cost_per_period(deployment),
            signature=deployment_signature(
                deployment.llm_active, deployment.tool_replicas
            ),
        )

    def _total_gpu(self, deployment: DeploymentDecision) -> int:
        total = 0
        for candidate_id, active in deployment.llm_active.items():
            if not active:
                continue
            config = self.scenario.llm_configs[self.scenario.candidates[candidate_id].config]
            total += int(config.gpu_count)
        return total

    def _cost_per_period(self, deployment: DeploymentDecision) -> float:
        period = self.scenario.simulation.orchestration_period_s
        cost = 0.0
        for candidate_id, active in deployment.llm_active.items():
            if not active:
                continue
            config = self.scenario.llm_configs[self.scenario.candidates[candidate_id].config]
            cost += config.running_cost_per_slot * period
        for (tool_id, _server_id), replicas in deployment.tool_replicas.items():
            cost += replicas * self.scenario.tools[tool_id].running_cost_per_slot * period
        return cost

    def _candidate_gpu_total(self) -> int:
        return sum(
            int(self.scenario.llm_configs[candidate.config].gpu_count)
            for candidate in self.scenario.candidates.values()
        )

    def _candidate_list(self) -> list[CandidateInstance]:
        return sorted(
            self.scenario.candidates.values(),
            key=lambda item: (item.model, item.server, item.config, item.id),
        )

    def _candidates_by_flops(self, model: str | None = None) -> list[CandidateInstance]:
        candidates = [
            candidate
            for candidate in self.scenario.candidates.values()
            if model is None or candidate.model == model
        ]
        return sorted(
            candidates,
            key=lambda item: (
                -self.scenario.llm_configs[item.config].effective_flops,
                self.scenario.llm_configs[item.config].running_cost_per_slot,
                item.server,
                item.config,
                item.id,
            ),
        )

    def _candidates_by_cost(self, model: str | None = None) -> list[CandidateInstance]:
        candidates = [
            candidate
            for candidate in self.scenario.candidates.values()
            if model is None or candidate.model == model
        ]
        return sorted(
            candidates,
            key=lambda item: (
                self.scenario.llm_configs[item.config].running_cost_per_slot,
                item.server,
                item.config,
                item.id,
            ),
        )

    def _placement_order(self, tier: str) -> list[CandidateInstance]:
        by_cost = self._candidates_by_cost()
        if tier == "cheap":
            return by_cost
        if tier == "expensive":
            return list(reversed(by_cost))
        if tier != "balanced":
            raise ValueError(f"Unknown placement tier {tier}")
        interleaved: list[CandidateInstance] = []
        low, high = 0, len(by_cost) - 1
        while low <= high:
            interleaved.append(by_cost[low])
            if low != high:
                interleaved.append(by_cost[high])
            low += 1
            high -= 1
        return interleaved

    def _servers_by_tool_rate(self, tool_id: str) -> list[str]:
        """Servers ordered by the service rate of ``tool_id`` (fastest first)."""
        tool = self.scenario.tools[tool_id]
        return sorted(
            self.scenario.servers,
            key=lambda server_id: (-tool.service_rate[server_id], server_id),
        )


def _rotate(values: Sequence[Any], offset: int) -> list[Any]:
    if not values:
        return []
    pivot = offset % len(values)
    return list(values[pivot:]) + list(values[:pivot])


def _strata_counts(entries: Iterable[DeploymentEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
    return counts


def _active_model_sets(
    entries: Iterable[DeploymentEntry], candidate_models: dict[str, str]
) -> set[tuple[str, ...]]:
    model_sets: set[tuple[str, ...]] = set()
    for entry in entries:
        model_sets.add(
            tuple(
                sorted(
                    {
                        candidate_models[candidate_id]
                        for candidate_id, active in entry.llm_active.items()
                        if active and candidate_id in candidate_models
                    }
                )
            )
        )
    return model_sets


COVERAGE_COLUMNS = (
    "index",
    "stratum",
    "n_models",
    "active_models",
    "n_llm",
    "n_tool_replicas",
    "total_gpu",
    "cost_per_period",
)


def write_coverage_csv(library: DeploymentLibrary, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COVERAGE_COLUMNS))
        writer.writeheader()
        for row in library.coverage_rows():
            writer.writerow(row)

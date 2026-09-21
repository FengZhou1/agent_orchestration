from __future__ import annotations

import json
import math

import pytest

from agent_orch.deployment import DeploymentLibrary, DeploymentLibraryBuilder
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec, ReferenceScales
from agent_orch.routing import CompositionSolution, CompositionSolver


def _active_models(scenario, deployment) -> set[str]:
    return {
        scenario.candidates[candidate_id].model
        for candidate_id, active in deployment.llm_active.items()
        if active
    }


@pytest.fixture(scope="module")
def library(scenario) -> DeploymentLibrary:
    return DeploymentLibraryBuilder(scenario).build()


@pytest.fixture(scope="module")
def evaluator(scenario, library) -> ObjectiveEvaluator:
    spec = ObjectiveSpec.slo_constrained(0.9)
    return ObjectiveEvaluator(
        scenario, spec, ReferenceScales.from_scenario(scenario, spec, library)
    )


@pytest.fixture(scope="module")
def solver(scenario, evaluator) -> CompositionSolver:
    return CompositionSolver(scenario, evaluator, mapping_samples=64, seed=0)


@pytest.fixture(scope="module")
def arrival_rates(scenario) -> dict[tuple[str, str], float]:
    return {
        (app.id, ingress): rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }


@pytest.fixture(scope="module")
def multi_model_entry(library, scenario) -> int:
    """Index of the first library deployment that activates more than one model."""

    for entry in library.entries:
        if len(entry.active_models(scenario)) > 1:
            return entry.index
    raise AssertionError("The toy library has no multi-model deployment")


def test_canonical_candidates_are_complete_and_normalised(solver, library, scenario):
    deployment = library.to_deployment(0)
    active = _active_models(scenario, deployment)
    candidates = solver.canonical_candidates(deployment)

    assert [candidate.name for candidate in candidates] == list(solver.CANONICAL_NAMES)
    for candidate in candidates:
        assert candidate.share, candidate.name
        assert set(candidate.share) == {
            (app.id, ingress, model)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
            for model in scenario.models
        }
        for model in scenario.models:
            if model not in active:
                for app in scenario.applications.values():
                    for ingress in app.ingress_rates:
                        assert candidate.share[(app.id, ingress, model)] == 0.0
        totals: dict[tuple[str, str], float] = {}
        for (app_id, ingress, _model), share in candidate.share.items():
            assert share >= 0.0
            totals[(app_id, ingress)] = totals.get((app_id, ingress), 0.0) + share
        for group, total in totals.items():
            assert total == pytest.approx(1.0, abs=1e-9), (candidate.name, group)


def test_canonical_candidates_skip_groups_without_active_models(scenario, evaluator):
    solver = CompositionSolver(scenario, evaluator, mapping_samples=32)
    empty = DeploymentLibraryBuilder(scenario).planner.empty_deployment()
    for candidate in solver.canonical_candidates(empty):
        assert all(share == 0.0 for share in candidate.share.values())


def test_quality_greedy_picks_the_best_quality_model(
    solver, library, scenario, multi_model_entry
):
    deployment = library.to_deployment(multi_model_entry)
    active = _active_models(scenario, deployment)
    assert len(active) > 1
    candidate = next(
        item for item in solver.canonical_candidates(deployment) if item.name == "quality_greedy"
    )
    for app in scenario.applications.values():
        best = min(active, key=lambda model: (-app.quality.get(model, 0.0), model))
        for ingress in app.ingress_rates:
            for model in scenario.models:
                expected = 1.0 if model == best else 0.0
                assert candidate.share[(app.id, ingress, model)] == pytest.approx(expected)


def test_latency_greedy_assigns_every_group_to_one_active_model(
    solver, library, multi_model_entry, scenario
):
    deployment = library.to_deployment(multi_model_entry)
    candidate = next(
        item for item in solver.canonical_candidates(deployment) if item.name == "latency_greedy"
    )
    for app in scenario.applications.values():
        for ingress in app.ingress_rates:
            shares = {
                model: candidate.share[(app.id, ingress, model)] for model in scenario.models
            }
            assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
            assert sum(1.0 for share in shares.values() if share > 0.0) == 1


def test_quality_uniform_mix_is_the_half_and_half_blend(solver, library):
    deployment = library.to_deployment(0)
    by_name = {item.name: item for item in solver.canonical_candidates(deployment)}
    for key, share in by_name["quality_uniform_mix"].share.items():
        expected = 0.5 * by_name["uniform"].share[key] + 0.5 * by_name["quality_greedy"].share[key]
        assert share == pytest.approx(expected)


def test_evaluate_share_is_deterministic(solver, library, arrival_rates):
    deployment = library.to_deployment(0)
    share = solver.uniform_share(deployment)
    first = solver.evaluate_share(deployment, share, arrival_rates)
    second = solver.evaluate_share(deployment, share, arrival_rates)
    assert first == second
    assert math.isfinite(first)


def test_solve_beats_every_canonical_candidate(solver, library, multi_model_entry, arrival_rates):
    deployment = library.to_deployment(multi_model_entry)
    solution = solver.solve(deployment, arrival_rates, budget=32)
    assert solution.candidate_utilities
    assert solution.utility >= max(solution.candidate_utilities.values()) - 1e-12
    assert solution.utility >= solution.candidate_utilities["uniform"] - 1e-12


def test_solve_respects_the_budget(solver, library, multi_model_entry, arrival_rates):
    deployment = library.to_deployment(multi_model_entry)
    for budget in (1, 6, 12):
        solution = solver.solve(deployment, arrival_rates, budget=budget)
        assert solution.evaluations <= budget
        assert solution.evaluations >= 1
    with pytest.raises(ValueError):
        solver.solve(deployment, arrival_rates, budget=0)


def test_solve_with_a_small_budget_still_beats_uniform(solver, library, arrival_rates):
    """A budget of six spends every evaluation on the canonical candidates."""

    for entry in library.entries[::3]:
        deployment = entry.to_deployment()
        solution = solver.solve(deployment, arrival_rates, budget=6)
        assert solution.evaluations == 6
        assert solution.utility >= solution.candidate_utilities["uniform"] - 1e-12


def test_solve_is_deterministic(solver, library, multi_model_entry, arrival_rates):
    deployment = library.to_deployment(multi_model_entry)
    first = solver.solve(deployment, arrival_rates, budget=16)
    second = solver.solve(deployment, arrival_rates, budget=16)
    assert first == second
    assert first.source == second.source


def test_solve_source_names_the_winning_candidate(solver, library, arrival_rates):
    for entry in library.entries[::5]:
        solution = solver.solve(entry.to_deployment(), arrival_rates, budget=8)
        assert solution.source in set(solver.CANONICAL_NAMES) | {"greedy"}
        if solution.source != "greedy":
            assert solution.candidate_utilities[solution.source] == pytest.approx(
                solution.utility
            )


def test_solution_json_round_trip(solver, library, multi_model_entry, arrival_rates):
    solution = solver.solve(library.to_deployment(multi_model_entry), arrival_rates, budget=16)
    payload = json.loads(json.dumps(solution.to_json()))
    restored = CompositionSolution.from_json(payload)
    assert restored == solution
    assert set(restored.model_share) == set(solution.model_share)
    for key in restored.model_share:
        assert isinstance(key, tuple) and len(key) == 3
    assert all(isinstance(key, str) for key in payload["model_share"])


def test_train_test_split_partitions_the_library(library):
    train, test = library.train_test_split(test_fraction=0.25, seed=2026)

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(library)
    train_indices = {entry.index for entry in train.entries}
    test_indices = {entry.index for entry in test.entries}
    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == {entry.index for entry in library.entries}

    counts = dict(library.strata_counts())
    for name, count in train.strata_counts().items():
        assert test.strata_counts().get(name, 0) + count == counts[name]
    assert set(train.strata_counts()) | set(test.strata_counts()) == set(counts)
    for entry in train.entries:
        assert entry.stratum in train.strata_counts()

    again = library.train_test_split(test_fraction=0.25, seed=2026)
    assert [entry.index for entry in again[0].entries] == sorted(train_indices)
    assert [entry.index for entry in again[1].entries] == sorted(test_indices)


def test_train_test_split_keeps_a_stratum_in_train(library):
    train, test = library.train_test_split(test_fraction=0.9, seed=7)
    for name, count in library.strata_counts().items():
        if count >= 2:
            assert train.strata_counts().get(name, 0) >= 1, name
        else:
            assert train.strata_counts().get(name, 0) == count
    assert len(train) > 0 and len(test) > 0


def test_train_test_split_without_stratification(library):
    train, test = library.train_test_split(test_fraction=0.25, seed=11, stratify=False)
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(library)
    assert train.metadata["split_stratified"] is False
    assert library.train_test_split(stratify=False, seed=11)[0].entries == train.entries
    with pytest.raises(ValueError):
        library.train_test_split(test_fraction=1.5)


def test_subset_keeps_the_source_indices_and_metadata(library):
    indices = [entry.index for entry in library.entries[::3]]
    subset = library.subset(indices)

    assert [entry.index for entry in subset.entries] == indices
    assert len(subset) == len(indices)
    assert subset.scenario_id == library.scenario_id
    assert subset.scenario_hash == library.scenario_hash
    assert subset.metadata["subset_size"] == len(indices)
    assert "subset_of" in subset.metadata
    for index in indices:
        assert subset.entry(index) == library.entry(index)
        assert subset.to_deployment(index) == library.to_deployment(index)
    assert subset.strata_counts()
    assert set(subset.strata_counts()) <= set(library.strata_counts())

    train, test = library.train_test_split(test_fraction=0.25, seed=3)
    for subset in (train, test):
        for index in (entry.index for entry in subset.entries):
            assert subset.entry(index).signature == library.entry(index).signature
    assert train.metadata["subset_size"] == len(train)
    assert test.metadata["split_role"] == "test"


def test_n_active_models_alias(library, scenario):
    for entry in library.entries:
        assert entry.n_active_models == entry.n_models
        assert entry.n_active_models == len(entry.active_models(scenario))

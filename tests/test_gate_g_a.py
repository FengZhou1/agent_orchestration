from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from agent_orch.deployment import DeploymentLibraryBuilder
from agent_orch.envs import CompositionLibraryEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_composition_response.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_module", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_gate_module"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scenario():
    return ScenarioLoader.load("configs/toy.yaml")


@pytest.fixture(scope="module")
def library(scenario):
    return DeploymentLibraryBuilder(scenario).build(min_entries=8, max_entries=32)


@pytest.fixture(scope="module")
def env(scenario, library):
    return CompositionLibraryEnv(
        scenario,
        max_slots=2,
        seed=0,
        arrival_trace=ArrivalTrace.stationary_poisson_intensity(scenario, 2, rate_scale=1.0),
        mapping_samples=8,
        objective=ObjectiveSpec.slo_constrained(0.9),
        library=library,
    )


def test_spearman_is_one_for_a_monotone_relation(gate):
    left = np.array([1.0, 2.0, 3.0, 4.0])
    assert gate._spearman(left, 2.0 * left + 5.0) == pytest.approx(1.0)
    assert gate._spearman(left, -left) == pytest.approx(-1.0)


def test_spearman_matches_the_manual_formula_on_distinct_ranks(gate):
    left = np.array([3.0, 1.0, 4.0, 2.0])
    right = np.array([2.0, 1.0, 4.0, 3.0])
    # ranks: left -> [3, 1, 4, 2], right -> [2, 1, 4, 3]; squared rank gaps = 1 + 0 + 0 + 1
    expected = 1.0 - 6.0 * 2.0 / (4 * (16 - 1))
    assert gate._spearman(left, right) == pytest.approx(expected)


def test_spearman_handles_ties_and_degenerate_input(gate):
    left = np.array([1.0, 1.0, 2.0, 2.0])
    right = np.array([1.0, 1.0, 1.0, 2.0])
    assert -1.0 <= gate._spearman(left, right) <= 1.0
    assert np.isnan(gate._spearman(np.array([1.0]), np.array([1.0])))
    assert np.isnan(gate._spearman(np.array([1.0, 1.0]), np.array([1.0, 1.0])))


def test_share_from_json_preserves_model_names_containing_delimiters(gate, scenario):
    payload = {"app-a|ingress-x|model|with|pipes": 0.25, "app-a|ingress-x|plain": 0.75}
    share = gate._share_from_json(scenario, payload)
    assert share[("app-a", "ingress-x", "model|with|pipes")] == pytest.approx(0.25)
    assert share[("app-a", "ingress-x", "plain")] == pytest.approx(0.75)


def test_constant_action_places_the_share_in_the_action_layout(gate, env):
    app_id, ingress = env.layout.model_groups[0]
    first_model = env.layout.models[0]
    action = gate._constant_action(env, {(app_id, ingress, first_model): 1.0})(None)
    dense = np.asarray(action["model"]).reshape(
        len(env.layout.model_groups), len(env.layout.models)
    )
    assert dense[0, 0] == pytest.approx(1.0)
    assert np.all(dense.sum(axis=1) <= 1.0 + 1.0e-6)


def _one_hot_reference(scenario, library, index: int = 0) -> dict[int, dict]:
    entries: dict[int, dict] = {}
    for entry in library.entries:
        share: dict[str, float] = {}
        for app in scenario.applications.values():
            for ingress in app.ingress_rates:
                for model_index, model in enumerate(scenario.models):
                    share[f"{app.id}|{ingress}|{model}"] = 1.0 if model_index == index else 0.0
        entries[entry.index] = {"model_share": share}
    return entries


def _uniform_reference(scenario, library) -> dict[int, dict]:
    entries: dict[int, dict] = {}
    width = max(1, len(scenario.models))
    for entry in library.entries:
        share: dict[str, float] = {}
        for app in scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in scenario.models:
                    share[f"{app.id}|{ingress}|{model}"] = 1.0 / width
        entries[entry.index] = {"model_share": share}
    return entries


def test_dirichlet_ceiling_follows_the_concentration_clamp(gate, scenario, library):
    entries = _uniform_reference(scenario, library)
    positions = list(range(len(library.entries)))
    ceiling = gate._dirichlet_ceiling(44.0, scenario, library, positions, entries)
    # C_hi = max(100, 2 * 44) = 100 and C_lo = 44, so two active models reach 100 / 144.
    # The reported value is rounded for display.
    assert ceiling["mean_ceiling_by_group_size"]["2"] == pytest.approx(100.0 / 144.0, abs=1.0e-6)
    assert ceiling["groups_above_ceiling"] == 0.0
    assert ceiling["worst_excess_over_ceiling"] == 0.0


def test_dirichlet_ceiling_flags_one_hot_targets_as_unreachable(gate, scenario, library):
    """A one-hot solver target always exceeds the reachable mean share."""

    entries = _one_hot_reference(scenario, library)
    positions = list(range(len(library.entries)))
    ceiling = gate._dirichlet_ceiling(1.0, scenario, library, positions, entries)
    assert ceiling["groups_above_ceiling"] > 0.0
    assert ceiling["worst_excess_over_ceiling"] > 0.0
    assert ceiling["groups_above_ceiling"] <= ceiling["groups_checked"]

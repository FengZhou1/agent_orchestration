from __future__ import annotations

import csv
import json

import pytest

from agent_orch.capacity import CapacityPlanner
from agent_orch.deployment import (
    DeploymentLibrary,
    DeploymentLibraryBuilder,
    FixedDeploymentSampler,
    StratifiedSampler,
    write_coverage_csv,
)


@pytest.fixture(scope="module")
def library(scenario) -> DeploymentLibrary:
    return DeploymentLibraryBuilder(scenario).build()


@pytest.fixture(scope="module")
def planner(scenario) -> CapacityPlanner:
    return CapacityPlanner(scenario)


def test_library_entries_are_feasible_and_distinct(library, scenario, planner):
    assert len(library.entries) >= 1
    assert library.scenario_id == scenario.id
    assert library.scenario_hash == DeploymentLibrary.scenario_hash_of(scenario)
    signatures = {entry.signature for entry in library.entries}
    assert len(signatures) == len(library.entries)
    assert [entry.index for entry in library.entries] == list(range(len(library)))
    for entry in library.entries:
        assert planner.deployment_feasible(entry.to_deployment())
        assert entry.n_llm == sum(1 for value in entry.llm_active.values() if value)
        assert entry.n_tool_replicas == sum(entry.tool_replicas.values())
        assert entry.n_models == len(entry.active_models(scenario))
        assert entry.cost_per_slot > 0.0
        for (tool_id, _server_id), replicas in entry.tool_replicas.items():
            assert 0 <= replicas <= scenario.simulation.max_tool_replicas_per_server
            assert tool_id in scenario.tools
        expected_gpu = sum(
            scenario.llm_configs[scenario.candidates[candidate_id].config].gpu_count
            for candidate_id, active in entry.llm_active.items()
            if active
        )
        assert entry.total_gpu == expected_gpu


def test_library_covers_several_active_model_sets(library, scenario):
    model_sets = library.active_model_sets(scenario)
    assert len(model_sets) > 1
    assert model_sets == library.active_model_sets()
    full_set = tuple(sorted(scenario.models))
    assert full_set in model_sets
    for model_set in model_sets:
        assert model_set
        assert set(model_set) <= set(scenario.models)


def test_model_subset_stratum_never_uses_the_full_model_set(library, scenario):
    subsets = {
        stratum: entries
        for stratum, entries in library.by_stratum().items()
        if stratum.startswith(DeploymentLibraryBuilder.SUBSET_PREFIX)
    }
    assert subsets
    for stratum, entries in subsets.items():
        declared = tuple(stratum.split(":", 1)[1].split("+"))
        assert len(declared) < len(scenario.models)
        for entry in entries:
            assert entry.active_models(scenario) == declared
            assert entry.n_models == len(declared)


def test_strata_cover_the_documented_families(library):
    counts = library.strata_counts()
    assert sum(counts.values()) == len(library.entries)
    assert counts[DeploymentLibraryBuilder.EXTREME_STRATUM] == 4
    assert any(name.startswith(DeploymentLibraryBuilder.CAPACITY_PREFIX) for name in counts)
    assert any(name.startswith(DeploymentLibraryBuilder.REPLICA_PREFIX) for name in counts)
    assert any(name.startswith(DeploymentLibraryBuilder.PLACEMENT_PREFIX) for name in counts)
    replicas = {
        int(name.split(":", 1)[1])
        for name in counts
        if name.startswith(DeploymentLibraryBuilder.REPLICA_PREFIX)
    }
    # A replica level can be absorbed by a higher-priority stratum with the same
    # signature (toy: one replica per service equals the cheapest L5 deployment).
    assert replicas and replicas <= {1, 2, 4}
    assert "shortfall_reason" in library.metadata


def test_save_and_load_round_trip(tmp_path, library, scenario):
    path = tmp_path / DeploymentLibrary.default_path(scenario.id).name
    library.save(path)
    restored = DeploymentLibrary.load(path)

    assert restored.scenario_id == library.scenario_id
    assert restored.scenario_hash == library.scenario_hash
    assert restored.strata_counts() == library.strata_counts()
    assert restored.metadata == library.metadata
    assert len(restored.entries) == len(library.entries)
    for original, reloaded in zip(library.entries, restored.entries):
        assert reloaded.index == original.index
        assert reloaded.stratum == original.stratum
        assert reloaded.llm_active == original.llm_active
        assert reloaded.tool_replicas == original.tool_replicas
        assert reloaded.signature == original.signature
        assert reloaded.cost_per_slot == pytest.approx(original.cost_per_slot)
        assert reloaded.active_models(scenario) == original.active_models(scenario)
        for key in reloaded.tool_replicas:
            assert isinstance(key, tuple) and len(key) == 2
    assert restored.active_model_sets() == library.active_model_sets()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert any("|" in key for key in payload["entries"][0]["tool_replicas"])
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        DeploymentLibrary.load(path)


def test_to_deployment_matches_the_planner_layout(library, planner):
    empty = planner.empty_deployment()
    for entry in library.entries:
        deployment = library.to_deployment(entry.index)
        assert set(deployment.llm_active) == set(empty.llm_active)
        assert set(deployment.tool_replicas) == set(empty.tool_replicas)
        assert deployment.llm_active == entry.llm_active
        assert deployment.tool_replicas == entry.tool_replicas
    with pytest.raises(IndexError):
        library.to_deployment(len(library.entries))


def test_max_entries_trims_the_tail(scenario):
    trimmed = DeploymentLibraryBuilder(scenario).build(min_entries=8, max_entries=12)
    assert 8 <= len(trimmed.entries) <= 12
    assert [entry.index for entry in trimmed.entries] == list(range(len(trimmed.entries)))


def test_cycle_sampler_balances_the_strata(library):
    sampler = StratifiedSampler(library, seed=7, mode="cycle")
    n_strata = len(library.strata_counts())
    assert len(sampler) == len(library.entries)
    for _ in range(3 * n_strata):
        sampler.next_index()
    counts = sampler.strata_coverage_counts()
    assert set(counts) == set(library.strata_counts())
    assert sum(counts.values()) == 3 * n_strata
    for stratum, count in counts.items():
        assert 1 <= count <= 3, stratum

    sampler.reset(seed=7)
    first = [sampler.next_index() for _ in range(4 * n_strata)]
    sampler.reset(seed=7)
    second = [sampler.next_index() for _ in range(4 * n_strata)]
    assert first == second


def test_cycle_sampler_weights_raise_a_stratum_quota(library):
    stratum = next(iter(library.strata_counts()))
    sampler = StratifiedSampler(library, seed=1, weights={stratum: 2.0})
    plan_size = len(library.strata_counts()) + 1
    for _ in range(2 * plan_size):
        sampler.next_entry()
    counts = sampler.strata_coverage_counts()
    assert counts[stratum] == 4
    for name, count in counts.items():
        if name != stratum:
            assert count == 2, name


def test_uniform_sampler_stays_inside_the_library(library):
    sampler = StratifiedSampler(library, seed=3, mode="uniform")
    indexes = [sampler.next_index() for _ in range(200)]
    assert all(0 <= index < len(library.entries) for index in indexes)
    assert len(set(indexes)) > 1

    weighted = StratifiedSampler(library, seed=3, mode="uniform", weights={"L5_extreme": 1.0})
    for _ in range(50):
        entry = weighted.next_entry()
        assert entry in library.entries


def test_fixed_modes_return_one_deployment(library):
    sampler = StratifiedSampler(library, mode="fixed")
    assert {sampler.next_index() for _ in range(5)} == {0}
    assert sampler.next_entry() == library.entry(0)

    stratum = next(iter(library.strata_counts()))
    scoped = StratifiedSampler(library, mode="fixed", stratum=stratum)
    expected = library.by_stratum()[stratum][0]
    assert scoped.next_entry() == expected
    assert scoped.next_deployment().llm_active == expected.llm_active

    fixed = FixedDeploymentSampler(library, index=len(library.entries) - 1)
    assert {fixed.next_index() for _ in range(3)} == {len(library.entries) - 1}
    assert fixed.next_entry() == library.entry(len(library.entries) - 1)
    assert len(fixed) == len(library.entries)

    with pytest.raises(ValueError):
        StratifiedSampler(library, mode="nonsense")
    with pytest.raises(ValueError):
        StratifiedSampler(library, stratum="L9_missing")


def test_coverage_rows_and_csv(tmp_path, library, scenario):
    rows = library.coverage_rows(scenario)
    assert len(rows) == len(library.entries)
    expected_fields = {
        "index",
        "stratum",
        "n_models",
        "active_models",
        "n_llm",
        "n_tool_replicas",
        "total_gpu",
        "cost_per_slot",
    }
    for row in rows:
        assert set(row) == expected_fields
        assert row["cost_per_slot"] > 0.0
        assert row["total_gpu"] >= 0
        assert row["active_models"]
    assert library.coverage_rows() == rows

    path = tmp_path / "coverage.csv"
    write_coverage_csv(library, path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert list(written[0]) == list(rows[0])
    assert float(written[0]["cost_per_slot"]) == pytest.approx(rows[0]["cost_per_slot"])


def test_default_path_uses_the_scenario_id(scenario):
    path = DeploymentLibrary.default_path(scenario.id)
    assert path.name == f"deployment_library_{scenario.id}.json"
    assert DeploymentLibrary.default_path(scenario.id, base="tmp/x").parent.name == "x"
